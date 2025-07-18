#!/usr/bin/env python3
"""
Call Transcriber - Audio transcription with speaker diarization for call centers
Transcribes MP3 files and separates agent/client speech for AI processing
"""

import os
import sys
import json
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv

import click
import whisper
import torch
import spacy
import torchaudio
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from pyannote.audio import Pipeline
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from audio_enhancer import AudioEnhancer

console = Console()
load_dotenv()
nlp = spacy.load("es_core_news_sm")

class CallTranscriber:
    def __init__(self, whisper_model: str = "./whisper_model", device: str = "auto", use_pyannote: bool = False):
        """Initialize the transcriber with models"""
        self.device = self._get_device(device)
        self.use_pyannote = use_pyannote

        console.print(f"[yellow]Loading fine-tuned Whisper model '{whisper_model}' on {self.device}...[/yellow]")
        self.processor = WhisperProcessor.from_pretrained(whisper_model)
        self.model = WhisperForConditionalGeneration.from_pretrained(whisper_model).to(self.device)

        # Initialize pyannote speaker diarization if requested
        if use_pyannote:
            console.print("[yellow]Loading pyannote speaker diarization pipeline...[/yellow]")
            try:
                # You'll need a HuggingFace token: https://huggingface.co/pyannote/speaker-diarization
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=os.getenv('HUGGINGFACE_TOKEN')  # Ensure you set this env variable
                )
                if self.device != "cpu":
                    self.diarization_pipeline.to(torch.device(self.device))
                console.print("[green]✅ Pyannote diarization loaded successfully[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not load pyannote: {e}[/yellow]")
                console.print("[yellow]Falling back to enhanced heuristic speaker separation[/yellow]")
                self.diarization_pipeline = None
        else:
            console.print("[yellow]Using enhanced heuristic speaker separation[/yellow]")
            self.diarization_pipeline = None

    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _split_into_sentences(self, text):
        """Splits text into sentences using spaCy"""
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def _split_audio(self, audio, sample_rate, chunk_length_s, overlap_s):
        chunk_size = int(chunk_length_s * sample_rate)
        step_size = int((chunk_length_s - overlap_s) * sample_rate)
        chunks = []

        for start in range(0, len(audio), step_size):
            end = start + chunk_size
            chunk = audio[start:end]

            # Skip too-short segments (final edge case)
            if len(chunk) < sample_rate * 2:  # 2 seconds
                break

            chunks.append((start / sample_rate, end / sample_rate, chunk))

        return chunks

    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio in chunks and split into sentence-level segments with timestamps."""
        console.print("[cyan]Transcribing with fine-tuned Whisper (sentence-level)...[/cyan]")

        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
            sample_rate = 16000

        # Convert to numpy
        waveform = waveform.squeeze().detach().cpu().numpy()

        # Split into overlapping chunks using your working logic
        chunks = self._split_audio(waveform, sample_rate, chunk_length_s=10, overlap_s=2)

        segments = []
        full_text = []

        for i, (chunk_start_time, chunk_end_time, chunk) in enumerate(chunks):
            print(f"🔊 Transcribing chunk {i + 1}/{len(chunks)} [{chunk_start_time:.2f}s → {chunk_end_time:.2f}s]")

            inputs = self.processor(
                chunk,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True
            )
            input_features = inputs.input_features.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)

            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    attention_mask=attention_mask,
                )
                chunk_text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

            if not chunk_text.strip():
                chunk_text = "[inaudible]"

            sentences = self._split_into_sentences(chunk_text)
            if not sentences:
                sentences = [chunk_text]

            total_chars = sum(len(s) for s in sentences)
            if total_chars == 0:
                total_chars = len(chunk_text)
                sentences = [chunk_text]

            sentence_start = chunk_start_time
            chunk_duration = chunk_end_time - chunk_start_time

            for sentence in sentences:
                proportion = len(sentence) / total_chars
                sentence_duration = proportion * chunk_duration
                sentence_end = sentence_start + sentence_duration

                segments.append({
                    "start": round(sentence_start, 2),
                    "end": round(sentence_end, 2),
                    "text": sentence.strip()
                })

                sentence_start = sentence_end

            full_text.append(chunk_text)

        total_duration = len(waveform) / 16000.0

        return {
            "text": " ".join(full_text),
            "segments": segments,
            "duration": total_duration
        }

    def pyannote_speaker_separation(self, audio_path: str) -> List[Dict]:
        """Diarize full audio, then transcribe each speaker segment separately with filtering."""
        console.print("[cyan]Running pyannote speaker diarization and direct transcription per segment...[/cyan]")

        try:
            # Load and prepare audio
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
                sample_rate = 16000
            waveform = waveform.squeeze().numpy()

            # Step 1: Diarize
            diarization = self.diarization_pipeline(audio_path)
            segments = []

            last_text = ""
            last_end = 0.0

            for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
                start = turn.start
                end = turn.end
                duration = end - start

                if duration < 0.7:
                    audio_chunk = waveform[int(start * sample_rate):int(end * sample_rate)]
                    energy = np.mean(np.abs(audio_chunk))
                    if energy < 0.01:
                        continue  # skip short + low energy
                else:
                    audio_chunk = waveform[int(start * sample_rate):int(end * sample_rate)]

                # Transcribe each speaker segment
                inputs = self.processor(audio_chunk, sampling_rate=16000, return_tensors="pt")
                input_features = inputs.input_features.to(self.device)

                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        input_features,
                        do_sample=False,
                        repetition_penalty=1.2
                    )
                    text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

                if not text:
                    continue

                # Deduplicate repeated fillers
                if (
                        text.lower() == last_text.lower() and
                        (start - last_end) < 1.0 and
                        text.lower() in ["gracias", "vale", "sí", "bueno"]
                ):
                    continue

                last_text = text
                last_end = end

                speaker_label = (
                    "AGENT" if speaker == "SPEAKER_00" else
                    "CLIENT" if speaker == "SPEAKER_01" else
                    speaker.upper()
                )

                segments.append({
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "text": text if text else "[inaudible]",
                    "speaker": speaker_label
                })

            console.print(
                f"[green]✅ Diarization and per-segment transcription complete ({len(segments)} segments)[/green]")
            return segments

        except Exception as e:
            console.print(f"[yellow]⚠️  Pyannote diarization failed: {e}[/yellow]")
            console.print("[yellow]Falling back to enhanced heuristic method[/yellow]")
            return self.enhanced_speaker_separation([], 0)



        except Exception as e:
            console.print(f"[yellow]⚠️  Pyannote diarization failed: {e}[/yellow]")
            console.print("[yellow]Falling back to enhanced heuristic method[/yellow]")
            return self.enhanced_speaker_separation([], 0)

    def enhanced_speaker_separation(self, segments: List[Dict], total_duration: float) -> List[Dict]:
        """
        Enhanced speaker separation using multiple heuristics
        Improved version of the basic method with better logic
        """
        console.print("[cyan]Performing enhanced heuristic speaker separation...[/cyan]")

        if not segments:
            console.print("[yellow]Warning: No segments to process[/yellow]")
            return []

        separated_segments = []
        current_speaker = "AGENT"  # Assume agent speaks first

        # Calculate segment statistics for better heuristics
        segment_durations = [seg['end'] - seg['start'] for seg in segments]
        avg_segment_duration = np.mean(segment_durations) if segment_durations else 0

        # Calculate pause durations
        pause_durations = []
        for i in range(1, len(segments)):
            pause = segments[i]['start'] - segments[i - 1]['end']
            pause_durations.append(pause)

        # Dynamic pause threshold based on conversation patterns
        if pause_durations:
            pause_threshold = np.percentile(pause_durations, 75)  # 75th percentile
            pause_threshold = max(1.0, min(pause_threshold, 3.0))  # Clamp between 1-3 seconds
        else:
            pause_threshold = 2.0

        console.print(f"[yellow]Using dynamic pause threshold: {pause_threshold:.1f}s[/yellow]")

        for i, segment in enumerate(segments):
            # Ensure segment has valid text
            segment_text = segment.get('text', '').strip()
            if not segment_text:
                continue

            # Enhanced speaker switching logic
            should_switch = False

            if i > 0:
                prev_end = segments[i - 1]['end']
                current_start = segment['start']
                pause_duration = current_start - prev_end

                # Multiple factors for speaker switching
                factors = []

                # 1. Pause duration (primary factor)
                if pause_duration > pause_threshold:
                    factors.append("long_pause")

                # 2. Segment length pattern (very short segments often indicate interruptions)
                current_duration = segment['end'] - segment['start']
                if i > 0:
                    prev_duration = segments[i - 1]['end'] - segments[i - 1]['start']
                    if current_duration < avg_segment_duration * 0.3 and prev_duration > avg_segment_duration:
                        factors.append("interruption_pattern")

                # 3. Text pattern analysis (basic)
                segment_lower = segment_text.lower()

                # Common agent phrases (Spanish and English)
                agent_phrases = [
                    "gracias por llamar", "mi nombre es", "¿en qué puedo ayudarle?", "¿cómo puedo ayudarle?",
                    "thank you for calling", "my name is", "how can I help", "how may I assist",
                    "servicio al cliente", "customer service", "un momento por favor", "one moment please"
                ]

                # Common client phrases
                client_phrases = [
                    "hola", "tengo un problema", "necesito ayuda", "no puedo", "no funciona",
                    "hello", "i have a problem", "i need help", "i can't", "it doesn't work",
                    "mi cuenta", "my account", "no entiendo", "i don't understand"
                ]

                if any(phrase in segment_lower for phrase in agent_phrases):
                    if current_speaker == "CLIENT":
                        factors.append("agent_phrase")
                elif any(phrase in segment_lower for phrase in client_phrases):
                    if current_speaker == "AGENT":
                        factors.append("client_phrase")

                # Decision logic: switch if we have evidence
                if "long_pause" in factors or len(factors) >= 2:
                    should_switch = True

            # Switch speaker if conditions are met
            if should_switch:
                current_speaker = "CLIENT" if current_speaker == "AGENT" else "AGENT"

            separated_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment_text,
                'speaker': current_speaker
            })

        # Post-processing: Fix obvious errors
        # If one speaker dominates too much (>90%), redistribute some segments
        agent_count = sum(1 for s in separated_segments if s['speaker'] == 'AGENT')
        total_count = len(separated_segments)

        if total_count > 0:
            agent_ratio = agent_count / total_count
            if agent_ratio > 0.9 or agent_ratio < 0.1:
                console.print("[yellow]Detected speaker imbalance, applying corrections...[/yellow]")
                # Simple correction: alternate every few segments in middle of conversation
                start_idx = total_count // 4
                end_idx = (3 * total_count) // 4
                for i in range(start_idx, end_idx, 3):  # Every 3rd segment
                    if i < len(separated_segments):
                        current_spk = separated_segments[i]['speaker']
                        separated_segments[i]['speaker'] = "CLIENT" if current_spk == "AGENT" else "AGENT"

        # If no valid segments were found, create a placeholder
        if not separated_segments:
            console.print("[yellow]Warning: No valid segments found, creating placeholder[/yellow]")
            separated_segments.append({
                'start': 0.0,
                'end': min(total_duration, 1.0),
                'text': "[No speech detected]",
                'speaker': "AGENT"
            })

        # Final statistics
        final_agent_count = sum(1 for s in separated_segments if s['speaker'] == 'AGENT')
        final_agent_ratio = final_agent_count / len(separated_segments) if separated_segments else 0

        console.print(
            f"[green]✅ Enhanced separation complete - {len(separated_segments)} segments (Agent: {final_agent_ratio:.1%})[/green]")
        return separated_segments

    def simple_speaker_separation(self, segments: List[Dict], total_duration: float) -> List[Dict]:
        """
        Simple speaker separation based on audio patterns
        This is a basic implementation - for production, use pyannote.audio with proper setup
        """
        console.print("[cyan]Performing basic speaker separation...[/cyan]")

        if not segments:
            console.print("[yellow]Warning: No segments to process[/yellow]")
            return []

        # Simple heuristic: alternate speakers based on pauses and segment patterns
        separated_segments = []
        current_speaker = "AGENT"  # Assume agent speaks first

        for i, segment in enumerate(segments):
            # Ensure segment has valid text
            segment_text = segment.get('text', '').strip()
            if not segment_text:
                continue

            # Switch speaker if there's a significant pause (>2 seconds) or change in speaking pattern
            if i > 0:
                prev_end = segments[i - 1]['end']
                current_start = segment['start']
                pause_duration = current_start - prev_end

                if pause_duration > 2.0:  # 2 second pause threshold
                    current_speaker = "CLIENT" if current_speaker == "AGENT" else "AGENT"

            separated_segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment_text,
                'speaker': current_speaker
            })

        # If no valid segments were found, create a placeholder
        if not separated_segments:
            console.print("[yellow]Warning: No valid segments found, creating placeholder[/yellow]")
            separated_segments.append({
                'start': 0.0,
                'end': min(total_duration, 1.0),
                'text': "[No speech detected]",
                'speaker': "AGENT"
            })

        console.print(f"[green]Processed {len(separated_segments)} speech segments[/green]")
        return separated_segments

    def format_for_ai(self, segments: List[Dict], metadata: Dict) -> Dict:
        """Format transcription for AI consumption"""

        # Group by speaker
        agent_parts = []
        client_parts = []
        conversation_flow = []

        for segment in segments:
            segment_data = {
                'timestamp': f"{segment['start']:.2f}s - {segment['end']:.2f}s",
                'text': segment['text'],
                'duration': segment['end'] - segment['start']
            }

            conversation_flow.append({
                'speaker': segment['speaker'],
                'timestamp': segment_data['timestamp'],
                'text': segment['text']
            })

            if segment['speaker'] == 'AGENT':
                agent_parts.append(segment_data)
            else:
                client_parts.append(segment_data)

        # Create summary statistics with safe division
        total_agent_time = sum(part['duration'] for part in agent_parts)
        total_client_time = sum(part['duration'] for part in client_parts)
        total_talk_time = total_agent_time + total_client_time
        total_duration = metadata.get('duration', 0)

        # Safe division with fallback values
        agent_talk_percentage = round((total_agent_time / total_duration) * 100, 1) if total_duration > 0 else 0
        client_talk_percentage = round((total_client_time / total_duration) * 100, 1) if total_duration > 0 else 0
        agent_dominance_ratio = round(total_agent_time / total_talk_time, 2) if total_talk_time > 0 else 0.5

        formatted_output = {
            'metadata': {
                'transcription_date': datetime.now().isoformat(),
                'total_duration': total_duration,
                'language': 'ca-es',
                'agent_talk_time': round(total_agent_time, 2),
                'client_talk_time': round(total_client_time, 2),
                'agent_talk_percentage': agent_talk_percentage,
                'client_talk_percentage': client_talk_percentage
            },
            'conversation_flow': conversation_flow,
            'ai_ready_format': {
                'call_summary': {
                    'agent_speech': ' '.join(part['text'] for part in agent_parts),
                    'client_speech': ' '.join(part['text'] for part in client_parts),
                    'key_metrics': {
                        'total_duration_seconds': total_duration,
                        'agent_dominance_ratio': agent_dominance_ratio,
                        'conversation_turns': len(conversation_flow)
                    }
                }
            }
        }

        return formatted_output

    def process_call(self, mp3_path: str, output_path: Optional[str] = None) -> str:
        """Main processing function — diarization-first when using pyannote"""

        if not os.path.exists(mp3_path):
            raise FileNotFoundError(f"Audio file not found: {mp3_path}")

        console.print(Panel(f"[bold blue]Processing Call: {mp3_path}[/bold blue]"))

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
        ) as progress:

            # Step 1: Preprocess audio
            task1 = progress.add_task("Preprocessing audio...", total=None)
            try:
                enhancer = AudioEnhancer()
                wav_path = enhancer.process_audio_file(
                    mp3_path,
                    enhance_speech=True,
                    reduce_noise=True,
                    apply_vad=False,
                    telephony_filter=True,
                    normalize_volume=True
                )
                progress.remove_task(task1)
            except Exception as e:
                progress.remove_task(task1)
                raise Exception(f"Failed to preprocess audio: {str(e)}")

            try:
                transcription_result = {}
                separated_segments = []
                duration = 0.0

                # Step 2: Diarization-first path
                if self.use_pyannote:
                    task2 = progress.add_task("Running diarization and transcription...", total=None)
                    try:
                        separated_segments = self.pyannote_speaker_separation(wav_path)

                        # Get duration from audio file
                        audio_info = torchaudio.info(wav_path)
                        duration = audio_info.num_frames / audio_info.sample_rate

                        progress.remove_task(task2)
                    except Exception as e:
                        progress.remove_task(task2)
                        raise Exception(f"Speaker diarization+transcription failed: {str(e)}")

                else:
                    # Whisper-first fallback
                    task2 = progress.add_task("Transcribing audio...", total=None)
                    try:
                        transcription_result = self.transcribe_audio(wav_path)
                        progress.remove_task(task2)

                        if not transcription_result or 'segments' not in transcription_result:
                            raise Exception("Transcription failed - no segments returned")

                    except Exception as e:
                        progress.remove_task(task2)
                        raise Exception(f"Transcription failed: {str(e)}")

                    # Step 3: Heuristic speaker separation
                    task3 = progress.add_task("Separating speakers...", total=None)
                    try:
                        separated_segments = self.enhanced_speaker_separation(
                            transcription_result['segments'],
                            transcription_result.get('duration', 0)
                        )
                        duration = transcription_result.get('duration', 0)
                        progress.remove_task(task3)
                    except Exception as e:
                        progress.remove_task(task3)
                        raise Exception(f"Speaker separation failed: {str(e)}")

                # Step 4: Format for AI
                task4 = progress.add_task("Formatting for AI consumption...", total=None)
                try:
                    formatted_result = self.format_for_ai(
                        separated_segments,
                        {
                            'duration': duration,
                            'language': transcription_result.get('language', 'es')  # fallback
                        }
                    )
                    progress.remove_task(task4)
                except Exception as e:
                    progress.remove_task(task4)
                    raise Exception(f"AI formatting failed: {str(e)}")

            finally:
                if os.path.exists(wav_path):
                    os.remove(wav_path)

        # Step 5: Save output
        if output_path is None:
            audio_dir = os.path.dirname(mp3_path)
            transcriptions_dir = os.path.join(audio_dir, "transcriptions")
            os.makedirs(transcriptions_dir, exist_ok=True)
            output_path = os.path.join(transcriptions_dir, f"{Path(mp3_path).stem}_transcription.json")

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise Exception(f"Failed to save output file: {str(e)}")

        console.print(f"[green]✅ Transcription saved to: {output_path}[/green]")
        return output_path


@click.command()
@click.argument('mp3_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output JSON file path')
@click.option('--model', '-m', default='whisper-model', help='Whisper model size (tiny, base, small, medium, large)')
@click.option('--device', '-d', default='auto', help='Device to use (auto, cpu, cuda, mps)')
@click.option('--preview', '-p', is_flag=True, help='Show preview of results in terminal')
@click.option('--use-pyannote', is_flag=True,
              help='Use advanced pyannote.audio speaker diarization (requires HuggingFace token)')
def main(mp3_file: str, output: str, model: str, device: str, preview: bool, use_pyannote: bool):
    """
    Call Transcriber - Transcribe MP3 call recordings with speaker separation

    MP3_FILE: Path to the MP3 audio file to transcribe

    Quality presets:
    - fast: Use tiny model, basic processing (fastest)
    - balanced: Use base/small model, enhanced processing (recommended)
    - high: Use medium/large model, all enhancements (most accurate)

    Examples:
    python3 main.py call.mp3 --language es --quality high
    python3 main.py call.mp3 --use-pyannote --enhanced
    python3 main.py call.mp3 --quality fast --preview
    """

    console.print(Panel.fit(
        "[bold blue]Call Transcriber[/bold blue]\n"
        "AI-Ready Call Transcription with Enhanced Speaker Separation",
        border_style="blue"
    ))

    # Pyannote.audio setup warning
    if use_pyannote:
        if not os.getenv('HUGGINGFACE_TOKEN'):
            console.print("[red]⚠️  Warning: HUGGINGFACE_TOKEN environment variable not set![/red]")
            console.print("[yellow]To use pyannote.audio, you need a HuggingFace token:[/yellow]")
            console.print("1. Get a token from https://huggingface.co/settings/tokens")
            console.print("2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1")
            console.print("3. Set: export HUGGINGFACE_TOKEN='your_token_here'")
            console.print("[cyan]Proceeding with enhanced heuristic method...[/cyan]")
            use_pyannote = False

    try:
        transcriber = CallTranscriber(
            whisper_model=model,
            device=device,
            use_pyannote=use_pyannote
        )
        output_file = transcriber.process_call(mp3_file, output)

        if preview:
            # Load and display preview
            with open(output_file, 'r', encoding='utf-8') as f:
                result = json.load(f)

            # Display summary table
            table = Table(title="Call Analysis Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            metadata = result['metadata']
            table.add_row("Total Duration", f"{metadata['total_duration']:.1f} seconds")
            table.add_row("Agent Talk Time",
                          f"{metadata['agent_talk_time']:.1f}s ({metadata['agent_talk_percentage']}%)")
            table.add_row("Client Talk Time",
                          f"{metadata['client_talk_time']:.1f}s ({metadata['client_talk_percentage']}%)")
            table.add_row("Conversation Turns", str(len(result['conversation_flow'])))
            table.add_row("Language", metadata.get('language', 'Unknown'))

            # Add quality metrics
            agent_ratio = metadata['agent_talk_time'] / (
                        metadata['agent_talk_time'] + metadata['client_talk_time']) if (metadata['agent_talk_time'] +
                                                                                        metadata[
                                                                                            'client_talk_time']) > 0 else 0
            table.add_row("Speaker Balance", f"Agent {agent_ratio:.1%} / Client {1 - agent_ratio:.1%}")

            console.print(table)

            # Show conversation preview
            console.print("\n[bold]Conversation Preview:[/bold]")
            for i, turn in enumerate(result['conversation_flow'][:10]):  # Show first 10 turns
                speaker_color = "blue" if turn['speaker'] == 'AGENT' else "green"
                console.print(
                    f"[{speaker_color}]{turn['speaker']}[/{speaker_color}] ({turn['timestamp']}): {turn['text']}")

            if len(result['conversation_flow']) > 10:
                console.print(f"[dim]... and {len(result['conversation_flow']) - 10} more turns[/dim]")

        console.print(f"\n[bold green]✅ Processing complete![/bold green]")
        console.print(f"📄 Transcription saved to: [cyan]{output_file}[/cyan]")
        console.print(f"🤖 Ready for AI consumption!")

        # Show quality recommendations
        with open(output_file, 'r', encoding='utf-8') as f:
            result = json.load(f)

        total_segments = len(result['conversation_flow'])
        if total_segments < 5:
            console.print(
                "\n[yellow]💡 Quality tip: Very few segments detected. Try using --quality high for better accuracy.[/yellow]")

        agent_ratio = len([s for s in result['conversation_flow'] if
                           s['speaker'] == 'AGENT']) / total_segments if total_segments > 0 else 0
        if agent_ratio > 0.9 or agent_ratio < 0.1:
            console.print(
                "\n[yellow]💡 Speaker tip: Unbalanced speaker detection. Consider using --use-pyannote for better speaker separation.[/yellow]")

    except Exception as e:
        console.print(f"[bold red]❌ Error: {str(e)}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()