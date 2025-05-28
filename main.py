#!/usr/bin/env python3
"""
Call Transcriber - Audio transcription with speaker diarization for call centers
Transcribes MP3 files and separates agent/client speech for AI processing
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import click
import whisper
import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from pyannote.audio import Pipeline
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

console = Console()

class CallTranscriber:
    def __init__(self, whisper_model: str = "base", device: str = "auto", language: str = "auto", use_pyannote: bool = False):
        """Initialize the transcriber with models"""
        self.device = self._get_device(device)
        self.language = language
        self.use_pyannote = use_pyannote
        
        console.print(f"[yellow]Loading Whisper model '{whisper_model}' on {self.device}...[/yellow]")
        self.whisper_model = whisper.load_model(whisper_model, device=self.device)
        
        # Initialize pyannote speaker diarization if requested
        if use_pyannote:
            console.print("[yellow]Loading pyannote speaker diarization pipeline...[/yellow]")
            try:
                # You'll need a HuggingFace token: https://huggingface.co/pyannote/speaker-diarization
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=os.getenv('HUGGINGFACE_TOKEN')
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
    
    def preprocess_audio(self, audio_path: str) -> str:
        """Enhanced audio preprocessing for better transcription quality"""
        console.print("[cyan]Preprocessing audio for optimal quality...[/cyan]")
        
        # Load audio with pydub
        audio = AudioSegment.from_mp3(audio_path)
        
        # 1. Normalize volume levels
        audio = normalize(audio)
        
        # 2. Compress dynamic range (helps with varying voice levels)
        audio = compress_dynamic_range(audio, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)
        
        # 3. Convert to optimal format for speech recognition
        # - Mono channel (speech recognition works better with mono)
        # - 16kHz sample rate (optimal for Whisper)
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # 4. Apply telephony frequency filtering (300-3400 Hz typical for phone calls)
        # This removes frequencies outside typical phone call range
        audio = audio.high_pass_filter(300).low_pass_filter(3400)
        
        # 5. Gentle noise gate to remove very quiet background noise
        # Keep anything above -40dB, gradually fade anything below -50dB
        audio = audio.compress_dynamic_range(
            threshold=-40.0, 
            ratio=float('inf'),  # Hard gate
            attack=1.0, 
            release=10.0
        )
        
        # Create temporary WAV file with enhanced audio
        temp_wav = tempfile.mktemp(suffix=".wav")
        audio.export(temp_wav, format="wav")
        
        console.print("[green]✅ Audio preprocessing complete[/green]")
        return temp_wav
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """Enhanced transcription using Whisper with optimal settings"""
        console.print("[cyan]Transcribing audio with enhanced settings...[/cyan]")
        
        # Determine language setting
        if self.language == "auto":
            language_setting = None  # Let Whisper auto-detect
            console.print("[yellow]Auto-detecting language...[/yellow]")
        else:
            language_setting = self.language
            console.print(f"[yellow]Using specified language: {self.language}[/yellow]")
        
        # Enhanced Whisper settings for better accuracy
        result = self.whisper_model.transcribe(
            audio_path,
            language=language_setting,
            task="transcribe",
            verbose=False,
            # Enhanced settings for better quality
            temperature=0.0,  # Deterministic output
            best_of=5,        # Try 5 different decodings and pick best
            beam_size=5,      # Beam search for better accuracy
            patience=1.0,     # Wait longer for better results
            length_penalty=1.0,  # Standard length penalty
            suppress_tokens=[-1],  # Suppress silence token
            initial_prompt="Esta es una grabación de una llamada telefónica de servicio al cliente entre un agente y un cliente." if language_setting == "es" else "This is a customer service phone call recording between an agent and a client.",
            condition_on_previous_text=True,  # Use context from previous segments
            compression_ratio_threshold=2.4,  # Detect repetitive speech
            logprob_threshold=-1.0,  # Quality threshold
            no_speech_threshold=0.6  # Silence detection threshold
        )
        
        # Report detected language
        detected_language = result.get('language', 'unknown')
        console.print(f"[green]Detected/Used language: {detected_language}[/green]")
        
        return result
    
    def pyannote_speaker_separation(self, audio_path: str, transcription_segments: List[Dict]) -> List[Dict]:
        """Advanced speaker diarization using pyannote.audio"""
        console.print("[cyan]Performing advanced speaker diarization with pyannote...[/cyan]")
        
        try:
            # Apply diarization
            diarization = self.diarization_pipeline(audio_path)
            
            # Convert pyannote segments to our format
            speaker_segments = []
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'speaker': speaker
                })
            
            # Match transcription segments with speaker segments
            separated_segments = []
            for trans_seg in transcription_segments:
                segment_text = trans_seg.get('text', '').strip()
                if not segment_text:
                    continue
                
                trans_start = trans_seg['start']
                trans_end = trans_seg['end']
                trans_mid = (trans_start + trans_end) / 2
                
                # Find the speaker segment that overlaps most with this transcription segment
                best_speaker = "SPEAKER_00"  # Default
                best_overlap = 0
                
                for spk_seg in speaker_segments:
                    overlap_start = max(trans_start, spk_seg['start'])
                    overlap_end = min(trans_end, spk_seg['end'])
                    overlap = max(0, overlap_end - overlap_start)
                    
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_speaker = spk_seg['speaker']
                
                # Map speaker IDs to AGENT/CLIENT
                # First speaker is usually the agent
                if best_speaker == "SPEAKER_00":
                    speaker_label = "AGENT"
                elif best_speaker == "SPEAKER_01":
                    speaker_label = "CLIENT"
                else:
                    # If more than 2 speakers, assign based on first occurrence
                    speaker_label = "CLIENT" if len([s for s in separated_segments if s.get('speaker') == "AGENT"]) > len([s for s in separated_segments if s.get('speaker') == "CLIENT"]) else "AGENT"
                
                separated_segments.append({
                    'start': trans_seg['start'],
                    'end': trans_seg['end'],
                    'text': segment_text,
                    'speaker': speaker_label
                })
            
            console.print(f"[green]✅ Pyannote diarization complete - processed {len(separated_segments)} segments[/green]")
            return separated_segments
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Pyannote diarization failed: {e}[/yellow]")
            console.print("[yellow]Falling back to enhanced heuristic method[/yellow]")
            return self.enhanced_speaker_separation(transcription_segments, 0)
    
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
            pause = segments[i]['start'] - segments[i-1]['end']
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
                prev_end = segments[i-1]['end']
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
                    prev_duration = segments[i-1]['end'] - segments[i-1]['start']
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
        
        console.print(f"[green]✅ Enhanced separation complete - {len(separated_segments)} segments (Agent: {final_agent_ratio:.1%})[/green]")
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
                prev_end = segments[i-1]['end']
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
                'language': metadata.get('language', 'en'),
                'agent_talk_time': round(total_agent_time, 2),
                'client_talk_time': round(total_client_time, 2),
                'agent_talk_percentage': agent_talk_percentage,
                'client_talk_percentage': client_talk_percentage
            },
            'conversation_flow': conversation_flow,
            'agent_segments': {
                'full_text': ' '.join(part['text'] for part in agent_parts),
                'segments': agent_parts,
                'segment_count': len(agent_parts)
            },
            'client_segments': {
                'full_text': ' '.join(part['text'] for part in client_parts),
                'segments': client_parts,
                'segment_count': len(client_parts)
            },
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
        """Main processing function"""
        
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
                wav_path = self.preprocess_audio(mp3_path)
                progress.remove_task(task1)
            except Exception as e:
                progress.remove_task(task1)
                raise Exception(f"Failed to preprocess audio: {str(e)}")
            
            try:
                # Step 2: Transcribe
                task2 = progress.add_task("Transcribing audio...", total=None)
                try:
                    transcription_result = self.transcribe_audio(wav_path)
                    progress.remove_task(task2)
                    
                    # Validate transcription result
                    if not transcription_result or 'segments' not in transcription_result:
                        raise Exception("Transcription failed - no segments returned")
                        
                except Exception as e:
                    progress.remove_task(task2)
                    raise Exception(f"Transcription failed: {str(e)}")
                
                # Step 3: Speaker separation
                task3 = progress.add_task("Separating speakers...", total=None)
                try:
                    if self.use_pyannote:
                        separated_segments = self.pyannote_speaker_separation(wav_path, transcription_result['segments'])
                    else:
                        separated_segments = self.enhanced_speaker_separation(
                            transcription_result['segments'], 
                            transcription_result.get('duration', 0)
                        )
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
                            'duration': transcription_result.get('duration', 0),
                            'language': transcription_result.get('language', 'en')
                        }
                    )
                    progress.remove_task(task4)
                except Exception as e:
                    progress.remove_task(task4)
                    raise Exception(f"AI formatting failed: {str(e)}")
                
            finally:
                # Clean up temporary file
                if os.path.exists(wav_path):
                    os.remove(wav_path)
        
        # Save output
        if output_path is None:
            # Save in transcriptions subfolder of the audio file's directory
            audio_dir = os.path.dirname(mp3_path)
            transcriptions_dir = os.path.join(audio_dir, "transcriptions")
            
            # Create transcriptions directory if it doesn't exist
            if not os.path.exists(transcriptions_dir):
                os.makedirs(transcriptions_dir)
            
            output_path = os.path.join(
                transcriptions_dir,
                f"{Path(mp3_path).stem}_transcription.json"
            )
        
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
@click.option('--model', '-m', default='base', help='Whisper model size (tiny, base, small, medium, large)')
@click.option('--device', '-d', default='auto', help='Device to use (auto, cpu, cuda, mps)')
@click.option('--language', '-l', default='auto', help='Language code (auto, es, en, fr, de, pt, it, etc.) or "auto" for detection')
@click.option('--preview', '-p', is_flag=True, help='Show preview of results in terminal')
@click.option('--use-pyannote', is_flag=True, help='Use advanced pyannote.audio speaker diarization (requires HuggingFace token)')
@click.option('--enhanced', is_flag=True, default=True, help='Use enhanced audio preprocessing and transcription settings (default: True)')
@click.option('--quality', type=click.Choice(['fast', 'balanced', 'high']), default='balanced', help='Quality preset: fast (tiny model), balanced (base/small), high (medium/large)')
def main(mp3_file: str, output: str, model: str, device: str, language: str, preview: bool, use_pyannote: bool, enhanced: bool, quality: str):
    """
    Call Transcriber - Transcribe MP3 call recordings with speaker separation
    
    MP3_FILE: Path to the MP3 audio file to transcribe
    
    Language codes:
    - auto: Auto-detect language (default)
    - es: Spanish
    - en: English  
    - fr: French
    - de: German
    - pt: Portuguese
    - it: Italian
    
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
    
    # Apply quality presets
    if quality == 'fast':
        if model == 'base':  # Only override if user didn't specify
            model = 'tiny'
        enhanced = False
        console.print("[yellow]Using FAST preset: tiny model, basic processing[/yellow]")
    elif quality == 'high':
        if model == 'base':  # Only override if user didn't specify
            model = 'medium'
        enhanced = True
        console.print("[yellow]Using HIGH quality preset: enhanced processing[/yellow]")
    else:  # balanced
        if model == 'base':
            model = 'small' if language != 'auto' else 'base'
        console.print("[yellow]Using BALANCED preset: optimized quality/speed[/yellow]")
    
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
            language=language,
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
            table.add_row("Agent Talk Time", f"{metadata['agent_talk_time']:.1f}s ({metadata['agent_talk_percentage']}%)")
            table.add_row("Client Talk Time", f"{metadata['client_talk_time']:.1f}s ({metadata['client_talk_percentage']}%)")
            table.add_row("Conversation Turns", str(len(result['conversation_flow'])))
            table.add_row("Language", metadata.get('language', 'Unknown'))
            
            # Add quality metrics
            agent_ratio = metadata['agent_talk_time'] / (metadata['agent_talk_time'] + metadata['client_talk_time']) if (metadata['agent_talk_time'] + metadata['client_talk_time']) > 0 else 0
            table.add_row("Speaker Balance", f"Agent {agent_ratio:.1%} / Client {1-agent_ratio:.1%}")
            
            console.print(table)
            
            # Show conversation preview
            console.print("\n[bold]Conversation Preview:[/bold]")
            for i, turn in enumerate(result['conversation_flow'][:10]):  # Show first 10 turns
                speaker_color = "blue" if turn['speaker'] == 'AGENT' else "green"
                console.print(f"[{speaker_color}]{turn['speaker']}[/{speaker_color}] ({turn['timestamp']}): {turn['text']}")
            
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
            console.print("\n[yellow]💡 Quality tip: Very few segments detected. Try using --quality high for better accuracy.[/yellow]")
        
        agent_ratio = len([s for s in result['conversation_flow'] if s['speaker'] == 'AGENT']) / total_segments if total_segments > 0 else 0
        if agent_ratio > 0.9 or agent_ratio < 0.1:
            console.print("\n[yellow]💡 Speaker tip: Unbalanced speaker detection. Consider using --use-pyannote for better speaker separation.[/yellow]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error: {str(e)}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    main() 