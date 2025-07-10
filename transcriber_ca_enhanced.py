"""
Slightly enhanced version of the Whisper Caesar transcription script.
"""

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
import os
import argparse
import re

CHUNK_LENGTH = 30  # seconds
CHUNK_OVERLAP = 1  # seconds

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper model.")
    parser.add_argument("-i", "--input", type=str, help="Path to the input audio file.")
    parser.add_argument("-o", "--output", type=str, default="./transcripcions_caesar", help="Path to save the transcription output.")
    args = parser.parse_args()

    model_path = "./whisper-model"

    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if not os.path.exists(args.input):
        print(f"Input file {args.input} does not exist.")
        return

    text = transcribe(args.input, processor, device, model)

    # Post-process to add line breaks at sentence boundaries
    text_with_lines = split_into_sentences(text)

    os.makedirs(args.output, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_file = os.path.join(args.output, base_name + ".txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text_with_lines)


def load_audio(input):
    waveform, sample_rate = torchaudio.load(input)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    waveform = waveform.mean(dim=0)  # mono
    waveform = waveform / waveform.abs().max()  # normalize
    return waveform


def split_audio(audio, sample_rate, chunk_length_s, overlap_s):
    chunk_size = int(chunk_length_s * sample_rate)
    step_size = int((chunk_length_s - overlap_s) * sample_rate)
    chunks = []

    for start in range(0, len(audio), step_size):
        end = start + chunk_size
        chunk = audio[start:end]
        if len(chunk) < sample_rate * 2:
            break
        chunks.append(chunk)

    return chunks


def transcribe_chunk(chunk, processor, device, model):
    inputs = processor(chunk, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    input_features = inputs.input_features.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # DO NOT force language or task: let model detect language automatically
    # Remove forced_decoder_ids to allow flexibility with language

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
        )

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def transcribe(input, processor, device, model):
    audio = load_audio(input)
    sample_rate = 16000
    chunks = split_audio(audio, sample_rate, CHUNK_LENGTH, CHUNK_OVERLAP)

    full_text = []
    for i, chunk in enumerate(chunks):
        print(f"🔊 Transcribing chunk {i+1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, processor, device, model)
        full_text.append(text)

    return " ".join(full_text)  # join with space to avoid unnecessary breaks


def split_into_sentences(text):
    """
    Split text into sentences for line breaks.
    Uses punctuation marks as delimiters.
    """
    # Regex splits on ., ! or ? followed by space(s) or end of text
    sentence_endings = re.compile(r'([.!?])\s+')
    sentences = sentence_endings.split(text)

    # sentences list will be like: [sentence1, punctuation, sentence2, punctuation,...]
    # rebuild sentences including punctuation
    combined = []
    for i in range(0, len(sentences)-1, 2):
        combined.append(sentences[i] + sentences[i+1])
    if len(sentences) % 2 == 1:
        combined.append(sentences[-1])  # add last piece if no punctuation

    return "\n".join(s.strip() for s in combined if s.strip())


if __name__ == "__main__":
    main()