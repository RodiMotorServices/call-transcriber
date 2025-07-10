from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
import os
import argparse

CHUNK_LENGTH = 30  # seconds
CHUNK_OVERLAP = 1  # seconds

def main():
    """
    Main function to load the Whisper model and processor, and prepare for transcription.
    """

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
    else:
        text = transcribe(args.input, processor, device, model)

        os.makedirs(args.output, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_file = os.path.join(args.output, base_name + ".txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)


def load_audio(input):
    """
    Load audio file and resample it to 16kHz if necessary.
    Normalize and convert to mono.
    """

    waveform, sample_rate = torchaudio.load(input)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    waveform = waveform.mean(dim=0)  # convert to mono
    waveform = waveform / waveform.abs().max()  # normalize
    return waveform


def split_audio(audio, sample_rate, chunk_length_s, overlap_s):
    """
    Split audio into overlapping chunks.
    """
    chunk_size = int(chunk_length_s * sample_rate)
    step_size = int((chunk_length_s - overlap_s) * sample_rate)
    chunks = []

    for start in range(0, len(audio), step_size):
        end = start + chunk_size
        chunk = audio[start:end]
        if len(chunk) < sample_rate * 2:  # skip too short chunks
            break
        chunks.append(chunk)

    return chunks


def transcribe_chunk(chunk, processor, device, model):
    """
    Transcribe a single chunk of audio.
    """
    inputs = processor(chunk, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    input_features = inputs.input_features.to(device)
    attention_mask = inputs.attention_mask.to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ca", task="transcribe")

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            forced_decoder_ids=forced_decoder_ids
        )

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def transcribe(input, processor, device, model):
    """
    Transcribe the input audio file using the Whisper model with chunking.
    """
    audio = load_audio(input)
    sample_rate = 16000
    chunks = split_audio(audio, sample_rate, CHUNK_LENGTH, CHUNK_OVERLAP)

    full_text = []
    for i, chunk in enumerate(chunks):
        print(f"🔊 Transcribing chunk {i+1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, processor, device, model)
        full_text.append(text.strip())

    return "\n".join(full_text)


if __name__ == "__main__":
    main()
