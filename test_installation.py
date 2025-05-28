#!/usr/bin/env python3
# Sample usage script for Call Transcriber

import os
from main import CallTranscriber

def test_transcriber():
    # Initialize the transcriber
    transcriber = CallTranscriber(whisper_model="tiny", device="cpu")
    
    print("Call Transcriber is ready!")
    print("Usage examples:")
    print("1. python main.py your_audio_file.mp3")
    print("2. python main.py audio.mp3 --model base --preview")
    print("3. python batch_process.py --directory ./audio_files")

if __name__ == "__main__":
    test_transcriber()
