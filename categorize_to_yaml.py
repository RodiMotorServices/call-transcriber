"""
Categorize a transcription into a compact YAML format.
"""

import argparse
import os
from dotenv import load_dotenv
import google.generativeai as genai

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Categorize a transcription into a compact YAML format.")
    parser.add_argument("transcription", type=str, help="Path to the output transcription file.")
    parser.add_argument("-p", "--prompt", type=str, default="./prompt.txt", help="Path to the input prompt file.")
    parser.add_argument("-o", "--output", type=str, default="./output.yaml", help="Path to the output YAML file.")

    args = parser.parse_args()

    if not os.path.exists(args.transcription):
        raise FileNotFoundError(f"Transcription file '{args.transcription}' does not exist.")

    if not os.path.exists(args.prompt):
        raise FileNotFoundError(f"Prompt file '{args.prompt}' does not exist.")

    with open(args.output, "w") as output_file:
        output_file.write(generate_yaml(args.transcription, args.prompt))

def generate_yaml(transcription, prompt):
    """
    Generate a YAML categorization of the transcription.
    """

    with open(transcription, "r") as transcription_file:
        transcription_content = transcription_file.read()

    with open(prompt, "r") as prompt_file:
        prompt_content = prompt_file.read()

    full_prompt = prompt_content + "\n\nTRANSCRIPCION:\n" + transcription_content

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")

    return model.generate_content(full_prompt).text


if __name__ == "__main__":
    main()
