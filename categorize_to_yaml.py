"""
Categorize transcriptions into compact YAML format.
Handles single files or directories and stores outputs in ./categorizations.
"""

import argparse
import os
from dotenv import load_dotenv
from google.generativeai.types import GenerationConfig
import google.generativeai as genai

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Categorize transcriptions into compact YAML format.")
    parser.add_argument("input", help="Path to a transcription file or directory of transcription files.")
    parser.add_argument("-p", "--prompt", type=str, default="./prompt_json.txt", help="Path to the input prompt file.")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input path '{args.input}' does not exist.")

    if not os.path.exists(args.prompt):
        raise FileNotFoundError(f"Prompt file '{args.prompt}' does not exist.")

    with open(args.prompt, "r") as prompt_file:
        prompt_content = prompt_file.read()

    # Create output directory
    output_dir = "./categorizations_json_v2"
    os.makedirs(output_dir, exist_ok=True)

    # Detect and collect transcription files
    transcription_files = get_transcription_files(args.input)

    if not transcription_files:
        raise ValueError(f"No transcription files found in input path '{args.input}'.")

    for file_path in transcription_files:
        try:
            yaml_output = generate_yaml(file_path, prompt_content)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.json")

            with open(output_path, "w") as out_file:
                out_file.write(yaml_output)

            print(f"✓ Processed: {file_path} → {output_path}")

        except Exception as e:
            print(f"✗ Failed to process {file_path}: {e}")

def get_transcription_files(input_path):
    """
    Return a list of transcription files from the input path.
    If input is a file, return a list with that file.
    If input is a directory, return all files inside it recursively.
    """
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        transcription_files = []
        for root, _, files in os.walk(input_path):
            for f in files:
                full_path = os.path.join(root, f)
                transcription_files.append(full_path)
        return transcription_files
    else:
        return []

def generate_yaml(transcription_path, prompt_content):
    """
    Generate a YAML categorization of a transcription.
    """
    with open(transcription_path, "r") as f:
        transcription_content = f.read()

    full_prompt = f"{prompt_content}\n\nTRANSCRIPCION:\n{transcription_content}"

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Parameter tuning for maximizing output determinism
    generation_cfg = GenerationConfig(
        temperature=0.0,  # Lower temperature for more deterministic output
        top_p=1.0,  # Use all tokens
    )

    response = model.generate_content(
        full_prompt,
        generation_config=generation_cfg,
    )

    return response.text

if __name__ == "__main__":
    main()
