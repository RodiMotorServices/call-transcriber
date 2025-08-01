import argparse
import os
from openai import OpenAI
from dotenv import load_dotenv
import yaml


def parse_yaml(yaml_path):
    """
    Parse a YAML file and return its content.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('audios', [])


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Categorize transcriptions using base GPT-4.1-mini with prompt caching.")
    parser.add_argument("input", help="Path to transcription file or directory.")
    parser.add_argument("-p", "--prompt", type=str, default="./prompt_json.txt", help="Path to prompt file.")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input path '{args.input}' does not exist.")

    if not os.path.exists(args.prompt):
        raise FileNotFoundError(f"Prompt file '{args.prompt}' does not exist.")

    with open(args.prompt, "r") as prompt_file:
        cached_prompt = prompt_file.read()

    output_dir = "./categorizations_gpt_v1"
    os.makedirs(output_dir, exist_ok=True)

    transcription_files = get_transcription_files(args.input)

    if not transcription_files:
        raise ValueError(f"No transcription files found in '{args.input}'.")

    client = OpenAI()

    for file_path in transcription_files:
        try:
            json_output = generate_json(client, file_path, cached_prompt)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.json")

            with open(output_path, "w") as out_file:
                out_file.write(json_output)

            print(f"✓ Processed: {file_path} → {output_path}")

        except Exception as e:
            print(f"✗ Failed to process {file_path}: {e}")

def get_transcription_files(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        transcription_files = []
        for root, _, files in os.walk(input_path):
            for f in files:
                transcription_files.append(os.path.join(root, f))
        return transcription_files
    else:
        return []

def generate_json(client, transcription_path, cached_prompt):
    with open(transcription_path, "r") as f:
        transcription_content = f.read()

    full_prompt = f"{cached_prompt}\n\nTRANSCRIPCION:\n{transcription_content}"

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=full_prompt,
        temperature=0.0,
        max_output_tokens=1024,
    )

    print(response)

    return response.output_text

if __name__ == "__main__":
    main()
