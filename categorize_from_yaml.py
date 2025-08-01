"""
Categorize all the transcriptions of a YAML file with GPT-4.1-mini.
"""

import argparse
import os
from openai import OpenAI
from dotenv import load_dotenv
import yaml

def load_and_split_calls(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data.get('audios', [])

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Categorize transcriptions using GPT-4.1-mini from one big YAML with multiple calls.")
    parser.add_argument("input", help="Path to big YAML file containing multiple calls under 'audios'.")
    parser.add_argument("-p", "--prompt", type=str, default="./prompt_json.txt", help="Path to prompt file.")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file '{args.input}' does not exist.")
    if not os.path.exists(args.prompt):
        raise FileNotFoundError(f"Prompt file '{args.prompt}' does not exist.")

    with open(args.prompt, "r", encoding="utf-8") as prompt_file:
        cached_prompt = prompt_file.read()

    output_dir = "./categorizations_gpt_v2"
    os.makedirs(output_dir, exist_ok=True)

    calls = load_and_split_calls(args.input)
    if not calls:
        raise ValueError(f"No calls found in YAML file '{args.input}'.")

    client = OpenAI()

    count = 1
    for call in calls:
        audio_id = call.get('audio', {}).get('audio_id')
        if audio_id is None:
            print("Warning: Found a call without audio_id, skipping...")
            continue

        # Convert the single call dict back to YAML string for prompt input
        transcription_content = yaml.dump({'audios': [call]}, allow_unicode=True)

        try:
            json_output = generate_json(client, transcription_content, cached_prompt, from_string=True)
            output_path = os.path.join(output_dir, f"call_{audio_id}.json")
            with open(output_path, "w", encoding="utf-8") as out_file:
                out_file.write(json_output)
            print(f"It {count} ✓ Processed call audio_id {audio_id} → {output_path}")
        except Exception as e:
            print(f"✗ Failed to process call audio_id {audio_id}: {e}")
        count += 1


def generate_json(client, transcription_input, cached_prompt, from_string=False):
    if from_string:
        transcription_content = transcription_input
    else:
        with open(transcription_input, "r", encoding="utf-8") as f:
            transcription_content = f.read()

    full_prompt = f"{cached_prompt}\n\nTRANSCRIPCION:\n{transcription_content}"

    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions="Eres un asistente experto en análisis de llamadas de servicio para talleres mecánicos.",
        input=full_prompt,
        temperature=0.0,
        max_output_tokens=1024,
    )

    return response.output_text

if __name__ == "__main__":
    main()
