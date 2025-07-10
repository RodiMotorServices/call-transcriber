"""
Takes a raw transcription file and corrects/enhances it based on added context using Gemini.
"""

import argparse
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_APE_KEY not found")

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel(model_name="models/gemini-2.5-pro")

def main():
    parser = argparse.ArgumentParser(description="Correct transcription using context.")
    parser.add_argument("-i",
                        "--input",
                        type=str,
                        help="Path to the raw transcription file to be corrected.",
                        )
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        default="./enhanced_transcriptions",
                        help="Path to the context file that provides additional information for correction.",
                        )

    args = parser.parse_args()

    raw_text = load_transcription(args.input)
    prompt = build_prompt(raw_text)
    enhanced_transcript = generate_response(prompt)

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_file = os.path.join(args.output, f"{base_name}_enhanced.txt")

    save_output(enhanced_transcript, output_file)


def load_transcription(input):
    with open(input, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(text):
    """
    Builds the main input prompt for Gemini
    """

    return f"""
                Dada la siguiente transcripción, corrígela teniendo en cuenta que se trata de una llamada de servicio de un taller llamado RODI entre un agente y un cliente.
                A su misma vez, indica las frases según si las ha dicho el cliente o las ha dicho el agente.
                El formato de la transcripción debe ser el siguiente:
                Agente: Frase 1
                Cliente: Frase 2
                ...
                Quiero que seas fidel a la transcripcion, de forma que si interpretas que uno de los dos dice dos frases seguidas aparezca algo del estilo:
                Cliente: Frase 1
                Cliente: Frase 2
                Quiero que no haya ningun salto de linea extra, de forma que la primera frase vaya en la linea 1 y la segunda en la 2, 
                {text}
            """

def generate_response(prompt):
    response = model.generate_content(prompt)
    return response.text


def save_output(text, output):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    main()
