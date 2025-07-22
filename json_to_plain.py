import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Convierte transcripciones JSON a texto plano (sin nombres de hablantes).")
    parser.add_argument("input", help="Ruta a un archivo .json o un directorio que contenga archivos .json")

    args = parser.parse_args()

    input_path = args.input
    output_dir = "./transcriptions_txt"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Ruta '{input_path}' no encontrada.")

    os.makedirs(output_dir, exist_ok=True)

    json_files = collect_json_files(input_path)

    if not json_files:
        raise ValueError("No se encontraron archivos JSON válidos.")

    for file_path in json_files:
        try:
            text = extract_text_from_json(file_path)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.txt")

            with open(output_path, "w", encoding="utf-8") as out_file:
                out_file.write(text)

            print(f"✓ Procesado: {file_path} → {output_path}")

        except Exception as e:
            print(f"✗ Error procesando {file_path}: {e}")

def collect_json_files(path):
    """
    Si es un archivo JSON, devuelve una lista con él.
    Si es un directorio, devuelve todos los archivos JSON dentro (recursivo).
    """
    if os.path.isfile(path) and path.lower().endswith(".json"):
        return [path]
    elif os.path.isdir(path):
        json_files = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".json"):
                    json_files.append(os.path.join(root, f))
        return json_files
    return []

def extract_text_from_json(file_path):
    """
    Lee un archivo JSON y concatena el texto de la conversación, ignorando nombres y timestamps.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "conversation_flow" not in data:
        raise ValueError(f"'conversation_flow' no encontrado en {file_path}.")

    return "\n".join(segment.get("text", "") for segment in data["conversation_flow"])

if __name__ == "__main__":
    main()
