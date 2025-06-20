#!/usr/bin/env python3
"""
Transcription Summarizer
Reads transcription JSON files, sends them to OpenAI for analysis,
and saves the summary to a text file for knowledge base creation.
"""

import os
import sys
import json
from pathlib import Path
import time
from typing import List, Dict

import click
import openai
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel

console = Console()

class Summarizer:
    def __init__(self, api_key: str):
        """Initialize the Summarizer with an OpenAI API key"""
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.client = openai.OpenAI(api_key=api_key)

    def _call_openai(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """Helper function to call the OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Eres un experto analizando llamadas de atención al cliente para una base de datos de conocimiento. Las llamadas pueden estar en español o catalán."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            return response.choices[0].message.content.strip()
        except openai.APIError as e:
            console.print(f"[red]OpenAI API Error: {e}[/red]")
            return f"Error: Could not generate response. {e}"
        except Exception as e:
            console.print(f"[red]An unexpected error occurred: {e}[/red]")
            return f"Error: An unexpected error occurred. {e}"

    def generate_summary(self, agent_speech: str, client_speech: str) -> str:
        """Generate a summary in Spanish using the OpenAI API"""
        
        prompt = f"""
        Analiza la siguiente transcripción de una llamada de atención al cliente y crea un resumen conciso en español para una base de conocimiento.
        El objetivo es entender el problema del cliente y la solución del agente. La transcripción puede estar en español o catalán, pero el resumen DEBE estar en español.

        Enfócate en estos puntos clave:
        1.  **Motivo de la llamada del cliente:** ¿Qué problema o pregunta específica tenía el cliente?
        2.  **Resolución del agente:** ¿Qué pasos siguió el agente? ¿Cuál fue la solución o resultado final?
        3.  **Información clave:** Anota cualquier detalle crítico (números de cuenta, productos, acciones de seguimiento, etc.).

        Mantén el resumen estructurado, claro y fácil de entender.

        ---
        **Diálogo del Cliente:**
        {client_speech}

        ---
        **Diálogo del Agente:**
        {agent_speech}
        ---

        **Resumen para Base de Conocimiento (en español):**
        """
        return self._call_openai(prompt)

    def generate_tags(self, agent_speech: str, client_speech: str) -> str:
        """Generate tags, categories, and other metadata from the transcription."""
        
        prompt = f"""
        Analiza la siguiente transcripción de una llamada para RODI MOTOR SERVICES, una red de talleres de reparación de vehículos. Extrae metadatos para categorización.
        La llamada puede estar en español o catalán. Responde únicamente con un objeto JSON.

        El JSON debe tener la siguiente estructura:
        {{
          "categoria_principal": "string",
          "sub_categoria": "string",
          "etiquetas": ["tag1", "tag2", "tag3"],
          "sentimiento_cliente": "Positivo | Negativo | Neutral",
          "estado_resolucion": "Resuelto | No Resuelto | Escalado a otro departamento | Requiere seguimiento"
        }}

        Utiliza las siguientes directrices para completar el JSON:
        - `categoria_principal`: El motivo general de la llamada. Usa una de estas opciones: "Cita Previa", "Consulta de Presupuesto", "Información de Servicios", "Estado del Vehículo", "Facturación y Pagos", "Reclamaciones o Incidencias", "Consulta General".
        - `sub_categoria`: Un tema más específico. Ejemplos: "Solicitar nueva cita", "Presupuesto para neumáticos", "Preguntar si el coche está listo", "Duda sobre factura", "Problema tras la reparación".
        - `etiquetas`: Palabras clave relevantes. Ejemplos: "cambio de aceite", "ITV", "revisión completa", "ruido motor", "batería", "garantía", "factura incorrecta".
        - `sentimiento_cliente`: La emoción general del cliente durante la llamada.
        - `estado_resolucion`: Si el problema se solucionó durante la llamada.

        ---
        **Diálogo del Cliente:**
        {client_speech}

        ---
        **Diálogo del Agente:**
        {agent_speech}
        ---

        **Objeto JSON de Metadatos:**
        """
        return self._call_openai(prompt, model="gpt-4o")

    def process_transcription(self, json_path: str, resumes_dir: str, skip_processed: bool = True) -> Dict:
        """Process a single transcription file, generating a summary and tags."""
        
        file_stem = Path(json_path).stem.replace("_transcription", "")
        summary_output_path = os.path.join(resumes_dir, f"{file_stem}.txt")
        tags_output_path = os.path.join(resumes_dir, f"{file_stem}_tags.txt")
        
        if skip_processed and os.path.exists(summary_output_path) and os.path.exists(tags_output_path):
            return {
                'file_path': json_path,
                'status': 'skipped'
            }
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            ai_data = data.get('ai_ready_format', {}).get('call_summary', {})
            agent_speech = ai_data.get('agent_speech')
            client_speech = ai_data.get('client_speech')
            
            if not agent_speech and not client_speech:
                return {
                    'file_path': json_path,
                    'status': 'failed',
                    'reason': "No agent or client speech found in the JSON file."
                }

            # Generate Summary
            summary = self.generate_summary(agent_speech, client_speech)
            if summary.startswith("Error:"):
                 return { 'file_path': json_path, 'status': 'failed', 'reason': summary }

            with open(summary_output_path, 'w', encoding='utf-8') as f:
                f.write(summary)

            # Generate Tags
            tags_json_str = self.generate_tags(agent_speech, client_speech)
            if tags_json_str.startswith("Error:"):
                return { 'file_path': json_path, 'status': 'failed', 'reason': tags_json_str }
            
            # Clean up and save tags
            try:
                # The model sometimes wraps the JSON in markdown, so we strip it
                clean_json_str = tags_json_str.strip().replace("```json", "").replace("```", "")
                tags_data = json.loads(clean_json_str)
                
                # Pretty print the tags to the file
                tags_output_str = json.dumps(tags_data, indent=2, ensure_ascii=False)
                with open(tags_output_path, 'w', encoding='utf-8') as f:
                    f.write(tags_output_str)

            except json.JSONDecodeError:
                # If JSON parsing fails, save the raw string for debugging
                with open(tags_output_path, 'w', encoding='utf-8') as f:
                    f.write(tags_json_str)
                return { 'file_path': json_path, 'status': 'failed', 'reason': 'Failed to parse JSON from tags output.' }
                
            return {
                'file_path': json_path,
                'status': 'success'
            }

        except json.JSONDecodeError:
            return { 'file_path': json_path, 'status': 'failed', 'reason': 'Invalid JSON format in transcription file.' }
        except Exception as e:
            return { 'file_path': json_path, 'status': 'failed', 'reason': f"An unexpected error occurred: {e}" }


@click.command()
@click.option('--directory', '-d', required=True, type=click.Path(exists=True, file_okay=False), help='Directory containing the transcription JSON files (e.g., ./mp3/transcriptions).')
@click.option('--output-dir', '-o', help='Directory to save the resume text files. Defaults to a "resumes" subdirectory inside the parent of the input directory.')
@click.option('--skip-processed', is_flag=True, default=True, help='Skip files that have already been summarized (default: True).')
@click.option('--api-key', envvar='OPENAI_API_KEY', help='OpenAI API key. Can also be set via OPENAI_API_KEY environment variable.')
def main(directory, output_dir, skip_processed, api_key):
    """
    Analyzes call transcriptions to create knowledge base summaries.
    
    This script reads all `_transcription.json` files from the specified
    directory, uses the OpenAI API to understand the customer's issue
    and the agent's solution, and saves a summary in .txt format.
    """
    
    console.print(Panel.fit(
        "[bold blue]AI Transcription Summarizer[/bold blue]\n"
        "Creating knowledge base articles from call transcriptions",
        border_style="blue"
    ))
    
    if not api_key:
        console.print("[red]Error: OpenAI API key not found.[/red]")
        console.print("Please provide it using the --api-key flag or by setting the OPENAI_API_KEY environment variable.")
        sys.exit(1)
        
    try:
        summarizer = Summarizer(api_key=api_key)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    
    # Determine output directory
    if not output_dir:
        parent_dir = Path(directory).parent
        resumes_dir = os.path.join(parent_dir, "resumes")
    else:
        resumes_dir = output_dir

    if not os.path.exists(resumes_dir):
        os.makedirs(resumes_dir)
        console.print(f"Created output directory: [cyan]{resumes_dir}[/cyan]")

    # Find transcription files
    json_files = sorted(list(Path(directory).glob("*_transcription.json")))
    
    if not json_files:
        console.print(f"[yellow]No `_transcription.json` files found in '{directory}'[/yellow]")
        sys.exit(0)
    
    console.print(f"Found {len(json_files)} transcription files to process.")
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Summarizing transcriptions...", total=len(json_files))
        
        for json_file in json_files:
            progress.update(task, description=f"Processing {json_file.name}...")
            result = summarizer.process_transcription(str(json_file), resumes_dir, skip_processed)
            
            if result['status'] == 'success':
                console.log(f"[green]✅ Processed:[/green] {json_file.name}")
            elif result['status'] == 'skipped':
                console.log(f"[yellow]⏩ Skipped:[/yellow] {json_file.name}")
            elif result['status'] == 'failed':
                console.log(f"[red]❌ Failed:[/red] {json_file.name} - Reason: {result['reason']}")
            
            progress.advance(task)
            time.sleep(0.5) # Add a small delay to avoid hitting API rate limits too quickly

    console.print("\n[bold green]✅ Summarization complete![/bold green]")
    console.print(f"📄 Summaries and tags saved to: [cyan]{resumes_dir}[/cyan]")

if __name__ == "__main__":
    main() 