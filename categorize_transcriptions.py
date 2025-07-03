#!/usr/bin/env python3
"""
Transcription Categorizer
Reads transcription JSON files, sends them to a Large Language Model for analysis based on a dynamic prompt,
and saves the output to a text file.
"""

import os
import sys
import json
from pathlib import Path
import shutil
import time
from typing import Dict

import click
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel

# Import centralized Gemini configuration
from config import get_gemini_client, validate_gemini_config, create_gemini_prompt, GEMINI_SETTINGS

console = Console()

class Categorizer:
    def __init__(self):
        """Initialize the Categorizer with centralized Gemini configuration"""
        try:
            # Validate configuration first
            is_valid, message = validate_gemini_config()
            if not is_valid:
                raise ValueError(message)
            
            # Get configured Gemini client
            self.model = get_gemini_client()
            
            # Store configuration for reference
            self.config = GEMINI_SETTINGS
            
            console.print(f"[green]✅ Gemini API initialized with model: {self.config['default_model']}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to initialize Gemini API: {e}[/red]")
            raise

    def _call_llm(self, prompt: str) -> str:
        """Helper function to call the Gemini API with enhanced error handling and retry logic"""
        max_retries = self.config.get('retry_attempts', 3)
        retry_delay = self.config.get('retry_delay', 1.0)
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                if response.text:
                    return response.text
                else:
                    raise Exception("Empty response from Gemini API")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    console.print(f"[yellow]⚠️  Attempt {attempt + 1} failed, retrying in {retry_delay}s: {e}[/yellow]")
                    time.sleep(retry_delay)
                    continue
                else:
                    console.print(f"[red]❌ Gemini API Error after {max_retries} attempts: {e}[/red]")
                    return f"Error: Could not generate response after {max_retries} attempts. {e}"

    def generate_output(self, agent_speech: str, client_speech: str, prompt_template: str) -> str:
        """Generate a category using the Gemini API and a dynamic prompt."""
        
        # Use the centralized prompt creation helper
        prompt = create_gemini_prompt(
            prompt_template,
            client_speech=client_speech,
            agent_speech=agent_speech
        )
        
        # Check if prompt exceeds max chunk size
        max_chunk_size = self.config.get('max_chunk_size', 30000)
        if len(prompt) > max_chunk_size:
            console.print(f"[yellow]⚠️  Prompt length ({len(prompt)}) exceeds recommended size ({max_chunk_size}). Consider splitting the content.[/yellow]")
        
        return self._call_llm(prompt)

    def process_transcription(self, json_path: str, output_dir: str, prompt: str, skip_processed: bool = True) -> Dict:
        """Process a single transcription file, generating a category."""
        
        file_stem = Path(json_path).stem.replace("_transcription", "")
        output_file_path = os.path.join(output_dir, f"{file_stem}_category.json")
        
        if skip_processed and os.path.exists(output_file_path):
            return {
                'file_path': json_path,
                'status': 'skipped'
            }
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            ai_data = data.get('ai_ready_format', {}).get('call_summary', {})
            agent_speech = ai_data.get('agent_speech', '')
            client_speech = ai_data.get('client_speech', '')
            
            if not agent_speech and not client_speech:
                return {
                    'file_path': json_path,
                    'status': 'failed',
                    'reason': "No agent or client speech found in the JSON file."
                }

            llm_output_str = self.generate_output(agent_speech, client_speech, prompt)
            if llm_output_str.startswith("Error:"):
                 return { 'file_path': json_path, 'status': 'failed', 'reason': llm_output_str }

            try:
                # The model sometimes wraps the JSON in markdown, so we strip it
                clean_json_str = llm_output_str.strip().replace("```json", "").replace("```", "")
                category_data = json.loads(clean_json_str)
                
                # Pretty print the tags to the file
                output_json_str = json.dumps(category_data, indent=2, ensure_ascii=False)
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(output_json_str)

            except json.JSONDecodeError:
                # If JSON parsing fails, save the raw string for debugging
                with open(output_file_path.replace(".json", "_raw.txt"), 'w', encoding='utf-8') as f:
                    f.write(llm_output_str)
                return { 'file_path': json_path, 'status': 'failed', 'reason': 'Failed to parse JSON from LLM output.' }
                
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
@click.option('--output-dir', '-o', help='Directory to save the category JSON files. Defaults to a "categories" subdirectory inside the parent of the input directory.')
@click.option('--prompt-file', '-p', type=click.Path(exists=True, dir_okay=False), help='Path to a text file containing the prompt template. Overrides the default prompt.')
@click.option('--skip-processed', is_flag=True, default=True, help='Skip files that have already been categorized (default: True).')
@click.option('--limit', '-n', type=int, default=None, help='Limit the number of files to process.')
def main(directory, output_dir, prompt_file, skip_processed, limit):
    """
    Analyzes call transcriptions to categorize them based on a given prompt.
    
    This script reads `_transcription.json` files, uses the Gemini API
    to process the content based on a user-defined prompt, and saves the
    output to a .json file.
    """
    
    console.print(Panel.fit(
        "[bold blue]AI Transcription Categorizer[/bold blue]\n"
        "Categorizing call transcriptions using Gemini API\n"
        f"Model: {GEMINI_SETTINGS['default_model']}",
        border_style="blue"
    ))
    
    # Validate Gemini configuration early
    is_valid, validation_message = validate_gemini_config()
    if not is_valid:
        console.print(f"[red]❌ Configuration Error: {validation_message}[/red]")
        console.print("[yellow]💡 Please ensure GEMINI_API_KEY is set in your .env file[/yellow]")
        sys.exit(1)
    
    console.print("[green]✅ Gemini API configuration validated[/green]")

    # Default prompt based on the user's detailed requirements
    default_prompt = """
Analiza la siguiente transcripción de una llamada para RODI MOTOR SERVICES y clasifícala. Responde únicamente con un objeto JSON.

La estructura del JSON debe ser la siguiente:
{{
  "categoria_principal": "string",
  "detalles": {{}}
}}

1.  **Determina la `categoria_principal`**. Debe ser una de las siguientes:
    *   "Quiere ir al taller"
    *   "Cambio de cita"
    *   "Coche en taller"
    *   "Otros"

2.  **Rellena el objeto `detalles`** según la `categoria_principal` que hayas elegido. No añadas campos que no apliquen.

    *   **Si `categoria_principal` es "Quiere ir al taller"**:
        - `tipologia_reparacion`: Una de ["Neumáticos", "Revisión/Mantenimiento", "Mecánica Ligera", "Mecánica Pesada", "Recambio", "Avería/Diagnóstico"].
        - `terminos_clave`: array de strings con palabras clave específicas de la reparación (ej: "cambio de aceite", "ruido motor", "ITV").
        - `menciones_marca_neumaticos`: array de strings con marcas de neumáticos mencionadas (ej: "Falken", "Pirelli", "Michelin").
        - `es_llamada_comercial`: "SI" o "NO".
        - `presupuesto`: "SI", "NO" o "ORIENTATIVO".
        - `cita_concertada`: Una de ["SI", "NO", "VUELVE A LLAMAR CLIENTE", "VUELVE A LLAMAR OPERADOR", "RECHAZA CITA RODI", "AUTORIZACION"].

    *   **Si `categoria_principal` es "Cambio de cita"**:
        - `tipo_cambio`: Una de ["Avanzar", "Retrasar", "Cancelar"].
        - `persistencia_operador_en_cancelacion`: (Solo si `tipo_cambio` es "Cancelar") "SI" o "NO".

    *   **Si `categoria_principal` es "Coche en taller"**:
        - `consulta_estado`: Una de ["Estado del vehículo", "Estado del Material", "Estado autorización"].
        - `prevision_finalizacion`: "SI" o "NO".
        - `prevision_llegada_pieza`: "SI" o "NO".

    *   **Si `categoria_principal` es "Otros"**:
        - `subcategoria_otros`: Una de ["Información general", "Facturas", "Comunicación interna", "Proveedores", "Reclamación"].

---
**Diálogo del Cliente:**
{client_speech}

---
**Diálogo del Agente:**
{agent_speech}
---

**Objeto JSON de Categorización:**
"""
    
    prompt_template = default_prompt
    if prompt_file:
        console.print(f"Loading custom prompt from: [cyan]{prompt_file}[/cyan]")
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

    if "{client_speech}" not in prompt_template or "{agent_speech}" not in prompt_template:
        console.print("[red]Error: Prompt must be a string containing {client_speech} and {agent_speech} placeholders.[/red]")
        sys.exit(1)
        
    try:
        categorizer = Categorizer()
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize categorizer: {e}[/red]")
        sys.exit(1)
    
    # Determine output directory
    if not output_dir:
        parent_dir = Path(directory).parent
        categories_dir = os.path.join(parent_dir, "categories")
    else:
        categories_dir = output_dir

    if not os.path.exists(categories_dir):
        os.makedirs(categories_dir)
        console.print(f"Created output directory: [cyan]{categories_dir}[/cyan]")

    # Create a directory to move processed files
    parent_dir = Path(directory).parent
    categorized_dir = parent_dir / "categorized"
    os.makedirs(categorized_dir, exist_ok=True)
    console.print(f"Processed files will be moved to: [cyan]{categorized_dir}[/cyan]")

    # Find transcription files
    json_files = sorted(list(Path(directory).glob("*_transcription.json")))
    
    if not json_files:
        console.print(f"[yellow]No `_transcription.json` files found in '{directory}'[/yellow]")
        sys.exit(0)
    
    # Limit the number of files to process if the option is provided
    if limit is not None and limit > 0:
        json_files = json_files[:limit]
        console.print(f"[yellow]Processing a limit of {len(json_files)} files.[/yellow]")

    console.print(f"Found {len(json_files)} transcription files to process.")
    
    # Track processing statistics
    success_count = 0
    failed_count = 0
    skipped_count = 0
    failed_files = []
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Categorizing transcriptions...", total=len(json_files))
        
        for json_file in json_files:
            progress.update(task, description=f"Processing {json_file.name}...")
            result = categorizer.process_transcription(str(json_file), categories_dir, prompt_template, skip_processed)
            
            if result['status'] == 'success':
                success_count += 1
                console.log(f"[green]✅ Processed:[/green] {json_file.name}")
                # Move the successfully processed file
                try:
                    source_path = Path(json_file)
                    dest_path = categorized_dir / source_path.name
                    shutil.move(source_path, dest_path)
                    console.log(f"[dim]Moved {source_path.name} to {dest_path}[/dim]")
                except Exception as e:
                    console.log(f"[red]❌ Error moving file {json_file.name}: {e}[/red]")
            elif result['status'] == 'skipped':
                skipped_count += 1
                console.log(f"[yellow]⏩ Skipped:[/yellow] {json_file.name}")
            elif result['status'] == 'failed':
                failed_count += 1
                failed_files.append((json_file.name, result['reason']))
                console.log(f"[red]❌ Failed:[/red] {json_file.name} - Reason: {result['reason']}")
            
            progress.advance(task)

    # Show completion summary
    console.print(f"\n[bold]📊 Processing Summary:[/bold]")
    console.print(f"   ✅ Successful: {success_count}")
    console.print(f"   ⏩ Skipped: {skipped_count}")
    console.print(f"   ❌ Failed: {failed_count}")
    
    if failed_count > 0:
        console.print(f"\n[bold red]⚠️  {failed_count} file(s) failed to process![/bold red]")
        console.print("[yellow]Failed files and reasons:[/yellow]")
        for filename, reason in failed_files:
            console.print(f"   • {filename}: {reason}")
        console.print(f"\n[red]❌ Categorization completed with errors![/red]")
        console.print(f"📄 Partial results saved to: [cyan]{categories_dir}[/cyan]")
        sys.exit(1)  # Exit with error code
    else:
        console.print(f"\n[bold green]✅ Categorization completed successfully![/bold green]")
        console.print(f"📄 Categories saved to: [cyan]{categories_dir}[/cyan]")

if __name__ == "__main__":
    main() 