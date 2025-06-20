#!/usr/bin/env python3
"""
Batch Processing Script for Call Transcriber
Process multiple MP3 files in a directory or from a list
"""

import os
import sys
import glob
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from datetime import datetime

import click
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

# Import our main transcriber
from main import CallTranscriber

console = Console()

class BatchProcessor:
    def __init__(self, model: str = "base", device: str = "auto", language: str = "auto", max_workers: int = 2, use_pyannote: bool = False, enhanced: bool = True):
        """Initialize batch processor"""
        self.transcriber = CallTranscriber(whisper_model=model, device=device, language=language, use_pyannote=use_pyannote)
        self.max_workers = max_workers
        self.results = []
        self.enhanced = enhanced
        
    def find_audio_files(self, directory: str, pattern: str = "*.mp3") -> List[str]:
        """Find audio files in directory"""
        search_path = os.path.join(directory, pattern)
        files = glob.glob(search_path)
        
        # Also search for other common audio formats
        for ext in ['*.wav', '*.m4a', '*.flac', '*.ogg']:
            if ext != pattern:
                search_path = os.path.join(directory, ext)
                files.extend(glob.glob(search_path))
        
        return sorted(files)
    
    def check_already_transcribed(self, file_path: str, output_dir: str = None) -> bool:
        """Check if a file has already been transcribed"""
        
        console.print(f"[bold magenta]Checking for transcription of:[/bold magenta] {file_path}")

        file_stem = Path(file_path).stem
        
        # Define potential transcription filenames
        transcription_filenames = [
            f"{file_stem}_transcription.json", # Standard name
            f"{file_stem}.json" # For backward compatibility or different naming
        ]
        
        # Check in specified output directory
        if output_dir:
            console.print(f"Checking in specified output directory: {output_dir}")
            for fname in transcription_filenames:
                check_path = os.path.join(output_dir, fname)
                exists = os.path.exists(check_path)
                console.print(f"  - Checking path: {check_path} -> {'[green]Exists[/green]' if exists else '[red]Not Found[/red]'}")
                if exists:
                    return True
            return False

        # If no output_dir, check default locations
        audio_dir = os.path.dirname(file_path)
        console.print(f"Checking in default locations relative to: {audio_dir}")

        # Location 1: New standard location in "transcriptions" subdirectory
        transcriptions_dir = os.path.join(audio_dir, "transcriptions")
        console.print(f"  [bold]1. Transcriptions subdirectory:[/bold] {transcriptions_dir}")
        for fname in transcription_filenames:
            check_path = os.path.join(transcriptions_dir, fname)
            exists = os.path.exists(check_path)
            console.print(f"    - Checking path: {check_path} -> {'[green]Exists[/green]' if exists else '[red]Not Found[/red]'}")
            if exists:
                console.print(f"[bold green]Found existing transcription.[/bold green]")
                return True

        # Location 2: Old location in the same directory as the audio file
        console.print(f"  [bold]2. Same directory as audio:[/bold] {audio_dir}")
        for fname in transcription_filenames:
            check_path = os.path.join(audio_dir, fname)
            exists = os.path.exists(check_path)
            console.print(f"    - Checking path: {check_path} -> {'[green]Exists[/green]' if exists else '[red]Not Found[/red]'}")
            if exists:
                console.print(f"[bold green]Found existing transcription.[/bold green]")
                return True

        # Location 3: Old location in the current working directory
        cwd = os.getcwd()
        console.print(f"  [bold]3. Current working directory:[/bold] {cwd}")
        for fname in transcription_filenames:
            # Note: This checks relative to CWD, which might be unexpected.
            check_path = fname 
            exists = os.path.exists(check_path)
            console.print(f"    - Checking path: {check_path} -> {'[green]Exists[/green]' if exists else '[red]Not Found[/red]'}")
            if exists:
                console.print(f"[bold green]Found existing transcription.[/bold green]")
                return True
        
        console.print("[bold yellow]No existing transcription found.[/bold yellow]")
        return False
    
    def filter_unprocessed_files(self, file_paths: List[str], output_dir: str = None) -> Tuple[List[str], List[str]]:
        """Filter out already processed files and return unprocessed and already processed lists"""
        unprocessed = []
        already_processed = []
        
        for file_path in file_paths:
            if self.check_already_transcribed(file_path, output_dir):
                already_processed.append(file_path)
            else:
                unprocessed.append(file_path)
        
        return unprocessed, already_processed
    
    def move_processed_files(self, file_path: str, transcription_path: str, processed_dir: str = "processed"):
        """Move processed audio and transcription files to processed directory"""
        try:
            # Create processed directory if it doesn't exist
            if not os.path.exists(processed_dir):
                os.makedirs(processed_dir)
            
            # Move audio file
            audio_filename = os.path.basename(file_path)
            new_audio_path = os.path.join(processed_dir, audio_filename)
            if os.path.exists(file_path):
                shutil.move(file_path, new_audio_path)
            
            # Move transcription file
            if transcription_path and os.path.exists(transcription_path):
                transcription_filename = os.path.basename(transcription_path)
                new_transcription_path = os.path.join(processed_dir, transcription_filename)
                shutil.move(transcription_path, new_transcription_path)
            
            return new_audio_path, new_transcription_path
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not move files to processed directory: {e}[/yellow]")
            return file_path, transcription_path

    def process_single_file(self, file_path: str, output_dir: str = None, move_processed: bool = False) -> Dict:
        """Process a single file and return results"""
        try:
            start_time = datetime.now()
            
            # Determine output path
            if output_dir:
                # Use specified output directory
                output_path = os.path.join(
                    output_dir, 
                    f"{Path(file_path).stem}_transcription.json"
                )
            else:
                # Save in transcriptions subfolder of the audio file's directory
                audio_dir = os.path.dirname(file_path)
                transcriptions_dir = os.path.join(audio_dir, "transcriptions")
                
                # Create transcriptions directory if it doesn't exist
                if not os.path.exists(transcriptions_dir):
                    os.makedirs(transcriptions_dir)
                
                output_path = os.path.join(
                    transcriptions_dir,
                    f"{Path(file_path).stem}_transcription.json"
                )
            
            # Process the file
            result_file = self.transcriber.process_call(file_path, output_path)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Load results for analysis
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Move files to processed directory if requested
            new_audio_path = file_path
            new_transcription_path = result_file
            
            if move_processed:
                processed_dir = os.path.join(os.path.dirname(file_path), "processed")
                new_audio_path, new_transcription_path = self.move_processed_files(
                    file_path, result_file, processed_dir
                )
            
            return {
                'file_path': file_path,
                'output_path': result_file,
                'final_audio_path': new_audio_path,
                'final_transcription_path': new_transcription_path,
                'moved_to_processed': move_processed,
                'success': True,
                'processing_time': processing_time,
                'duration': data['metadata']['total_duration'],
                'agent_talk_time': data['metadata']['agent_talk_time'],
                'client_talk_time': data['metadata']['client_talk_time'],
                'conversation_turns': len(data['conversation_flow']),
                'error': None
            }
            
        except Exception as e:
            return {
                'file_path': file_path,
                'output_path': None,
                'final_audio_path': file_path,
                'final_transcription_path': None,
                'moved_to_processed': False,
                'success': False,
                'processing_time': 0,
                'duration': 0,
                'agent_talk_time': 0,
                'client_talk_time': 0,
                'conversation_turns': 0,
                'error': str(e)
            }
    
    def process_files(self, file_paths: List[str], output_dir: str = None, parallel: bool = True, move_processed: bool = False) -> List[Dict]:
        """Process multiple files"""
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        results = []
        processing_messages = []
        
        if parallel and len(file_paths) > 1:
            # Parallel processing
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                
                task = progress.add_task("Processing files...", total=len(file_paths))
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all tasks
                    future_to_file = {
                        executor.submit(self.process_single_file, file_path, output_dir, move_processed): file_path 
                        for file_path in file_paths
                    }
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_file):
                        result = future.result()
                        results.append(result)
                        
                        file_name = Path(result['file_path']).name
                        if result['success']:
                            if result['moved_to_processed']:
                                processing_messages.append(f"✅ {file_name} (moved to processed/)")
                            else:
                                processing_messages.append(f"✅ {file_name}")
                        else:
                            processing_messages.append(f"❌ {file_name}: {result['error']}")
                        
                        progress.advance(task)
        else:
            # Sequential processing
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                
                task = progress.add_task("Processing files...", total=len(file_paths))
                
                for file_path in file_paths:
                    progress.update(task, description=f"Processing {Path(file_path).name}...")
                    result = self.process_single_file(file_path, output_dir, move_processed)
                    results.append(result)
                    
                    file_name = Path(file_path).name
                    if result['success']:
                        if result['moved_to_processed']:
                            processing_messages.append(f"✅ {file_name} (moved to processed/)")
                        else:
                            processing_messages.append(f"✅ {file_name}")
                    else:
                        processing_messages.append(f"❌ {file_name}: {result['error']}")
                    
                    progress.advance(task)
        
        # Print all processing messages after progress is complete
        console.print("\n[bold]Processing Results:[/bold]")
        for message in processing_messages:
            console.print(message)
        
        return results
    
    def generate_batch_report(self, results: List[Dict], output_file: str = None) -> str:
        """Generate a comprehensive batch processing report"""
        
        # Calculate statistics
        total_files = len(results)
        successful = sum(1 for r in results if r['success'])
        failed = total_files - successful
        moved_count = sum(1 for r in results if r.get('moved_to_processed', False))
        
        total_duration = sum(r['duration'] for r in results if r['success'])
        total_processing_time = sum(r['processing_time'] for r in results)
        total_agent_time = sum(r['agent_talk_time'] for r in results if r['success'])
        total_client_time = sum(r['client_talk_time'] for r in results if r['success'])
        total_turns = sum(r['conversation_turns'] for r in results if r['success'])
        
        # Create report
        report = {
            'batch_summary': {
                'processing_date': datetime.now().isoformat(),
                'total_files': total_files,
                'successful_files': successful,
                'failed_files': failed,
                'moved_to_processed': moved_count,
                'success_rate': round((successful / total_files) * 100, 1) if total_files > 0 else 0,
                'total_audio_duration_seconds': round(total_duration, 2),
                'total_processing_time_seconds': round(total_processing_time, 2),
                'processing_speed_ratio': round(total_duration / total_processing_time, 2) if total_processing_time > 0 else 0
            },
            'aggregated_metrics': {
                'total_agent_talk_time': round(total_agent_time, 2),
                'total_client_talk_time': round(total_client_time, 2),
                'agent_dominance_ratio': round(total_agent_time / (total_agent_time + total_client_time), 2) if (total_agent_time + total_client_time) > 0 else 0,
                'average_conversation_turns': round(total_turns / successful, 1) if successful > 0 else 0,
                'total_conversation_turns': total_turns
            },
            'file_results': results,
            'errors': [
                {'file': r['file_path'], 'error': r['error']} 
                for r in results if not r['success']
            ],
            'processed_files': [
                {
                    'original_path': r['file_path'],
                    'final_audio_path': r.get('final_audio_path', r['file_path']),
                    'final_transcription_path': r.get('final_transcription_path', r.get('output_path')),
                    'moved_to_processed': r.get('moved_to_processed', False)
                }
                for r in results if r['success']
            ]
        }
        
        # Save report
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"batch_report_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return output_file

@click.command()
@click.option('--directory', '-d', help='Directory containing MP3 files')
@click.option('--files', '-f', multiple=True, help='Specific files to process')
@click.option('--pattern', '-p', default='*.mp3', help='File pattern to match (default: *.mp3)')
@click.option('--output-dir', '-o', help='Output directory for transcriptions')
@click.option('--model', '-m', default='base', help='Whisper model size')
@click.option('--device', default='auto', help='Device to use (auto, cpu, cuda, mps)')
@click.option('--language', '-l', default='auto', help='Language code (auto, es, en, fr, de, pt, it, etc.)')
@click.option('--workers', '-w', default=2, help='Number of parallel workers')
@click.option('--sequential', is_flag=True, help='Process files sequentially instead of parallel')
@click.option('--report', '-r', help='Output file for batch report')
@click.option('--force-reprocess', is_flag=True, help='Force reprocessing of all files, overriding the default of skipping them.')
@click.option('--move-processed', is_flag=True, help='Move processed files (audio + transcription) to a "processed" subdirectory')
@click.option('--use-pyannote', is_flag=True, help='Use advanced pyannote.audio speaker diarization (requires HuggingFace token)')
@click.option('--enhanced', is_flag=True, default=True, help='Use enhanced audio preprocessing and transcription settings (default: True)')
@click.option('--quality', type=click.Choice(['fast', 'balanced', 'high']), default='balanced', help='Quality preset: fast (tiny model), balanced (base/small), high (medium/large)')
def main(directory, files, pattern, output_dir, model, device, language, workers, sequential, report, force_reprocess, move_processed, use_pyannote, enhanced, quality):
    """
    Batch process multiple MP3 files for call transcription
    
    Examples:
    python3 batch_process.py --directory ./mp3 --quality high --language es
    python3 batch_process.py --directory ./mp3 --use-pyannote --enhanced
    python3 batch_process.py --directory ./mp3 --sequential --move-processed
    python3 batch_process.py --files call1.mp3 call2.mp3 --quality fast
    """
    
    console.print(Panel.fit(
        "[bold blue]Call Transcriber - Enhanced Batch Processing[/bold blue]\n"
        "Process multiple audio files with advanced speaker separation",
        border_style="blue"
    ))
    
    # Apply quality presets
    if quality == 'fast':
        if model == 'base':  # Only override if user didn't specify
            model = 'tiny'
        enhanced = False
        workers = min(workers, 4)  # More workers for faster processing
        console.print("[yellow]Using FAST preset: tiny model, basic processing[/yellow]")
    elif quality == 'high':
        if model == 'base':  # Only override if user didn't specify
            model = 'large'
        enhanced = True
        workers = min(workers, 2)  # Fewer workers for memory management
        console.print("[yellow]Using HIGH quality preset: large model, enhanced processing[/yellow]")
    else:  # balanced
        if model == 'base':
            model = 'medium' if language != 'auto' else 'base'
        console.print("[yellow]Using BALANCED preset: optimized quality/speed[/yellow]")
    
    # Pyannote.audio setup warning
    if use_pyannote:
        if not os.getenv('HUGGINGFACE_TOKEN'):
            console.print("[red]⚠️  Warning: HUGGINGFACE_TOKEN environment variable not set![/red]")
            console.print("[yellow]To use pyannote.audio, you need a HuggingFace token:[/yellow]")
            console.print("1. Get a token from https://huggingface.co/settings/tokens")
            console.print("2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1")
            console.print("3. Set: export HUGGINGFACE_TOKEN='your_token_here'")
            console.print("[cyan]Proceeding with enhanced heuristic method...[/cyan]")
            use_pyannote = False
    
    # Determine files to process
    file_paths = []
    
    if directory:
        if not os.path.exists(directory):
            console.print(f"[red]Error: Directory '{directory}' not found[/red]")
            sys.exit(1)
        
        processor = BatchProcessor(
            model=model, 
            device=device, 
            language=language, 
            max_workers=workers,
            use_pyannote=use_pyannote,
            enhanced=enhanced
        )
        all_files = processor.find_audio_files(directory, pattern)
        
        if not all_files:
            console.print(f"[yellow]No audio files found in '{directory}' matching pattern '{pattern}'[/yellow]")
            sys.exit(0)
        
        console.print(f"Found {len(all_files)} audio files in '{directory}'")
        
        # Filter out already processed files unless --force-reprocess is used
        if not force_reprocess:
            unprocessed_files, already_processed = processor.filter_unprocessed_files(all_files, output_dir)
            file_paths = unprocessed_files
            
            if already_processed:
                console.print(f"[yellow]Skipping {len(already_processed)} already processed files[/yellow]")
                for processed_file in already_processed[:5]:  # Show first 5
                    console.print(f"  - {Path(processed_file).name}")
                if len(already_processed) > 5:
                    console.print(f"  ... and {len(already_processed) - 5} more")
            
            if not file_paths:
                console.print("[green]✅ All files in directory have already been processed![/green]")
                if move_processed and already_processed:
                    console.print("\n[cyan]Moving already processed files to 'processed' directory...[/cyan]")
                    processed_dir = os.path.join(directory, "processed")
                    for audio_file in already_processed:
                        # Find corresponding transcription file in new or old locations
                        transcription_file = None
                        audio_dir = os.path.dirname(audio_file)
                        
                        # Check new transcriptions directory first
                        new_transcription_path = os.path.join(
                            audio_dir, "transcriptions",
                            f"{Path(audio_file).stem}_transcription.json"
                        )
                        
                        # Check old locations for backward compatibility
                        old_same_dir_path = os.path.join(
                            audio_dir,
                            f"{Path(audio_file).stem}_transcription.json"
                        )
                        old_current_dir_path = f"{Path(audio_file).stem}_transcription.json"
                        
                        if os.path.exists(new_transcription_path):
                            transcription_file = new_transcription_path
                        elif os.path.exists(old_same_dir_path):
                            transcription_file = old_same_dir_path
                        elif os.path.exists(old_current_dir_path):
                            transcription_file = old_current_dir_path
                        
                        processor.move_processed_files(audio_file, transcription_file, processed_dir)
                    console.print(f"[green]✅ Moved {len(already_processed)} processed files to 'processed' directory[/green]")
                sys.exit(0)
            
            console.print(f"[green]Processing {len(file_paths)} unprocessed files...[/green]")
        else:
            file_paths = all_files
            console.print(f"Processing all {len(file_paths)} files (forcing re-process)...")
    
    elif files:
        file_paths = list(files)
        # Verify all files exist
        for file_path in file_paths:
            if not os.path.exists(file_path):
                console.print(f"[red]Error: File '{file_path}' not found[/red]")
                sys.exit(1)
        
        processor = BatchProcessor(
            model=model, 
            device=device, 
            language=language, 
            max_workers=workers,
            use_pyannote=use_pyannote,
            enhanced=enhanced
        )
        
        # Filter out already processed files if requested
        if not force_reprocess:
            unprocessed_files, already_processed = processor.filter_unprocessed_files(file_paths, output_dir)
            file_paths = unprocessed_files
            
            if already_processed:
                console.print(f"[yellow]Skipping {len(already_processed)} already processed files[/yellow]")
            
            if not file_paths:
                console.print("[green]✅ All specified files have already been processed![/green]")
                sys.exit(0)
    
    else:
        console.print("[red]Error: Must specify either --directory or --files[/red]")
        sys.exit(1)
    
    # Process files
    console.print(f"\nProcessing {len(file_paths)} files...")
    features = []
    if enhanced:
        features.append("Enhanced Audio Processing")
    if use_pyannote:
        features.append("Advanced Speaker Diarization")
    else:
        features.append("Enhanced Heuristic Speaker Separation")
    
    console.print(f"Model: [cyan]{model}[/cyan] | Language: [cyan]{language}[/cyan] | Device: [cyan]{device}[/cyan]")
    console.print(f"Features: [cyan]{', '.join(features)}[/cyan]")
    
    if move_processed:
        console.print("[cyan]Files will be moved to 'processed' directory after transcription[/cyan]")
    
    try:
        results = processor.process_files(
            file_paths, 
            output_dir=output_dir, 
            parallel=not sequential,
            move_processed=move_processed
        )
        
        # Generate and save report
        report_file = processor.generate_batch_report(results, report)
        
        # Display summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        moved_count = sum(1 for r in results if r.get('moved_to_processed', False))
        
        summary_table = Table(title="Enhanced Batch Processing Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Files Processed", str(len(results)))
        summary_table.add_row("Successful", str(successful))
        summary_table.add_row("Failed", str(failed))
        summary_table.add_row("Success Rate", f"{(successful/len(results)*100):.1f}%" if len(results) > 0 else "0%")
        
        if move_processed:
            summary_table.add_row("Moved to Processed", str(moved_count))
        
        total_duration = sum(r['duration'] for r in results if r['success'])
        total_processing = sum(r['processing_time'] for r in results)
        
        summary_table.add_row("Total Audio Duration", f"{total_duration:.1f}s")
        summary_table.add_row("Total Processing Time", f"{total_processing:.1f}s")
        summary_table.add_row("Speed Ratio", f"{total_duration/total_processing:.1f}x" if total_processing > 0 else "N/A")
        
        # Quality metrics
        if successful > 0:
            avg_segments = sum(r['conversation_turns'] for r in results if r['success']) / successful
            avg_agent_ratio = sum(r['agent_talk_time'] / (r['agent_talk_time'] + r['client_talk_time']) for r in results if r['success'] and (r['agent_talk_time'] + r['client_talk_time']) > 0) / successful if successful > 0 else 0
            
            summary_table.add_row("Avg Conversation Turns", f"{avg_segments:.1f}")
            summary_table.add_row("Avg Agent/Client Balance", f"{avg_agent_ratio:.1%} / {1-avg_agent_ratio:.1%}")
        
        console.print(summary_table)
        
        if failed > 0:
            console.print(f"\n[yellow]⚠️  {failed} files failed to process. Check the report for details.[/yellow]")
        
        console.print(f"\n[bold green]✅ Enhanced batch processing complete![/bold green]")
        console.print(f"📄 Report saved to: [cyan]{report_file}[/cyan]")
        
        if output_dir:
            console.print(f"📁 Transcriptions saved to: [cyan]{output_dir}[/cyan]")
        
        if move_processed and moved_count > 0:
            console.print(f"📦 {moved_count} files moved to processed directory")
        
        # Quality recommendations
        if successful > 0:
            avg_segments = sum(r['conversation_turns'] for r in results if r['success']) / successful
            if avg_segments < 5:
                console.print("\n[yellow]💡 Quality tip: Low segment count detected. Consider using --quality high for better accuracy.[/yellow]")
            
            unbalanced_count = sum(1 for r in results if r['success'] and (
                r['agent_talk_time'] / (r['agent_talk_time'] + r['client_talk_time']) > 0.9 or 
                r['agent_talk_time'] / (r['agent_talk_time'] + r['client_talk_time']) < 0.1
            ) if (r['agent_talk_time'] + r['client_talk_time']) > 0)
            
            if unbalanced_count > successful * 0.3:  # More than 30% unbalanced
                console.print("\n[yellow]💡 Speaker tip: Many files have unbalanced speaker detection. Consider using --use-pyannote for better results.[/yellow]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error during batch processing: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main() 