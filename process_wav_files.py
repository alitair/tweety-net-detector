import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
from src.inference import Inference

def get_default_model_path():
    """Get the path to the default model directory."""
    script_dir = Path(__file__).parent
    return str(script_dir / "files" / "canary_fall_nerve_llb-.01")

def format_time(seconds):
    """Convert seconds to a human-readable format."""
    return str(timedelta(seconds=int(seconds)))

def format_progress(desc, current, total, elapsed_time, extra_info=""):
    """Format progress information for a processing step."""
    percentage = (current / total) * 100 if total > 0 else 0
    speed = current / elapsed_time if elapsed_time > 0 else 0
    return f"{desc}: {percentage:.1f}% [{current}/{total}] - {format_time(elapsed_time)} elapsed - {speed:.2f} units/s {extra_info}"

def should_process_file(wav_file: Path, output_dir: str, plot_spec_results: bool) -> bool:
    """
    Check if a wav file needs to be processed by checking for existing output files.
    
    Args:
        wav_file (Path): Path to the wav file
        output_dir (str): Output directory path
        plot_spec_results (bool): Whether spectrograms should be generated
        
    Returns:
        bool: True if the file needs processing, False if outputs already exist
    """
    wav_name = wav_file.stem
    json_path = Path(output_dir) / f"{wav_file.name}_results.json"
    spec_path = Path(output_dir) / "specs" / f"{wav_file.name}_detection.png"
    
    # If JSON file doesn't exist, we need to process
    if not json_path.exists():
        return True
        
    # If spectrograms are required and the spectrogram doesn't exist, we need to process
    if plot_spec_results and not spec_path.exists():
        return True
        
    print(f"Skipping {wav_file.name} - output files already exist")
    return False

class FileProgressTracker:
    def __init__(self, total_steps=4):
        self.start_time = time.time()
        self.step_times = {}
        self.current_step = None
        self.total_steps = total_steps
    
    def start_step(self, step_name):
        """Start timing a new processing step."""
        self.current_step = step_name
        self.step_times[step_name] = {
            'start': time.time(),
            'last_update': time.time(),
            'progress': 0,
            'total': 100
        }
    
    def update_step(self, progress, total=100, extra_info=""):
        """Update progress for current step."""
        if self.current_step:
            now = time.time()
            self.step_times[self.current_step]['progress'] = progress
            self.step_times[self.current_step]['total'] = total
            elapsed = now - self.step_times[self.current_step]['start']
            
            # Only update display every 0.5 seconds to avoid flooding the console
            if now - self.step_times[self.current_step]['last_update'] >= 0.5:
                print(f"\r{format_progress(self.current_step, progress, total, elapsed, extra_info)}", end="")
                self.step_times[self.current_step]['last_update'] = now
    
    def finish_step(self):
        """Finish current processing step."""
        if self.current_step:
            elapsed = time.time() - self.step_times[self.current_step]['start']
            print(f"\n{self.current_step} completed in {format_time(elapsed)}")
            self.current_step = None

def get_wav_files_from_json_list(json_list_path: str) -> list:
    """
    Read a list of JSON file paths and convert them to corresponding WAV file paths.
    
    Args:
        json_list_path (str): Path to the text file containing JSON paths
        
    Returns:
        list: List of Path objects for WAV files
    """
    wav_files = []
    
    with open(json_list_path, 'r') as f:
        for line in f:
            # Clean the line and get the JSON path
            json_path = line.strip()
            if not json_path:
                continue
                
            # Convert JSON path to WAV path by removing _f0.json suffix
            if json_path.endswith('_f0.json'):
                wav_path = json_path[:-8] + '.wav'  # Remove _f0.json and add .wav
            else:
                print(f"Warning: Unexpected JSON path format: {json_path}")
                continue
            
            # Use path relative to current directory
            wav_file = Path(wav_path)
            if wav_file.exists():
                wav_files.append(wav_file)
            else:
                print(f"Warning: WAV file not found: {wav_file}")
    
    return wav_files

def process_wav_files(model_path: str = None, json_list_path: str = None):
    """
    Process WAV files based on a list of JSON paths.
    
    Args:
        model_path (str): Path to the model directory (optional)
        json_list_path (str): Path to the text file containing JSON paths (optional)
    """
    # Use default model path if none provided
    if model_path is None:
        model_path = get_default_model_path()
        print(f"Using default model path: {model_path}")
    
    # Get list of WAV files
    if json_list_path:
        wav_files = get_wav_files_from_json_list(json_list_path)
        print(f"Found {len(wav_files)} WAV files from JSON list")
    else:
        # Fallback to current directory
        wav_files = list(Path('.').rglob("*.wav"))
        print(f"Found {len(wav_files)} WAV files by recursive search")
    
    if not wav_files:
        print(f"No wav files found")
        return
    
    # First pass to count files that need processing
    files_to_process = []
    for wav_file in wav_files:
        if should_process_file(wav_file, str(wav_file.parent), True):
            files_to_process.append(wav_file)
    
    if not files_to_process:
        print("No files need processing - all outputs exist.")
        return
        
    print(f"\nWill process {len(files_to_process)} files")
    print(f"Skipping {len(wav_files) - len(files_to_process)} files (outputs already exist)")
    
    # Process files with progress bar
    start_time = time.time()
    processing_times = []
    
    with tqdm(total=len(files_to_process), desc="Processing files") as pbar:
        for wav_file in files_to_process:
            try:
                # Get the directory containing the wav file
                output_dir = str(wav_file.parent)
                file_size_mb = wav_file.stat().st_size / (1024 * 1024)  # File size in MB
                
                print(f"\nProcessing: {wav_file}")
                print(f"File size: {file_size_mb:.2f} MB")
                print(f"Output directory: {output_dir}")
                
                file_start_time = time.time()
                progress_tracker = FileProgressTracker()
                
                # Initialize the inference object for this file
                progress_tracker.start_step("Initializing")
                sorter = Inference(
                    input_path=str(wav_file),
                    output_path=output_dir,
                    model_path=model_path,
                    plot_spec_results=True,  # Generate spectrograms
                    create_json=True,        # Create JSON output
                    separate_json=True,      # Create separate JSON for each file
                    threshold=0.5,           # Default threshold
                    min_length=500,          # Minimum segment length (ms)
                    pad_song=50,            # Padding around segments (ms)
                    progress_callback=progress_tracker.update_step  # Add progress tracking
                )
                progress_tracker.finish_step()
                
                # Process the file
                progress_tracker.start_step("Processing audio")
                result = sorter.sort_single_song(str(wav_file))
                progress_tracker.finish_step()
                
                # Calculate processing time and speed
                processing_time = time.time() - file_start_time
                processing_times.append(processing_time)
                speed_mbps = file_size_mb / processing_time
                
                # Calculate statistics
                avg_time = sum(processing_times) / len(processing_times)
                remaining_files = len(files_to_process) - len(processing_times)
                est_remaining_time = remaining_files * avg_time
                
                print(f"Processing time: {format_time(processing_time)} ({speed_mbps:.2f} MB/s)")
                print(f"Average processing time: {format_time(avg_time)}")
                if remaining_files > 0:
                    print(f"Estimated time remaining: {format_time(est_remaining_time)}")
                
                if result is None:
                    print(f"Successfully processed {wav_file}")
                else:
                    print(f"Warning: Unexpected result while processing {wav_file}")
                    
                pbar.update(1)
                
            except Exception as e:
                print(f"Error processing {wav_file}: {str(e)}")
                continue
    
    total_time = time.time() - start_time
    avg_time_per_file = total_time / len(files_to_process)
    
    print("\nProcessing Summary:")
    print(f"Total processing time: {format_time(total_time)}")
    print(f"Average time per file: {format_time(avg_time_per_file)}")
    print(f"Files processed: {len(files_to_process)}")
    print(f"Files skipped: {len(wav_files) - len(files_to_process)}")

def main():
    parser = argparse.ArgumentParser(description="Process wav files recursively using TweetyNet")
    parser.add_argument("--json_list", type=str, required=True,
                      help="Path to text file containing list of JSON paths")
    parser.add_argument("--model_path", type=str,
                      help="Path to the model directory (optional, will use default if not specified)")
    
    args = parser.parse_args()
    
    process_wav_files(args.model_path, args.json_list)

if __name__ == "__main__":
    main() 