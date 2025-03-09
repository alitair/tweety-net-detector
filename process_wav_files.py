import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import librosa
import soundfile as sf
import numpy as np
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

def split_wav_channels(wav_file: Path) -> list:
    """
    Split a multi-channel WAV file into separate mono files.
    
    Args:
        wav_file (Path): Path to the multi-channel WAV file
        
    Returns:
        list: List of (channel_number, channel_path, channel_id) tuples
    """
    # Parse the filename to get timestamp and channel IDs
    filename = wav_file.stem
    parts = filename.split('-')
    if len(parts) < 2:  # Need at least timestamp and one channel
        print(f"Warning: Unexpected filename format: {filename}")
        return []
        
    timestamp = parts[0]
    channel_ids = parts[1:]  # List of channel IDs
    
    # Create temporary directory for split files if it doesn't exist
    temp_dir = wav_file.parent / "temp_channels"
    temp_dir.mkdir(exist_ok=True)
    
    channel_files = []
    existing_channels = []
    missing_channels = []
    
    # First check which channels already exist
    for i, channel_id in enumerate(channel_ids):
        channel_filename = f"{timestamp}-{channel_id}.wav"
        channel_path = temp_dir / channel_filename
        if channel_path.exists():
            channel_files.append((i, channel_path, channel_id))
            existing_channels.append(channel_id)
        else:
            missing_channels.append((i, channel_id))
    
    # If all channels exist, return them
    if len(existing_channels) == len(channel_ids):
        print(f"All channel files already exist for {wav_file.name}")
        return channel_files
    
    # If some channels are missing, we need to read and split the audio
    print(f"Splitting channels for {wav_file.name} (missing: {[id for _, id in missing_channels]})")
    
    # Read the multi-channel audio
    audio, sr = librosa.load(wav_file, sr=None, mono=False)
    if len(audio.shape) == 1:
        audio = audio.reshape(1, -1)  # Convert mono to shape (1, samples)
    
    # Verify we have enough channels in the audio file
    if audio.shape[0] < len(channel_ids):
        print(f"Warning: File has {audio.shape[0]} channels but filename specifies {len(channel_ids)} channels")
        return []
    
    # Write only the missing channel files
    for channel_num, channel_id in missing_channels:
        if channel_num >= audio.shape[0]:
            print(f"Warning: Channel {channel_num} requested but file only has {audio.shape[0]} channels")
            continue
            
        # Create filename for this channel
        channel_filename = f"{timestamp}-{channel_id}.wav"
        channel_path = temp_dir / channel_filename
        
        # Write the channel data to a new file
        sf.write(str(channel_path), audio[channel_num], sr)
        
        channel_files.append((channel_num, channel_path, channel_id))
    
    # Sort by channel number to maintain order
    channel_files.sort(key=lambda x: x[0])
    return channel_files

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
    
    # If JSON file doesn't exist, we need to process
    if not json_path.exists():
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
    
    # Create a single Inference object to reuse for all files
    print("\nInitializing TweetyNet model...")
    sorter = Inference(
        input_path="",  # Will be set for each file
        output_path="",  # Will be set for each file
        model_path=model_path,
        plot_spec_results=False,  # Disable spectrogram generation
        create_json=True,
        separate_json=True,
        threshold=0.5,
        min_length=500,
        pad_song=50,
        progress_callback=None  # Will be set for each file
    )
    print("Model initialized successfully")
    
    # Process each WAV file
    start_time = time.time()
    processing_times = []
    total_channels_processed = 0
    
    for wav_file in tqdm(wav_files, desc="Processing files"):
        try:
            print(f"\nProcessing multi-channel file: {wav_file}")
            
            # Split the channels
            channel_files = split_wav_channels(wav_file)
            if not channel_files:
                print(f"Warning: No channels extracted from {wav_file}")
                continue
            
            print(f"Processing {len(channel_files)} channels")
            
            # Process each channel
            for channel_num, channel_path, channel_id in channel_files:
                try:
                    output_dir = str(wav_file.parent)
                    file_size_mb = channel_path.stat().st_size / (1024 * 1024)
                    
                    print(f"\nProcessing channel {channel_num} ({channel_id})")
                    print(f"File size: {file_size_mb:.2f} MB")
                    print(f"Output directory: {output_dir}")
                    
                    if not should_process_file(channel_path, output_dir, False):
                        continue
                    
                    file_start_time = time.time()
                    progress_tracker = FileProgressTracker()
                    
                    # Update paths and callback for this channel
                    sorter.input_path = str(channel_path)
                    sorter.output_path = output_dir
                    sorter.progress_callback = progress_tracker.update_step
                    
                    # Process the channel
                    progress_tracker.start_step(f"Processing channel {channel_id}")
                    result = sorter.sort_single_song(str(channel_path))
                    progress_tracker.finish_step()
                    
                    # Calculate processing time and speed
                    processing_time = time.time() - file_start_time
                    processing_times.append(processing_time)
                    speed_mbps = file_size_mb / processing_time
                    
                    print(f"Processing time: {format_time(processing_time)} ({speed_mbps:.2f} MB/s)")
                    
                    if result is None:
                        print(f"Successfully processed channel {channel_id}")
                        total_channels_processed += 1
                    else:
                        print(f"Warning: Unexpected result while processing channel {channel_id}")
                    
                except Exception as e:
                    print(f"Error processing channel {channel_id}: {str(e)}")
                    continue
                
        except Exception as e:
            print(f"Error processing {wav_file}: {str(e)}")
            continue
    
    total_time = time.time() - start_time
    
    print("\nProcessing Summary:")
    print(f"Total processing time: {format_time(total_time)}")
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        print(f"Average time per channel: {format_time(avg_time)}")
    print(f"Total channels processed: {total_channels_processed}")

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