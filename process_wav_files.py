import os
import sys
import argparse
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from src.inference import Inference

def get_default_model_path():
    """Get the path to the default model directory."""
    script_dir = Path(__file__).parent
    return str(script_dir / "files" / "canary_fall_nerve_llb-.01")

def split_wav_channels(wav_file: Path) -> list:
    """
    Split a multi-channel WAV file into separate mono files.
    
    Args:
        wav_file (Path): Path to the multi-channel WAV file
        
    Returns:
        list: List of (channel_number, channel_path, channel_id) tuples
    """
    # First check if results already exist for this file
    results_json = wav_file.parent / f"{wav_file.name}_results.json"
    if results_json.exists():
        print(f"Skipping {wav_file.name} - results already exist")
        return []

    # Parse the filename to get timestamp and channel IDs
    filename = wav_file.stem
    parts = filename.split('-')
    if len(parts) < 2:  # Need at least timestamp and one channel
        print(f"Warning: Unexpected filename format: {filename}")
        return []
        
    timestamp = parts[0]
    channel_ids = parts[1:]  # List of channel IDs
    
    channel_files = []
    existing_channels = []
    missing_channels = []
    
    # First check which channels already exist
    for i, channel_id in enumerate(channel_ids):
        channel_filename = f"{timestamp}-{channel_id}.wav"
        channel_path = wav_file.parent / channel_filename
        if channel_path.exists():
            channel_files.append((i, channel_path, channel_id))
            existing_channels.append(channel_id)
        else:
            missing_channels.append((i, channel_id))
    
    # If all channels exist, return them
    if len(existing_channels) == len(channel_ids):
        return channel_files
    
    # If some channels are missing, we need to read and split the audio
    if missing_channels:
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
            channel_path = wav_file.parent / channel_filename
            
            try:
                # Write the channel data to a new file
                sf.write(str(channel_path), audio[channel_num], sr)
                
                # Verify the file was written
                if channel_path.exists():
                    channel_files.append((channel_num, channel_path, channel_id))
                else:
                    print(f"ERROR: Failed to write {channel_path}")
            except Exception as e:
                print(f"Error writing channel {channel_id}: {str(e)}")
                continue
    
    # Sort by channel number to maintain order
    channel_files.sort(key=lambda x: x[0])
    return channel_files

def should_process_file(wav_file: Path) -> bool:
    """
    Check if a wav file needs to be processed by checking for existing output files.
    
    Args:
        wav_file (Path): Path to the wav file
        
    Returns:
        bool: True if the file needs processing, False if outputs already exist
    """
    results_json = wav_file.parent / f"{wav_file.name}_results.json"
    return not results_json.exists()

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
        wav_files = []
        with open(json_list_path, 'r') as f:
            for line in f:
                json_path = line.strip()
                if json_path.endswith('_f0.json'):
                    wav_path = json_path[:-8] + '.wav'
                    wav_file = Path(wav_path)
                    if wav_file.exists():
                        wav_files.append(wav_file)
    else:
        wav_files = list(Path('.').rglob("*.wav"))
    
    if not wav_files:
        print("No WAV files found")
        return
    
    # Create Inference object
    sorter = Inference(
        input_path="",  # Will be set for each file
        output_path="",  # Will be set for each file
        model_path=model_path,
        plot_spec_results=False,
        create_json=True,
        separate_json=True,
        threshold=0.5,
        min_length=500,
        pad_song=50
    )
    
    # Process each WAV file
    for wav_file in wav_files:
        try:
            # Check if results already exist
            if not should_process_file(wav_file):
                print(f"Skipping {wav_file.name} - results already exist")
                continue

            # First check if any channels need processing
            filename = wav_file.stem
            parts = filename.split('-')
            if len(parts) < 2:  # Need at least timestamp and one channel
                print(f"Warning: Unexpected filename format: {filename}")
                continue
                
            timestamp = parts[0]
            channel_ids = parts[1:]  # List of channel IDs
            
            # Check if all channels are already processed
            all_processed = True
            for channel_id in channel_ids:
                channel_filename = f"{timestamp}-{channel_id}.wav"
                channel_path = wav_file.parent / channel_filename
                if should_process_file(channel_path):
                    all_processed = False
                    break
            
            if all_processed:
                print(f"Skipping {wav_file.name} - all channels already processed")
                continue
            
            # Only split channels if we need to process at least one
            channel_files = split_wav_channels(wav_file)
            if not channel_files:
                continue
            
            # Process each channel
            for channel_num, channel_path, channel_id in channel_files:
                try:
                    output_dir = str(wav_file.parent)
                    
                    if not should_process_file(channel_path):
                        print(f"Skipping {channel_path.name} - results already exist")
                        continue
                    
                    print(f"Processing {channel_path.name}")
                    
                    # Update paths for this channel
                    sorter.input_path = str(channel_path)
                    sorter.output_path = output_dir
                    
                    # Process the channel
                    result = sorter.sort_single_song(str(channel_path))
                    
                except Exception as e:
                    print(f"Error processing channel {channel_id}: {str(e)}")
                    continue
                
        except Exception as e:
            print(f"Error processing {wav_file}: {str(e)}")
            continue

def find_f0_json_files(root_dir: str) -> list:
    """
    Recursively find all _f0.json files in the given root directory.
    
    Args:
        root_dir (str): Root directory to search in
        
    Returns:
        list: List of Path objects for _f0.json files
    """
    root_path = Path(root_dir)
    f0_json_files = list(root_path.rglob("*_f0.json"))
    print(f"Found {len(f0_json_files)} _f0.json files")
    return f0_json_files

def main():
    parser = argparse.ArgumentParser(description="Process wav files recursively using TweetyNet")
    parser.add_argument("--json_list", type=str,
                      help="Path to text file containing list of JSON paths")
    parser.add_argument("--model_path", type=str,
                      help="Path to the model directory (optional, will use default if not specified)")
    parser.add_argument("--root_dir", type=str,
                      help="Root directory to search for _f0.json files")
    
    args = parser.parse_args()
    
    if args.root_dir:
        # Find all _f0.json files
        f0_json_files = find_f0_json_files(args.root_dir)
        if not f0_json_files:
            print("No _f0.json files found in the specified directory")
            sys.exit(1)
            
        # Create a temporary file to store the list of f0_json files
        temp_list = "temp_f0_list.txt"
        with open(temp_list, 'w') as f:
            for json_file in f0_json_files:
                f.write(f"{json_file}\n")
        
        # Process files using the temporary list
        process_wav_files(args.model_path, temp_list)
        
        # Clean up temporary file
        os.remove(temp_list)
        
    elif args.json_list:
        # Process files using the JSON list
        process_wav_files(args.model_path, args.json_list)
    else:
        print("Error: Either --root_dir or --json_list must be provided")
        sys.exit(1)

if __name__ == "__main__":
    main() 