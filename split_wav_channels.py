#!/usr/bin/env python3

import argparse
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

def split_wav_channels(wav_file: Path, verbose: bool = True) -> list:
    """
    Split a multi-channel WAV file into separate mono files.
    Only processes files that:
    1. Have a filename pattern of timestamp-channel1-channel2-...channelN.wav
    2. Actually contain multiple channels in the audio data
    
    The channel mapping is:
    - Channel 0 = Left channel in stereo file
    - Channel 1 = Right channel in stereo file
    
    Args:
        wav_file (Path): Path to the multi-channel WAV file
        verbose (bool): Whether to print detailed progress
        
    Returns:
        list: List of (channel_number, channel_path, channel_id) tuples
    """
    if verbose:
        print(f"\nExamining file: {wav_file}")
    
    # First check if this is actually a multi-channel file
    try:
        # Use soundfile to read the audio data as it preserves channel order
        with sf.SoundFile(wav_file) as audio_file:
            if audio_file.channels == 1:
                if verbose:
                    print(f"Skipping: File is mono: {wav_file}")
                return []
            num_channels = audio_file.channels
            # Read the entire file
            audio = audio_file.read().T  # Transpose to get (channels, samples)
            sr = audio_file.samplerate
            
            if verbose:
                print(f"Audio file has {num_channels} channels at {sr}Hz")
    except Exception as e:
        print(f"Error reading audio file {wav_file}: {str(e)}")
        return []
    
    # Parse the filename to get timestamp and channel IDs
    filename = wav_file.stem
    parts = filename.split('-')
    
    # Check for correct filename pattern
    if len(parts) < 3:  # Need timestamp and at least 2 channel IDs
        if verbose:
            print(f"Skipping: Not a multi-channel filename format: {filename}")
        return []
        
    timestamp = parts[0]
    channel_ids = parts[1:]  # List of channel IDs
    
    # Verify the number of channels matches the filename
    if len(channel_ids) != num_channels:
        if verbose:
            print(f"Skipping: Filename specifies {len(channel_ids)} channels but audio has {num_channels} channels")
        return []
    
    if verbose:
        print(f"Found {len(channel_ids)} channels: {channel_ids}")
        print("Channel mapping:")
        for i, channel_id in enumerate(channel_ids):
            print(f"  Audio channel {i} -> {channel_id}")
    
    channel_files = []
    missing_channels = []
    
    # First check which channels already exist
    for i, channel_id in enumerate(channel_ids):
        channel_filename = f"{timestamp}-{channel_id}.wav"
        channel_path = wav_file.parent / channel_filename
        if channel_path.exists():
            if verbose:
                print(f"Found existing channel file: {channel_path}")
            channel_files.append((i, channel_path, channel_id))
        else:
            if verbose:
                print(f"Channel file missing: {channel_path}")
            missing_channels.append((i, channel_id))
    
    # If some channels are missing, write them out
    if missing_channels:
        if verbose:
            print(f"Need to create files for channels: {[id for _, id in missing_channels]}")
        
        # Write only the missing channel files
        for channel_num, channel_id in missing_channels:
            # Create filename for this channel
            channel_filename = f"{timestamp}-{channel_id}.wav"
            channel_path = wav_file.parent / channel_filename
            
            try:
                # Write the channel data to a new file
                if verbose:
                    print(f"Writing channel {channel_id} to {channel_path}")
                # Extract single channel and ensure it's 1D
                channel_data = audio[channel_num].reshape(-1)
                sf.write(str(channel_path), channel_data, sr)
                
                # Verify the file was written
                if channel_path.exists():
                    if verbose:
                        print(f"Successfully wrote {channel_path}")
                    channel_files.append((channel_num, channel_path, channel_id))
                else:
                    print(f"ERROR: Failed to write {channel_path}")
            except Exception as e:
                print(f"Error writing channel {channel_id}: {str(e)}")
                continue
    
    # Sort by channel number to maintain order
    channel_files.sort(key=lambda x: x[0])
    return channel_files

def process_directory(root_dir: str, verbose: bool = True):
    """
    Process all WAV files in a directory and its subdirectories.
    
    Args:
        root_dir (str): Root directory to search for WAV files
        verbose (bool): Whether to print detailed progress
    """
    root_path = Path(root_dir)
    wav_files = list(root_path.rglob("*.wav"))
    
    if not wav_files:
        print("No WAV files found")
        return
    
    print(f"Found {len(wav_files)} WAV files")
    
    for wav_file in wav_files:
        split_wav_channels(wav_file, verbose)

def main():
    parser = argparse.ArgumentParser(description="Split multi-channel WAV files into single channel files")
    parser.add_argument("--root_dir", type=str, required=True,
                      help="Root directory to search for WAV files")
    parser.add_argument("--quiet", action="store_true",
                      help="Reduce verbosity of output")
    
    args = parser.parse_args()
    
    if not Path(args.root_dir).exists():
        print(f"Error: Directory {args.root_dir} does not exist")
        return
    
    process_directory(args.root_dir, verbose=not args.quiet)

if __name__ == "__main__":
    main() 