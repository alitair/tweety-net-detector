#!/usr/bin/env python3

import argparse
from pathlib import Path
import soundfile as sf
import os

def is_single_channel_wav(wav_file: Path, verbose: bool = True) -> bool:
    """
    Check if a WAV file is single channel both by filename pattern and actual audio channels.
    
    Args:
        wav_file (Path): Path to the WAV file
        verbose (bool): Whether to print detailed progress
        
    Returns:
        bool: True if the file is a single channel WAV file, False otherwise
    """
    # Check filename pattern first
    parts = wav_file.stem.split('-')
    if len(parts) > 2:  # Has timestamp and multiple channel IDs
        return False
        
    # Verify audio channels
    try:
        with sf.SoundFile(wav_file) as audio_file:
            is_single = audio_file.channels == 1
            if verbose and is_single:
                print(f"Found single channel file: {wav_file}")
            return is_single
    except Exception as e:
        print(f"Error reading {wav_file}: {str(e)}")
        return False

def remove_single_channel_wavs(root_dir: str, dry_run: bool = False, verbose: bool = True):
    """
    Remove all single channel WAV files in a directory and its subdirectories.
    
    Args:
        root_dir (str): Root directory to search for WAV files
        dry_run (bool): If True, only print what would be done without actually deleting
        verbose (bool): Whether to print detailed progress
    """
    root_path = Path(root_dir)
    wav_files = list(root_path.rglob("*.wav"))
    
    if not wav_files:
        print("No WAV files found")
        return
    
    if verbose:
        print(f"Found {len(wav_files)} WAV files to examine")
    
    files_to_remove = []
    
    # First identify all files to remove
    for wav_file in wav_files:
        if is_single_channel_wav(wav_file, verbose):
            files_to_remove.append(wav_file)
    
    # Report findings
    if len(files_to_remove) == 0:
        print("No single channel WAV files found to remove")
        return
        
    print(f"\nFound {len(files_to_remove)} single channel WAV files:")
    for file in files_to_remove:
        print(f"  {file}")
    
    # Remove files if not in dry run mode
    if not dry_run:
        print(f"\nRemoving {len(files_to_remove)} files...")
        for file in files_to_remove:
            try:
                os.remove(file)
                if verbose:
                    print(f"Removed: {file}")
            except Exception as e:
                print(f"Error removing {file}: {str(e)}")
    else:
        print("\nDRY RUN - No files were actually removed")

def main():
    parser = argparse.ArgumentParser(
        description="Remove single channel WAV files while preserving multi-channel files"
    )
    parser.add_argument(
        "--root_dir", 
        type=str, 
        required=True,
        help="Root directory to search for WAV files"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Print what would be done without actually removing files"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Reduce verbosity of output"
    )
    
    args = parser.parse_args()
    
    if not Path(args.root_dir).exists():
        print(f"Error: Directory {args.root_dir} does not exist")
        return
    
    remove_single_channel_wavs(args.root_dir, args.dry_run, verbose=not args.quiet)

if __name__ == "__main__":
    main() 