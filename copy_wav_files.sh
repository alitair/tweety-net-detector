#!/bin/bash

# Check if source and destination directories are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_directory> <destination_directory>"
    echo "Example: $0 /path/to/source /path/to/destination"
    exit 1
fi

SOURCE_DIR="$1"
DEST_DIR="$2"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist"
    exit 1
fi

# Check if destination directory exists
if [ ! -d "$DEST_DIR" ]; then
    echo "Error: Destination directory '$DEST_DIR' does not exist"
    exit 1
fi

# Find all WAV files in source directory
find "$SOURCE_DIR" -type f -name "*.wav" | while read -r wav_file; do
    # Get the relative path of the file
    rel_path="${wav_file#$SOURCE_DIR/}"
    # Get the directory part of the relative path
    dir_path=$(dirname "$rel_path")
    # Construct the destination path
    dest_path="$DEST_DIR/$dir_path"
    
    # Check if the destination subdirectory exists
    if [ -d "$dest_path" ]; then
        # Create full destination path for the file
        dest_file="$DEST_DIR/$rel_path"
        # Copy the file
        echo "Copying: $rel_path"
        cp "$wav_file" "$dest_file"
    else
        echo "Skipping: $rel_path (destination directory does not exist)"
    fi
done

echo "Copy complete!" 