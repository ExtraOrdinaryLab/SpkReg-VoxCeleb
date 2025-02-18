#!/bin/bash

# Define the source and target base directories
SRC_DIR="/mnt/data1_HDD_14TB/yang/corpus/audio/VoxCeleb2/test/aac"
DEST_DIR="/mnt/data1_HDD_14TB/yang/corpus/audio/VoxCeleb2/test/wav"

# Find all .m4a files and process them
find "$SRC_DIR" -type f -name "*.m4a" | while read -r src_file; do
    # Add missing "/" at the beginning if needed
    if [[ "$src_file" != /* ]]; then
        src_file="/$src_file"
    fi

    # Construct the corresponding destination path
    dest_file="${src_file/$SRC_DIR/$DEST_DIR}"
    dest_file="${dest_file%.m4a}.wav"

    # Create the destination directory if it doesn't exist
    mkdir -p "$(dirname "$dest_file")"
    
    # Convert the file using ffmpeg
    ffmpeg -n -i "$src_file" -acodec pcm_s16le -ar 16000 "$dest_file"

done