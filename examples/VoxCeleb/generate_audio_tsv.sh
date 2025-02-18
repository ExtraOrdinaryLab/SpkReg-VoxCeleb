#!/bin/bash

# Define output files
MANIFEST_SAVE_PATH="/mnt/data1_HDD_14TB/yang/corpus/audio/RIRS_NOISES/rir.tsv"

# Define root paths
ROOT_DIR="/mnt/data1_HDD_14TB/yang/corpus/audio/RIRS_NOISES"

# Function to process files and save to TSV
generate_tsv() {
    local dataset_dir=$1
    local output_file=$2
    
    # Find all .wav files, extract speaker ID and save
    find "$dataset_dir" -type f -name "*.wav" | while read -r filepath; do
        echo -e "$filepath"
    done > "$output_file"

    # Count and display the number of lines
    echo "$(wc -l < "$output_file") lines written to $output_file"
}

# Generate train.tsv and test.tsv
generate_tsv "$ROOT_DIR" "$MANIFEST_SAVE_PATH"

echo "TSV files generated:"
echo " - $MANIFEST_SAVE_PATH"