#!/bin/bash

# Define output files
TRAIN_TSV="/path/to/save/train.tsv"
TEST_TSV="/path/to/save/test.tsv"

# Define root paths
DEV_DIR="/path/to/dev/wav"
TEST_DIR="/path/to/test/wav"

# Function to process files and save to TSV
generate_tsv() {
    local dataset_dir=$1
    local output_file=$2
    
    # Find all .wav files, extract speaker ID and save
    find "$dataset_dir" -type f -name "*.wav" | while read -r filepath; do
        # Extract speaker ID (idXXXX from the path)
        speaker_id=$(echo "$filepath" | awk -F'/' '{print $(NF-2)}')
        echo -e "$filepath\t$speaker_id"
    done > "$output_file"

    # Count and display the number of lines
    echo "$(wc -l < "$output_file") lines written to $output_file"
}

# Generate train.tsv and test.tsv
generate_tsv "$DEV_DIR" "$TRAIN_TSV"
generate_tsv "$TEST_DIR" "$TEST_TSV"

echo "TSV files generated:"
echo " - $TRAIN_TSV"
echo " - $TEST_TSV"