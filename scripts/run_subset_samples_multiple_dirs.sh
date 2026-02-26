#!/bin/bash

# Script to run subset_samples_for_score.py for multiple directories
# This script handles the conda environment switching for convert_cg2all

# Configuration
SCRIPT_DIR="$HOME/proteina"
MAIN_SCRIPT="$SCRIPT_DIR/subset_samples_for_score_main.py"
CONVERT_SCRIPT="$SCRIPT_DIR/convert_cg2all_files.py"
ALN_FILE="aln.tsv"
BIN_NUM=10
NUM_SAMPLES_PER_BIN=10

# Directories to process
DIRECTORIES=(
    "$HOME/proteina/inference/inference_seq_cath_cond_sampling_finetune-all_8-seq_purge-7bny-7kww-7ad5_045-noise/7ad5_A/"
    "$HOME/proteina/inference/inference_seq_cond_sampling_finetune-all_8-seq_purge-7bny-7kww-7ad5_045-noise/7ad5_A/"
)

# Function to run the main sampling script
run_main_sampling() {
    local dir="$1"
    echo "=========================================="
    echo "Running main sampling for directory: $dir"
    echo "=========================================="
    
    conda run -n proteina python "$MAIN_SCRIPT" \
        -i "$dir" \
        -f "$ALN_FILE" \
        -b "$BIN_NUM" \
        -n "$NUM_SAMPLES_PER_BIN"
    
    if [ $? -eq 0 ]; then
        echo "Main sampling completed successfully for $dir"
    else
        echo "Error: Main sampling failed for $dir"
        return 1
    fi
}

# Function to run the convert_cg2all script in the cg2all conda environment
run_convert_cg2all() {
    local dir="$1"
    echo "=========================================="
    echo "Running convert_cg2all for directory: $dir"
    echo "=========================================="
    
    # Activate the cg2all conda environment and run the conversion
    conda run -n cg2all python "$CONVERT_SCRIPT" \
        -i "$dir" \
        -f "$ALN_FILE"
    
    if [ $? -eq 0 ]; then
        echo "convert_cg2all completed successfully for $dir"
    else
        echo "Error: convert_cg2all failed for $dir"
        return 1
    fi
}

# Main execution
main() {
    echo "Starting batch processing of directories..."
    echo "Number of directories to process: ${#DIRECTORIES[@]}"
    echo ""
    
    for dir in "${DIRECTORIES[@]}"; do
        echo "Processing directory: $dir"
        
        # Step 1: Run the main sampling script
        if run_main_sampling "$dir"; then
            echo "Main sampling successful for $dir"
        else
            echo "Skipping convert_cg2all for $dir due to main sampling failure"
            continue
        fi
        
        # Step 2: Run the convert_cg2all script
        if run_convert_cg2all "$dir"; then
            echo "convert_cg2all successful for $dir"
        else
            echo "convert_cg2all failed for $dir"
        fi
        
        echo ""
    done
    
    echo "Batch processing completed!"
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Check if the required conda environments exist
if ! conda env list | grep -q "proteina"; then
    echo "Error: conda environment 'proteina' not found"
    echo "Please create the proteina environment first"
    exit 1
fi

if ! conda env list | grep -q "cg2all"; then
    echo "Error: conda environment 'cg2all' not found"
    echo "Please create the cg2all environment first"
    exit 1
fi

# Run the main function
main "$@" 