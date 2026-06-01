#!/bin/bash
# Metrics Calculation Script for Nuclear Segmentation Evaluation
#
# This script calculates evaluation metrics for nuclear segmentation results
# across different model checkpoints and epochs. It processes both instance
# segmentation and type classification metrics.
#
# Usage: bash metrics.sh

# ============================================================================
# CONFIGURATION
# ============================================================================

# Define epochs to evaluate (model checkpoints)
# Modify this array to evaluate different checkpoints
EPOCHS=(150 160 165 170 175 180 186 190 195 197 200)

# Define prediction result directories
# Each directory should contain subdirectories named by epoch number,
# and each epoch subdirectory should contain a 'mat' folder with .mat files
PRED_DIRS=(
    '/path/to/results/experiment_1'
    # '/path/to/results/experiment_2'  # Add more experiments as needed
    # '/path/to/results/experiment_3'
)

# Script path for computing statistics
# Update this path to point to your compute_stats.py script
COMPUTE_STATS_SCRIPT='/path/to/compute_stats.py'

# ============================================================================
# VALIDATION
# ============================================================================

# Ensure pred_dir array is not empty
if [ ${#PRED_DIRS[@]} -eq 0 ]; then
    echo "Error: PRED_DIRS array must not be empty."
    echo "Please add at least one prediction directory to evaluate."
    exit 1
fi

# ============================================================================
# INSTANCE SEGMENTATION METRICS
# ============================================================================

echo "Starting instance segmentation metrics calculation..."

# Loop through each prediction directory
for pred_dir_base in "${PRED_DIRS[@]}"
do
    echo "Processing directory: ${pred_dir_base}"

    # Loop through each epoch/checkpoint
    for epoch in "${EPOCHS[@]}"
    do
        # Construct prediction directory path for current epoch
        PRED_DIR="${pred_dir_base}/${epoch}/mat"

        # Check if directory exists before processing
        if [ ! -d "${PRED_DIR}" ]; then
            echo "Warning: Directory ${PRED_DIR} does not exist, skipping..."
            continue
        fi

        echo "Calculating instance metrics for epoch ${epoch}..."

        # Execute instance segmentation metrics calculation
        python ${COMPUTE_STATS_SCRIPT} --mode instance --pred_dir "${PRED_DIR}"

        echo "Finished instance metrics for epoch ${epoch}"
    done
done

echo "Instance segmentation metrics calculation completed."

# ============================================================================
# TYPE CLASSIFICATION METRICS
# ============================================================================

echo "Starting type classification metrics calculation..."

# Loop through each prediction directory
for pred_dir_base in "${PRED_DIRS[@]}"
do
    echo "Processing directory: ${pred_dir_base}"

    # Loop through each epoch/checkpoint
    for epoch in "${EPOCHS[@]}"
    do
        # Construct prediction directory path for current epoch
        PRED_DIR="${pred_dir_base}/${epoch}/mat"

        # Check if directory exists before processing
        if [ ! -d "${PRED_DIR}" ]; then
            echo "Warning: Directory ${PRED_DIR} does not exist, skipping..."
            continue
        fi

        echo "Calculating type metrics for epoch ${epoch}..."

        # Execute type classification metrics calculation
        python ${COMPUTE_STATS_SCRIPT} --mode type --pred_dir "${PRED_DIR}"

        echo "Finished type metrics for epoch ${epoch}"
    done
done

echo "Type classification metrics calculation completed."

# ============================================================================
# SUMMARY
# ============================================================================

echo "=========================================="
echo "All metrics calculations completed!"
echo "Processed ${#PRED_DIRS[@]} experiment(s)"
echo "Evaluated ${#EPOCHS[@]} epoch(s) per experiment"
echo "=========================================="