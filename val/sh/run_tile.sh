#!/bin/bash
# Tile-based Inference Script for Nuclear Segmentation
#
# This script performs tile-based inference on large pathology images
# using pre-trained nuclear segmentation models. It processes multiple
# models and checkpoints, generating segmentation results for each.
#
# Usage: bash run_tile.sh

# ============================================================================
# CONFIGURATION
# ============================================================================

# GPU Configuration
GPU='0'                                    # GPU device ID to use
NR_TYPES=5                                 # Number of nuclear types
TYPE_INFO_PATH='type_info.json'           # Path to type information JSON file
BATCH_SIZE=1                              # Batch size for inference
MODEL_MODE='fast'                          # Model mode (fast/standard)

# Worker Configuration
NR_INFERENCE_WORKERS=8                     # Number of parallel inference workers
NR_POST_PROC_WORKERS=16                   # Number of post-processing workers

# Input/Output Configuration
INPUT_DIR='/path/to/test/images'          # Directory containing test images
MEM_USAGE=0.1                             # Memory usage per tile (0.0-1.0)

# Output Options
DRAW_DOT=''                               # Set to '--draw_dot' to enable dot visualization
SAVE_QUPATH=''                            # Set to '--save_qupath' to save Qupath format

# Model and Output Configuration Arrays
# Each model path corresponds to an output directory
# Add/remove pairs as needed for your experiments

MODEL_BASE_PATHS=(
    '/path/to/models/experiment1/net_epoch='
    # '/path/to/models/experiment2/net_epoch='    # Additional model (commented out)
    # '/path/to/models/experiment3/net_epoch='    # Additional model (commented out)
)

OUTPUT_DIRS=(
    '/path/to/results/experiment1'
    # '/path/to/results/experiment2'              # Additional output dir (commented out)
    # '/path/to/results/experiment3'              # Additional output dir (commented out)
)

# Epochs to evaluate for each model
EPOCHS=(150 160 165 170 175 180 186 190 195 197 200)

# Inference script path
INFERENCE_SCRIPT='/path/to/run_infer.py'

# ============================================================================
# VALIDATION
# ============================================================================

# Ensure model paths and output directories arrays have same length
if [ ${#MODEL_BASE_PATHS[@]} -ne ${#OUTPUT_DIRS[@]} ]; then
    echo "Error: MODEL_BASE_PATHS and OUTPUT_DIRS arrays must have the same length."
    echo "Model paths: ${#MODEL_BASE_PATHS[@]}"
    echo "Output dirs: ${#OUTPUT_DIRS[@]}"
    exit 1
fi

# Check if arrays are empty
if [ ${#MODEL_BASE_PATHS[@]} -eq 0 ]; then
    echo "Error: No model paths defined. Please add at least one model."
    exit 1
fi

# ============================================================================
# TILE-BASED INFERENCE
# ============================================================================

echo "Starting tile-based inference..."
echo "Number of models to process: ${#MODEL_BASE_PATHS[@]}"
echo "Epochs to evaluate: ${EPOCHS[@]}"
echo "GPU device: ${GPU}"

# Loop through each model and corresponding output directory
for i in "${!MODEL_BASE_PATHS[@]}"; do
    MODEL_BASE_PATH=${MODEL_BASE_PATHS[$i]}
    OUTPUT_DIR_BASE=${OUTPUT_DIRS[$i]}

    echo "=========================================="
    echo "Processing model ${i}: ${MODEL_BASE_PATH}"
    echo "Output base: ${OUTPUT_DIR_BASE}"

    # Loop through each epoch/checkpoint
    for epoch in "${EPOCHS[@]}"; do
        echo "----------------------------------------"
        echo "Processing epoch ${epoch}..."

        # Construct full model path for current epoch
        MODEL_PATH="${MODEL_BASE_PATH}${epoch}.tar"

        # Check if model file exists
        if [ ! -f "${MODEL_PATH}" ]; then
            echo "Warning: Model file ${MODEL_PATH} does not exist, skipping..."
            continue
        fi

        # Construct output directory for current epoch
        OUTPUT_DIR="${OUTPUT_DIR_BASE}/${epoch}"

        # Create output directory if it doesn't exist
        mkdir -p "${OUTPUT_DIR}"

        echo "Model: ${MODEL_PATH}"
        echo "Output: ${OUTPUT_DIR}"

        # Execute inference command
        python ${INFERENCE_SCRIPT} \
            --gpu="${GPU}" \
            --nr_types="${NR_TYPES}" \
            --type_info_path="${TYPE_INFO_PATH}" \
            --batch_size="${BATCH_SIZE}" \
            --model_mode="${MODEL_MODE}" \
            --model_path="${MODEL_PATH}" \
            --nr_inference_workers="${NR_INFERENCE_WORKERS}" \
            --nr_post_proc_workers="${NR_POST_PROC_WORKERS}" \
            tile \
            --input_dir="${INPUT_DIR}" \
            --output_dir="${OUTPUT_DIR}" \
            --mem_usage="${MEM_USAGE}" \
            ${DRAW_DOT} \
            ${SAVE_QUPATH}

        # Check if inference was successful
        if [ $? -eq 0 ]; then
            echo "Successfully completed epoch ${epoch}"
        else
            echo "Error occurred during epoch ${epoch}"
        fi

        echo "Finished epoch ${epoch}"
        echo "----------------------------------------"
    done

    echo "Finished processing model ${i}"
    echo "=========================================="
done

echo "=========================================="
echo "Tile-based inference completed!"
echo "Total models processed: ${#MODEL_BASE_PATHS[@]}"
echo "Total epochs processed per model: ${#EPOCHS[@]}"
echo "=========================================="