#!/bin/bash

# Number of GPUs to use
NUM_GPUS=2

# List of models to pre-train
MODELS=("inception_v3" "resnet50" "densenet201" "mobilenet_v3_large" "efficientnet_b0" "vit_b_16")

# Base directories for logs and checkpoints
BASE_LOG_DIR="/home/ubuntu/Paper4_CNN_ViT_Comparison/pretraining/logs"
BASE_CHECKPOINT_DIR="/home/ubuntu/Paper4_CNN_ViT_Comparison/pretraining/checkpoints"

# Path to the training script
TRAIN_SCRIPT="/home/ubuntu/Paper4_CNN_ViT_Comparison/pretraining/main_pretraining.py"

for MODEL in "${MODELS[@]}"
do
    echo "Starting pretraining for $MODEL..."

    LOG_DIR="$BASE_LOG_DIR/logs_$MODEL"
    CKPT_DIR="$BASE_CHECKPOINT_DIR/checkpoint_$MODEL"

    # Start torchrun in background
    torchrun --nproc-per-node=$NUM_GPUS "$TRAIN_SCRIPT" \
        --model_name "$MODEL" \
        --log_dir "$LOG_DIR" \
        --checkpoint_dir "$CKPT_DIR" &

    # Capture PID of the background process
    PID=$!

    # Wait for the process to finish
    wait $PID
    EXIT_STATUS=$?

    if [ $EXIT_STATUS -ne 0 ]; then
        echo "Pretraining failed for $MODEL with exit status $EXIT_STATUS. Exiting."
        exit $EXIT_STATUS
    fi

    echo "Finished pretraining for $MODEL. Waiting 10 seconds before next..."
    sleep 10
done

echo "All models pre-trained successfully."
