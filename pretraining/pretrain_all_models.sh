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

# Pre-train all models
for MODEL in "${MODELS[@]}"
do
    LOG_DIR="$BASE_LOG_DIR/logs_$MODEL"
    CKPT_DIR="$BASE_CHECKPOINT_DIR/checkpoint_$MODEL"

    torchrun --nproc-per-node=$NUM_GPUS $TRAIN_SCRIPT \
        --model_name $MODEL \
        --log_dir $LOG_DIR \
        --checkpoint_dir $CKPT_DIR
done

