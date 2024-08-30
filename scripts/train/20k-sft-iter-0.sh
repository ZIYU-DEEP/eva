#!/bin/bash
set -e
# set -x  # Print the commands

# Set the environmental variable
export WANDB_PROJECT="sft"

SFT_MODEL_PATH="google/gemma-2-9b-it"
MODEL_PATH=${SFT_MODEL_PATH}

# ##################################################################
# 1. Training
# ##################################################################
# ------------------------------------------------------------------
# Run the training
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file ./recipes/accelerate_configs/deepspeed_zero3.yaml \
    --main_process_port 8964 \
    eva/run_sft.py "./recipes/sft/config_full.yaml" \
    --learning_rate=2.0e-05 \
    --optim="adamw_torch" \
    --output_dir="checkpoints/NSPLIT3-gemma-2-9b-it-sft-iter-0" \
    --model_name_or_path=${MODEL_PATH} \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --num_train_epochs=3
# ------------------------------------------------------------------
