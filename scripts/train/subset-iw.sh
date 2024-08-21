#!/bin/bash
set -e
# set -x  # Print the commands

# Set the environmental variable
export WANDB_PROJECT="dpo"

# ##################################################################
# 1. Training
# ##################################################################
# ------------------------------------------------------------------
# Run the training
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file ./recipes/accelerate_configs/deepspeed_zero3.yaml \
    --main_process_port 8964 \
    eva/run_sppo.py "./recipes/misc/config_full_iw.yaml" \
    --learning_rate=5.0e-7 \
    --beta=0.05 \
    --optim="adamw_torch" \
    --output_dir="checkpoints/gemma-2-9b-it-dpo-iter-1-evol-1-subset-iw" \
    --loss_type="sigmoid" \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --model_name_or_path="cat-searcher/gemma-2-9b-it-dpo-iter-1" \
    --num_train_epochs=2
# ------------------------------------------------------------------
