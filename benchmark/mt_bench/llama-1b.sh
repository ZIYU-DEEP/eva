#!/bin/bash

# Define the model paths and IDs in arrays
MODEL_PATHS=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct-dpo-iter-1"
    "meta-llama/Llama-3.2-1B-Instruct-dpo-iter-1-evol-1"
)

MODEL_IDS=(
    "llama-3.2-1b-instruct"
    "llama-3.2-3b-instruct-dpo-iter-1"
    "llama-3.2-3b-instruct-dpo-iter-1-evol-1"
)

# # --------------------------------------------------------------------------
# 0. Download models
for i in "${!MODEL_PATHS[@]}"; do
    CUDA_VISIBLE_DEVICES=4,5,6,7 python download_model.py \
        --model-path "${MODEL_PATHS[$i]}" \
        --num-gpus-total 4
done
# # --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 1. Generate answers (saved to ./data/mt_bench/model_answer/*.jsonl)
for i in "${!MODEL_PATHS[@]}"; do
    CUDA_VISIBLE_DEVICES=4,5,6,7 python gen_model_answer.py \
        --model-path "${MODEL_PATHS[$i]}" \
        --model-id "${MODEL_IDS[$i]}" \
        --num-gpus-total 4
done
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 2. Generate judgements (saved to ./data/mt_bench/model_answer/*.jsonl)
python gen_judgment_single.py \
    --model-list "${MODEL_IDS[@]}" \
    --parallel 40 \
    --filename-suffix _new_llama
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 3. Show results
python show_result.py \
    --model-list "${MODEL_IDS[@]}" \
    --filename-suffix _combined_new_llama
# --------------------------------------------------------------------------
