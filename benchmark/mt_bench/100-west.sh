#!/bin/bash

# Define the model paths and IDs in arrays
MODEL_PATHS=(
    # "cat-searcher/gemma-2-9b-it-dpo-iter-1-evol-1-reward_dts-0.25"
    # "cat-searcher/gemma-2-9b-it-dpo-iter-1-evol-1-reward_loo-0.25"
    # "cat-searcher/gemma-2-9b-it-dpo-iter-1-evol-1-reward_var-0.25"
    "cat-searcher/gemma-2-9b-it-dpo-iter-1-evol-1-reward_mean-0.25"
    "cat-searcher/gemma-2-9b-it-dpo-iter-1-evol-1-reward_gap_inv-0.25"
    "cat-searcher/gemma-2-9b-it-dpo-iter-1-evol-1-reward_mean_inv-0.25"
)

MODEL_IDS=(
    # "gemma-2-9b-it-dpo-iter-1-evol-1-reward_dts-0.25"
    # "gemma-2-9b-it-dpo-iter-1-evol-1-reward_loo-0.25"
    # "gemma-2-9b-it-dpo-iter-1-evol-1-reward_var-0.25"
    "gemma-2-9b-it-dpo-iter-1-evol-1-reward_mean-0.25"
    "gemma-2-9b-it-dpo-iter-1-evol-1-reward_gap_inv-0.25"
    "gemma-2-9b-it-dpo-iter-1-evol-1-reward_mean_inv-0.25"
)

# # --------------------------------------------------------------------------
# 0. Download models
for i in "${!MODEL_PATHS[@]}"; do
    python download_model.py \
        --model-path "${MODEL_PATHS[$i]}"
done
# # --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 1. Generate answers (saved to ./data/mt_bench/model_answer/*.jsonl)
for i in "${!MODEL_PATHS[@]}"; do
    python gen_model_answer.py \
        --model-path "${MODEL_PATHS[$i]}" \
        --model-id "${MODEL_IDS[$i]}" \
        --num-gpus-total 8
done
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 2. Generate judgements (saved to ./data/mt_bench/model_answer/*.jsonl)
python gen_judgment_single.py \
    --model-list "${MODEL_IDS[@]}" \
    --parallel 40 \
    --filename-suffix _new
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 3. Show results
python show_result.py \
    --model-list "${MODEL_IDS[@]}" \
    --filename-suffix _combined_new
# --------------------------------------------------------------------------
