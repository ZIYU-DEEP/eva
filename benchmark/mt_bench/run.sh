#!/bin/bash

# Define the model paths and IDs in arrays
MODEL_PATHS=(
    "google/gemma-1.1-2b-it"
    "cat-searcher/gemma-1.1-2b-it-sppo-iter0"
    "cat-searcher/gemma-1.1-2b-it-sppo-iter0-evol-mixed"
)

MODEL_IDS=(
    "gemma-1.1-2b-it"
    "gemma-1.1-2b-it-sppo-iter0"
    "gemma-1.1-2b-it-sppo-iter0-evol-mixed"
)


# --------------------------------------------------------------------------
# 0. Download models
for i in "${!MODEL_PATHS[@]}"; do
    python download_model.py \
        --model-path "${MODEL_PATHS[$i]}"
done
# --------------------------------------------------------------------------


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
python gen_judgment.py \
    --model-list "${MODEL_IDS[@]}" \
    --parallel 40
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 3. Show results
python show_result.py \
    --model-list "${MODEL_IDS[@]}"
# --------------------------------------------------------------------------
