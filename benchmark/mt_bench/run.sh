#!/bin/bash

# Define the model paths and IDs in arrays
MODEL_PATHS=(
    "cat-searcher/gemma-2-9b-it-sppo-iter-0"
    "cat-searcher/gemma-2-9b-it-sppo-iter-1"
    "cat-searcher/gemma-2-9b-it-sppo-iter-2"
    "cat-searcher/gemma-2-9b-it-sppo-iter-3"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

MODEL_IDS=(
    "gemma-2-9b-it-sppo-iter-0"
    "gemma-2-9b-it-sppo-iter-1"
    "gemma-2-9b-it-sppo-iter-2"
    "gemma-2-9b-it-sppo-iter-3"
    "llama-3.1-8b-instruct"
)

# # --------------------------------------------------------------------------
# 0. Download models
for i in "${!MODEL_PATHS[@]}"; do
    python download_model.py \
        --model-path "${MODEL_PATHS[$i]}"
        --num-gpus-total 8 \
        --dtype bfloat16
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
