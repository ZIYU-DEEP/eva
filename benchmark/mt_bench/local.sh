#!/bin/bash

# Define the model paths and IDs in arrays
MODEL_PATHS=(
    # "cat-searcher/gemma-1.1-2b-it-sppo-iter-1"
    # "google/gemma-1.1.2b-it"
    "google/gemma-2-9b-it"
    "princeton-nlp/gemma-2-9b-it-DPO"

)

MODEL_IDS=(
    # "gemma-1.1-2b-it-sppo-iter-1"
    # "gemma-1.1-2b-it"
    "gemma-2-9b-it"
    "gemma-2-9b-it-DPO"
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
python gen_judgment.py \
    --model-list "${MODEL_IDS[@]}" \
    --parallel 40 \
    --filename-suffix _9b
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# 3. Show results
python show_result.py \
    --model-list "${MODEL_IDS[@]}" \
    --filename-suffix _9b
# --------------------------------------------------------------------------
