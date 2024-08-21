#!/bin/bash

MODEL_PATHS=(
    "cat-searcher/gemma-2-9b-it-dpo-iter-1-evol-1-subset-greedy"
    "cat-searcher/gemma-2-9b-it-dpo-iter-1-evol-1-subset-iw"
)

MODEL_IDS=(
    "gemma-2-9b-it-dpo-iter-1-evol-1-subset-greedy"
    "gemma-2-9b-it-dpo-iter-1-evol-1-subset-iw"
)

# _MODEL="princeton-nlp/gemma-2-9b-it-DPO"
# _MODEL_ID="princeton-nlp-gemma-2-9b-it-DPO"


# MODEL_PATHS+=($_MODEL)
# MODEL_IDS+=($_MODEL_ID)

# python download_model.py \
#     --model-path $_MODEL

# CUDA_VISIBLE_DEVICES=4,5,6,7 python gen_model_answer.py \
#     --model-path $_MODEL \
#     --model-id $_MODEL_ID \
#     --num-gpus-total 4

# 3. Generate judgement
python gen_judgment_single.py \
    --model-list "${MODEL_IDS[@]}" \
    --parallel 60 \
    --filename-suffix _quick  # Use any suffix to differentiate

# 4. Show result
python show_result.py \
    --model-list "${MODEL_IDS[@]}" \
    --filename-suffix _combined_quick
