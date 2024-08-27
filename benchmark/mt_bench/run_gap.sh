#!/bin/bash

MODEL_PATHS=(
    "cat-searcher/gemma-2-9b-it-dpo-iter-1-evol-1-reward_gap-0.25-iw_topic-0.25"
    # "cat-searcher/gemma-2-9b-it-dpo-iter-1-evol-1-reward_maxmean-0.25"
)

MODEL_IDS=(
    "gemma-2-9b-it-dpo-iter-1-evol-1-reward_gap-0.25-iw_topic-0.25"
    # "gemma-2-9b-it-dpo-iter-1-evol-1-reward_maxmean-0.25"
)

_MODEL="cat-searcher/gemma-2-9b-it-dpo-iter-1-evol-1-reward_gap-0.25-iw_topic-0.25"
_MODEL_ID="gemma-2-9b-it-dpo-iter-1-evol-1-reward_gap-0.25-iw_topic-0.25"


# MODEL_PATHS+=($_MODEL)
# MODEL_IDS+=($_MODEL_ID)

python download_model.py \
    --model-path $_MODEL \
    --num-gpus-total 8 \
    --dtype bfloat16

python gen_model_answer.py \
    --model-path $_MODEL \
    --model-id $_MODEL_ID \
    --num-gpus-total 8 \
    --dtype bfloat16

# 3. Generate judgement
python gen_judgment_single.py \
    --model-list "${MODEL_IDS[@]}" \
    --parallel 60 \
    --filename-suffix _quick  # Use any suffix to differentiate

# 4. Show result
python show_result.py \
    --model-list "${MODEL_IDS[@]}" \
    --filename-suffix _combined_quick
