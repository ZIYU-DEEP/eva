#!/bin/bash

MODEL_PATHS=(
    "cat-searcher/NSPLIT3-gemma-2-9b-it-dpo-iter-3"
)

MODEL_IDS=(
    "NSPLIT3-gemma-2-9b-it-dpo-iter-3"
)

_MODEL=cat-searcher/NSPLIT3-gemma-2-9b-it-dpo-iter-3
_MODEL_ID=NSPLIT3-gemma-2-9b-it-dpo-iter-3


# MODEL_PATHS+=($_MODEL)
# MODEL_IDS+=($_MODEL_ID)

python download_model.py \
    --model-path $_MODEL

python gen_model_answer.py \
    --model-path $_MODEL \
    --model-id $_MODEL_ID \
    --num-gpus-total 8

# 3. Generate judgement
python gen_judgment_single.py \
    --model-list "${MODEL_IDS[@]}" \
    --parallel 60 \
    --filename-suffix _quick  # Use any suffix to differentiate

# 4. Show result
python show_result.py \
    --model-list "${MODEL_IDS[@]}" \
    --filename-suffix _combined_quick
