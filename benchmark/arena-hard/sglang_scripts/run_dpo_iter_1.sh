#!/bin/bash
set -e
sleep 20

MODEL_PATH="cat-searcher/gemma-2-9b-it-dpo-iter-1"
MODEL_NAME="gemma-2-9b-it-dpo-iter-1"
port=8000
dtype="bfloat16"

# python gen_answer.py \
#     --setting-file config/gen_answer_config_dpo_iter_1.yaml \
#     --endpoint-file config/api_config.yaml

python gen_judgment.py \
    --setting-file config/judge_config_dpo_iter_1.yaml \
    --endpoint-file config/api_config.yaml

python show_result.py