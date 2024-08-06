#!/bin/bash
set -e
set -x

MODEL_PATH="cat-searcher/gemma-2-9b-it-sppo-iter-5"
MODEL_NAME="gemma-2-9b-it-sppo-iter-5"
port=8005
dtype="bfloat16"

python gen_answer.py \
    --setting-file config/gen_answer_config_5.yaml \
    --endpoint-file config/api_config.yaml

python gen_judgment.py \
    --setting-file config/judge_config_5.yaml \
    --endpoint-file config/api_config.yaml

python show_result.py
