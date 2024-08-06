#!/bin/bash
set -e
set -x

MODEL_PATH="cat-searcher/gemma-2-9b-it-sppo-iter-1"
MODEL_NAME="gemma-2-9b-it-sppo-iter-1"
cuda_visible_devices="1"
port=8001
dtype="bfloat16"

python gen_answer.py \
    --setting-file config/gen_answer_config_1.yaml \
    --endpoint-file config/api_config.yaml

python gen_judgment.py \
    --setting-file config/judge_config_1.yaml \
    --endpoint-file config/api_config.yaml

python show_result.py