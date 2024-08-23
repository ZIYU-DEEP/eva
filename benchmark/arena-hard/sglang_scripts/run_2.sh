#!/bin/bash
set -e
sleep 20

MODEL_PATH="cat-searcher/gemma-2-9b-it-dpo-iter-0"
MODEL_NAME="gemma-2-9b-it-dpo-iter-0"
cuda_visible_devices="0,1,2,3,4,5,6,7"
port=8000
dtype="bfloat16"
tensor_parallel_size=8

python gen_answer.py \
    --setting-file config/gen_answer_config_2.yaml \
    --endpoint-file config/api_config.yaml

python gen_judgment.py \
    --setting-file config/judge_config_2.yaml \
    --endpoint-file config/api_config.yaml

python show_result.py