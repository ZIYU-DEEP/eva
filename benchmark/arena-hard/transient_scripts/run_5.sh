#!/bin/bash
set -e
sleep 20

MODEL_PATH="cat-searcher/gemma-2-9b-it-sppo-iter-5"
MODEL_NAME="gemma-2-9b-it-sppo-iter-5"
cuda_visible_devices="5"
port=8005
dtype="bfloat16"
tensor_parallel_size=1

CUDA_VISIBLE_DEVICES=5 python gen_answer.py \
    --setting-file config/gen_answer_config_5.yaml \
    --endpoint-file config/api_config.yaml

CUDA_VISIBLE_DEVICES=5 python gen_judgment.py \
    --setting-file config/judge_config_5.yaml \
    --endpoint-file config/api_config.yaml

CUDA_VISIBLE_DEVICES=5 python show_result.py