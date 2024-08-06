#!/bin/bash
set -e
sleep 20

MODEL_PATH="cat-searcher/gemma-2-9b-it-sppo-iter-2"
MODEL_NAME="gemma-2-9b-it-sppo-iter-2"
cuda_visible_devices="2"
port=8002
dtype="bfloat16"
tensor_parallel_size=1

CUDA_VISIBLE_DEVICES=2 python gen_answer.py \
    --setting-file config/gen_answer_config_2.yaml \
    --endpoint-file config/api_config.yaml

CUDA_VISIBLE_DEVICES=2 python gen_judgment.py \
    --setting-file config/judge_config_2.yaml \
    --endpoint-file config/api_config.yaml

CUDA_VISIBLE_DEVICES=2 python show_result.py