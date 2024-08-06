#!/bin/bash
set -e
set -x

MODEL_PATH="cat-searcher/gemma-2-9b-it-sppo-iter-3"
MODEL_NAME="gemma-2-9b-it-sppo-iter-3"
cuda_visible_devices="3"
port=8003
dtype="bfloat16"
tensor_parallel_size=1

CUDA_VISIBLE_DEVICES=3 python gen_answer.py \
    --setting-file config/gen_answer_config_3.yaml \
    --endpoint-file config/api_config.yaml

CUDA_VISIBLE_DEVICES=3 python gen_judgment.py \
    --setting-file config/judge_config_3.yaml \
    --endpoint-file config/api_config.yaml

CUDA_VISIBLE_DEVICES=3 python show_result.py