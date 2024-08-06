#!/bin/bash
set -e
set -x

MODEL_PATH="cat-searcher/gemma-2-9b-it-sppo-iter-4"
MODEL_NAME="gemma-2-9b-it-sppo-iter-4"
cuda_visible_devices="4"
port=8004
dtype="bfloat16"
tensor_parallel_size=1

CUDA_VISIBLE_DEVICES=4 python gen_answer.py \
    --setting-file config/gen_answer_config_4.yaml \
    --endpoint-file config/api_config.yaml

CUDA_VISIBLE_DEVICES=4 python gen_judgment.py \
    --setting-file config/judge_config_4.yaml \
    --endpoint-file config/api_config.yaml

CUDA_VISIBLE_DEVICES=4 python show_result.py