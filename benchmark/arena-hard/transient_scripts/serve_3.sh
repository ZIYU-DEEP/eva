#!/bin/bash
set -e
set -x

model_name=${1:-"cat-searcher/gemma-2-9b-it-sppo-iter-3"}
port=${2:-8003}
cuda_visible_devices=${3:-"3"}

export VLLM_ATTENTION_BACKEND=FLASHINFER
CUDA_VISIBLE_DEVICES=${cuda_visible_devices} \
    python -m sglang.launch_server --model-path $model_name \
    --dtype bfloat16 \
    --host localhost \
    --port $port \
    --tp 1 \
    --api-key eva
