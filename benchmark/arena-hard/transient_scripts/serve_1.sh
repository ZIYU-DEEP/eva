#!/bin/bash
set -e
set -x

model_name=${1:-"cat-searcher/gemma-2-9b-it-sppo-iter-1"}
port=${2:-8001}
cuda_visible_devices=${3:-"0,1,2,3,4,5,6,7,8"}

export VLLM_ATTENTION_BACKEND=FLASHINFER
CUDA_VISIBLE_DEVICES=${cuda_visible_devices} \
    python -m sglang.launch_server --model-path $model_name \
    --dtype bfloat16 \
    --host localhost \
    --port $port \
    --tp 8 \
    --api-key eva
