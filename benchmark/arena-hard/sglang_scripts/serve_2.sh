#!/bin/bash
set -e

model_name=${1:-"cat-searcher/gemma-2-9b-it-sppo-iter-2"}
cuda_visible_devices=${7:-"2"}
dtype=${2:-"bfloat16"}
host=${3:-"localhost"}
port=${4:-8002}
tensor_parallel_size=${5:-1}
attention_backend=${6:-"FLASHINFER"}  # Use FLASHINFER for gemma-2 models and XFORMERS for other models

export VLLM_ATTENTION_BACKEND=$attention_backend
CUDA_VISIBLE_DEVICES=$cuda_visible_devices \
    python -m sglang.launch_server --model-path $model_name \
    --dtype $dtype \
    --host $host \
    --port $port \
    --tp $tensor_parallel_size \
    --api-key eva
