#!/bin/bash
set -e

model_name=${1:-"cat-searcher/gemma-2-9b-it-dpo-iter-1-evol-1-reward_dts-0.25"}
dtype=${2:-"bfloat16"}
host=${3:-"localhost"}
port=${4:-8964}
tensor_parallel_size=${5:-8}
attention_backend=${6:-"FLASHINFER"}  # Use FLASHINFER for gemma-2 models and XFORMERS for other models
cuda_visible_devices=${7:-"0,1,2,3,4,5,6,7"}

export VLLM_ATTENTION_BACKEND=$attention_backend
CUDA_VISIBLE_DEVICES=$cuda_visible_devices \
    python -m sglang.launch_server --model-path $model_name \
    --dtype $dtype \
    --host $host \
    --port $port \
    --tp $tensor_parallel_size \
    --api-key eva
