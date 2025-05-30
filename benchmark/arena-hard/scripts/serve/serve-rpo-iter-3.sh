#!/bin/bash
set -e

model_name=${1:-"cat-searcher/NSPLIT3-gemma-2-9b-it-rpo-iter-3"}
dtype=${2:-"bfloat16"}
host=${3:-"localhost"}
port=${4:-8964}
tensor_parallel_size=${5:-8}
attention_backend=${6:-"FLASHINFER"}  # Use FLASHINFER for gemma-2 models and XFORMERS for other models
cuda_visible_devices=${7:-"0,1,2,3,4,5,6,7,8"}

export VLLM_ATTENTION_BACKEND=$attention_backend
CUDA_VISIBLE_DEVICES=$cuda_visible_devices \
    python -m sglang.launch_server --model-path $model_name \
    --dtype $dtype \
    --host $host \
    --port $port \
    --tp $tensor_parallel_size \
    --api-key eva
  
