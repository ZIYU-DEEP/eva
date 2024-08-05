#!/bin/bash
set -e

model_name=${1:-"cat-searcher/gemma-2-9b-it-sppo-iter-1-evol-1"}
cuda_visible_devices=${7:-"0,1,2,3,4,5,6,7"}
dtype=${2:-"bfloat16"}
host=${3:-"localhost"}
port=${4:-8964}
tensor_parallel_size=${5:-8}
attention_backend=${6:-"FLASHINFER"}  # Use FLASHINFER for gemma-2 models and XFORMERS for other models

export VLLM_ATTENTION_BACKEND=$attention_backend
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m sglang.launch_server --model-path "cat-searcher/gemma-2-9b-it-sppo-iter-1-evol-1" \
    --dtype bfloat16 \
    --host localhost \
    --port 8964 \
    --tp 8 \
    --api-key eva
