#!/bin/bash
set -e

model_name=${1:-"google/gemma-1.1-2b-it"}
dtype=${2:-"float16"}  # Use bfloat16 on A100/H100
host=${3:-"localhost"}
port=${4:-8964}
tensor_parallel_size=${5:-8}
attention_backend=${6:-"XFORMERS"}  # Use FLASHINFER for gemma-2 models
cuda_visible_devices=${7:-"0,1,2,3,4,5,6,7"}

export VLLM_ATTENTION_BACKEND=$attention_backend
CUDA_VISIBLE_DEVICES=$cuda_visible_devices \
    vllm serve $model_name \
    --dtype $dtype \
    --host $host \
    --port $port \
    --tensor-parallel-size $tensor_parallel_size \
    --api-key eva