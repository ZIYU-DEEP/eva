#!/bin/bash
set -e

MODEL_PATH="cat-searcher/gemma-2-9b-it-sppo-iter-2"
MODEL_NAME="gemma-2-9b-it-sppo-iter-2"
cuda_visible_devices="2"
port=8002
dtype="bfloat16"
tensor_parallel_size=1

# Start the serve command in the background
CUDA_VISIBLE_DEVICES=$cuda_visible_devices \
    python -m sglang.launch_server --model-path $MODEL_PATH \
    --dtype $dtype \
    --host localhost  \
    --port $port \
    --tp $tensor_parallel_size \
    --api-key eva > local_sglang_serve_${MODEL_NAME}.log 2>&1
