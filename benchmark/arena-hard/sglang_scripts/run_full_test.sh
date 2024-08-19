#!/bin/bash
set -e

MODEL_PATH="cat-searcher/NSPLIT3-gemma-2-9b-it-dpo-iter-1-evol-1"
MODEL_NAME="NSPLIT3-gemma-2-9b-it-dpo-iter-1-evol-1"
cuda_visible_devices="0,1,2,3,4,5,6,7"
port=8964
dtype="bfloat16"
tensor_parallel_size=8

# Start the serve command in the background
CUDA_VISIBLE_DEVICES=$cuda_visible_devices \
    python -m sglang.launch_server --model-path $MODEL_PATH \
    --dtype $dtype \
    --host localhost  \
    --port $port \
    --tp $tensor_parallel_size \
    --api-key eva > local_sglang_serve_1.log 2>&1 &

# Capture the PID of the  serve process
SERVE_PID=$!
sleep 30

# Function to check if the server is running
check_server() {
  nc -z localhost $port
}

# Wait until the server is ready
echo "Waiting for the server to start..."
until check_server; do
  sleep 5
done

echo "Server is up and running."

# Define the source and temporary configuration file paths for gen_answer_config
SOURCE_CONFIG_GEN="config/gen_answer_config.yaml"
TEMP_CONFIG_GEN="config/gen_answer_config_${MODEL_NAME}.yaml"

# Copy the source configuration file to the temporary configuration file
cp $SOURCE_CONFIG_GEN $TEMP_CONFIG_GEN

# Remove everything on and after model_list: and create a new entry in gen_answer_config
awk -v model_name="$MODEL_NAME" '
  BEGIN { in_model_list = 0 }
  /^model_list:/ { in_model_list = 1; print "model_list:\n  - " model_name; next }
  !in_model_list { print }
' $SOURCE_CONFIG_GEN > $TEMP_CONFIG_GEN

# Define the source and temporary configuration file paths for judge_config
SOURCE_CONFIG_JUDGE="config/judge_config.yaml"
TEMP_CONFIG_JUDGE="config/judge_config_${MODEL_NAME}.yaml"

# Copy the source configuration file to the temporary configuration file
cp $SOURCE_CONFIG_JUDGE $TEMP_CONFIG_JUDGE

# Remove everything on and after model_list: and create a new entry in judge_config
awk -v model_name="$MODEL_NAME" '
  BEGIN { in_model_list = 0 }
  /^model_list:/ { in_model_list = 1; print "model_list:\n  - " model_name; next }
  !in_model_list { print }
' $SOURCE_CONFIG_JUDGE > $TEMP_CONFIG_JUDGE

# -----------------------------------------------------------------------
# Generate answer for the specific model using the temporary config file
python gen_answer.py \
    --setting-file $TEMP_CONFIG_GEN \
    --endpoint-file config/api_config.yaml
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Generate judgement for the specific model using the temporary config file
python gen_judgment.py \
    --setting-file $TEMP_CONFIG_JUDGE \
    --endpoint-file config/api_config.yaml
# -----------------------------------------------------------------------

# Show the results
python show_result.py

# Stop the vllm server
echo "Stopping the server..."
kill $SERVE_PID
echo "Server stopped."
