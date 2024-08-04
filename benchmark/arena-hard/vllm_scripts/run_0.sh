#!/bin/bash
set -e

MODEL_PATH="cat-searcher/gemma-2-9b-it-sppo-iter-0"
MODEL_NAME="gemma-2-9b-it-sppo-iter-0"

# Start the vllm serve command in the background
CUDA_VISIBLE_DEVICES=0 \
nohup vllm serve $MODEL_PATH \
--dtype bfloat16 \
--host localhost \
--port 8000 \
--tensor-parallel-size 1 \
--api-key eva > local_vllm_serve_0.log 2>&1 &

# Capture the PID of the vllm serve process
VLLM_PID=$!

# Function to check if the server is running
check_server() {
  nc -z localhost 8000
}

# Wait until the server is ready
echo "Waiting for the server to start..."
until check_server; do
  sleep 5
done

echo "Server is up and running."

# Define the source and temporary configuration file paths for gen_answer_config
SOURCE_CONFIG_GEN="config/gen_answer_config.yaml"
TEMP_CONFIG_GEN="config/temp_gen_answer_config_${MODEL_NAME}.yaml"

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
TEMP_CONFIG_JUDGE="config/temp_judge_config_${MODEL_NAME}.yaml"

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

# Clean up by removing the temporary configuration file for gen_answer
rm $TEMP_CONFIG_GEN

# -----------------------------------------------------------------------
# Generate judgement for the specific model using the temporary config file
python gen_judgment.py \
    --setting-file $TEMP_CONFIG_JUDGE \
    --endpoint-file config/api_config.yaml
# -----------------------------------------------------------------------

# Clean up by removing the temporary configuration file for judge_config
rm $TEMP_CONFIG_JUDGE

# Show the results
python show_result.py

# Stop the vllm server
echo "Stopping the vllm server..."
kill $VLLM_PID
echo "vllm server stopped."
