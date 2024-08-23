#!/bin/bash
set -e

MODEL_PATH=${1:-"cat-searcher/NSPLIT3-gemma-2-9b-it-rpo-iter-1-evol-1"}
MODEL_NAME=${2:-"NSPLIT3-gemma-2-9b-it-rpo-iter-1-evol-1"}
cuda_visible_devices=${3:-"0,1,2,3"}
port=${4:-8964}
dtype=${5:-"bfloat16"}
tensor_parallel_size=${5:-4}

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

# Stop the sglang server
echo "Stopping the server..."
pkill -9 -f "sglang.launch_server"
echo "Server stopped."

# -----------------------------------------------------------------------
# Generate judgement for the specific model using the temporary config file
python gen_judgment.py \
    --setting-file $TEMP_CONFIG_JUDGE \
    --endpoint-file config/api_config.yaml
# -----------------------------------------------------------------------

# Show the results
python show_result.py

