#!/bin/bash
# Define default paths
DEFAULT_MODEL_NAME="gemma-2-9b-it"
DEFAULT_CONFIG_PATH=$(pwd)/models_configs/${DEFAULT_MODEL_NAME}

# Take the target model name as input or use the default value
TARGET_MODEL_NAME=${1:-"gemma-2-9b-it-dpo-iter-1-evol-1-reward_var-0.25"}

# Define the new config folder path based on the target model name
CONFIG_PATH=$(pwd)/models_configs/${TARGET_MODEL_NAME}
echo "Creating config at ${CONFIG_PATH}"

# # Check if the target directory already exists
# if [ -d "${CONFIG_PATH}" ]; then
#   echo "Directory ${CONFIG_PATH} already exists. Exiting to prevent overwriting."
#   exit 1
# fi

# Copy the default config folder to the new folder
cp -r ${DEFAULT_CONFIG_PATH} ${CONFIG_PATH}

# Replace the model name in configs.yaml
sed -i "s/${DEFAULT_MODEL_NAME}/${TARGET_MODEL_NAME}/g" "${CONFIG_PATH}/configs.yaml"

# Display the updated config path
echo "Updated configs.yaml at ${CONFIG_PATH}"

# Run alpaca_eval with the modified configuration
CUDA_VISIBLE_DEVICES=4 alpaca_eval evaluate_from_model \
  --model_configs "${CONFIG_PATH}/configs.yaml" \
  --annotators_config 'alpaca_eval_gpt4_turbo_fn' 
