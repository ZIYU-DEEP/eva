#!/bin/bash
# Take the target model name as input or use the default value
TARGET_MODEL_NAME=${1:-"Llama-3.2-3B-Instruct-dpo-iter-1"}

# Define the new config folder path based on the target model name
CONFIG_PATH=$(pwd)/models_configs/${TARGET_MODEL_NAME}


# Run alpaca_eval with the modified configuration
alpaca_eval evaluate_from_model \
  --model_configs "${CONFIG_PATH}/configs.yaml" \
  --annotators_config 'alpaca_eval_gpt4_turbo_fn' 
