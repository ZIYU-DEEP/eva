#!/bin/bash
# Make sure you run at the alpaca_eval benchmark directory.
# e.g., YOUR_ROOT/benchmark/alpaca_eval

# Define CONFIG_PATH to point to your configuration directory
CONFIG_PATH=$(pwd)/models_configs/gemma-2-9b-it-sppo-iter-0
export CONFIG_PATH
echo ${CONFIG_PATH}

# Substitute the environment variables in configs.yaml and save to local_configs.yaml
envsubst '${CONFIG_PATH}' < "${CONFIG_PATH}/configs.yaml" > "${CONFIG_PATH}/local_configs.yaml"

# Run alpaca_eval with the substituted configuration file
alpaca_eval evaluate_from_model \
  --model_configs "${CONFIG_PATH}/local_configs.yaml" \
  --annotators_config 'weighted_alpaca_eval_gpt4_turbo' 