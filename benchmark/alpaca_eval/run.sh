# Make sure you run at the alpaca_eval benchmark directory.
# e.g., YOUR_ROOT/benchmark/alpaca_eval

CONFIG_PATH=$(pwd)/models_configs/gemma-2-9b-it-sppo-iter-0
echo ${CONFIG_PATH}

alpaca_eval evaluate_from_model \
  --model_configs $CONFIG_PATH \
  --annotators_config 'weighted_alpaca_eval_gpt4_turbo' \
  --reference_output 'gpt4_turbo'
 