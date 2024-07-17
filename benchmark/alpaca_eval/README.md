# need a GPU for local models
alpaca_eval evaluate_from_model \
  --model_configs '/localscratch/hsun409/github/eva/benchmark/alpaca_eval/models_configs/gemma' \
  --annotators_config 'alpaca_eval_gpt4_turbo_fn'      