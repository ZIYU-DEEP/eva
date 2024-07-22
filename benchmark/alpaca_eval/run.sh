alpaca_eval evaluate_from_model \
  --model_configs './models_configs/gemma' \
  --annotators_config 'alpaca_eval_gpt4_turbo_fn'   

alpaca_eval evaluate_from_model \
  --model_configs './models_configs/gemma-evol' \
  --annotators_config 'alpaca_eval_gpt4_turbo_fn'   

alpaca_eval evaluate_from_model \
  --model_configs '/localscratch/hsun409/github/eva/benchmark/alpaca_eval/models_configs/gemma-baseline' \
  --annotators_config 'alpaca_eval_gpt4_turbo_fn'   