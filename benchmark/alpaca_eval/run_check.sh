# need a GPU for local models
alpaca_eval evaluate_from_model \
  --model_configs 'SPPO-Gemma-2-9B-It-PairRM' \
  --annotators_config 'alpaca_eval_gpt4_turbo_fn'      
