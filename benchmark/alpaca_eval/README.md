# README
```bash
alpaca_eval evaluate_from_model \
  --model_configs './models_configs/gemma' \
  --annotators_config 'alpaca_eval_gpt4_turbo_fn'   

alpaca_eval evaluate_from_model \
  --model_configs './models_configs/gemma-evol' \
  --annotators_config 'alpaca_eval_gpt4_turbo_fn'  
```      

```bash
                            length_controlled_winrate  win_rate  standard_error  n_total  avg_length
gemma-1.1-2b-it-sppo-iter0                      19.85     13.04            1.19      805        1220
gemma-1.1-2b-it-sppo-iter0-evol 
```