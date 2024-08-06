# README

## Requirements
```bash
pip install alpaca-eval
pip install vllm==0.5.4
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```
<!-- Tested on h100-c. -->

## Scripts
```bash
CONFIG_PATH=$(pwd)/models_configs/gemma-2-9b-it-sppo-iter-0
export CONFIG_PATH
echo ${CONFIG_PATH}

# Substitute the environment variables in configs.yaml and save to local_configs.yaml
envsubst '${CONFIG_PATH}' < "${CONFIG_PATH}/configs.yaml" > "${CONFIG_PATH}/local_configs.yaml"

# Run alpaca_eval with the substituted configuration file
alpaca_eval evaluate_from_model \
  --model_configs "${CONFIG_PATH}/local_configs.yaml" \
  --annotators_config 'weighted_alpaca_eval_gpt4_turbo' \
  --reference_outputs 'gpt4_turbo' 
```      

<!-- ```bash
                            length_controlled_winrate  win_rate  standard_error  n_total  avg_length
gemma-1.1-2b-it-sppo-iter0                      19.85     13.04            1.19      805        1220
gemma-1.1-2b-it-sppo-iter0-evol 
``` -->