set -e
# set -x

new_config_file="./recipes/default/config-gemma-1.1-2b-it-dpo-iter-1.yaml"



# ##################################################################
# 1. Training
# ##################################################################
# ------------------------------------------------------------------
# Run the training
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file ./recipes/accelerate_configs/deepspeed_zero3.yaml \
    --main_process_port 8964 \
    eva/run_dpo.py "$new_config_file" 
# 2>&1 | tee "logs/train_$log_file.log"
# ------------------------------------------------------------------
