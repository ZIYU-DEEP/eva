#!/bin/bash
set -e
# set -x  # Print the commands

export VLLM_ATTENTION_BACKEND=XFORMERS

# ------------------------------------------------------------------
# Below is to be re-written by source iterate.sh in other bash files
ITER=${ITER:-0}
MODEL_FAMILY=${MODEL_FAMILY:-"Mistral-7B-Instruct-v0.2"}
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"mistralai/Mistral-7B-Instruct-v0.2"}
LOSS_TYPE=${LOSS_TYPE:-"sppo"}
PREF=${PREF:-"sppo_score"}
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Other general parameter to be reset
HF_USERNAME=${HF_USERNAME:-'cat-searcher'}
LEARNING_RATE=${LEARNING_RATE:-"5.0e-7"}
BETA=${BETA:-"0.001"}
OPTIM=${OPTIM:-"rmsprop"}
N_EPOCHS=${N_EPOCHS:-9}
BATCH_SIZE=${BATCH_SIZE:-2}
ACCUMULATE=${ACCUMULATE:-4}
# ------------------------------------------------------------------


# ##################################################################
# 0. PREPARATION
# ##################################################################
# ------------------------------------------------------------------
NEXT_ITER=$((ITER + 1))

# The base model to train
if [ "$ITER" -eq 0 ]; then
    MODEL_PATH=${SFT_MODEL_PATH}
else
    MODEL_PATH="${HF_USERNAME}/${MODEL_FAMILY}-${LOSS_TYPE}-iter-${ITER}"
fi


# The preference data from the base model
DATASET="${HF_USERNAME}/ultrafeedback-${MODEL_FAMILY}-split-${SPLIT}-iter-${ITER}-pair"

# The directory for the saved model
SAVE_DIR="checkpoints/${MODEL_FAMILY}-${LOSS_TYPE}-iter-${NEXT_ITER}"
HUB_MODEL_ID="${HF_USERNAME}/${MODEL_FAMILY}-${LOSS_TYPE}-iter-${NEXT_ITER}"
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# export OMP_NUM_THREADS=2
export OMP_NUM_THREADS=$(nproc)

# Set the name for the log file
log_file="iter-${ITER}"
log_file+="_${LEARNING_RATE}"
log_file+="_${BETA}"
log_file+="_${OPTIM}"
log_file+="_${LOSS_TYPE}"
log_file+="_${PREF}"
log_file+="_${N_EPOCHS}"
echo "logging to $log_file.log"

# Save the new recipe
# TODO: make the config as an argument
config_name=$(echo "$DATASET" | cut -d '/' -f2) # identify with model-split-iter
new_config_file="./recipes/default/config_full_${config_name}.yaml"

# TODO: make this optional
cp ./recipes/default/config_full.yaml "$new_config_file"

# Update the dataset, model name, and hub model ID
python src/update_config.py \
    --dataset $DATASET \
    --model_name $MODEL_PATH \
    --hub_model_id $HUB_MODEL_ID \
    --config_path "$new_config_file" >"logs/train_$log_file.log"
# ------------------------------------------------------------------


# ##################################################################
# 1. Training
# ##################################################################
# ------------------------------------------------------------------
# Run the training
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file ./recipes/accelerate_configs/deepspeed_zero3.yaml \
    --main_process_port 8964 \
    eva/run_sppo.py "$new_config_file" \
    --learning_rate=$LEARNING_RATE \
    --beta=$BETA \
    --optim="$OPTIM" \
    --output_dir="$SAVE_DIR" \
    --loss_type=$LOSS_TYPE \
    --per_device_train_batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=$ACCUMULATE \
    --model_name_or_path=$MODEL_PATH \
    --num_train_epochs=$N_EPOCHS
# 2>&1 | tee "logs/train_$log_file.log"
# ------------------------------------------------------------------
