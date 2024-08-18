#!/bin/bash
set -e
# set -x  # Print the commands

# Set the environmental variable
export WANDB_PROJECT="dpo"
export VLLM_ATTENTION_BACKEND=FLASHINFER

# ------------------------------------------------------------------
# Below is to be re-written by source iterate.sh in other bash files
ITER=${ITER:-0}
SPLIT=${SPLIT:-1}
MODEL_FAMILY=${MODEL_FAMILY:-"gemma-2-9b-it"}
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"google/gemma-2-9b-it"}
LOSS_TYPE=${LOSS_TYPE:-"dpo"}
PREF=${PREF:-"dpo_score"}
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Other general parameter to be reset
HF_USERNAME=${HF_USERNAME:-'cat-searcher'}
LEARNING_RATE=${LEARNING_RATE:-"5.0e-7"}
BETA=${BETA:-"0.05"}
OPTIM=${OPTIM:-"adamw_torch"}
N_EPOCHS=${N_EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-1}
ACCUMULATE=${ACCUMULATE:-8}
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# The prefix
EXP_PREFIX=${EXP_PREFIX:-"NSPLIT3-"}
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
    MODEL_PATH="${HF_USERNAME}/${EXP_PREFIX}${MODEL_FAMILY}-${LOSS_TYPE}-iter-${ITER}"
fi

# Set the loss type in trainer
if [ "$LOSS_TYPE" = 'dpo' ]; then
    LOSS_TYPE_TRAIN="sigmoid"
else
    LOSS_TYPE_TRAIN=${LOSS_TYPE}
fi

echo "Loss type for training set to be ${LOSS_TYPE_TRAIN}."

# The preference data from the base model
# TODO: to update the naming convention with parameters
DATASET="${HF_USERNAME}/${EXP_PREFIX}ultrafeedback-${LOSS_TYPE}-${MODEL_FAMILY}-split-${SPLIT}-iter-${ITER}-pair"

# The directory for the saved model
SAVE_DIR="checkpoints/${EXP_PREFIX}${MODEL_FAMILY}-${LOSS_TYPE}-iter-${NEXT_ITER}"
HUB_MODEL_ID="${HF_USERNAME}/${EXP_PREFIX}${MODEL_FAMILY}-${LOSS_TYPE}-iter-${NEXT_ITER}"

echo "The dataset used is $DATASET."
echo "The model will be pushed to $HUB_MODEL_ID."
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# export OMP_NUM_THREADS=2
export OMP_NUM_THREADS=$(nproc)

# Set the name for the log file
log_file="${EXP_PREFIX}iter-${ITER}"
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
new_config_file="./recipes/iterative-dpo/config_full_${config_name}.yaml"

# TODO: make this optional
cp ./recipes/iterative-dpo/config_full.yaml "$new_config_file"

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
    --loss_type=$LOSS_TYPE_TRAIN \
    --per_device_train_batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=$ACCUMULATE \
    --model_name_or_path=$MODEL_PATH \
    --num_train_epochs=$N_EPOCHS
# 2>&1 | tee "logs/train_$log_file.log"
# ------------------------------------------------------------------
