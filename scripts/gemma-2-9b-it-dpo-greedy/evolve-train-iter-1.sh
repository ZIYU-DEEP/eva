#!/bin/bash
set -e
# set -x  # Print the commands

# Set the environmental variable
export WANDB_PROJECT="dpo"

# ------------------------------------------------------------------
# Below is to be re-written by source iterate.sh in other bash files
ITER=${ITER:-1}
SPLIT=${SPLIT:-1}  # Specifically for evol
MODEL_FAMILY=${MODEL_FAMILY:-"gemma-2-9b-it"}
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
# Specifically for evol
SAMPLE_METRIC=${SAMPLE_METRIC:-'reward_gap'}
SAMPLE_FRAC=${SAMPLE_FRAC:-0.25}
SAMPLE_METHOD=${SAMPLE_METHOD:-'greedy'}

# For identify the evol 
EVOL_NO=${EVOL_NO:-1}
RATIO_BASE=${RATIO_BASE:-0.2}  # Use a relatively low ratio to avoid overfitting
RATIO_EVOL=${RATIO_EVOL:-0.8}  # Use more new evolved
# ------------------------------------------------------------------


# ##################################################################
# 0. PREPARATION
# ##################################################################
# ------------------------------------------------------------------
# The base model to train
MODEL_PATH="${HF_USERNAME}/${MODEL_FAMILY}-${LOSS_TYPE}-iter-${ITER}"   

# Set the loss type in trainer
if [ "$LOSS_TYPE" = 'dpo' ]; then
    LOSS_TYPE_TRAIN="sigmoid"
else
    LOSS_TYPE_TRAIN=${LOSS_TYPE}
fi

# The preference data from the base model
DATASET_PREFIX="${HF_USERNAME}/ultrafeedback-${LOSS_TYPE}-${MODEL_FAMILY}-split-${SPLIT}-iter-${ITER}"
DATASET_BASE="${DATASET_PREFIX}-pair"

# Default naming for subset and the evolved new set
DATASET_SUBSET="${HF_USERNAME}/${OUTPUT_DIR}-subset-${SAMPLE_METRIC}-${SAMPLE_FRAC}"
DATASET_EVOL="${DATASET_PREFIX}-evol-${SAMPLE_METRIC}-${SAMPLE_FRAC}-pair"

# The name of the dataset used to train the model
# TODO: a slight issue - we did not add the sample method to the dataset name
DATASET="${DATASET_BASE}-evol-${EVOL_NO}-mixed-${RATIO_BASE}-${RATIO_EVOL}-pair"

# Default directory for the saved model
SAVE_DIR="checkpoints/${MODEL_FAMILY}-${LOSS_TYPE}-iter-${ITER}-evol-${EVOL_NO}"
HUB_MODEL_ID="${HF_USERNAME}/${MODEL_FAMILY}-${LOSS_TYPE}-iter-${ITER}-evol-${EVOL_NO}"

# ---------------------------------------------
# FEAT
# Check if SAMPLE_METHOD is greedy
if [ "$SAMPLE_METHOD" == "greedy" ]; then
    DATASET_SUBSET="${DATASET_SUBSET}-greedy"
    DATASET_EVOLVED="${DATASET_EVOLVED}-greedy"
    DATASET="${DATASET}-greedy"
    SAVE_DIR="${SAVE_DIR}-greedy"
    HUB_MODEL_ID="${HUB_MODEL_ID}-greedy"
fi
# ---------------------------------------------


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
# Mix the datasets with generated responses
python src/snippets/combine_ds.py \
    --datasets $DATASET_BASE $DATASET_EVOL \
    --ratios $RATIO_BASE $RATIO_EVOL \
    --output $DATASET

# ------------------------------------------------------------------

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
