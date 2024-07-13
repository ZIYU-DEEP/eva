set -e
# set -x


# ##################################################################
# 0. PREPARATION
# ##################################################################
# ------------------------------------------------------------------
# export OMP_NUM_THREADS=2
export OMP_NUM_THREADS=$(nproc)

# Set the parameters
LEARNING_RATE="5.0e-7"
ITER="1"
BETA="0.001"
LOSS_TYPE="sppo"
OPTIM="rmsprop"
PREF="sppo_score"
N_EPOCHS=18
MODEL="google/gemma-1.1-2b-it"
DATASET="cat-searcher/responses-gemma-1.1-2b-it-split-0-pair"
BATCH_SIZE=1
ACCUMULATE=1
SAVE_DIR="checkpoints/gemma-1.1-2b-it-iter-0"
RUN_NAME="sppo"

# Set the name for the log file
log_file="iter${ITER}"
log_file+="_${LEARNING_RATE}"
log_file+="_${BETA}"
log_file+="_${OPTIM}"
log_file+="_${LOSS_TYPE}"
log_file+="_${PREF}"
log_file+="_${N_EPOCHS}"
echo "logging to $log_file.log"

# Save the new recipe (in fact, only data mixer name is updated)
dataset_name=$(echo "$DATASET" | cut -d '/' -f2) 
new_config_file="./recipes/default/config_full_${dataset_name}.yaml"
cp ./recipes/default/config_full.yaml "$new_config_file"

# Update the dataset (todo: do this in place in bash files)
python src/update_config.py \
    --dataset $DATASET \
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
    eva/run_dpo.py "$new_config_file" \
    --learning_rate=$LEARNING_RATE \
    --beta=$BETA \
    --optim="$OPTIM" \
    --output_dir="$SAVE_DIR" \
    --run_name="$RUN_NAME" \
    --loss_type=$LOSS_TYPE \
    --per_device_train_batch_size=$BATCH_SIZE \
    --gradient_accumulation_steps=$ACCUMULATE \
    --model_name_or_path=$MODEL \
    --num_train_epochs=$N_EPOCHS
# 2>&1 | tee "logs/train_$log_file.log"
# ------------------------------------------------------------------
