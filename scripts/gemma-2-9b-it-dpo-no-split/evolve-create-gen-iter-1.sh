#!/bin/bash
set -e  # Exit if failing
# set -x  # Print the commands

# Set the environmental variable
export WANDB_PROJECT="dpo"

# ------------------------------------------------------------------
# Below is to be re-written by source generate.sh in other bash files
ITER=${ITER:-1}
SPLIT=${SPLIT:-1}
MODEL_FAMILY=${MODEL_FAMILY:-"gemma-2-9b-it"}
SFT_MODEL_PATH=${SFT_MODEL_PATH:-"google/gemma-2-9b-it"}
LOSS_TYPE=${LOSS_TYPE:-"dpo"}
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Other general parameter to be reset
N_PAIRS=${N_PAIRS:-6}  # number of response generated for each prompt (better change name)
DATA_ROOT=${DATA_ROOT:-"./data"}  # assume the script is run at the project directory
MAX_TOKENS=${MAX_TOKENS:-2048}
DTYPE=${DTYPE:-"bfloat16"}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.9}
HF_USERNAME=${HF_USERNAME:-'cat-searcher'}
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# The prefix
PROMPT_SET_NAME_PREFIX=${PROMPT_SET_NAME_PREFIX:-"ultrafeedback-split"}
EXP_PREFIX=${EXP_PREFIX:-"NSPLIT3-"}
# ------------------------------------------------------------------



# ##################################################################
# 0. PREPARATION
# ##################################################################
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
N_GPUS=8
VLLM_WORLD_SIZE=1

# ------------------------------------------------------------------
# TODO - The naming fashion is a bit inconvenient
# Currently 0 refers to the SFT model - to fix later
# So dataset iter starts with 1, while model iter starts with 0
# X_1 & theta_0 --> Y_1
# X_1 & Y_1 --> theta_1
# NEXT_ITER=$((ITER + 1))

if [ "$ITER" -eq 0 ]; then
    MODEL_PATH=${SFT_MODEL_PATH}
else
    MODEL_PATH="${HF_USERNAME}/${EXP_PREFIX}${MODEL_FAMILY}-${LOSS_TYPE}-iter-${ITER}"
fi

# The original prompt set
PROMPT_SET_NAME="${HF_USERNAME}/${PROMPT_SET_NAME_PREFIX}-${SPLIT}"

# The below will be used for local folder, and HF upload
OUTPUT_DIR="${EXP_PREFIX}ultrafeedback-${LOSS_TYPE}-${MODEL_FAMILY}-split-${SPLIT}-iter-${ITER}"  

echo "The prompt set being used to generate responses is $PROMPT_SET_NAME."
echo "The base model used to generate responses is set to $MODEL_PATH."
echo "The generated responses will be uploaded to $OUTPUT_DIR with suffix pair and all."
# ------------------------------------------------------------------

# ------------------------------------------------------------------
TIMESTAMP=$(date +"%b-%d-%H-%M")
START_TIME=$(date +%s)
mkdir -p ./logs
# ------------------------------------------------------------------


# ##################################################################
# 1. RESPONSE GENERATION
# ##################################################################

# ------------------------------------------------------------------
# 1.1. Generate Y | X
for gpu_id in $(seq 0 $((N_GPUS-1))); do

    ################################################################
    # Common command for all GPUs
    gen="CUDA_VISIBLE_DEVICES=$gpu_id python src/generate.py \
        --output_dir $OUTPUT_DIR \
        --dataset_name $PROMPT_SET_NAME \
        --model_path $MODEL_PATH \
        --n_gpus $N_GPUS \
        --vllm_world_size $VLLM_WORLD_SIZE \
        --local_rank $gpu_id \
        --n_pairs $N_PAIRS \
        --max_tokens $MAX_TOKENS \
        --data_root $DATA_ROOT \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --dtype $DTYPE"
    ################################################################

    if [ $gpu_id -eq 0 ]; then
        # Print output for GPU 0 to screen and log file
        eval "$gen 2>&1 | tee logs/gen_${TIMESTAMP}_${gpu_id}.log" &
    else
        # Log output for other GPUs
        eval "$gen > logs/gen_${TIMESTAMP}_${gpu_id}.log 2>&1" &
    fi
done
wait
# ------------------------------------------------------------------

# Record Time in the log files
END_TIME=$(date +%s)  
DURATION=$((END_TIME - START_TIME)) 
echo "All generations completed at $(date)"
echo "Time elapsed: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s" | tee -a logs/gen_${TIMESTAMP}_*.log 


# ------------------------------------------------------------------
# 1.2. Combine Generated Results
################################################################
python src/combine_generate.py \
    --output_dir $OUTPUT_DIR \
    --data_root $DATA_ROOT \
    --n_gpus $N_GPUS \
    --n_pairs $N_PAIRS
################################################################
# ------------------------------------------------------------------


# ##################################################################
# 2. Get the chosen and rejected responses by reward models
# ##################################################################
# ------------------------------------------------------------------
# 2.1. Push the generated responses in local to huggingface
DATASET_TO_REWARD="${HF_USERNAME}/${OUTPUT_DIR}-all"

echo "Dataset to reward is: $DATASET_TO_REWARD."

python src/generate_to_hub.py \
        --output_dir $OUTPUT_DIR \
        --dataset_name $PROMPT_SET_NAME \
        --model_path $MODEL_PATH \
        --data_root $DATA_ROOT \
        --n_pairs $N_PAIRS \
        --to_hf_dataset $DATASET_TO_REWARD

# This will push a dataset to https://huggingface.co/datasets/${HF_USERNAME}/ultrafeeback-${LOSS_TYPE}-${MODEL_FAMILY}-split-${SPLIT}-iter-${ITER}-all
# ------------------------------------------------------------------

# We do not need to rank the responses here as the next script will take care of it.
