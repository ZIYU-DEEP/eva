#!/bin/bash
set -e  # Exit if failing
# set -x  # Print the commands

# ------------------------------------------------------------------
# Below is to be re-written by source generate.sh in other bash files
ITER=${ITER:-1}
SPLIT=${SPLIT:-1}  # Specifically for evol
MODEL_FAMILY=${MODEL_FAMILY:-"mistral-7b-it-v0.2"}
LOSS_TYPE=${LOSS_TYPE:-"sppo"}
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
# Specifically for evol
SAMPLE_METRIC=${SAMPLE_METRIC:-'reward_gap'}
SAMPLE_FRAC=${SAMPLE_FRAC:-0.25}
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

MODEL_PATH="${HF_USERNAME}/${MODEL_FAMILY}-${LOSS_TYPE}-iter-${ITER}"

# ------------------------------------------------------------------
# Specifically for evol

# The below will be used for local folder, and HF upload
OUTPUT_DIR="ultrafeedback-${MODEL_FAMILY}-split-${SPLIT}-iter-${ITER}-evol-${SAMPLE_METRIC}-${SAMPLE_FRAC}" 
PROMPT_SET_NAME="${HF_USERNAME}/${OUTPUT_DIR}"

# PROMPT_SET_NAME="cat-searcher/ultrafeedback-2-9b-it-split-1-iter-1-evol-reward_gap-0.25"
# That is the name generated previously.
# ------------------------------------------------------------------

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
# 2. RANK the data
# ##################################################################
# ------------------------------------------------------------------
# 2.1. Preload the reward model
python src/preload.py  # Initialize the checkpoints
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 2.2. Rank the responses
for gpu_id in $(seq 0 $((N_GPUS-1))); do

    ################################################################
    # Common command for all GPUs
    rank="CUDA_VISIBLE_DEVICES=$gpu_id python src/rank.py \
        --output_dir $OUTPUT_DIR \
        --dataset_name $PROMPT_SET_NAME \
        --model_path $MODEL_PATH \
        --data_root $DATA_ROOT \
        --n_pairs $N_PAIRS \
        --n_gpus $N_GPUS \
        --local_rank $gpu_id"
    ################################################################

    if [ $gpu_id -eq 0 ]; then
        # Print output for GPU 0 to screen and log file
        eval "$rank 2>&1 | tee logs/rank_${TIMESTAMP}_${gpu_id}.log" &
    else
        # Log output for other GPUs
        eval "$rank > logs/rank_${TIMESTAMP}_${gpu_id}.log 2>&1" &
    fi
done
wait
# ------------------------------------------------------------------

# Record Time in the log files
END_TIME_=$(date +%s)  
DURATION=$((END_TIME_ - END_TIME)) 
echo "All ranking completed at $(date)"
echo "Time elapsed: $((DURATION / 3600))h $((DURATION % 3600 / 60))m $((DURATION % 60))s" | tee -a logs/rank_${TIMESTAMP}_*.log 

# ------------------------------------------------------------------
# 2.3. Compute the probability
################################################################
python src/compute_prob.py \
    --output_dir $OUTPUT_DIR \
    --dataset_name $PROMPT_SET_NAME \
    --data_root $DATA_ROOT \
    --n_pairs $N_PAIRS \
    --n_gpus $N_GPUS \
    --hf_username $HF_USERNAME
################################################################
# ------------------------------------------------------------------

# In the end, we will push two datasets to HF

# One is ${OUTPUT_DIR}-all
# See https://huggingface.co/datasets/${HF_USERNAME}/ultrafeeback-${MODEL_FAMILY}-split-${SPLIT}-iter-${ITER}-evol-${SAMPLE_METRIC}-${SAMPLE_FRAC}-all

# One is ${OUTPUT_DIR}-pair, 
# With columns: chosen, rejected, chosen_probs, chosen_probs_win, chosen_probs_lose
# See https://huggingface.co/datasets/${HF_USERNAME}/ultrafeedback-${MODEL_FAMILY}-split-${SPLIT}-iter-${ITER}-evol-${SAMPLE_METRIC}-${SAMPLE_FRAC}-pair
