#!/bin/bash
set -e  # Exit if failing
# set -x  # Print the commands


# ##################################################################
# 0. PREPARATION
# ##################################################################
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# ------------------------------------------------------------------
# Below is to be re-written by source generate.sh in other bash files
MODEL_PATH="google/gemma-1.1-2b-it"
DATASET_NAME="cat-searcher/ultra-feedback-split-0"
HF_USERNAME='cat-searcher'

N_GPUS=8
VLLM_WORLD_SIZE=1

N_PAIRS=5  # number of response generated for each prompt (better change name)
MAX_TOKENS=2048
OUTPUT_DIR="responses-gemma-1.1-2b-it-split-0"
DATA_ROOT="./data"  # assume . is the root of the project
DTYPE="float16"

TEMPERATURE=0.7
TOP_P=0.9

TIMESTAMP=$(date +"%b-%d-%H-%M")
START_TIME=$(date +%s)
# ------------------------------------------------------------------


# ##################################################################
# 1. RESPONSE GENERATION
# ##################################################################

# ------------------------------------------------------------------
# 1.1. Generate Y | X
for gpu_id in $(seq 0 $((N_GPUS-1))); do

    # Common command for all GPUs
    gen="CUDA_VISIBLE_DEVICES=$gpu_id python src/generate.py \
        --model_path $MODEL_PATH \
        --dataset_name $DATASET_NAME \
        --n_gpus $N_GPUS \
        --vllm_world_size $VLLM_WORLD_SIZE \
        --local_rank $gpu_id \
        --n_pairs $N_PAIRS \
        --max_tokens $MAX_TOKENS \
        --output_dir $OUTPUT_DIR \
        --data_root $DATA_ROOT \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --dtype $DTYPE"

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
python src/combine_generate.py \
    --output_dir $OUTPUT_DIR \
    --data_root $DATA_ROOT \
    --n_gpus $N_GPUS \
    --n_pairs $N_PAIRS
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
    # Common command for all GPUs
    rank="CUDA_VISIBLE_DEVICES=$gpu_id python src/rank.py \
        --model_path $MODEL_PATH \
        --output_dir $OUTPUT_DIR \
        --data_root $DATA_ROOT \
        --n_pairs $N_PAIRS \
        --n_gpus $N_GPUS \
        --local_rank $gpu_id \
        --dataset_name $DATASET_NAME"

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
python src/compute_prob.py \
    --output_dir $OUTPUT_DIR \
    --data_root $DATA_ROOT \
    --n_pairs $N_PAIRS \
    --n_gpus $N_GPUS \
    --dataset_name $DATASET_NAME \
    --hf_username $HF_USERNAME
# ------------------------------------------------------------------

# In the end, we will push two datasets to HF

# One is ${OUTPUT_DIR}-all
# See https://huggingface.co/datasets/cat-searcher/responses-gemma-1.1-2b-it-split-0-all

# One is ${OUTPUT_DIR}-pair, 
# With columns: chosen, rejected, chosen_probs, chosen_probs_win, chosen_probs_lose
# See https://huggingface.co/datasets/cat-searcher/responses-gemma-1.1-2b-it-split-0-pair
