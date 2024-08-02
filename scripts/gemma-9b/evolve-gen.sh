#!/bin/bash
set -e  # Exit if failing
# set -x  # Print the commands

# GENERAL IDEA
# 1. Given X_t and theta_t, we first generate responses from theta_t
#    (Notice that theta_t is trained from X_t)
# 2. based on the responses, we apply a point_wise RM to reward all the responses
#    We then use the rewards to provide a metric to prompt in X_t
# 3. We then build a set with evolved X_t^evol
# 4. We then mix the evolved X_t^evol with the original X_t to form X_t^plus
# 5. We then train a new model theta_{t}^{evol} on X_t

# Below is to be re-written by source generate.sh in other bash files
ITER=${ITER:-1}  # the ind should start from at least 1, as 0 means unaligned in our notation
MODEL_FAMILY=${MODEL_FAMILY:-"gemma-2-9b-it"}
LOSS_TYPE=${LOSS_TYPE:-"sppo"}
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Other general parameter to be reset
N_PAIRS=${N_PAIRS:-6}  # number of response generated for each prompt (better change name)
DATA_ROOT=${DATA_ROOT:-"./data"}  # assume the script is run at the project directory
MAX_TOKENS=${MAX_TOKENS:-2048}
DTYPE=${DTYPE:-"bfloat16"}
TEMPERATURE=${TEMPERATURE:-0.9}
TOP_P=${TOP_P:-0.9}
HF_USERNAME=${HF_USERNAME:-'cat-searcher'}
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
NEXT_ITER=$((ITER + 1))

MODEL_PATH="${HF_USERNAME}/${MODEL_FAMILY}-${LOSS_TYPE}-iter-${ITER}"  # TODO: this naming fashion only works at iter-1
DATASET_NAME="${HF_USERNAME}/ultrafeedback-gemma-split-${ITER}"  # INPUT - Only to get prompts from X_t
OUTPUT_DIR="ultrafeedback-${MODEL_FAMILY}-split-${ITER}-buffer"  # OUTPUT - Used to save responses from this model | TODO: this naming fashion only works at iter-1

echo "The base model used to generate responses is set to $MODEL_PATH."
echo "The generated responses will be uploaded to $OUTPUT_DIR with suffix pair and all."
# ------------------------------------------------------------------

# ------------------------------------------------------------------
TIMESTAMP=$(date +"%b-%d-%H-%M")
START_TIME=$(date +%s)
mkdir -p ./logs
# ------------------------------------------------------------------



# ##################################################################
# 3. EVOLVE-RELEVANT Get Rewards for the responses
# ##################################################################
# TODO: REPLACE ALL WITH POINTWISE RM - REMOVE THE PAIRRM PART
DATASET_TO_REWARD="${HF_USERNAME}/ultrafeedback-gemma-split-${SPLIT}-iter-${ITER}-all"

python src/reward_hf.py \
    --input_dataset $DATASET_TO_REWARD \
    --output_dir $OUTPUT_DIR \
    --n_generations $N_PAIRS \
    --data_root $DATA_ROOT \
    --hf_username  $HF_USERNAME\
    --reward_model_path RLHFlow/ArmoRM-Llama3-8B-v0.1 \
    --torch_dtype $DTYPE

echo "Pushed the annotated data to ${HF_USERNAME}/${OUTPUT_DIR}-all-hf-rewards."


# ##################################################################
# 4. EVOLVE-RELEVANT Create a new dataset with ONLY evolved prompts
# ##################################################################
DATASET_WITH_REWARDS="${HF_USERNAME}/${OUTPUT_DIR}-all-hf-rewards"
DATASET_EVOLVED="${HF_USERNAME}/${OUTPUT_DIR}-all-hf-rewards-resample-evol"  

python data_gen/evol_prompt.py \
    --hf_username  $HF_USERNAME \
    --input_dataset $DATASET_WITH_REWARDS \
    --output_dataset $DATASET_EVOLVED \
    --data_root $DATA_ROOT \
    --gen_model_name gpt-4-turbo \
    --num_evolutions 4 \
    --num_workers 30 \
    --do_adaptive_sample 1 \
    --sample_metric reward_mean \
    --sample_frac 0.25 \
    --sample_method importance_weighted

echo "Pushed the annotated data to ${DATASET_EVOLVED}."