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
SPLIT=${SPLIT:-1}
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
NUM_EVOLUTIONS=${NUM_EVOLUTIONS:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
EVOLVE_TEMPERATURE=${EVOLVE_TEMPERATURE:-1.0}
SAMPLE_METHOD=${SAMPLE_METHOD:-'importance_weighted'}
GEN_MODEL_NAME=${GEN_MODEL_NAME:-'gpt-4-0125-preview'}
# ------------------------------------------------------------------


# ##################################################################
# 0. PREPARATION
# ##################################################################
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
N_GPUS=8
VLLM_WORLD_SIZE=1

# ------------------------------------------------------------------
MODEL_PATH="${HF_USERNAME}/${MODEL_FAMILY}-${LOSS_TYPE}-iter-${ITER}"  # TODO: this naming fashion only works at iter-1
# DATASET_NAME="${HF_USERNAME}/ultrafeedback-split-${ITER}"  # INPUT - Only to get prompts from X_t
OUTPUT_DIR="ultrafeedback-${MODEL_FAMILY}-split-${SPLIT}-iter-${ITER}"  # OUTPUT - Used to save responses from this model | TODO: this naming fashion only works at iter-1
# Notice this is different from default training, where the split is plus one of iter.

echo "The base model used to generate responses is set to $MODEL_PATH."
echo "The generated responses will be uploaded to $OUTPUT_DIR with suffix pair and all."
# ------------------------------------------------------------------

# ------------------------------------------------------------------
TIMESTAMP=$(date +"%b-%d-%H-%M")
START_TIME=$(date +%s)
mkdir -p ./logs
# ------------------------------------------------------------------

# We skipped the generation part, as this is the same as the gen.sh, only using different iter number.


# ##################################################################
# 2. EVOLVE-RELEVANT Get Rewards for the responses
# ##################################################################
# TODO: REPLACE ALL WITH POINTWISE RM - REMOVE THE PAIRRM PART
DATASET_TO_REWARD="${HF_USERNAME}/${OUTPUT_DIR}-all"

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
# 3. EVOLVE-RELEVANT Create a new dataset with ONLY evolved prompts
# ##################################################################
DATASET_WITH_REWARDS="${HF_USERNAME}/${OUTPUT_DIR}-all-hf-rewards"
DATASET_SUBSET="${HF_USERNAME}/${OUTPUT_DIR}-subset-${SAMPLE_METRIC}-${SAMPLE_FRAC}"
DATASET_EVOLVED="${HF_USERNAME}/${OUTPUT_DIR}-evol-${SAMPLE_METRIC}-${SAMPLE_FRAC}"  

echo $DATASET_EVOLVED

python src/evolve_prompt.py \
    --hf_username  $HF_USERNAME \
    --input_dataset $DATASET_WITH_REWARDS \
    --subset_dataset $DATASET_SUBSET \
    --output_dataset $DATASET_EVOLVED \
    --data_root $DATA_ROOT \
    --gen_model_name $GEN_MODEL_NAME \
    --num_evolutions $NUM_EVOLUTIONS \
    --num_workers 20 \
    --do_adaptive_sample 1 \
    --sample_metric $SAMPLE_METRIC \
    --sample_frac $SAMPLE_FRAC \
    --sample_method $SAMPLE_METHOD \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --evolve_temperature $EVOLVE_TEMPERATURE

echo "Pushed the annotated data to ${DATASET_EVOLVED}."

# Next, we will run the gen again, with the newly specified dataset name.
# Then, we will combine the two pairs dataset, and train a new model on the combined dataset.
