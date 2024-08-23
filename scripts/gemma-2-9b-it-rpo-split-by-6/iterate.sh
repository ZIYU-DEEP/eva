#!/bin/bash
set -x
set -e

# Set the environmental variable
export WANDB_PROJECT="rpo"
folder_name="gemma-2-9b-it-rpo-split-by-6"
EXP_PREFIX="NSPLIT6-"  # added on all the huggingface uploads

# This file should be run under the project directory
n_splits=${1:-3}

# General parameters
MODEL_FAMILY="gemma-2-9b-it"
SFT_MODEL_PATH="google/gemma-2-9b-it"
LOSS_TYPE="rpo"
PREF="rpo_score"
HF_USERNAME='cat-searcher'
PROMPT_SET_NAME_PREFIX='ultrafeedback-gemma-split'
N_PAIRS=6
DATA_ROOT="./data"
MAX_TOKENS=2048
DTYPE="bfloat16"
TEMPERATURE=0.7  # To Test
TOP_P=0.9
LEARNING_RATE="5.0e-7"
BETA="0.1"
OPTIM="adamw_torch"
N_EPOCHS=2
BATCH_SIZE=1
ACCUMULATE=8

for ((i=1; i<=n_splits; i++))
do
  echo "Split $i"

  pre_iter=$((i - 1))

  # Set parameters for each iteration and export them
  export SPLIT=$i ITER=$pre_iter\
         MODEL_FAMILY SFT_MODEL_PATH LOSS_TYPE PREF HF_USERNAME \
         N_PAIRS DATA_ROOT MAX_TOKENS DTYPE TEMPERATURE TOP_P \
         LEARNING_RATE BETA OPTIM N_EPOCHS BATCH_SIZE ACCUMULATE \
         PROMPT_SET_NAME_PREFIX EXP_PREFIX

  if [[ $i -ne 1 ]]; then
    # Source the gen.sh script only if i is not equal to 1
    source ./scripts/${folder_name}/gen.sh
  fi

  # Source the train.sh script
  source ./scripts/${folder_name}/train.sh 

done
