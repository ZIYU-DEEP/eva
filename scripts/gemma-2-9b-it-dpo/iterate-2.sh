#!/bin/bash
set -x
set -e

# Set the environmental variable
export WANDB_PROJECT="dpo"

# This file should be run under the project directory
# Number of iterations
n_splits=${1:-2}
folder_name="gemma-2-9b-it-dpo"

# General parameters
MODEL_FAMILY="gemma-2-9b-it"
SFT_MODEL_PATH="google/gemma-2-9b-it"
LOSS_TYPE="dpo"
PREF="dpo_score"
HF_USERNAME='cat-searcher'
N_PAIRS=6
DATA_ROOT="./data"
MAX_TOKENS=2048
DTYPE="bfloat16"
TEMPERATURE=0.7  # To Test
TOP_P=0.9
LEARNING_RATE="5.0e-7"
BETA="0.05"
OPTIM="adamw_torch"
N_EPOCHS=2
BATCH_SIZE=1
ACCUMULATE=8

for ((i=2; i<=n_splits; i++))
do
  echo "Split $i"

  pre_iter=$((i - 1))

  # Set parameters for each iteration and export them
  export SPLIT=$i ITER=$pre_iter\
         MODEL_FAMILY SFT_MODEL_PATH LOSS_TYPE PREF HF_USERNAME \
         N_PAIRS DATA_ROOT MAX_TOKENS DTYPE TEMPERATURE TOP_P \
         LEARNING_RATE BETA OPTIM N_EPOCHS BATCH_SIZE ACCUMULATE 

  # # # Source the gen.sh script
  source ./scripts/${folder_name}/gen.sh

  # Source the train.sh script
  source ./scripts/${folder_name}/train.sh 

done
