#!/bin/bash
# This file should be run under the project directory
set -e
set -x  # Print the commands

# Set the environmental variable
folder_name="gemma-2-9b-it-dpo-no-split"
export WANDB_PROJECT="dpo"
EXP_PREFIX="NOSPLIT-"  # added on all the huggingface uploads

# Number of iterations
n_splits=${1:-1}

# General parameters
MODEL_FAMILY="gemma-2-9b-it"
SFT_MODEL_PATH="google/gemma-2-9b-it"
LOSS_TYPE="dpo"
PREF="dpo_score"
N_PAIRS=8
DATA_ROOT="./data"
MAX_TOKENS=2048
TEMPERATURE=0.7  # To Test
TOP_P=0.9
HF_USERNAME='cat-searcher'
PROMPT_SET_NAME_PREFIX='uf-split'

DTYPE="bfloat16"

LEARNING_RATE="5.0e-7"
BETA="0.1"
OPTIM="adamw_torch"
N_EPOCHS=1
BATCH_SIZE=1
ACCUMULATE=8

SAMPLE_METRIC=reward_gap
SAMPLE_FRAC=0.25
NUM_EVOLUTIONS=4
MAX_PROMPT_LENGTH=666
EVOLVE_TEMPERATURE=0.88
SAMPLE_METHOD=importance_weighted
GEN_MODEL_NAME=gpt-4-0125-preview

EVOL_NO=1
RATIO_BASE=0.2
RATIO_EVOL=0.8

for ((i=1; i<=n_splits; i++))
do
  echo "Split $i"

  # Set parameters for each iteration and export them
  export SPLIT=$i ITER=$i \
         MODEL_FAMILY SFT_MODEL_PATH LOSS_TYPE PREF HF_USERNAME \
         N_PAIRS DATA_ROOT MAX_TOKENS DTYPE TEMPERATURE TOP_P \
         LEARNING_RATE BETA OPTIM N_EPOCHS BATCH_SIZE ACCUMULATE \
         SAMPLE_METRIC SAMPLE_FRAC NUM_EVOLUTIONS \
         MAX_PROMPT_LENGTH EVOLVE_TEMPERATURE SAMPLE_METHOD GEN_MODEL_NAME \
         EVOL_NO RATIO_BASE RATIO_EVOL \
         EXP_PREFIX PROMPT_SET_NAME_PREFIX

  source ./scripts/${folder_name}/evolve-create-gen-iter-1.sh
  source ./scripts/${folder_name}/evolve-create-iter-1.sh
  source ./scripts/${folder_name}/evolve-gen-iter-1.sh
  source ./scripts/${folder_name}/evolve-train-iter-1.sh

done
