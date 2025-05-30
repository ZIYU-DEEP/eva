#!/bin/bash
set -x

# This file should be run under the project directory
# Number of iterations
n_splits=${1:-1}

# General parameters
MODEL_FAMILY="Mistral-7B-Instruct-v0.2"
SFT_MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
LOSS_TYPE="sppo"
PREF="sppo_score"
HF_USERNAME='cat-searcher'
N_PAIRS=6
DATA_ROOT="./data"
MAX_TOKENS=2048
DTYPE="bfloat16"
TEMPERATURE=1.0  # To Test
TOP_P=1.0
LEARNING_RATE="5.0e-7"
BETA="0.001"
OPTIM="rmsprop"
N_EPOCHS=9
BATCH_SIZE=2
ACCUMULATE=4

for ((i=1; i<=n_splits; i++))
do
  echo "Split $i"

  pre_iter=$((i - 1))

  # Set parameters for each iteration and export them
  export SPLIT=$i ITER=$pre_iter\
         MODEL_FAMILY SFT_MODEL_PATH LOSS_TYPE PREF HF_USERNAME \
         N_PAIRS DATA_ROOT MAX_TOKENS DTYPE TEMPERATURE TOP_P \
         LEARNING_RATE BETA OPTIM N_EPOCHS BATCH_SIZE ACCUMULATE 

  # # # Source the gen.sh script
  source ./scripts/mistral-7b/gen.sh

  # Source the train.sh script
  source ./scripts/mistral-7b/train.sh 

done
