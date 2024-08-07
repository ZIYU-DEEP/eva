#!/bin/bash
set -e
set -x

# This file should be run under the project directory
# Number of iterations
n_splits=${1:-3}

# General parameters
MODEL_FAMILY="mistral-7b-it-v0.2"
SFT_MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
LOSS_TYPE="sppo"
PREF="sppo_score"
HF_USERNAME='cat-searcher'
N_PAIRS=6
DATA_ROOT="./data"
MAX_TOKENS=2048  # TODO: This looks to be inconsistent with training which uses 1024
DTYPE="bfloat16"
TEMPERATURE=0.7  # TODO: To Test
TOP_P=0.9        # TODO: To test
LEARNING_RATE="5.0e-7"
BETA="0.001"
OPTIM="rmsprop"
N_EPOCHS=1
BATCH_SIZE=8
ACCUMULATE=1

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
  source ./scripts/split-3-mistral-7b/gen.sh

  # Source the train.sh script
  source ./scripts/split-3-mistral-7b/train.sh 

done
