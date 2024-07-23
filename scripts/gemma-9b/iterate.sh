#!/bin/bash
set -x

# This file should be run under the project directory
# Number of iterations
n_iters=${1:-6}

# General parameters
MODEL_FAMILY="gemma-2-9b-it"
LOSS_TYPE="sppo"
PREF="sppo_score"
HF_USERNAME='cat-searcher'
N_PAIRS=5
DATA_ROOT="./data"
MAX_TOKENS=2048
DTYPE="bfloat16"
TEMPERATURE=0.7  # To Test
TOP_P=0.9
LEARNING_RATE="5.0e-7"
BETA="0.001"
OPTIM="rmsprop"
N_EPOCHS=9
BATCH_SIZE=8
ACCUMULATE=1

for ((i=0; i<n_iters; i++))
do
  echo "Iteration $i"

  # Set parameters for each iteration and export them
  export ITER=$i \
         MODEL_FAMILY LOSS_TYPE PREF HF_USERNAME \
         N_PAIRS DATA_ROOT MAX_TOKENS DTYPE TEMPERATURE TOP_P \
         LEARNING_RATE BETA OPTIM N_EPOCHS BATCH_SIZE ACCUMULATE

  # Source the gen.sh script
  source ./scripts/gemma-9b/gen.sh 

  # Source the train.sh script
  source ./scripts/gemma-9b/train.sh 
done
