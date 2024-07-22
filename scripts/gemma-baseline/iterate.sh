#!/bin/bash

# This file should be run under the project directory
# Number of iterations
n_iters=${1:-3}

# General parameters
MODEL_FAMILY="gemma-1.1-2b-it"
LOSS_TYPE="sppo"
PREF="sppo_score"
HF_USERNAME='cat-searcher'
N_PAIRS=5
DATA_ROOT="./data"
MAX_TOKENS=2048
DTYPE="bfloat16"
TEMPERATURE=0.7
TOP_P=0.9
LEARNING_RATE="5.0e-7"
BETA="0.001"
OPTIM="rmsprop"
N_EPOCHS=18
BATCH_SIZE=4
ACCUMULATE=2

for ((i=0; i<n_iters; i++))
do
  echo "Iteration $i"

  # Set parameters for each iteration and export them
  export ITER=$i \
         MODEL_FAMILY LOSS_TYPE PREF HF_USERNAME \
         N_PAIRS DATA_ROOT MAX_TOKENS DTYPE TEMPERATURE TOP_P \
         LEARNING_RATE BETA OPTIM N_EPOCHS BATCH_SIZE ACCUMULATE

  # Source the generate.sh script
  source ./scripts/gemma-baseline/generate.sh 

  # Source the train.sh script
  source ./scripts/gemma-baseline/train.sh 
done
