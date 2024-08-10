#!/bin/bash
set -e
set -x  # Print the commands

# bash ./scripts/split-3-llama-3.1-8b/evolve-create-gen-iter-1.sh
# bash ./scripts/split-3-llama-3.1-8b/evolve-create-iter-1.sh
# bash ./scripts/split-3-llama-3.1-8b/evolve-gen-iter-1.sh
# bash ./scripts/split-3-llama-3.1-8b/evolve-train-iter-1.sh

# General parameters
SPLIT=1
ITER=1
MODEL_FAMILY="meta-llama-3.1-8b-it"
SFT_MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
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

SAMPLE_METRIC="reward_gap"
SAMPLE_FRAC=0.25
NUM_EVOLUTIONS=4  # TODO: Using a different number
MAX_PROMPT_LENGTH=512  # TODO: check if we need longer
EVOLVE_TEMPERATURE=1.0  # TODO: see different versions
SAMPLE_METHOD="importance_weighted"
GEN_MODEL_NAME="gpt-4o-mini"
EVOL_NO=1
RATIO_BASE=0.2
RATIO_EVOL=0.8

# Set parameters for each iteration and export them
export SPLIT ITER \
        MODEL_FAMILY SFT_MODEL_PATH LOSS_TYPE PREF HF_USERNAME \
        N_PAIRS DATA_ROOT MAX_TOKENS DTYPE TEMPERATURE TOP_P \
        LEARNING_RATE BETA OPTIM N_EPOCHS BATCH_SIZE ACCUMULATE \
        SAMPLE_METRIC SAMPLE_FRAC NUM_EVOLUTIONS MAX_PROMPT_LENGTH \
        EVOLVE_TEMPERATURE SAMPLE_METHOD GEN_MODEL_NAME EVOL_NO \
        RATIO_BASE RATIO_EVOL \

source ./scripts/split-3-llama-3.1-8b/evolve-create-gen-iter-1.sh
source ./scripts/split-3-llama-3.1-8b/evolve-create-iter-1.sh
source ./scripts/split-3-llama-3.1-8b/evolve-gen-iter-1.sh
source ./scripts/split-3-llama-3.1-8b/evolve-train-iter-1.sh


