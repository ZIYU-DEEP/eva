#!/bin/bash
set -e
# set -x  # Print the commands

bash ./scripts/llama-8b/evolve-create-gen-iter-1.sh
bash ./scripts/llama-8b/evolve-create-iter-1.sh
bash ./scripts/llama-8b/evolve-gen-iter-1.sh
bash ./scripts/llama-8b/evolve-train-iter-1.sh
