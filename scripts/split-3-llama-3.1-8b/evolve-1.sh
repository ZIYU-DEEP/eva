#!/bin/bash
set -e
# set -x  # Print the commands

bash ./scripts/split-3-llama-3.1-8b/evolve-create-gen-iter-1.sh
bash ./scripts/split-3-llama-3.1-8b/evolve-create-iter-1.sh
bash ./scripts/split-3-llama-3.1-8b/evolve-gen-iter-1.sh
bash ./scripts/split-3-llama-3.1-8b/evolve-train-iter-1.sh
