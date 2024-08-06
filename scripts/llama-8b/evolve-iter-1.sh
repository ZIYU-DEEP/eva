#!/bin/bash
set -e
# set -x  # Print the commands

bash ./scripts/mistral-7b/evolve-create-gen-iter-1.sh
bash ./scripts/mistral-7b/evolve-create-iter-1.sh
bash ./scripts/mistral-7b/evolve-gen-iter-1.sh
bash ./scripts/mistral-7b/evolve-train-iter-1.sh
