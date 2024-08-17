#!/bin/bash
set -e
set -x  # Print the commands

sleep 4h  # wait for me

# Set the environmental variable
export WANDB_PROJECT="ipo"

# Set the folder name
folder_name="gemma-2-9b-it-ipo"

bash ./scripts/${folder_name}/evolve-create-gen-iter-1.sh
bash ./scripts/${folder_name}/evolve-create-iter-1.sh
bash ./scripts/${folder_name}/evolve-gen-iter-1.sh
bash ./scripts/${folder_name}/evolve-train-iter-1.sh
