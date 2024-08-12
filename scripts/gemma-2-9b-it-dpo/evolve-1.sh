#!/bin/bash
set -e
set -x  # Print the commands
folder_name="gemma-2-9b-it-dpo"

bash ./scripts/${folder_name}/evolve-create-gen-iter-1.sh
bash ./scripts/${folder_name}/evolve-create-iter-1.sh
bash ./scripts/${folder_name}/evolve-gen-iter-1.sh
bash ./scripts/${folder_name}/evolve-train-iter-1.sh
