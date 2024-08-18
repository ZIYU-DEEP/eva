#!/bin/bash

# Candidate tasks
# gsm_plus_mini
# gsm8k
# wmdp
# leaderboard_mmlu_pro
# leaderboard_musr 
# truthfulqa
# toxigen


task=${1:-mathqa}
hf_username=${2:-cat-searcher}

MODEL_PATHS=(
    "google/gemma-2-9b-it"
    "cat-searcher/gemma-2-9b-it-dpo-iter-1"
    "cat-searcher/gemma-2-9b-it-dpo-iter-1-evol-1"
)

for i in "${!MODEL_PATHS[@]}"; do

    accelerate launch -m lm_eval --model hf \
        --model_args pretrained="${MODEL_PATHS[$i]}" \
        --tasks ${task} \
        --device cuda:0,1,2,3,4,5,6,7 \
        --batch_size 4 \
        --log_samples \
        --trust_remote_code \
        --output_path results \
        --hf_hub_log_args hub_results_org=${hf_username},hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 

    sleep 20

done
