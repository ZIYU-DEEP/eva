#!/bin/bash

# Candidate tasks
# gsm_plus_mini
# mathqa


task=${1:-mathqa}

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=cat-searcher/gemma-2-9b-it-sppo-iter-0 \
    --tasks ${task} \
    --device cuda:0,1,2,3,4,5,6,7 \
    --batch_size 4 \
    --log_samples \
    --trust_remote_code \
    --output_path results \
    # --hf_hub_log_args hub_results_org=cat-searcher,hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 

sleep 20

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=cat-searcher/gemma-2-9b-it-sppo-iter-1 \
    --tasks ${task} \
    --device cuda:0,1,2,3,4,5,6,7 \
    --batch_size 4 \
    --log_samples \
    --trust_remote_code \
    --output_path results \
    # --hf_hub_log_args hub_results_org=cat-searcher,hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


sleep 20

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=cat-searcher/gemma-2-9b-it-sppo-iter-1-evol-1 \
    --tasks ${task} \
    --device cuda:0,1,2,3,4,5,6,7 \
    --batch_size 4 \
    --log_samples \
    --trust_remote_code \
    --output_path results \
    # --hf_hub_log_args hub_results_org=cat-searcher,hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 


