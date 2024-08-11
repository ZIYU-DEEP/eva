lm_eval --model hf \
    --model_args pretrained=google/gemma-2-9b-it \
    --tasks hellaswag \
    --device cuda:0,1,2,3,4,5,6,7 \
    --batch_size 4 \
    --log_samples \
    --output_path results \
    --hf_hub_log_args hub_results_org=cat-searcher,hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False 
