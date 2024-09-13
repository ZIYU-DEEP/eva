ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file ./recipes/accelerate_configs/deepspeed_zero3.yaml \
    --main_process_port 8964 \
    eva/run_dpo_online.py \
    --model_name_or_path trl-lib/pythia-1b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-1b-deduped-tldr-rm \
    --dataset_name cat-searcher/ultrafeedback-dpo-gemma-2-9b-it-split-1-iter-1-pair-evol-reward_var-0.25-mixed-0.2-0.8-pair \
    --learning_rate 5.0e-7 \
    --output_dir checkpoints/pythia-1b-deduped-tldr-online-dpo \
    --beta 0.1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --max_new_tokens 50 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --logging_steps 20 \
    --save_steps 0.1 \
    --gradient_checkpointing \
    --bf16 
    # --push_to_hub