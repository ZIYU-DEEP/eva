

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file ./recipes/accelerate_configs/deepspeed_zero3.yaml \
    --main_process_port 8964 \
    eva/run_dpo_online.py \
    --model_name_or_path google/gemma-2-9b-it     \
    --reward_model_path trl-lib/pythia-1b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-7 \
    --lr_scheduler_type "cosine" \
    --output_dir checkpoints/pythia-1b-deduped-tldr-online-dpo \
    --beta 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --optim "adamw_torch" \
    --max_new_tokens 50 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --logging_steps 20 \
    --gradient_checkpointing \
    --save_strategy "epoch" \
    --do_eval "no" \
    --eval_strategy "no" \
    --bf16 
    # --push_to_hub