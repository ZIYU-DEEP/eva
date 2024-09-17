
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file ./recipes/accelerate_configs/deepspeed_zero3.yaml \
    --main_process_port 8964 \
    eva/run_dpo_online.py \
    --model_name_or_path "google/gemma-2-2b-it"   \
    --reward_model_path "Skywork/Skywork-Reward-Gemma-2-27B"\
    --dataset_name cat-searcher/ultrafeedback-gemma-split-1 \
    --hub_model_id "cat-searcher/gemma-2-2b-it-odpo-rm-27b" \
    --output_dir checkpoints/gemma-2-2b-it-odpo-rm-27b \
    --learning_rate 5.0e-7 \
    --lr_scheduler_type "cosine" \
    --beta 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --optim "adamw_torch" \
    --max_new_tokens 512 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --logging_steps 10 \
    --gradient_checkpointing \
    --do_eval "no" \
    --eval_strategy "no" \
    --n_completions 2 \
    --bf16 \
    --push_to_hub \
    --hub_private_repo \
    --attn_implementation "eager" \
    --log_level "info" \
    --hub_strategy "checkpoint" \
    --save_strategy "steps" \
    --save_steps 0.2 \
    # --save_strategy "epoch" \


# The eager mode is essential for gemma models!!