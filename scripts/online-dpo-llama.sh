
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file ./recipes/accelerate_configs/deepspeed_zero3.yaml \
    --main_process_port 8964 \
    eva/run_dpo_online.py \
    --model_name_or_path "meta-llama/Meta-Llama-3.1-8B-Instruct"   \
    --reward_model_path "Skywork/Skywork-Reward-Llama-3.1-8B"\
    --dataset_name cat-searcher/ultrafeedback-gemma-split-1 \
    --hub_model_id "cat-searcher/llama-3.1-8b-it-odpo-rm-8b" \
    --output_dir checkpoints/llama-3.1-8b-it-odpo-rm-8b \
    --learning_rate 5.0e-7 \
    --lr_scheduler_type "cosine" \
    --beta 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --optim "adamw_torch" \
    --max_new_tokens 512 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --logging_steps 20 \
    --gradient_checkpointing \
    --save_strategy "epoch" \
    --do_eval "no" \
    --eval_strategy "no" \
    --n_completions 2 \
    --bf16 \
    --push_to_hub \
    --hub_private_repo \
    --attn_implementation "eager" \
    # --push_to_hub

# The eager mode is essential for gemma models!!