python gen_judgment.py \
    --setting-file config/judge_config_dpo_iter_1_evol_1.yaml \
    --endpoint-file config/api_config.yaml

python gen_judgment.py \
    --setting-file config/judge_config_dpo_iter_1.yaml \
    --endpoint-file config/api_config.yaml

python show_result.py