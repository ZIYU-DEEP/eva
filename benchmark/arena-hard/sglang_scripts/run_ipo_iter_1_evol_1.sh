#!/bin/bash
set -e
sleep 20

MODEL_PATH="cat-searcher/gemma-2-9b-it-ipo-iter-1-evol-1"
MODEL_NAME="gemma-2-9b-it-ipo-iter-1-evol-1"
dtype="bfloat16"

python gen_answer.py \
    --setting-file config/gen_answer_config_ipo_iter_1_evol_1.yaml \
    --enipoint-file config/api_config.yaml

python gen_judgment.py \
    --setting-file config/judge_config_ipo_iter_1_evol_1.yaml \
    --enipoint-file config/api_config.yaml

python show_result.py