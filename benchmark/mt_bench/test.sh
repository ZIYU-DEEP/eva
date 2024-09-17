MODEL_PATHS=(
    # "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # "mistralai/Mistral-7B-Instruct-v0.2"
    # "cat-searcher/meta-llama-3.1-8b-it-sppo-iter-1"
    # "cat-searcher/mistral-7b-it-v0.2-sppo-iter-1"
    # "cat-searcher/meta-llama-3.1-8b-it-sppo-iter-1-evol-1"
    "cat-searcher/gemma-2-9b-it-dpo-iter-1-skywork27-evol-1-reward_gap-0.25"
)

MODEL_IDS=(
    # "meta-llama-3.1-8b-it"
    # "mistral-7b-instruct-v0.2"
    # "meta-llama-3.1-8b-it-sppo-iter-1"
    # "mistral-7b-it-v0.2-sppo-iter-1"
    # "meta-llama-3.1-8b-it-sppo-iter-1-evol-1"
    "gemma-2-9b-it-dpo-iter-1-skywork27-evol-1-reward_gap-0.25"
)

_MODEL="cat-searcher/gemma-2-9b-it-dpo-iter-1-skywork27-evol-1-reward_gap-0.25"
_MODEL_ID="gemma-2-9b-it-dpo-iter-1-skywork27-evol-1-reward_gap-0.25"

# 1. Download models
python download_model.py \
    --model-path $_MODEL


# 2. Generate answers
python gen_model_answer.py \
    --model-path $_MODEL \
    --model-id $_MODEL_ID \
    --num-gpus-total 8


# 3. Generate judgement
python gen_judgment_single.py \
    --model-list "${MODEL_IDS[@]}" \
    --parallel 60 \
    --filename-suffix _latest  # Use any suffix to differentiate


# 4. Show result
python show_result.py \
    --model-list "${MODEL_IDS[@]}" \
    --filename-suffix _combined_latest
