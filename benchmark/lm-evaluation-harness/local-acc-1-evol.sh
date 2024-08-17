task=${1:-arc_challenge}

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=cat-searcher/gemma-2-9b-it-sppo-iter-1-evol-1 \
    --tasks ${task} \
    --device cuda:0,1,2,3,4,5,6,7 \
    --batch_size 4 \
    --log_samples \
    --trust_remote_code \
    --output_path results  
