#!/bin/bash
iter_num=3
for i in $(seq 1 $iter_num); do
    if [ "$i" -eq 1 ]; then
        MODEL="google/gemma-1.1-2b-it"
    else
        MODEL=$OUTPUT_DIR
    fi

    # -------------------------------------------------------------
    # 0. PEREPARATION
    OUTPUT_DIR="checkpoints/gemma-1.1-2b-it-rloo-iter${i}"      # directory to save the checkpoint
    PROMPT="cat-searcher/data-gemma-1.1-2b-it-rloo-iter${i}"    # prompt dataset to use
    OUT="data-gemma-2-9b-it-sppo-iter${i}"                      # output data path
    echo "runing epoch $i"
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # 1. GENERATE Y_t | X_t, obtain the reward for each one
    bash scripts/generate.sh \
        --model $MODEL \
        --prompt $PROMPT \
        --out_path $OUT
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # 2. TRAIN with Y_t | X_t
    bash scripts/pipeline.sh \
        --model $MODEL \
        --iter $i \
        --dataset "response_data_gemma-1.1-2b-it-rloo-iter${i}_score" \
        --output_dir $OUTPUT_DIR \
        --num 1 \
        --batch_size 4 \
        --accumulate 2
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # 3. GENERATE X_t' | X_t; first filter by threshold; then expand
    bash scripts/generate_x.sh \
        --model $MODEL \
        --prompt $PROMPT \
        --out_path $OUT \
        --threshold 0.0
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # 4. TRAIN with Y_t' | X_t'
    bash scripts/pipeline.sh \
        --model $MODEL \
        --iter $i \
        --dataset "response_data_gemma-1.1-2b-it-rloo-iter${i}_score_prime" \
        --output_dir $OUTPUT_DIR \
        --num 1 \
        --batch_size 4 \
        --accumulate 2
    # -------------------------------------------------------------
done
