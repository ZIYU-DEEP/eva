#!/bin/bash
iter_num=3
for i in $(seq 1 $iter_num); do
    if [ "$i" -eq 1 ]; then
        MODEL="google/gemma-1.1-2b-it"
    else
        MODEL=$OUTPUT_DIR
    fi

    # Set necessary variables
    OUTPUT_DIR="checkpoints/gemma-1.1-2b-it-sppo-iter${i}"      # directory to save the checkpoint
    PROMPT="UCLA-AGI/data-mistral-7b-instruct-sppo-iter${i}"    # prompt dataset to use
    OUT="data-gemma-2-9b-it-sppo-iter${i}"                      # output data path
    echo "runing epoch $i"

    # Generate responses (variables get overwritten)
    bash scripts/generate.sh \
        --model $MODEL \
        --prompt $PROMPT \
        --out_path $OUT

    # Run pipeline (train with synthetic model)
    bash scripts/pipeline.sh \
        --model $MODEL \
        --iter $i \
        --dataset "synthetic_data_gemma-2-9b-it-sppo-iter${i}_score" \
        --output_dir $OUTPUT_DIR \
        --num 1 \
        --batch_size 4 \
        --accumulate 2
done
