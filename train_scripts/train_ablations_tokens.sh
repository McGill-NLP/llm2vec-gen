#!/bin/bash

model="qwen3-4"
lr=3e-4

for tokens in 2 6 10 50 100; do

    if [ $tokens -eq 2 ]; then
        compression=1
    elif [ $tokens -eq 6 ]; then
        compression=3
    elif [ $tokens -eq 10 ]; then
        compression=5
    elif [ $tokens -eq 50 ]; then
        compression=25
    elif [ $tokens -eq 100 ]; then
        compression=50
    else
        echo "Invalid number of tokens: $tokens"
        continue
    fi

    python scripts/train.py \
    training=llm2vec-gen \
    training.learning_rate=$lr \
    training.per_device_train_batch_size=32 \
    model=$model \
    model.encoding_mode=last_${compression}_tokens \
    data=$model \
    special_tokens=total_${tokens} \
    run.wandb_run_id=$model-tokens-${tokens}
done
