#!/bin/bash

model="qwen3-4"
lr=3e-4

for tokens in 1 5 20 50 100; do

    python scripts/train.py \
    training=llm2vec-gen \
    training.learning_rate=$lr \
    training.per_device_train_batch_size=32 \
    model=$model \
    model.encoding_mode=last_${tokens}_tokens \
    data=$model \
    special_tokens=total_${tokens} \
    run.wandb_run_id=$model-tokens-${tokens}
done
