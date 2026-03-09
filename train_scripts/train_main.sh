#!/bin/bash

for model in "qwen3-0.6" "qwen3-1.7" "qwen3-4" "qwen2.5-0.5" "qwen2.5-1.5" "qwen2.5-3" "llama3.2-1" "llama3.2-3"
    if [[ $model == *"qwen3"* ]]; then
        lr=3e-4
    else
        lr=5e-4
    fi
    
    python scripts/train.py \
    training=llm2vec-gen \
    training.learning_rate=$lr \
    training.per_device_train_batch_size=32 \
    model=$model \
    data=$model \
    special_tokens=total_20 \
    run.wandb_run_id=$model-v2
done

# 2GPU
for model in "qwen2.5-7" "llama3.1-8" "qwen3-8" ; do  
    if [[ $model == *"qwen3"* ]]; then
        lr=3e-4
    else
        lr=5e-4
    fi
    
    accelerate launch --main_process_port $((29500 + RANDOM % 1000)) scripts/train.py \
    training=llm2vec-gen \
    training.learning_rate=$lr \
    training.per_device_train_batch_size=16 \
    model=$model \
    data=$model \
    special_tokens=total_20 \
    run.wandb_run_id=$model
done
