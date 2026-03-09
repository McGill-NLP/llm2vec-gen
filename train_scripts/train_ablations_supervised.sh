#!/bin/bash

model="qwen3-4"
lr=3e-4

for model in "qwen3-1.7" "qwen3-4"; do 

    if [ $model == "qwen3-0.6" ]; then
        teacher_path="McGill-NLP/LLM2Vec-Qwen3-06B-mntp"
    elif [ $model == "qwen3-1.7" ]; then
        teacher_path="McGill-NLP/LLM2Vec-Qwen3-17B-mntp"
    elif [ $model == "qwen3-4" ]; then
        teacher_path="McGill-NLP/LLM2Vec-Qwen3-4B-mntp"
    elif [ $model == "qwen3-8" ]; then
        teacher_path="McGill-NLP/LLM2Vec-Qwen3-8B-mntp"
    fi

    # supervised teacher
    python scripts/train.py \
    training=llm2vec-gen \
    training.learning_rate=$lr \
    training.per_device_train_batch_size=32 \
    model=$model \
    model.pretrained_teacher_path=${teacher_path} \
    model.pretrained_teacher_path_2=${teacher_path}-supervised \
    data=$model \
    special_tokens=total_20 \
    run.wandb_run_id=$model-teacher-llm2vec-sup

    # supervised teacher + original echo responses
    python scripts/train.py \
    training=llm2vec-gen \
    training.learning_rate=$lr \
    training.per_device_train_batch_size=32 \
    model=$model \
    model.pretrained_teacher_path=${teacher_path} \
    model.pretrained_teacher_path_2=${teacher_path}-supervised \
    data=tulu \
    data.hf_dataset_name=McGill-NLP/llm2vec-gen-echo-rewritten-w-hard-negative \
    special_tokens=total_20 \
    run.wandb_run_id=$model-teacher-llm2vec-sup-data-echo-original
    
    # supervised teacher + LLM-generated echo responses
    python scripts/train.py \
    training=llm2vec-gen \
    training.learning_rate=$lr \
    training.per_device_train_batch_size=32 \
    model=$model \
    model.pretrained_teacher_path=${teacher_path} \
    model.pretrained_teacher_path_2=${teacher_path}-supervised \
    data=$model \
    data.hf_dataset_name=McGill-NLP/llm2vec-gen-echo-rewritten-w-hard-negative \
    special_tokens=total_20 \
    run.wandb_run_id=$model-teacher-llm2vec-sup-data-echo-${model}

    # 2GPU
    # supervised teacher + hard negative + lora
    th=10
    accelerate launch --main_process_port $((29500 + RANDOM % 1000)) scripts/train.py \
    training=llm2vec-gen \
    training.learning_rate=$lr \
    training.per_device_train_batch_size=16 \
    training.use_hard_negatives=true \
    training.other_sub_losses_weight=1.0 \
    training.margin_loss_weight=1.0 \
    training.margin_threshold=$th \
    training.encoder_lora_r=8 \
    training.encoder_lora_alpha=16 \
    training.use_peft_for_encoder=true \
    model=$model \
    model.pretrained_teacher_path=${teacher_path} \
    model.pretrained_teacher_path_2=${teacher_path}-supervised \
    data=$model \
    data.hf_dataset_name=McGill-NLP/llm2vec-gen-tulu-w-hard-negative \
    special_tokens=total_20 \
    run.wandb_run_id=$model-teacher-llm2vec-sup-data-tulu-${model}-hard-neg-th-${th}-lora


    # 2GPU
    # supervised teacher + hard negative
    th=10
    accelerate launch --main_process_port $((29500 + RANDOM % 1000)) scripts/train.py \
    training=llm2vec-gen \
    training.learning_rate=$lr \
    training.per_device_train_batch_size=16 \
    training.use_hard_negatives=true \
    training.other_sub_losses_weight=1.0 \
    training.margin_loss_weight=1.0 \
    training.margin_threshold=$th \
    model=$model \
    model.pretrained_teacher_path=${teacher_path} \
    model.pretrained_teacher_path_2=${teacher_path}-supervised \
    data=$model \
    data.hf_dataset_name=McGill-NLP/llm2vec-gen-tulu-w-hard-negative \
    special_tokens=total_20 \
    run.wandb_run_id=$model-teacher-llm2vec-sup-data-tulu-${model}-hard-neg-th-${th}

    # 2GPU
    # supervised teacher + LLM-generated echo responses + hard negative + lora
    th=10
    accelerate launch --main_process_port $((29500 + RANDOM % 1000)) scripts/train.py \
    training=llm2vec-gen \
    training.learning_rate=$lr \
    training.per_device_train_batch_size=16 \
    training.use_hard_negatives=true \
    training.other_sub_losses_weight=1.0 \
    training.margin_loss_weight=1.0 \
    training.margin_threshold=$th \
    training.encoder_lora_r=8 \
    training.encoder_lora_alpha=16 \
    training.use_peft_for_encoder=true \
    model=$model \
    model.pretrained_teacher_path=${teacher_path} \
    model.pretrained_teacher_path_2=${teacher_path}-supervised \
    data=$model \
    data.hf_dataset_name=McGill-NLP/llm2vec-gen-echo-rewritten-w-hard-negative \
    special_tokens=total_20 \
    run.wandb_run_id=$model-teacher-llm2vec-sup-data-echo-${model}-hard-neg-th-${th}-lora
done

for model in "qwen3-8"; do

    if [ $model == "qwen3-0.6" ]; then
        teacher_path="McGill-NLP/LLM2Vec-Qwen3-06B-mntp"
    elif [ $model == "qwen3-1.7" ]; then
        teacher_path="McGill-NLP/LLM2Vec-Qwen3-17B-mntp"
    elif [ $model == "qwen3-4" ]; then
        teacher_path="McGill-NLP/LLM2Vec-Qwen3-4B-mntp"
    elif [ $model == "qwen3-8" ]; then
        teacher_path="McGill-NLP/LLM2Vec-Qwen3-8B-mntp"
    fi

    # 2GPU
    # supervised teacher
    accelerate launch --main_process_port $((29500 + RANDOM % 1000)) scripts/train.py \
    training=llm2vec-gen \
    training.learning_rate=$lr \
    training.per_device_train_batch_size=16 \
    model=$model \
    model.pretrained_teacher_path=${teacher_path} \
    model.pretrained_teacher_path_2=${teacher_path}-supervised \
    data=$model \
    special_tokens=total_20 \
    run.wandb_run_id=$model-teacher-llm2vec-sup

    # 2GPU
    # supervised teacher + original echo responses
    accelerate launch --main_process_port $((29500 + RANDOM % 1000)) scripts/train.py \
    training=llm2vec-gen \
    training.learning_rate=$lr \
    training.per_device_train_batch_size=16 \
    model=$model \
    model.pretrained_teacher_path=${teacher_path} \
    model.pretrained_teacher_path_2=${teacher_path}-supervised \
    data=tulu \
    data.hf_dataset_name=McGill-NLP/llm2vec-gen-echo-rewritten-w-hard-negative \
    special_tokens=total_20 \
    run.wandb_run_id=$model-teacher-llm2vec-sup-data-echo-original

    # 2GPU
    # supervised teacher + LLM-generated echo responses
    accelerate launch --main_process_port $((29500 + RANDOM % 1000)) scripts/train.py \
    training=llm2vec-gen \
    training.learning_rate=$lr \
    training.per_device_train_batch_size=16 \
    model=$model \
    model.pretrained_teacher_path=${teacher_path} \
    model.pretrained_teacher_path_2=${teacher_path}-supervised \
    data=$model \
    data.hf_dataset_name=McGill-NLP/llm2vec-gen-echo-rewritten-w-hard-negative \
    special_tokens=total_20 \
    run.wandb_run_id=$model-teacher-llm2vec-sup-data-echo-${model}

    # 4GPU
    # supervised teacher + hard negative + lora
    th=10
    accelerate launch --main_process_port $((29500 + RANDOM % 1000)) scripts/train.py \
    training=llm2vec-gen \
    training.learning_rate=$lr \
    training.per_device_train_batch_size=8 \
    training.use_hard_negatives=true \
    training.other_sub_losses_weight=1.0 \
    training.margin_loss_weight=1.0 \
    training.margin_threshold=$th \
    training.encoder_lora_r=8 \
    training.encoder_lora_alpha=16 \
    training.use_peft_for_encoder=true \
    model=$model \
    model.pretrained_teacher_path=${teacher_path} \
    model.pretrained_teacher_path_2=${teacher_path}-supervised \
    data=$model \
    data.hf_dataset_name=McGill-NLP/llm2vec-gen-tulu-w-hard-negative \
    special_tokens=total_20 \
    run.wandb_run_id=$model-teacher-llm2vec-sup-data-tulu-${model}-hard-neg-th-${th}-lora

    # 4GPU
    # supervised teacher + hard negative
    th=10
    accelerate launch --main_process_port $((29500 + RANDOM % 1000)) scripts/train.py \
    training=llm2vec-gen \
    training.learning_rate=$lr \
    training.per_device_train_batch_size=8 \
    training.use_hard_negatives=true \
    training.other_sub_losses_weight=1.0 \
    training.margin_loss_weight=1.0 \
    training.margin_threshold=$th \
    model=$model \
    model.pretrained_teacher_path=${teacher_path} \
    model.pretrained_teacher_path_2=${teacher_path}-supervised \
    data=$model \
    data.hf_dataset_name=McGill-NLP/llm2vec-gen-tulu-w-hard-negative \
    special_tokens=total_20 \
    run.wandb_run_id=$model-teacher-llm2vec-sup-data-tulu-${model}-hard-neg-th-${th}

    # 4GPU
    # supervised teacher + LLM-generated echo responses + hard negative + lora
    th=10
    accelerate launch --main_process_port $((29500 + RANDOM % 1000)) scripts/train.py \
    training=llm2vec-gen \
    training.learning_rate=$lr \
    training.per_device_train_batch_size=8 \
    training.use_hard_negatives=true \
    training.other_sub_losses_weight=1.0 \
    training.margin_loss_weight=1.0 \
    training.margin_threshold=$th \
    training.encoder_lora_r=8 \
    training.encoder_lora_alpha=16 \
    training.use_peft_for_encoder=true \
    model=$model \
    model.pretrained_teacher_path=${teacher_path} \
    model.pretrained_teacher_path_2=${teacher_path}-supervised \
    data=$model \
    data.hf_dataset_name=McGill-NLP/llm2vec-gen-echo-rewritten-w-hard-negative \
    special_tokens=total_20 \
    run.wandb_run_id=$model-teacher-llm2vec-sup-data-echo-${model}-hard-neg-th-${th}-lora

done
