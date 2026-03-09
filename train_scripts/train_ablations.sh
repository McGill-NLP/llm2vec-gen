#!/bin/bash

model="qwen3-4"
lr=3e-4

# recon loss only
python scripts/train.py \
training=llm2vec-gen \
training.learning_rate=$lr \
training.per_device_train_batch_size=32 \
training.recon_loss_weight=1.0 \
training.align_loss_weight=0.0 \
model=$model \
data=$model \
special_tokens=total_20 \
run.wandb_run_id=$model-only-recon-loss

# align loss only
python scripts/train.py \
training=llm2vec-gen \
training.learning_rate=$lr \
training.per_device_train_batch_size=32 \
training.recon_loss_weight=0.0 \
training.align_loss_weight=1.0 \
model=$model \
data=$model \
special_tokens=total_20 \
run.wandb_run_id=$model-only-align-loss

################################################################################

# no thinking: 0 thought tokens and 20 compression tokens
python scripts/train.py \
training=llm2vec-gen \
training.learning_rate=$lr \
training.per_device_train_batch_size=32 \
model=$model \
model.encoding_mode=last_20_tokens \
data=$model \
special_tokens=total_20 \
run.wandb_run_id=$model-no-thinking

# all thinking: 19 thought tokens and 1 compression tokens
python scripts/train.py \
training=llm2vec-gen \
training.learning_rate=$lr \
training.per_device_train_batch_size=32 \
model=$model \
model.encoding_mode=last_1_tokens \
data=$model \
special_tokens=total_20 \
run.wandb_run_id=$model-all-thinking

################################################################################

# original tulu responses
python scripts/train.py \
training=llm2vec-gen \
training.learning_rate=$lr \
training.per_device_train_batch_size=32 \
model=$model \
data=tulu \
special_tokens=total_20 \
run.wandb_run_id=$model-original-tulu-responses

# qwen3-8b tulu responses
python scripts/train.py \
training=llm2vec-gen \
training.learning_rate=$lr \
training.per_device_train_batch_size=32 \
model=$model \
data=qwen3-8 \
special_tokens=total_20 \
run.wandb_run_id=$model-qwen3-8-tulu-responses

# gemini-3-flash tulu responses
python scripts/train.py \
training=llm2vec-gen \
training.learning_rate=$lr \
training.per_device_train_batch_size=32 \
model=$model \
data=gemini \
special_tokens=total_20 \
run.wandb_run_id=$model-gemini-tulu-responses

################################################################################

# LLM2Vec-Qwen-3-8B embedding teacher
python scripts/train.py \
training=llm2vec-gen \
training.learning_rate=$lr \
training.per_device_train_batch_size=32 \
model=$model \
model.pretrained_teacher_path=McGill-NLP/LLM2Vec-Qwen3-8B-mntp \
model.pretrained_teacher_path_2=McGill-NLP/LLM2Vec-Qwen3-8B-mntp-unsup-simcse \
data=$model \
special_tokens=total_20 \
run.wandb_run_id=$model-teacher-llm2vec-qwen3-8b

# LLM2Vec-Llama-3.1-8B embedding teacher
python scripts/train.py \
training=llm2vec-gen \
training.learning_rate=$lr \
training.per_device_train_batch_size=32 \
model=$model \
model.pretrained_teacher_path=McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp \
model.pretrained_teacher_path_2=McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-unsup-simcse \
data=$model \
special_tokens=total_20 \
run.wandb_run_id=$model-teacher-llm2vec-llama31-8b

################################################################################

# lora r=8 alpha=16
python scripts/train.py \
training=llm2vec-gen \
training.learning_rate=$lr \
training.per_device_train_batch_size=32 \
training.encoder_lora_r=8 \
training.encoder_lora_alpha=16 \
training.use_peft_for_encoder=true \
model=$model \
data=$model \
special_tokens=total_20 \
run.wandb_run_id=$model-lora-r8-alpha16

# lora r=32 alpha=64
python scripts/train.py \
training=llm2vec-gen \
training.learning_rate=$lr \
training.per_device_train_batch_size=32 \
training.encoder_lora_r=32 \
training.encoder_lora_alpha=64 \
training.use_peft_for_encoder=true \
model=$model \
data=$model \
special_tokens=total_20 \
run.wandb_run_id=$model-lora-r32-alpha64
