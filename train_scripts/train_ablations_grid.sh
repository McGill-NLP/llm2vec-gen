#!/bin/bash

model="qwen3-4"
lr=3e-4

for teacher in "qwen3-0.6b" "qwen3-1.7b" "qwen3-4b" "qwen3-8b"; do
    for data in "qwen3-0.6" "qwen3-1.7" "qwen3-4" "qwen3-8"; do

        if [ $teacher == "qwen3-0.6b" ]; then
            teacher_path="McGill-NLP/LLM2Vec-Qwen3-06B-mntp"
        elif [ $teacher == "qwen3-1.7b" ]; then
            teacher_path="McGill-NLP/LLM2Vec-Qwen3-17B-mntp"
        elif [ $teacher == "qwen3-4b" ]; then
            teacher_path="McGill-NLP/LLM2Vec-Qwen3-4B-mntp"
        elif [ $teacher == "qwen3-8b" ]; then
            teacher_path="McGill-NLP/LLM2Vec-Qwen3-8B-mntp"
        fi

        if [ $data == "qwen3-4" ] && [ $teacher == "qwen3-4b" ]; then
            echo "Skipping: model $model teacher $teacher data $data"
            continue
        # elif [ $data == "qwen3-4" ] && [ $teacher == "qwen3-8b" ]; then
        #     echo "Skipping: model $model teacher $teacher data $data"
        #     continue
        # fi

        counter=$((counter + 1))
        eai job new -f $FINAL_LAUNCH_CONFIG \
        --field id -- /bin/bash -c \
        "source /opt/conda/bin/activate /home/toolkit/generative-embeddings-clean/.conda && \
        python scripts/train.py \
        training=llm2vec-gen \
        training.learning_rate=$lr \
        training.per_device_train_batch_size=32 \
        model=$model \
        model.pretrained_teacher_path=${teacher_path} \
        model.pretrained_teacher_path_2=${teacher_path}-unsup-simcse \
        data=$data \
        special_tokens=total_20 \
        run.wandb_run_id=$model-teacher-llm2vec-${teacher}-data-${data} \
        >> /home/toolkit/generative-embeddings-clean/logs/llm2vec-gen-${model}-teacher-llm2vec-${teacher}-data-${data}.log 2>&1"
    done
done

echo "Total number of runs: $counter"
