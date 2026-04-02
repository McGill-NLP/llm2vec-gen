FINAL_LAUNCH_CONFIG="bash_scripts/configs/train_final.yaml"
FINAL_2GPU_LAUNCH_CONFIG="bash_scripts/configs/train_2gpu_final.yaml"

counter=0

for model in "qwen3-1.7" "qwen3-4"; do
    if [[ $model == *"qwen3"* ]]; then
        lr=3e-4
    else
        lr=5e-4
    fi
    counter=$((counter + 1))

    python scripts/train.py \
    training=llm2vec-gen \
    training.learning_rate=$lr \
    training.per_device_train_batch_size=32 \
    model=$model \
    data=$model \
    special_tokens=total_10 \
    model.pretrained_teacher_path=BAAI/bge-m3-unsupervised \
    model.pretrained_teacher_path_2=none \
    run.wandb_run_id=$model-teacher-bge
done

for model in "qwen3-8" ; do  
    if [[ $model == *"qwen3"* ]]; then
        lr=3e-4
    else
        lr=5e-4
    fi
    counter=$((counter + 1))
    accelerate launch --main_process_port $((29500 + RANDOM % 1000)) scripts/train.py \
    training=llm2vec-gen \
    training.learning_rate=$lr \
    training.per_device_train_batch_size=16 \
    model=$model \
    data=$model \
    special_tokens=total_10 \
    model.pretrained_teacher_path=BAAI/bge-m3-unsupervised \
    model.pretrained_teacher_path_2=none \
    run.wandb_run_id=$model-teacher-bge
done
echo "Total number of runs: $counter"
