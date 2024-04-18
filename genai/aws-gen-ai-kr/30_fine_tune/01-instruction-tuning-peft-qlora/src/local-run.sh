#!/bin/bash

set -e
pip install -r requirements.txt

mkdir -p /tmp/huggingface-cache/
export HF_DATASETS_CACHE="/tmp/huggingface-cache"

declare -a OPTS=(
    --base_model nlpai-lab/kullm-polyglot-12.8b-v2
    --pretrained_model_path /home/ec2-user/SageMaker/models/kullm-polyglot-12-8b-v2/
    --cache_dir $HF_DATASETS_CACHE
    --data_path ../chunk-train
    --output_dir output
    --chkpt_dir chkpt
    --save_path ./model
    --batch_size 2
    --gradient_accumulation_steps 2
    --num_epochs 1
    --learning_rate 3e-4
    --lora_r 8
    --lora_alpha 16
    --lora_dropout 0.05
    --lora_target_modules "[query_key_value, xxx]"
    --logging_steps 1
    --save_steps 40
    --eval_steps 40
    --weight_decay 0.
    --warmup_steps 50
    --warmup_ratio 0.03
    --lr_scheduler_type "linear"
)

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ $NUM_GPUS -eq 1 ]
then
    echo python train.py "${OPTS[@]}" "$@"
    CUDA_VISIBLE_DEVICES=0 python train.py "${OPTS[@]}" "$@"
else
    echo torchrun --nnodes 1 --nproc_per_node "$NUM_GPUS" train.py "${OPTS[@]}" "$@"
    torchrun --nnodes 1 --nproc_per_node "$NUM_GPUS" train.py "${OPTS[@]}" "$@"
fi