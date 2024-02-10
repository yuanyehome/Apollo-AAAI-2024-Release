#!/bin/bash

cd /home1/yy/Apollo-AAAI-2024/ || exit

export http_proxy="localhost:7890"
export https_proxy="localhost:7890"

export WANDB_API_KEY="<WANDB_API_KEY>"
export WANDB_PROJECT="Apollo-Reproduce"
export WANDB_NAME="GPT2-Scratch-10epochs-4gpus-3e-4lr"
export WANDB_MODE="offline"
export OMP_NUM_THREADS=8

export CUDA_VISIBLE_DEVICES="2,3,4,5"
# count the number of GPUs
NODE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
MASTER_PORT=47489

DISTRIBUTED_ARGS="
    --nproc_per_node $NODE_DEVICES \
    --master_addr localhost \
    --master_port $MASTER_PORT 
"

TRAINING_ARGS="
    --tokenizer_name gpt2 \
    --config_name gpt2 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 10 \
    --do_train \
    --bf16 \
    --streaming \
    --ddp_timeout 36000 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --learning_rate 0.0003 \
    --weight_decay 0.01 \
    --seed 42 \
    --warmup_ratio 0.025 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --output_dir experiments/gpt2-scratch
"

cmd="torchrun $DISTRIBUTED_ARGS run_pretrain_gpt2.py \
              $TRAINING_ARGS
"

# shellcheck disable=SC2086
echo $cmd
# shellcheck disable=SC2086
eval $cmd
