#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0
export NCCL_DEBUG=DEBUG
export NCCL_IB_TIMEOUT=30
export NCCL_IB_RETRY_CNT=5
export OMP_NUM_THREADS=4
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_0,mlx5_1

export WANDB_MODE="disabled"
export OMP_NUM_THREADS=8

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# count the number of GPUs
NODE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
MASTER_PORT=47489


DISTRIBUTED_ARGS="
    --nproc_per_node $NODE_DEVICES \
    --master_addr localhost \
    --master_port $MASTER_PORT 
"

TRAINING_ARGS="
    --tokenizer_name bert-base-uncased \
    --config_name bert-base-uncased \
    --per_device_train_batch_size 48 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10 \
    --do_train \
    --bf16 \
    --streaming \
    --ddp_timeout 36000 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --learning_rate 0.0001 \
    --weight_decay 0.01 \
    --seed 42 \
    --warmup_ratio 0.025 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --output_dir experiments/bert-scratch-example \
    --my_debug
"

cmd="torchrun $DISTRIBUTED_ARGS run_pretrain_bert.py \
              $TRAINING_ARGS
"

# shellcheck disable=SC2086
echo $cmd
# shellcheck disable=SC2086
eval $cmd
