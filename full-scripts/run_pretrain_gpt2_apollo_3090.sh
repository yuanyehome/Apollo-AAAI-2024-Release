#!/bin/bash

export http_proxy="localhost:7890"
export https_proxy="localhost:7890"

export WANDB_API_KEY="<WANDB_API_KEY>"
export WANDB_PROJECT="Apollo-Reproduce"
export WANDB_NAME="GPT2-Apollo-Debug"
export WANDB_MODE="offline"
export OMP_NUM_THREADS=8

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

export CUDA_VISIBLE_DEVICES="1,2,4,6"
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
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 32 \
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
    --output_dir experiments/gpt2-apollo-4gpus \
    --apollo_info_output_file logs/gpt2-apollo-4gpus/apollo-info/apollo_info \
    --apollo_epoch_list 1,2,5,10 \
    --apollo_layers 1,3,6,12 \
    --dist_function lvps \
    --grow_method extend
"

mkdir -p logs/gpt2-apollo-4gpus/apollo-info/
cmd="torchrun $DISTRIBUTED_ARGS run_pretrain_gpt2_apollo.py \
              $TRAINING_ARGS
"

# shellcheck disable=SC2086
echo $cmd
# shellcheck disable=SC2086
eval $cmd
