#!/bin/bash

cd /share/yuanye/apollo-aaai-24/ || exit

export http_proxy="localhost:7890"
export https_proxy="localhost:7890"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0
export NCCL_DEBUG=DEBUG
export NCCL_IB_TIMEOUT=30
export NCCL_IB_RETRY_CNT=5
export OMP_NUM_THREADS=4
export GLOO_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_0,mlx5_1

export WANDB_API_KEY="<WANDB_API_KEY>"
export WANDB_PROJECT="Apollo-Reproduce"
export WANDB_NAME="BERT-Apollo"

export PYTHONPATH=/share/yuanye/apollo-aaai-24/:$PYTHONPATH

HOSTFILE=/share/yuanye/apollo-aaai-24/hostfile
echo "HOSTFILE: $HOSTFILE"
NUM_NODES=$(awk 'BEGIN {cnt=0} !/^#/ && NF {$1=$1; cnt++} END {print cnt}' "$HOSTFILE")
echo "NUM_NODES: $NUM_NODES"
# shellcheck disable=SC2020
# NODE_ADDR=$(ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2;}'|tr -d "addr:"|tail -n 3|head -n 1)
# shellcheck disable=SC2020
NODE_ADDR=$(ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2;}'|tr -d "addr:"|head -n 1)
NODE_NAME=$(hostname)
NODE_RANK_BY_ADDR=$(awk -v node="$NODE_ADDR" 'BEGIN {cnt=0} !/^#/ && NF {ranks[$1]=cnt; cnt++;} END {print ranks[node];}' "$HOSTFILE")
echo "NODE_RANK_BY_ADDR: $NODE_RANK_BY_ADDR"
NODE_RANK_BY_NAME=$(awk -v node="$NODE_NAME" 'BEGIN {cnt=0} !/^#/ && NF {ranks[$1]=cnt; cnt++;} END { print ranks[node];}' "$HOSTFILE")
echo "NODE_RANK_BY_NAME: $NODE_RANK_BY_NAME"
if [ -n "$NODE_RANK_BY_ADDR" ]; then
    NODE_RANK=$NODE_RANK_BY_ADDR
    echo "NODE_RANK: $NODE_RANK"
    NODE_DEVICES=$(awk -v node="$NODE_ADDR" '!/^#/ && NF && $1==node {split($2, arr, "="); print arr[2]}' "$HOSTFILE")
    echo "NODE_DEVICES: $NODE_DEVICES"
    NODE_TYPE=$(awk -v node="$NODE_ADDR" '!/^#/ && NF && $1==node {print $3}' "$HOSTFILE")
    echo "NODE_TYPE: $NODE_TYPE"
elif [ -n "$NODE_RANK_BY_NAME" ]; then
    NODE_RANK=$NODE_RANK_BY_NAME
    echo "NODE_RANK: $NODE_RANK"
    NODE_DEVICES=$(awk -v node="$NODE_NAME" '!/^#/ && NF && $1==node {split($2, arr, "="); print arr[2]}' "$HOSTFILE")
    echo "NODE_DEVICES: $NODE_DEVICES"
    NODE_TYPE=$(awk -v node="$NODE_NAME" '!/^#/ && NF && $1==node {print $3}' "$HOSTFILE")
    echo "NODE_TYPE: $NODE_TYPE"
else
    echo "Error: NODE_RANK not found"
    exit 1
fi
MASTER_ADDR=$(grep -v '^#\|^$' $HOSTFILE | head -n1 | awk '{print $1;}')
echo "MASTER_ADDR: $MASTER_ADDR"
MASTER_PORT=47429
echo "MASTER_PORT: $MASTER_PORT"
echo "NODE_ADDR: $NODE_ADDR"


DISTRIBUTED_ARGS="
    --nproc_per_node $NODE_DEVICES \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT 
"

TRAINING_ARGS="
    --tokenizer_name bert-base-uncased \
    --config_name bert-base-uncased \
    --per_device_train_batch_size 48 \
    --num_train_epochs 40 \
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
    --output_dir experiments/bert-apollo \
    --apollo_info_output_file logs/bert-apollo/apollo-info/apollo_info \
    --apollo_epoch_list 2,4,10,40 \
    --apollo_layers 1,3,6,12 \
    --dist_function lvps \
    --grow_method extend
"

mkdir -p logs/bert-apollo/apollo-info/
cmd="torchrun $DISTRIBUTED_ARGS run_pretrain_bert_apollo.py \
              $TRAINING_ARGS
"

# shellcheck disable=SC2086
echo $cmd
# shellcheck disable=SC2086
eval $cmd
