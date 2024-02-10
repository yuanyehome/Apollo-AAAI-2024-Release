#!/bin/bash

HOSTFILE=/share/yuanye/apollo-aaai-24/hostfile

mkdir -p /share/yuanye/apollo-aaai-24/logs/gpt2-scratch/
LOG_FILE=/share/yuanye/apollo-aaai-24/logs/gpt2-scratch/worker_log.log
SCRIPT_FILE=/share/yuanye/apollo-aaai-24/run_pretrain_gpt2.sh

hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
COUNT=0
# shellcheck disable=SC2068
for host in ${hostlist[@]}; do
  echo "$host"
  echo "ssh -f -n $host sh -c 'nohup bash $SCRIPT_FILE >> $LOG_FILE.$COUNT 2>&1 &'"
  ssh -f -n "$host" "sh -c 'nohup bash $SCRIPT_FILE >> $LOG_FILE.$COUNT 2>&1 &'"
  ((COUNT++))
done
