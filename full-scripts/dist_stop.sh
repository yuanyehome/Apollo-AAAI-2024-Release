#!/bin/bash

HOSTFILE=/share/yuanye/apollo-aaai-24/hostfile
# shellcheck disable=SC2126
NUM_NODES=$(grep -v '^#\|^$' "$HOSTFILE" | wc -l)
echo "NUM_NODES: $NUM_NODES"

hostlist=$(grep -v '^#\|^$' "$HOSTFILE" | awk '{print $1}' | xargs)
# shellcheck disable=SC2068
for host in ${hostlist[@]}; do
    ssh "$host" "pkill -f '/usr/local/bin/torchrun'" 
    echo "$host is killed."
done
