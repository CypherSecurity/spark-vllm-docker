#!/bin/bash
# Launch DeepSeek-V4.1-Flash TP4 with node-local Engram rows (no Ray). Run on the head.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/home/jeff/models
export VLLM_SPARK_EXTRA_DOCKER_ARGS="-v /var/tmp/engram-local/DeepSeek-V4.1-Flash:/engram-local:ro"
for h in $(grep -E '^CLUSTER_NODES=' .env | sed -E 's/^CLUSTER_NODES="?([^"]*)"?$/\1/' | tr ',' ' '); do
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" 'test -f /var/tmp/engram-local/DeepSeek-V4.1-Flash/engram-local.json' \
    || { echo "missing node-local Engram rows on $h (run ~/dsv41-engram-copy.sh first)"; exit 1; }
done
exec ./run-recipe.py recipes/4x-spark-cluster/deepseek-v41-flash-tp4-localengram.yaml --no-ray "$@"
