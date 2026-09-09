#!/bin/bash
# Pre-start for vllm-glm53.service: wait until every worker node in .env answers over
# SSH and has a working docker daemon, then make sure no stale cluster container is left.
# Workers mount this head's /home/jeff/models over NFS, so after a reboot they may lag.
set -u
cd /home/jeff/spark-vllm-docker || exit 1
NODES=$(grep -E '^CLUSTER_NODES=' .env | sed -E 's/^CLUSTER_NODES="?([^"]*)"?$/\1/' | tr ',' ' ')
HEAD=$(grep -E '^LOCAL_IP=' .env | cut -d= -f2)
deadline=$(( $(date +%s) + ${WAIT_FOR_WORKERS_SECS:-600} ))
for h in $NODES; do
  [ "$h" = "$HEAD" ] && continue
  until ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" 'docker info >/dev/null 2>&1 && test -f /home/jeff/models/GLM-5.3-Tech2wild/config.json'; do
    [ $(date +%s) -ge $deadline ] && { echo "worker $h not ready (ssh/docker/NFS) after timeout"; exit 1; }
    echo "waiting for worker $h ..."; sleep 10
  done
  ssh -o BatchMode=yes "$h" 'docker rm -f vllm_node >/dev/null 2>&1' || true
done
docker rm -f vllm_node >/dev/null 2>&1 || true
echo "workers ready: $NODES"
