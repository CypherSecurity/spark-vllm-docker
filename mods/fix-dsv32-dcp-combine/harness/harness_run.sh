#!/bin/bash
# harness_run.sh <recipe-name> <label> [--wait-idle]
# Stops whatever cluster is running, boots the recipe under a pty (launcher needs one),
# waits for health, dumps parity data, stops the cluster. Logs to $S/harness_<label>.log.
set -u
export VLLM_SPARK_EXTRA_DOCKER_ARGS="-v /home/jeff/models:/models"
export HF_HOME=/home/jeff/models
S="${S:-$(cd "$(dirname "$0")" && pwd)}"
REPO=/home/jeff/spark-vllm-docker
RECIPE=$1; LABEL=$2; WAIT_IDLE=${3:-}
NODES="192.168.177.11 192.168.177.13 192.168.177.14"
LOG=$S/harness_$LABEL.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ "$WAIT_IDLE" = "--wait-idle" ]; then
  idle=0
  while [ $idle -lt 2 ]; do
    r=$(curl -s --max-time 3 localhost:8000/metrics | grep -E '^vllm:num_requests_running\{' | awk '{print $2}')
    if [ -z "$r" ] || [ "${r%.*}" = "0" ]; then idle=$((idle+1)); else idle=0; fi
    [ $idle -lt 2 ] && sleep 20
  done
  log "production idle (2 consecutive checks); stopping it"
fi

# stop + remove any cluster containers on all nodes
docker stop vllm_node >/dev/null 2>&1; docker rm -f vllm_node >/dev/null 2>&1
for h in $NODES; do ssh -o BatchMode=yes $h 'docker stop vllm_node >/dev/null 2>&1; docker rm -f vllm_node >/dev/null 2>&1'; done
sleep 5
log "launching $RECIPE"
( cd $REPO && script -qfc "./run-recipe.py recipes/4x-spark-cluster/$RECIPE --no-ray" $S/launch_$LABEL.log >/dev/null 2>&1 ) &
LPID=$!
t0=$(date +%s)
while :; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 localhost:8000/health)
  [ "$code" = "200" ] && { log "healthy after $(( $(date +%s)-t0 ))s"; break; }
  if grep -qE "WorkerProc hit an exception|WorkerProc failed to start|Engine core initialization failed|RuntimeError:|ValueError:" $S/launch_$LABEL.log 2>/dev/null; then
    log "BOOT FAILED:"; grep -E "Error|error:" $S/launch_$LABEL.log | grep -vE "NCCL|frame #" | tail -3 | cut -c1-300 | tee -a "$LOG"; break; fi
  if ! kill -0 $LPID 2>/dev/null; then log "launcher exited before health"; break; fi
  [ $(( $(date +%s)-t0 )) -gt 1200 ] && { log "TIMEOUT waiting for health"; break; }
  sleep 15
done
if [ "$code" = "200" ]; then
  grep -oE "GPU KV cache size: [0-9,]+ tokens|RoCEnante all-reduce is live[^\n]{0,60}|Using B12X[^\n]{0,60}|fix-dsv32-dcp-combine\] applied[^\n]{0,40}" $S/launch_$LABEL.log | sort -u | head -6 | tee -a "$LOG"
  ( cd $S && python3 parity.py dump $LABEL ) 2>&1 | tee -a "$LOG"
fi
log "stopping cluster"
docker stop vllm_node >/dev/null 2>&1; docker rm -f vllm_node >/dev/null 2>&1
for h in $NODES; do ssh -o BatchMode=yes $h 'docker stop vllm_node >/dev/null 2>&1; docker rm -f vllm_node >/dev/null 2>&1'; done
kill $LPID 2>/dev/null; wait $LPID 2>/dev/null
log "done"
