#!/bin/bash
# Restart the DeepSeek SGLang cluster only once the engine has been idle for 90 s (max wait 45 min).
cd /home/jeff/dsv41-knapcio
for i in $(seq 1 270); do
  # health probes (1-token prefill, empty decode every ~15 s) do not count as activity
  last=$(docker logs --since 90s dsv41-head 2>&1 | grep -E "Prefill batch|Decode batch" \
    | grep -vcE "#new-token: 1, #cached-token: 0,.*#running-req: 0,|Decode batch, #running-req: 1, #full token: 0,")
  if [ "$last" -eq 0 ]; then
    echo "idle at $(date +%T) after $((i*10)) s of waiting; restarting"
    ./start-tp4.sh stop 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E "API down|no dsv41"
    echo "down at $(date +%T)"
    ./start-tp4.sh serve > /home/jeff/dsv41-knapcio/serve5.log 2>&1
    rc=$?; echo "serve exit $rc at $(date +%T)"; exit $rc
  fi
  sleep 10
done
echo "NOT IDLE within 45 min; did not restart"; exit 3
