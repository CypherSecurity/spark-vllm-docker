#!/bin/bash
# Wait for 90 s without real requests (health probes ignored), then swap the served weights to 1017.
cd /home/jeff/dsv41-knapcio
for i in $(seq 1 540); do
  last=$(docker logs --since 90s dsv41-head 2>&1 | grep -E "Prefill batch|Decode batch" \
    | grep -vcE "#new-token: 1, #cached-token: 0,.*#running-req: 0,|Decode batch, #running-req: 1, #full token: 0,")
  if [ "$last" -eq 0 ]; then
    echo "idle at $(date +%T)"
    ./start-tp4.sh stop 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E "API down|no dsv41"
    echo "down at $(date +%T)"
    cp .env.tp4-1017 .env.tp4 && echo "config switched to 1017 (rollback: cp .env.tp4-0910 .env.tp4)"
    ./start-tp4.sh serve > /home/jeff/dsv41-knapcio/serve-1017.log 2>&1
    rc=$?; echo "serve exit $rc at $(date +%T)"; exit $rc
  fi
  sleep 10
done
echo "NOT IDLE within 90 min; did not swap"; exit 3
