#!/bin/bash
# sweep_driver.sh <recipe>...  : boot each full-model recipe, benchmark, append to sweep_results.md
export VLLM_SPARK_EXTRA_DOCKER_ARGS="-v /home/jeff/models:/models" HF_HOME=/home/jeff/models
S="${S:-$(cd "$(dirname "$0")" && pwd)}"
OUT=$S/sweep_results.md; NODES="192.168.177.11 192.168.177.13 192.168.177.14"
stopall(){ docker rm -f vllm_node >/dev/null 2>&1; for h in $NODES; do ssh -o BatchMode=yes $h 'docker rm -f vllm_node >/dev/null 2>&1'; done; }
for RECIPE in "$@"; do
  echo -e "\n## $RECIPE  ($(date +%H:%M))" >> $OUT
  stopall; sleep 3
  ( cd /home/jeff/spark-vllm-docker && script -qfc "./run-recipe.py recipes/4x-spark-cluster/$RECIPE --no-ray" $S/launch_$RECIPE.log >/dev/null 2>&1 ) & LP=$!; t0=$(date +%s); code=000
  while :; do code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 localhost:8000/health); [ "$code" = "200" ] && break
    if grep -qE "WorkerProc hit an exception|WorkerProc failed to start|Engine core initialization failed" $S/launch_$RECIPE.log 2>/dev/null; then echo "BOOT FAILED: $(tr -d '\r' < $S/launch_$RECIPE.log | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'Error' | grep -vE 'NCCL|frame #' | tail -2 | cut -c1-240)" >> $OUT; break; fi
    kill -0 $LP 2>/dev/null || { echo "launcher exited before health" >> $OUT; break; }; [ $(( $(date +%s)-t0 )) -gt 1800 ] && { echo "TIMEOUT" >> $OUT; break; }; sleep 20; done
  if [ "$code" = "200" ]; then
    echo "healthy after $(( $(date +%s)-t0 ))s; $(curl -s localhost:8000/metrics | grep -oE 'kv_cache_size_tokens="[0-9]+"')" >> $OUT
    python3 - >> $OUT <<'PY'
import json,urllib.request
body={"model":"glm5.3","messages":[{"role":"user","content":"Count from 1 to 12, one number per line, nothing else."}],"max_tokens":60,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
r=json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8000/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=600))
t=r["choices"][0]["message"]["content"]; print("coherence:", "OK" if t.strip().startswith("1\n2\n3") else "SUSPECT "+repr(t[:60]))
PY
    python3 $S/bench1.py code | tail -1 | sed 's/^/warmup: /' >> $OUT
    python3 $S/bench1.py code prose count code >> $OUT
    python3 $S/bench_long.py 14000 >> $OUT
    grep -E "MemAvailable" /proc/meminfo >> $OUT
  fi
  stopall; kill $LP 2>/dev/null; wait $LP 2>/dev/null
done
echo -e "\n## sweep done $(date +%H:%M)" >> $OUT
