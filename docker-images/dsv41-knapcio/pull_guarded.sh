#!/bin/bash
# Pull the base image on each node in turn; abort a node's pull if MemAvailable < 300 MB.
IMG=lmsysorg/sglang:dev-dsv41
for h in 192.168.177.12 192.168.177.13 192.168.177.11 192.168.177.14; do
  echo "=== $h start $(date +%T)"
  ssh -o BatchMode=yes $h "docker image inspect $IMG >/dev/null 2>&1 && echo 'already present' && exit 0
    docker pull --platform linux/arm64 -q $IMG > /tmp/sglang-pull.log 2>&1 & P=\$!
    minm=999999999
    while kill -0 \$P 2>/dev/null; do
      m=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo); [ \$m -lt \$minm ] && minm=\$m
      if [ \$m -lt 307200 ]; then echo \"GUARD: MemAvailable \$((m/1024)) MB, killing pull\"; kill \$P; pkill -f 'docker pull.*dev-dsv41'; exit 3; fi
      sleep 2
    done
    wait \$P; rc=\$?; echo \"pull rc=\$rc min MemAvailable \$((minm/1024)) MB\"; tail -1 /tmp/sglang-pull.log; exit \$rc"
  rc=$?; echo "=== $h done rc=$rc $(date +%T)"
  [ $rc -ne 0 ] && { echo "STOPPING: $h failed"; exit $rc; }
done
echo "ALL PULLS DONE"
