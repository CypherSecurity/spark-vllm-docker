#!/bin/bash
# Re-check that the served DeepSeek prompt format matches DeepSeek's reference encoder.
# 1) SGLang's V4.1 encoder vs DeepSeek's five golden prompts (offline, no server needed)
# 2) live server prompt token counts vs the reference, for chat / thinking / tools / thinking history,
#    on /v1/chat/completions and /v1/messages (needs the server up on :8000)
set -e
D=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-/home/jeff/models/dsv41-1017-flat}
IMG=${IMG:-dsv41-4x-spark:canary-roce}
run() { docker run --rm --network host -v "$MODEL":/model:ro -v "$MODEL/encoding":/ref:ro -v "$D":/chk:ro --entrypoint python3 "$IMG" "/chk/$1" 2>&1 | grep -vE "Warning|warn|^\[transformers"; }
echo "== encoder vs golden prompts"; run enc_check.py
echo "== live server vs reference";  run enc_probe.py; run enc_probe2.py
