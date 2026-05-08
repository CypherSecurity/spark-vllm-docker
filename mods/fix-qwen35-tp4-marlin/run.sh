#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[fix-qwen35-tp4-marlin] Applying disable_tp=True patch to qwen3_5.py..."
patch -p0 -d /usr/local/lib/python3.12/dist-packages/vllm < "${SCRIPT_DIR}/qwen3_5.patch" \
  || echo "[fix-qwen35-tp4-marlin] Patch not applicable, skipping..."
echo "[fix-qwen35-tp4-marlin] Done."
