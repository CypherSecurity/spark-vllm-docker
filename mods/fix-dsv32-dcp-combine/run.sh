#!/usr/bin/env bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[fix-dsv32-dcp-combine] applying"
python3 "$D/fix-dsv32-dcp-combine.py"
python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v32/attention.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/b12x_mla_sparse.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v32/common/kernels.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v32/nvidia/model.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/cuda_communicator.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/b12x_indexer.py
find /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v32 /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v32/common /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "[fix-dsv32-dcp-combine] py_compile OK"
