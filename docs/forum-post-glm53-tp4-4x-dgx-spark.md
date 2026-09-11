# GLM-5.3 (743B, Int4-Int8Mix) on 4x DGX Spark at TP=4 + DCP=4: 33 tok/s code, 327K context, full recipe

There is very little on the forum about the *full* GLM-5.3 (743B) at TP=4, as opposed to
GLM-5.3-Flash, so here is a working, measured recipe plus the one code fix that was needed to
get there. Everything below is reproducible with the community `spark-vllm-docker` launcher.

## TL;DR

| | before (glm-eldritch image, July fork) | now (b12x nightly + fix) |
|---|---|---|
| decode step | 137–150 ms | 115–121 ms |
| single stream, code | 27 tok/s | 33–34 tok/s |
| single stream, prose | 17 tok/s | 21.6 tok/s |
| single stream, list/counting | 36 tok/s | 40.6 tok/s |
| decode at 32K context | 25.7 tok/s | 35.8 tok/s |
| 30K cold prefill | ~500 tok/s | 818 tok/s |
| KV pool (fp8, DCP=4) | 657,664 tokens | 657,664 tokens |
| context | 327,680 (2 concurrent) | 327,680 (2 concurrent) |

Single stream, temperature 0, thinking off, quiet cluster. The spread between prose and
lists is MTP acceptance (2.5 vs 4.8 of 4 draft tokens per ~120 ms step), not the server.

## Hardware

- 4x DGX Spark (GB10, 121 GB unified each), Ubuntu 24.04, driver 580.142.
- ConnectX-7 ports on a RoCE switch (a direct-connect mesh also works; the fork's RDMA
  allreduce needs only ≤2 HCAs per node). GID index for RoCE v2 differs per node on our
  cluster (3 on two nodes, 4 on the other two); the recipe computes it from sysfs at launch.
- Weights on the head node's NVMe, exported read-only over NFS to the workers (one copy).

## Software

- Launcher: [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker), `run-recipe.py ... --no-ray`.
- Image: `eugr/spark-vllm-b12x:nightly-20260908` (pulled from Docker Hub; 24 GB unpacked).
  Inside: vLLM `0.1.dev20605+gf9dc27dde` (local-inference-lab/vllm `dev/jovian-judgement`),
  **b12x 1.3.0**, torch 2.13.0+cu130, FlashInfer 0.6.18, CUTLASS DSL 4.7.0.
- Checkpoint: [Tech2wild/GLM-5.3-Int4-Int8Mix](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix)
  (compressed-tensors, W4 g128 experts, W8 g128 attention/dense, W8-channel MTP head; 378 GB).
- One mod (below), applied by the launcher at boot on every node.

## The bug you will hit, and the fix

On this vLLM tree DSA models (GLM-5.x 743B, DeepSeek V3.2) use the CUDA-specific
`vllm/models/deepseek_v32` path. With `--decode-context-parallel-size 4` it crashes at
CUDA-graph capture, and if you patch the attention gather in, it boots but emits garbage.

Root cause: `deepseek_v32/common/kernels.py::fused_norm_rope` treats a **negative slot as
padding and skips the whole row**. Under DCP the slot mapping is -1 for every token another
rank owns, so 3/4 of the query/KV latents on every rank were zeros before the first
attention layer. Nothing downstream can fix that.

The mod `mods/fix-dsv32-dcp-combine` adds a `COMPUTE_UNOWNED` constexpr that keeps the row
computation and skips only the cache stores, adds the plain-DCP query gather + LSE combine
to the DSA attention path, opens the b12x full-CKV-gather prefill path to `glm_moe_dsa`,
and (optionally) routes the DCP group's gathers through the b12x RoCE runtime, which was
worth ~5 ms/step. Verified with a 6-layer parity harness (DCP=4 vs DCP=1: prefill top-1
logprob mean diff 0.02, attention outputs within 0.3%) and then on the full model.

Mod + recipe + harness: `<GITHUB BRANCH URL>` (branch `claude/glm-speed-optimization-911a80`
of the launcher repo). The kernel change is a few lines and should go upstream.

## Recipe (`recipes/4x-spark-cluster/glm-53-b12x-dcp4-prod`)

```yaml
recipe_version: "1"
name: GLM-5.3-b12x-dcp4-prod
model: /models/GLM-5.3-Tech2wild
container: eugr/spark-vllm-b12x:nightly-20260908

defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 4
  gpu_memory_utilization: 0.88
  max_model_len: 327680
  max_num_batched_tokens: 4096

env:
  NCCL_NET: IB
  NCCL_IB_DISABLE: "0"
  NCCL_IB_HCA: "mlx5_0,mlx5_2"
  NCCL_SOCKET_IFNAME: "enP7s7,enp1s0f0np0,enP2p1s0f0np0"
  GLOO_SOCKET_IFNAME: "enp1s0f0np0"
  NCCL_IB_GID_INDEX: "-1"
  NCCL_MAX_NCHANNELS: "4"
  NCCL_MIN_NCHANNELS: "4"
  NCCL_CUMEM_ENABLE: "0"
  NCCL_CROSS_NIC: "1"
  NCCL_IGNORE_CPU_AFFINITY: "1"
  NCCL_BUFFSIZE: "16777216"

  VLLM_ENABLE_ROCE_ALLREDUCE: "1"          # b12x one-shot RDMA allreduce for the TP group
  VLLM_ROCE_DCP_COLLECTIVES: "1"           # (mod) DCP-group gathers via the same runtime
  VLLM_B12X_MLA_CKV_GATHER: "1"            # full-CKV-gather prefill
  VLLM_B12X_MLA_CKV_GATHER_DSA: "1"        # (mod) ... enabled for glm_moe_dsa
  VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS: "524288"
  VLLM_ROCE_ALLREDUCE_MAX_SIZE: "2MB"
  VLLM_ENABLE_PCIE_ALLREDUCE: "0"

  CUTE_DSL_ARCH: "sm_121a"
  VLLM_USE_V2_MODEL_RUNNER: "1"
  VLLM_USE_AOT_COMPILE: "1"
  VLLM_SPARSE_INDEXER_MAX_LOGITS_MB: "256"
  VLLM_MARLIN_USE_ATOMIC_ADD: "1"
  VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"
  VLLM_WORKER_MULTIPROC_METHOD: "spawn"
  CUDA_DEVICE_MAX_CONNECTIONS: "32"
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
  OMP_NUM_THREADS: "8"
  TORCH_CUDA_ARCH_LIST: "12.1a"
  HF_HUB_OFFLINE: "1"
  TRANSFORMERS_OFFLINE: "1"
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: "1800"

command: |
  export B12X_ROCE_GID_INDEX=$(for i in 0 1 2 3 4 5 6 7; do g=$(cat /sys/class/infiniband/mlx5_0/ports/1/gids/$i 2>/dev/null); t=$(cat /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/$i 2>/dev/null); case "$g" in *:ffff:*) [ "$t" = "RoCE v2" ] && {{ echo $i; break; }};; esac; done)
  echo "[recipe] B12X_ROCE_GID_INDEX=$B12X_ROCE_GID_INDEX on $(hostname)"
  vllm serve /models/GLM-5.3-Tech2wild \
    --served-model-name glm5.3 \
    --quantization compressed-tensors \
    --tensor-parallel-size 4 \
    --decode-context-parallel-size 4 \
    --dcp-comm-backend ag_rs \
    --attention-backend B12X \
    --speculative-config '{{"method":"mtp","num_speculative_tokens":4,"attention_backend":"B12X","quantization":"compressed-tensors"}}' \
    --kv-cache-dtype fp8_ds_mla \
    --kv-cache-memory-bytes 9000000000 \
    --max-model-len 327680 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 4 \
    --compilation-config '{{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}}' \
    --max-cudagraph-capture-size 32 \
    --async-scheduling \
    --long-prefill-token-threshold 2048 \
    --gpu-memory-utilization 0.88 \
    --hf-overrides '{{"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}}' \
    --host {host} \
    --port {port} \
    --trust-remote-code \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    --enable-prefix-caching \
    --no-enable-flashinfer-autotune

mods:
  - mods/fix-dsv32-dcp-combine
```

## Running it

1. On every node: `docker pull eugr/spark-vllm-b12x:nightly-20260908`, passwordless SSH from
   the head, the weights visible at the same path (`/home/jeff/models/...` here, NFS-exported
   from the head), swappiness low, and the usual `~/.cache/{vllm,flashinfer}` dirs.
2. Head node `.env`: `CLUSTER_NODES`, `ETH_IF`, `IB_IF` for your interfaces
   (`./autodiscover.sh` fills them), `HF_HOME=/home/jeff/models`.
3. The recipe mounts the weights as `/models`; the launcher takes that from
   `VLLM_SPARK_EXTRA_DOCKER_ARGS="-v $HOME/models:/models"` in your shell.
4. Launch: `./run-recipe.py recipes/4x-spark-cluster/glm-53-b12x-dcp4-prod --no-ray`
   (no Ray: the RDMA collectives were integrated on the PyTorch-distributed path).
5. Boot is ~15 min (weights over NFS ~10 min, then compile/warmup/graph capture). Look for
   `RoCEnante all-reduce is live`, `Using [...] all-reduce backends ... for group 'dcp:0'`,
   and `GPU KV cache size: 657,664 tokens` in the log.

A systemd unit (`systemd/vllm-glm53.service`, with a pre-start that waits for the workers'
SSH/docker/NFS) is in the same branch.

## What did not help (each its own boot, same benchmark set)

MTP k=5 (worse on code/prose, better only on lists); 8192-token prefill chunks (noise);
page-aligned DCP ownership `--dcp-kv-cache-interleave-size 64` (noise); `--dcp-comm-backend
a2a` (noise on decode, prefill down). Query replication does not fit in memory at this size.
Run-to-run noise on the code prompt is about ±2 tok/s (MTP acceptance varies).

## Where the remaining time goes

Same image with DCP=1 (4x smaller KV pool) runs ~100 ms/step, so the DCP collectives cost
~15–20 ms after routing the gathers over RDMA; what is left is the per-layer reduce-scatter,
which the b12x RoCE runtime does not implement. Beyond that, the next step is 8 nodes at TP=8
without DCP (a recipe for that is in the branch, untested for lack of nodes).

## Credits

eugr (spark-vllm-docker), Luke Alonso / local-inference-lab (b12x, the RoCE one-shot
collectives, the Spark vLLM tree), Tech2wild (the Int4-Int8Mix quant with a native MTP head),
tonyd2wild and the GLM-5.2/5.3 Spark recipe authors whose notes this builds on.
