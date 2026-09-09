# fix-dsv32-dcp-combine

Makes plain decode-context-parallel (DCP > 1, no PCP) work for DSA models
(`glm_moe_dsa` = GLM-5.x 743B, DeepSeek V3.2) on the CUDA "nvidia" model path of
local-inference-lab/vllm `dev/jovian-judgement` (image
`eugr/spark-vllm-b12x:nightly-20260908`, b12x 1.3.0), verified on a 4x DGX Spark
cluster on 2026-09-09.

## Symptoms before the fix

- `--decode-context-parallel-size 4` crashes at CUDA-graph capture:
  `RuntimeError: shape '[32, 16, 512]' is invalid for input of size 1048576`.
- With only the attention gather/combine patched in, it boots but the output is
  garbage (MTP acceptance ~0.1/4, "Count 1 to 15" -> stray digits).

## Root cause (the one that matters)

`vllm/models/deepseek_v32/common/kernels.py::fused_norm_rope` fuses q_a_layernorm,
kv_a_layernorm, RoPE, the indexer-K norm/quant and both cache writes into one
Triton kernel. It reads the layer slot mapping and treats a negative slot as
"padding": it **returns before doing any work for that row**.

Under plain DCP the slot mapping is -1 for every token this rank does not own
(1 in `dcp_world_size`). The kernel therefore left `q_c`, `kv_c`, `k_pe` and the
indexer K **zero for three quarters of the rows on every rank**, before the first
attention layer ran. Every downstream stage (query gather, LSE combine, CKV
gather, RoCE collectives) was then operating on zeros, which is why none of the
attention-level experiments changed the output.

Measured with the 6-layer harness: `q_c` rows for non-owned tokens were exactly
zero on each rank; after the fix `q_c` is identical on all ranks and to the DCP=1
reference, attention output agrees to 0.3% (layer 0) / 1.3% (layer 3), prefill
top-1 logprobs agree to a mean of 0.02 (was 0.8, max 6.0).

## What the mod changes

1. **`kernels.py`** (root cause): new constexpr `COMPUTE_UNOWNED`. When set, the
   row computation always runs and only the indexer-cache and MLA-cache stores are
   skipped for negative slots. The attention layer passes
   `compute_unowned=(not use_pcp and dcp_world_size > 1)`, so single-node and PCP
   behaviour is unchanged.
2. **`models/deepseek_v32/attention.py`**: the plain-DCP query all-gather and LSE
   combine (mirrors `mla_attention.py`'s generic path), plus empty-shard masking.
   Needed for decode rows.
3. **`v1/attention/backends/mla/b12x_mla_sparse.py`**: opens the b12x
   full-CKV-gather prefill path (previously GLM5Next-only) to `glm_moe_dsa` behind
   `VLLM_B12X_MLA_CKV_GATHER_DSA=1`, so prefill does not depend on the extend-plan
   LSE, whose validity for the combine is unverified.
4. Env-gated debug dumps (`DSV32_DCP_DEBUG=1`), inert otherwise.

## Recipe env

```
VLLM_B12X_MLA_CKV_GATHER: "1"
VLLM_B12X_MLA_CKV_GATHER_DSA: "1"
VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS: "524288"   # >= max context
```

## Results on the full model (4x DGX Spark, TP4 + DCP4, MTP k=4, 327K ctx)

| | eldritch v2 (old stack) | nightly + this mod |
|---|---|---|
| decode step | 137-150 ms | 122-127 ms |
| code / prose / count | 27 / 17 / 36 tok/s | 32 / 20 / 39 tok/s |
| decode @ 32K ctx | 25.7 tok/s | 33.6 tok/s |
| 30K cold prefill | ~500 tok/s | 803 tok/s |

DCP=1 on the same image runs ~100 ms/step; the ~25 ms difference is the DCP
collectives, which still go through NCCL (only the TP group uses RoCEnante).

## Reproduction harness

`/home/jeff/models/GLM-5.3-mini6` (layers 0-5 of the int4 checkpoint, real shard
files), recipes `glm-53-mini6-dcp1` / `glm-53-mini6-dcp4`, and
`parity.py` (prompt logprob + greedy parity between DCP=1 and DCP=4). A boot
takes ~2.5 minutes. Greedy sequences on the truncated model flip on noise; use the
logprob statistics.

## Speed sweep (2026-09-09, full model, 8 GB KV, one variable per boot)

| variant | step | code / prose / count | decode @32K | prefill | kept |
|---|---|---|---|---|---|
| base (fix, k=4, 2048 chunks, interleave 1, ag_rs) | 122-127 ms | 32 / 20 / 39 | 33.6 | 803 | ref |
| MTP k=5 | 131-143 ms | 28-31 / 18 / 42 | 32.8 | 787 | no |
| long-prefill threshold 8192 | 120-126 ms | 29-32 / 20 / 39 | 31.2 | 813 | no |
| dcp-kv-cache-interleave-size 64 | 119-127 ms | 28 / 19 / 40 | 32.6 | 791 | no |
| dcp-comm-backend a2a | 118-127 ms | 32-34 / 20 / 41 | 30.5 | 687 | no |
| `VLLM_ROCE_DCP_COLLECTIVES=1` (mod part 6) | 115-122 ms | 32-33 / 21 / 42 | 35.6 | 809 | **yes** |

Run-to-run noise on the code prompt is about +/-2 tok/s (MTP acceptance varies between
runs even at temperature 0). Query replication (`--dcp-q-replicate`) was not tried: it needs
the full-head q projection on every rank (~3.6 GB/node at int8), which does not fit.
Remaining DCP cost is the per-layer reduce-scatter (NCCL); the b12x RoCE runtime has no
reduce-scatter. Production recipe: `recipes/4x-spark-cluster/glm-53-b12x-dcp4-prod`.

## Harness scripts

`harness/harness_run.sh <recipe> <label> [--wait-idle]` boots a recipe under a pty,
waits for health, runs `parity.py dump <label>`, and stops the cluster.
`harness/sweep_driver.sh <recipe>...` boots each recipe and appends the benchmark set
to `sweep_results.md`. Both default `S` (log/output dir) to their own directory and
export the `-v ~/models:/models` docker arg the recipes rely on.

## systemd unit

`systemd/vllm-glm53.service` runs the production recipe as a service (pre-start waits for
the worker nodes' SSH, docker and NFS mount of the weights, and removes stale
`vllm_node` containers). Install:

```
sudo cp systemd/vllm-glm53.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable vllm-glm53
sudo systemctl start vllm-glm53      # ~15 min to healthy; journalctl -u vllm-glm53 -f
```
