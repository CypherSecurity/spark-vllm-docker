**Title:** Field report: v2.2 and v2.3 on a switched 4x GB10 fleet (`mlx5_*` port names): published numbers reproduce, 900K needle, 16 slots

Thank you for this profile, and to MiaAI-Lab, rhys101, local-inference-lab and everyone credited in the README. On four DGX Sparks this is the fastest DeepSeek-V4.1-Flash we have run by a wide margin. Below is a second fleet's data on the published production line, with the few things we had to change and what we hit along the way.

## Fleet

- 4x DGX Spark (GB10, 121.7 GiB), switched RoCE fabric, MTU 9000. One cable per Spark; the ConnectX-7 presents it as two 200G ports, named `mlx5_0` (`enp1s0f0np0`) and `mlx5_2` (`enP2p1s0f0np0`) on this OS image. IPv4 RoCE v2 GID index 3 on both ports of every node.
- Kernel 6.17.0-1014-nvidia, driver 580.142, Docker 29.2.1. GPU clocks at defaults (no 2200 MHz cap).
- Base image `lmsysorg/sglang:dev-dsv41` pulled 2026-09-27: `sha256:4a5d132a06a7...`, image id `381b27ffa19b`, sglang `da64c5cbb`, created 2026-09-11. SGLang tree `f80c91a4b` via `scripts/fetch-sglang-canary.sh`.
- Checkpoint revision `dba1be0a` (the current `main`). Its weight files are byte-identical to the `fb2764a5` pin; only `encoding/` and the READMEs differ.

## What we changed from `.env.tp4.example`

Production line (last `EXTRA_CONTAINER_ENV`), `IMAGE=dsv41-4x-spark:canary-roce`, `BUILD_DOCKERFILE=Dockerfile.canary-roce`, `EP_SIZE=1`, everything else as shipped, plus:

| setting | value | why |
|---|---|---|
| `IB_HCA`, `B12X_ROCE_HCA` | `mlx5_0,mlx5_2` | local port names (yours: `rocep1s0f0,roceP2p1s0f0`) |
| `GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME` | `enp1s0f0np0` | bootstrap on the fabric; the dist-init address is the fabric IP |
| `NFS_SHARE` | `0` | the head already exports the model directory with the host `nfs-server`, and the workers mount it; the exporter container would clash on 2049 |
| `MODEL_DIR` | a flat folder of hard links into the HF cache snapshot (`cp -rlL <snapshot> <flat>`) | the launcher wants plain files; the snapshot's relative symlinks break inside a bind mount. No extra disk |
| worker volume `dsv41-weights` | `docker volume create --opt type=none --opt o=bind --opt device=<flat folder on the NFS mount>` | satisfies `nfs_worker_has_model` with `NFS_SHARE=0` on a switched fabric |
| `WORKER_ENGRAM_DIR` | the MiaAI-Lab trial's packed shards | `pack_engram.py` and the reader are identical between the two repos, so no repack |
| `PORT`, `SERVED_MODEL_NAME` | `8000`, `deepseek-v4.1-flash` | drop-in for our previous endpoint |

Result: `RoCEnante ready: world=4 hcas=mlx5_0,mlx5_2 gid_index=3 max_size=2097152`, and every line in the README's boot checklist present.

## Results

Client on the head (the README says bench from a worker; noted as a caveat), a plain OpenAI streaming client, thinking off, temperature 0. Decode = `completion_tokens / (wall - TTFT)`. Not sparkDash, so compare the shape rather than the exact cells.

| | v2.2 (e9ec61d) | v2.3 (58f2321) |
|---|---|---|
| boot to ready | 360 s (empty caches), 260 s (warm) | 248 s |
| KV pool | 6,469,120 / 6,622,720 tokens | 6,601,472 tokens |
| idle MemAvailable | head 13.6 GB, workers 15-16 GB | similar |
| code / prose / count, c1 | 114.7-119.1 / 59.2-61.3 / 155.3-161.9 | 115.8-118.1 / 60.0-61.8 / 154.7-160.5 |
| cold prefill, 54K random dictionary words | 4,450 tok/s | 4,810 tok/s |
| cold prefill, 84-88K real journal logs | 4,448-4,910 tok/s | not re-run |
| aggregate c4 / c8 / c16 (512 tokens each) | 241 / 369 / 516 tok/s | c16 494 (one run) |

Prose c1 at ~61 on our single essay prompt lines up with your varied-prompt figure (61.1) rather than the sparkDash prose cell, as you describe.

**Staged needle** (v2.2; random-word filler, one code at 50% depth, 4-node MemAvailable guard that POSTs `/abort_request` at 3 GB):

| prompt | prefill | needle | lowest MemAvailable (head / workers) |
|---|---|---|---|
| 255,345 tokens | 4,496 tok/s | PASS | 8.5 / 10.2-11.0 GB |
| 510,370 | 3,776 | PASS | 4.6 / 6.0-6.8 GB |
| 896,401 | 3,203 | PASS | 3.1 / 4.2-5.6 GB |

The head gets to within ~0.1 GB of our guard at 900K, so on this fleet (stock loader budget, 4 GiB Engram cache) we keep clients at 512-640K.

**Same four nodes, other stacks, same client and prompts** (for scale):

| | vLLM `dsv41-feat` e47aa780 (kilork/tonyd2wild stack, DSpark k=5) | MiaAI-Lab 6b40a5e, TP4 profile | this profile, v2.3 |
|---|---|---|---|
| code / prose, c1 | 74.4 / 32.4 | 63.7 / 27-30 | ~117 / ~61 |
| cold prefill, 54K | 1,238 | 2,263 | 4,810 |
| slots | 8 | 4 (see below) | 16 |

## Things we hit

1. **Anthropic `/v1/messages` rejects Z.ai-style thinking.** Clients built for the GLM API (zcode in our case) send `{"thinking": {"type": "enabled"}}` with no `budget_tokens`; SGLang's `AnthropicThinkingParam` returns 400. It also 400s on `budget_tokens < 1024` and on `redacted_thinking` history blocks. Since serving only logs the budget, we patch at build time (after the `COPY runtime/sglang-canary/python` line): accept a missing budget, raise sub-1024 budgets to 1024, and drop `redacted_thinking` blocks instead of raising. The patch is a small string replace that fails the build if the upstream text moves; happy to open a PR if you want it in the image. Also worth knowing for anyone coming from vLLM: `/v1/messages` with no `thinking` field runs with thinking off.
2. **MiaAI-Lab's image hung at decode CUDA-graph capture, bs=6, every boot with `MAX_RUNNING_REQUESTS=8`.** Ranks 0/2/3 sat in a torch barrier; rank 1 spun in `cuMemcpyDtoHAsync` (`seq_lens.sum().item()`, `deepseek_v4_backend.py:2141`) with its GPU idle. No Xid, clean RoCE counters, host Engram pool and CUDA callback threads idle. Capping at 4 slots worked around it. Your canary-roce image captured bs 1-16 on the first try, on the same nodes.
3. **Engram row cache on real text** (MiaAI-Lab stack, 4 GiB): 74.5% hit rate on journal logs, but random-word prefill was unchanged (2,243 tok/s with the cache, 2,263 without) and logs ran at 2,100 tok/s with it (not measured without), so that stack looked compute-bound. We have not A/B'd the cache on this profile.
4. **Upgrading v2.2 to v2.3 in place:** `./start-tp4.sh build` ran fine while v2.2 served (head had 8.7 GB free), so the downtime was just stop + boot, about 4 minutes. The replay guard works: `/v1/completions` with `echo` + `logprobs` now returns 400 and the engine stays up.

Happy to run anything specific on this fleet if a number here looks off.
