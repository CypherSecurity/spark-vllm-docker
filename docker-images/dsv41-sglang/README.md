# DeepSeek-V4.1-Flash on SGLang (trial, 4x DGX Spark)

Uses MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks (AGPL-3.0) at commit 6b40a5e, checked out
on the head at `~/dsv41-sglang`, driven by its `start-tp4.sh` (TP4/EP4). Nothing from that
repo is vendored here; `env.tp4` is our copy of `~/dsv41-sglang/.env.tp4`.

## What was changed from their `.env.tp4.example`

| setting | value | why |
|---|---|---|
| HEAD_IP / WORKER_IPS / WORKER_HOSTS | 192.168.177.12 / .13 .11 .14 | RoCE fabric addresses |
| GLOO/NCCL_SOCKET_IFNAME, FABRIC_IFACE | enp1s0f0np0 | bootstrap on the fabric (dist-init address is the fabric IP) |
| IB_HCA | mlx5_0,mlx5_2 | both logical ports; GID index 3 on both, every node (2026-09-24) |
| MODEL_DIR | /home/jeff/models/dsv41-flat | hard links to the HF snapshot dba1be0a (no extra disk) |
| NFS_SHARE | 0 | the head already exports /home/jeff/models (host nfs-server); their exporter container would clash |
| PORT, SERVED_MODEL_NAME | 8000, deepseek-v4.1-flash | drop-in for the vLLM endpoint |
| HF_REVISION | dba1be0a | weights byte-identical to their fb2764a5 pin; only encoding/ differs |

Workers: docker volume `dsv41-weights` = bind of /home/jeff/models/dsv41-flat (the NFS mount).
Packed Engram shards: head `~/dsv41-engram`, workers `~/dsv41-4x-spark/engram` (2 x 23.6 GiB each).
The vLLM node-local copies in /var/tmp/engram-local are separate and untouched.
Containers: `dsv41-head` / `dsv41-worker` (not `vllm_node`).

## Switch

    sudo systemctl stop vllm-dsv41
    cd ~/dsv41-sglang && ./start-tp4.sh serve      # streams the boot log, returns when ready
    ./start-tp4.sh logs -f                           # head log (docker logs dsv41-head works too)
    ./start-tp4.sh stop                              # back to vLLM: sudo systemctl start vllm-dsv41

## Not yet done
- DSpark SPS cost table (`python -m sglang.benchmark.dspark_sps_profiler all --base-url ...`,
  saved to `~/dsv41-sglang/state-tp4/dspark_sps.json`). Without it DSpark runs verify-all;
  their notes say the table mainly pays at concurrency >= 2.

## vLLM baseline with the same script (2026-09-24, `bench_any.py`)
See `vllm_baseline_20260924.txt`: code 74.4 / prose 32.4 / count 93.0 tok/s, 54K cold prefill
1,238 tok/s (image pull running concurrently), 8 streams 281 tok/s aggregate (37.6 per stream).

## Trial result (2026-09-25)

Boots 1 and 2 (MAX_RUNNING_REQUESTS=8) hung at decode CUDA-graph capture, bs=6, every time:
ranks 0/2/3 wait in a torch barrier while rank 1 (gx10253, .13) spins in cuMemcpyDtoHAsync
(`seq_lens.sum().item()`, deepseek_v4_backend.py:2141) with its GPU idle; host Engram pool and CUDA
callback threads idle, no Xid, clean RoCE counters. Boot 3 with MAX_RUNNING_REQUESTS=4 (graphs 1-4)
came up in 12.7 min: KV pool 7,515,648 tokens, 1M context. Also changed: DSPARK_BLOCK_SIZE=5 (their
issue #20), `--enable-metrics` (issue #12; metric names are `sglang:*`). Do not profile an SPS table
with Engram (issue #12 / sgl-project/sglang#39173).

| same bench_any.py, client on the head | vLLM (2026-09-24) | SGLang (2026-09-25) |
|---|---|---|
| code / prose / count, 1 stream | 74.4 / 32.4 / 93.0 | 63.7 / 27.4-29.9 / 87.7 |
| 54K cold prefill | 1,238 (image pull running) | 2,263 |
| same 54K prompt again, TTFT | 37.0 s (re-prefilled) | 0.38 s (cache hit) |
| concurrency | 8 streams: 281 agg, 37.6 each | 4 streams: 137 agg, 36.2 each |
| context / KV pool | 512K / 3.3M | 1M / 7.5M |

## Local changes (2026-09-25, boot 4)

- `anthropic_thinking_compat.py`, run at image build (Dockerfile.local.diff adds it after the
  adapter copy): `/v1/messages` accepts `{"thinking": {"type": "enabled"}}` with no budget
  (Z.ai/GLM clients such as zcode), raises budgets below 1024 to 1024, and drops
  `redacted_thinking` history blocks instead of returning 400. The backend never enforced the
  budget anyway. Copy it to `~/dsv41-sglang/local/` before `./start-tp4.sh build`.
- `DSV41_CACHE_GIB=4`: 4 GiB Engram row cache per node (host RAM). Their issue #21 measured
  +16-42% prefill on repetitive text at 8 GiB, ~0 on random text; logs are repetitive.
- Still `MAX_RUNNING_REQUESTS=4` (capture hang at bs=6 with 8).
- Note: `/v1/messages` without a `thinking` field runs with thinking OFF (vLLM defaulted on).

## Boot 4 checks (2026-09-25): thinking fix, Engram cache, long context

- zcode-style `{"thinking": {"type": "enabled"}}` and redacted-thinking history now return 200.
- KV pool 7,424,512 tokens (7,515,648 without the Engram cache).
- Engram cache 4 GiB/node: 74.5% hit rate on real journal text, but no prefill gain: logs 2,101-2,103
  tok/s at 84-88K tokens, random words 2,243 at 54K (2,263 without the cache). Compute-bound, not NVMe.
- Staged needle test (`staged_ctx.py`, 4-node MemAvailable guard at 3 GB, aborts via /abort_request):
  256K PASS 1,683 tok/s (head min 8.7 GB); 512K PASS 1,309 tok/s (head min 5.1 GB);
  900K aborted by the guard at ~767K prefilled tokens, head 2.55 GB, prefill down to ~700 tok/s.
  Server stayed healthy. Head idle MemAvailable afterwards 7.6 GB (14.3 before the test).
- Practical ceiling with this profile: 512K. The Engram cache takes ~4 GB of the head's headroom for no
  measured speed gain; setting DSV41_CACHE_GIB=0 is the first lever to raise the ceiling.

## Boot 5 (2026-09-25): Engram cache off again

DSV41_CACHE_GIB=0. KV pool 7,057,920 tokens (the budget is computed from MemAvailable at load, so it
varies by a few percent between boots). Idle free memory: head 14.4 GB, workers 16.3-18.2 GB. Thinking fix
still active. zcode context setting: 512K.
