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
