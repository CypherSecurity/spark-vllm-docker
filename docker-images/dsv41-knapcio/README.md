# DeepSeek-V4.1-Flash, knapcio v2.2 TP4 profile (test setup, 2026-09-27)

Source: knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4 (AGPL-3.0) at e9ec61d, checked out at
`~/dsv41-knapcio` on the head. A tuned fork of MiaAI-Lab's recipe (see ../dsv41-sglang).
`env.tp4` here is our copy of `~/dsv41-knapcio/.env.tp4`.

Production profile as published: `Dockerfile.canary-roce` (SGLang dsv4.1 branch f80c91a4b staged by
`scripts/fetch-sglang-canary.sh`, RoCEnante overlay, b12x_next MoE), `EP_SIZE=1`, 16 slots, 4096-token
prefill chunks, k=5, 4 GiB Engram cache, and their full production `EXTRA_CONTAINER_ENV`.

Local changes: the same fleet mapping as the MiaAI-Lab trial (fabric IPs, `enp1s0f0np0` bootstrap,
`IB_HCA=mlx5_0,mlx5_2`, hard-linked weights + worker volume `dsv41-weights`, `NFS_SHARE=0`, port 8000,
served name `deepseek-v4.1-flash`), `B12X_ROCE_HCA=mlx5_0,mlx5_2` (their `rocep1s0f0,roceP2p1s0f0`),
`WORKER_DIR=/home/jeff/dsv41-knapcio`, and the packed Engram rows from the MiaAI-Lab trial
(`WORKER_ENGRAM_DIR=/home/jeff/dsv41-4x-spark/engram`, head `~/dsv41-engram`; format identical).

Base image: `lmsysorg/sglang:dev-dsv41` (the moving tag, as knapcio built it; newer than MiaAI-Lab's
pinned digest). Pulled per node with `pull_guarded.sh` while GLM served (aborts a node's pull below
300 MB MemAvailable). The image build itself must run with GLM stopped: GLM leaves 0.8-1.8 GB free
and the build runs Python tests that import the whole SGLang stack.

Steps once GLM is stopped:

    cd ~/dsv41-knapcio && ./start-tp4.sh build     # all 4 nodes, runs the in-image tests
    ./start-tp4.sh serve

Expected boot lines (their README): shared-expert padding K 576->640, indexer chunked ARMED,
`gamma=5`, `max_running_requests=16`, `RoCEnante ready: world=4`, `[moe_b12x_next] armed`.

## Results (2026-09-27, client on the head, same scripts as ../dsv41-sglang)

Boot: ready in 6 min; all profile lines present (RoCEnante world=4 on mlx5_0,mlx5_2, b12x_next EP1,
indexer chunked, gamma=5, spec_sync_free, eager_glue, replicated_split); graphs to bs 16, no capture
hang. KV pool 6,469,120 tokens. Idle free memory: head 13.6 GB, workers 15-16 GB.

| | vLLM | MiaAI-Lab SGLang | knapcio v2.2 |
|---|---|---|---|
| code / prose / count, 1 stream | 74.4 / 32.4 / 93.0 | 63.7 / 27-30 / 87.7 | 114.7-119.1 / 59.2-61.3 / 156-162 |
| cold prefill 54K random words | 1,238 | 2,263 | 4,450 |
| cold prefill 84-88K journal logs | - | 2,101-2,103 | 4,448-4,910 |
| 4 / 8 / 16 streams aggregate | - / 281 / - | 137 / - / - | 241 / 369 / 516 |
| needle 256K / 512K / 900K | - | PASS / PASS / aborted at 767K | PASS 4,496 / PASS 3,776 / PASS 3,203 tok/s |
| head min free at 900K | - | 2.6 GB (guard) | 3.1 GB |

900K passed with only 3.1 GB left on the head (guard at 3.0): the ceiling is between 900K and 1M, so
keep clients at 512K-640K. Anthropic thinking patch (../dsv41-sglang/anthropic_thinking_compat.py)
added to Dockerfile.canary-roce after the SGLang tree copy (see the .local.diff): f80c91a4b has the
same strict validator.

## systemd

`systemd/sglang-dsv41.service` (Type=oneshot + RemainAfterExit: `start-tp4.sh serve` returns once the API is up;
`stop` tears down all four nodes). Conflicts with vllm-glm53 and vllm-dsv41. Pre-start is the shared
`systemd/wait-for-workers.sh` with `REQUIRE_PATH=/home/jeff/models/dsv41-flat/config.json`. Head log:
`journalctl -u sglang-dsv41` for the launcher, `docker logs -f dsv41-head` for the engine.
