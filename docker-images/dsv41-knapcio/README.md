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
