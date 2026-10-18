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

## v2.3 upgrade (2026-09-30, 58f2321)

Built while v2.2 served (head had 8.7 GB free); downtime 4 min (stop 23:32:03, ready 23:36:11).
Certified LM head armed (M 6..48), replay guard on: `/v1/completions` with echo+logprobs now returns 400
and the engine stays up (v2.2 crashed all four ranks). Thinking patch reapplied after the SGLang tree copy.
KV 6,601,472. Same scripts: code 115.8-118.1 / prose 60.0-61.8 / count 154.7-160.5, 54K cold prefill 4,810,
16 streams 494 aggregate (one run; v2.2 516). Within noise of v2.2 as expected.

## Local scheduler changes (2026-10-06, knapcio#9)

Problem, measured here: while a long prompt is mid-chunk, (a) a new request is not admitted until it
finishes (short chat TTFT 35.6 s behind a 199K prefill) and (b) requests already generating freeze
(stock SGLang always runs a pending prefill chunk before decode).

- `local/prefill_bypass.py` (loaded via adapter/sitecustomize.py on `sglang.srt.managers.schedule_policy`,
  `DSV41_PREFILL_BYPASS=4096`): while the chunked prefill continues, waiting requests get an extra
  4096 tokens of chunk budget per pass, admitted only if they fit whole (a second chunked request
  would trip `assert self.chunked_req is None`). Stock memory admission unchanged; deterministic, so
  every TP rank builds the same batch. Tested in the image build (`tests/test_prefill_bypass.py`).
  Result: short chat TTFT 35.6 s -> 3.3 s; the 199K prefill 43.6 s -> 45.1 s.
- `--prefill-decode-interval 2` (stock SGLang option, default 0): two decode steps after each prefill
  chunk, so running generations keep streaming (max gap 1.4 s observed during a long prefill).
- Known gap: with two long cold prompts queued, a short request behind the second still waits
  (FCFS loop stops at the first refused candidate). Mostly seen right after a restart, when every
  session re-prefills its context.
- `local/restart_when_idle.sh`: restarts only after 90 s without real batches (ignores the 1-token
  health probes every ~15 s).

Clean re-run (2026-10-06 23:58, no other clients): 199K cold prefill 47.0 s (43.6 s stock); a running generation
kept streaming at ~1/10 of its idle rate (116 SSE chunks in 47 s vs 107 in the 4 s before; longest gap 1.0 s);
a new short chat sent 8 s into the prefill: TTFT 1.4 s, complete in 1.5 s (stock: TTFT 35.6 s, complete 37.1 s).
Raising --prefill-decode-interval trades prefill speed for faster running generations.

## DeepSeek-V4.1-Flash-1017 (2026-10-18)

Revision 1bae0e10. Same config.json and weight index as the original; all 48 shards new (510 GB), including
the Engram shards, so it has its own hard-linked folder (/home/jeff/models/dsv41-1017-flat), worker volume
(dsv41-weights-1017) and packed Engram rows (~/dsv41-engram-1017 on every node, 10 min to pack). Encoder:
special tokens, templates and all five reference prompts identical; the 31 changed lines only assemble
images inside tool results. Same image, no rebuild. `env.tp4-1017` = 1017, `env.tp4-0910` = original;
`local/swap_to_1017.sh` swaps at the next 90 s quiet moment. Rollback: cp .env.tp4-0910 .env.tp4, restart.

Serving 1017 since 2026-10-18 20:15 (swap down 20:10:34; KV 6,679,040). Same scripts: code 97.8 (first run after boot) / 126.0,
prose 61.1-62.5, count 158-161 tok/s; 54K cold prefill 4,343; 16 streams 507 agg; during a 199K prefill (53.5 s) a running
generation kept streaming (max gap 1.7 s) and a new short chat answered in 2.4 s. Thinking via /v1/messages OK.

## Prompt-encoding check (2026-10-18, 1017)

No Jinja template ships; SGLang selects its own V4.1 encoder (`--tool-call-parser deepseekv41` -> spec "dsv41").
`encoding-check/run.sh`: SGLang encoder == DeepSeek golden prompts 5/5 as served (stock SGLang 4/5: default
effort 50 vs the publisher 75, fixed by the adapter). Live prompt token counts == reference for chat, thinking,
explicit effort, tools, thinking+tools and multi-turn thinking history, on both /v1/chat/completions and
/v1/messages. Only non-client shapes differ (tools attached inside a message get pydantic key order + strict:false;
DeepSeek `task`/`mask` fields). Reply cap raised to 131072 (DSV41_MAX_NEW_TOKENS) per the 1017 card (max_tokens >= 256K).
