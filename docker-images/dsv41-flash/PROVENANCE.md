# patches/ — vendored from tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark (MIT)

Files are byte-identical to the repo's `patch/` top-level set (boot 10),
verified at bake time by md5 (Dockerfile gate; manifest in
`patch/README.md` of the upstream repo, 2026-09-10):

| file | md5 | baked to (vllm package-relative) |
|---|---|---|
| engram.py | c0329107 | models/deepseek_v4_1/common/engram.py |
| model_state.py | 0a14bee6 | models/deepseek_v4_1/nvidia/model_state.py |
| weight_utils.py | 7e1027f1 | model_executor/model_loader/weight_utils.py |
| attention.py | da9ef196 | models/deepseek_v4_1/attention.py |
| flashinfer_sparse.py | af0f8447 | models/deepseek_v4_1/nvidia/flashinfer_sparse.py |
| sparse_swa.py | cc419353 | v1/attention/backends/mla/sparse_swa.py |
| sparse_attn_indexer.py | a9b73756 | model_executor/layers/sparse_attn_indexer.py |

`mounts.txt` is their bind-mount manifest (kept as the bake manifest).
`prewarm5.py` / `verify5.py` are their JIT prewarm/verify scripts
(upstream `build/`). The `patch/*/` subfolders (per-fix diffs + offline
tests) are intentionally NOT vendored — upstream docs, not build inputs.
