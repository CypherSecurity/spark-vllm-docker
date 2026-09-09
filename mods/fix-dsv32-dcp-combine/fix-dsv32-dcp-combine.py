#!/usr/bin/env python3
"""Add plain-DCP (no PCP) query gather + LSE combine to the CUDA DSA attention path.

BUG
---
On the local-inference-lab ``dev/jovian-judgement`` tree (eugr/spark-vllm-b12x
nightly-20260908), DSA models (``glm_moe_dsa`` = GLM-5.x 743B, DeepSeek V3.2) are
routed on CUDA to ``vllm/models/deepseek_v32/attention.py``. Its
``_sparse_indexer_and_attn`` only performs the decode-context-parallel query
all-gather and the LSE combine when ``self.use_pcp`` (prefill context parallel)
is set. With plain ``--decode-context-parallel-size 4`` the DCP-gathered
attention output (all 64 heads) reaches the 16-head local view and the boot dies
at CUDA-graph capture with::

    RuntimeError: shape '[32, 16, 512]' is invalid for input of size 1048576

The generic layer (``vllm/model_executor/layers/attention/mla_attention.py``,
``forward`` ~L1185-1230) handles plain DCP via ``self.dcp_manager.query_gather``
and ``self.dcp_manager.combine``. ``MLAAttention.__init__`` already builds
``self.dcp_manager`` (with ``query_gather`` populated when not use_pcp), so the
DSA subclass only lacks the two call sites.

FIX
---
Insert the ``not use_pcp`` branches beside the existing PCP-only ones. The
b12x full-CKV-gather mode (GLM5Next only) is honoured via
``impl.uses_full_ckv_dcp`` so the gather/combine is skipped when the backend
already gathered the complete cache.

SCOPE
-----
* Written 2026-09-09 against vllm 0.1.dev20605+gf9dc27dde (jovian-judgement).
* Idempotent; fails closed if either anchor is missing.
"""

import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v32/attention.py"

ANCHOR_GATHER = """        if self.use_pcp and self.impl.dcp_world_size > self.impl.pcp_world_size:
            if isinstance(mqa_q_arg, tuple):
                mqa_q_arg = torch.cat(mqa_q_arg, dim=-1)
            mqa_q_arg = get_tp_group().all_gather(mqa_q_arg, dim=1)
        attn_out, lse = self.impl.forward_mqa(  # type: ignore[attr-defined]
"""

PATCH_GATHER = """        if self.use_pcp and self.impl.dcp_world_size > self.impl.pcp_world_size:
            if isinstance(mqa_q_arg, tuple):
                mqa_q_arg = torch.cat(mqa_q_arg, dim=-1)
            mqa_q_arg = get_tp_group().all_gather(mqa_q_arg, dim=1)
        # [fix-dsv32-dcp-combine] plain DCP (no PCP): gather queries across the
        # DCP group exactly like the generic MLA layer does.
        _plain_dcp = (not self.use_pcp) and self.impl.dcp_world_size > 1
        _full_ckv_dcp = False
        if _plain_dcp:
            _uses_full_ckv = getattr(self.impl, "uses_full_ckv_dcp", None)
            if _uses_full_ckv is not None:
                _full_ckv_dcp = bool(_uses_full_ckv(attn_metadata, num_actual))
            if not _full_ckv_dcp:
                if isinstance(mqa_q_arg, tuple):
                    mqa_q_arg = torch.cat(mqa_q_arg, dim=-1)
                assert self.dcp_manager is not None
                assert self.dcp_manager.query_gather is not None
                if _DSV32_ROCE_DCP:
                    # dim-0 gather (eligible for the b12x RoCE runtime) + reshape to
                    # the [N, W*H, D] layout the sparse kernel expects.
                    from vllm.distributed import get_dcp_group as _gdg
                    _grp = _gdg(); _q = mqa_q_arg.contiguous(); _N, _H, _D = _q.shape
                    _gath = _grp.all_gather(_q, dim=0)
                    mqa_q_arg = (
                        _gath.view(_grp.world_size, _N, _H, _D).movedim(0, 1).reshape(_N, _grp.world_size * _H, _D)
                    )
                    if self.dcp_manager.padded_num_heads is not None:
                        from vllm.v1.attention.ops.dcp import reserve_query_head_storage as _rqhs
                        mqa_q_arg = _rqhs(mqa_q_arg, self.dcp_manager.padded_num_heads)
                else:
                    mqa_q_arg = self.dcp_manager.query_gather(mqa_q_arg)
        _dbg_q_in = None
        if _DSV32_DCP_DEBUG:
            _dbg_q_in = (torch.cat(mqa_q_arg, dim=-1) if isinstance(mqa_q_arg, tuple) else mqa_q_arg)[:num_actual].clone()
        attn_out, lse = self.impl.forward_mqa(  # type: ignore[attr-defined]
"""

DUMP_ANCHOR = """        # NOTE(woosuk): While the below does not need to be in the eager region,
"""

DUMP_PATCH = """        # [fix-dsv32-dcp-combine] optional debug dump (DSV32_DCP_DEBUG=1)
        if _DSV32_DCP_DEBUG and 70 <= num_actual <= 90:
            _dsv32_dcp_debug_dump(
                self, attn_metadata, num_actual, locals().get("_dbg_q_in"),
                locals().get("_dbg_partial"), attn_out, lse, kv_cache,
                extra={"q_c": q_c, "ql_nope": ql_nope, "mqa_q": mqa_q, "kv_c": kv_c, "k_pe": k_pe,
                       "index_q_fp8": index_q_fp8, "index_weights": index_weights_out},
            )
        # NOTE(woosuk): While the below does not need to be in the eager region,
"""

ANCHOR_COMBINE = """            attn_out = self.dcp_manager.combine(
                attn_out,
                lse,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
            )
            attn_out = finalize_mla_pcp_decode(attn_out, self.num_heads)
"""

PATCH_COMBINE = """            attn_out = self.dcp_manager.combine(
                attn_out,
                lse,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
            )
            attn_out = finalize_mla_pcp_decode(attn_out, self.num_heads)
        elif _plain_dcp and not _full_ckv_dcp:
            # [fix-dsv32-dcp-combine] plain DCP: LSE-combine the per-rank partial
            # outputs (reduce-scatter back to the local head set). Sparse MLA runs
            # every token through the MQA path, so pass the full metadata.
            assert lse is not None and self.dcp_manager is not None
            _dbg_partial = (attn_out.clone(), lse.clone()) if _DSV32_DCP_DEBUG else None
            # Rows whose causal key set on THIS rank is empty (early prompt
            # positions, short contexts) must not contribute: force lse=-inf and
            # zero the partial output so the LSE-weighted combine ignores them.
            _local_lens = getattr(attn_metadata, "cache_seq_lens_per_token", None)
            if _local_lens is not None:
                _empty = _local_lens[:num_actual] == 0
                lse = lse.masked_fill(_empty[:, None], float("-inf"))
                attn_out = attn_out.masked_fill(_empty[:, None, None], 0)
            attn_out = self.dcp_manager.combine(
                attn_out,
                lse,
                seq_lens=cast(torch.Tensor, attn_metadata.seq_lens),  # type: ignore[attr-defined]
                query_start_loc=attn_metadata.query_start_loc,
            )
"""

HELPER_ANCHOR = """from vllm.distributed.parallel_state import get_tp_group
"""

HELPER_PATCH = """from vllm.distributed.parallel_state import get_tp_group
import os as _dsv32_os
_DSV32_DCP_DEBUG = _dsv32_os.getenv("DSV32_DCP_DEBUG", "0") == "1"
_DSV32_ROCE_DCP = _dsv32_os.getenv("VLLM_ROCE_DCP_COLLECTIVES", "0") == "1"
_DSV32_DCP_DUMPED: dict = {}


def _dsv32_dcp_debug_dump(layer, md, num_actual, q_in, partial, attn_out, lse, kv_cache, extra=None):
    # [fix-dsv32-dcp-combine] one dump per layer, layers 0 and 3 only
    name = getattr(layer, "layer_name", getattr(layer, "prefix", "?"))
    if not (".layers.0." in name or ".layers.3." in name) or name in _DSV32_DCP_DUMPED:
        return
    _DSV32_DCP_DUMPED[name] = True
    from vllm.distributed import get_dcp_group, get_tensor_model_parallel_rank
    dcp = get_dcp_group()
    rank = dcp.rank_in_group if dcp.world_size > 1 else 0
    d = "/root/.cache/vllm/dcpdbg"
    _dsv32_os.makedirs(d, exist_ok=True)
    def cpu(t):
        return None if t is None else (t[:num_actual].detach().float().cpu() if hasattr(t, "shape") and t.shape[0] >= num_actual else t.detach().float().cpu())
    tb = getattr(layer, "topk_indices_buffer", None)
    blob = {
        "layer": name, "dcp_rank": rank, "dcp_world": dcp.world_size, "tp_rank": get_tensor_model_parallel_rank(),
        "num_actual": num_actual,
        "q_in": None if q_in is None else q_in.detach().float().cpu(),
        "partial_out": None if partial is None else partial[0][:num_actual].detach().float().cpu(),
        "partial_lse": None if partial is None else partial[1][:num_actual].detach().float().cpu(),
        "final_out": attn_out[:num_actual].detach().float().cpu(),
        "lse": None if lse is None else lse[:num_actual].detach().float().cpu(),
        "topk": None if tb is None else tb[:num_actual].detach().cpu(),
        "slot_mapping": getattr(md, "slot_mapping", None)[:num_actual].detach().cpu() if getattr(md, "slot_mapping", None) is not None else None,
        "seq_lens": getattr(md, "seq_lens", None).detach().cpu() if getattr(md, "seq_lens", None) is not None else None,
        "cache_seq_lens_per_token": cpu(getattr(md, "cache_seq_lens_per_token", None)),
        "query_start_loc": getattr(md, "query_start_loc", None).detach().cpu() if getattr(md, "query_start_loc", None) is not None else None,
        "block_table_row0": getattr(md, "block_table", None)[0, :4].detach().cpu() if getattr(md, "block_table", None) is not None else None,
        "kv_block0": kv_cache[0].detach().cpu() if kv_cache is not None and kv_cache.ndim >= 2 else None,
        "kv_block1": kv_cache[1].detach().cpu() if kv_cache is not None and kv_cache.ndim >= 2 and kv_cache.shape[0] > 1 else None,
        "cp_interleave": getattr(md, "cp_kv_cache_interleave_size", None),
    }
    for k, v in (extra or {}).items():
        try:
            blob[k] = None if v is None else v[:num_actual].detach().float().cpu()
        except Exception as e:  # noqa: BLE001
            blob[k] = f"unavailable: {e}"
    torch.save(blob, f"{d}/{name.replace('.', '_')}_dcp{dcp.world_size}_rank{rank}.pt")
"""

MARK = "[fix-dsv32-dcp-combine]"


def main() -> int:
    with open(TARGET, encoding="utf-8") as fh:
        src = fh.read()
    if MARK in src:
        print(f"{MARK} already applied")
        return 0
    for name, anchor in (("gather", ANCHOR_GATHER), ("combine", ANCHOR_COMBINE)):
        if src.count(anchor) != 1:
            print(f"{MARK} ERROR: {name} anchor found {src.count(anchor)} times", file=sys.stderr)
            return 1
    for name, anchor in (("dump", DUMP_ANCHOR), ("helper", HELPER_ANCHOR)):
        if src.count(anchor) != 1:
            print(f"{MARK} ERROR: {name} anchor found {src.count(anchor)} times", file=sys.stderr)
            return 1
    src = (src.replace(ANCHOR_GATHER, PATCH_GATHER, 1).replace(ANCHOR_COMBINE, PATCH_COMBINE, 1)
              .replace(DUMP_ANCHOR, DUMP_PATCH, 1).replace(HELPER_ANCHOR, HELPER_PATCH, 1))
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(src)
    print(f"{MARK} applied to {TARGET}")
    return 0


# ---------------------------------------------------------------------------
# Part 3: let the B12X sparse-MLA backend use its full-CKV-gather prefill path for
# glm_moe_dsa when VLLM_B12X_MLA_CKV_GATHER_DSA=1 (plus VLLM_B12X_MLA_CKV_GATHER=1).
# Why: the b12x extend-mode kernel does not return a real LSE (only the decode plan
# does), so prefill under plain DCP cannot be LSE-combined across ranks; measured
# 2026-09-09 on a 6-layer GLM-5.3: per-rank partial outputs are right, the LSE is
# 0.0 for rows with keys, and the combine diverges from prompt position 1. The
# GLM5Next lane avoids the combine by gathering the full compressed KV for prefill;
# the code path is generic (cp_gather_cache + all-gather of 656-byte DS-MLA records)
# but gated on GLM5Next. Decode rows keep the gather + LSE-combine route.
# ---------------------------------------------------------------------------
B12X_TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/b12x_mla_sparse.py"

B12X_EDITS = [
    ("""from vllm import envs
""", """from vllm import envs
import os as _dsv32_os
_DSV32_CKV_FOR_DSA = _dsv32_os.getenv("VLLM_B12X_MLA_CKV_GATHER_DSA", "0") == "1"  # [fix-dsv32-dcp-combine]
"""),
    ("""        self._ckv_gather_requested = (
            self.requires_glm_next_selector_metadata
            and self.dcp_world_size > 1
            and envs.VLLM_B12X_MLA_CKV_GATHER
        )
        if self._ckv_gather_requested:
            hf_config = vllm_config.model_config.hf_text_config
            ckv_topk_tokens = int(hf_config.index_topk) + int(hf_config.index_kpool) - 1
""", """        self._ckv_gather_requested = (
            (self.requires_glm_next_selector_metadata or _DSV32_CKV_FOR_DSA)
            and self.dcp_world_size > 1
            and envs.VLLM_B12X_MLA_CKV_GATHER
        )
        if self._ckv_gather_requested:
            hf_config = vllm_config.model_config.hf_text_config
            ckv_topk_tokens = int(hf_config.index_topk) + int(getattr(hf_config, "index_kpool", 1)) - 1
"""),
    ("""            is_glm_next=self.requires_glm_next_selector_metadata,
            dcp_world_size=self.dcp_world_size,
""", """            is_glm_next=(self.requires_glm_next_selector_metadata or _DSV32_CKV_FOR_DSA),
            dcp_world_size=self.dcp_world_size,
"""),
    ("""        self._ckv_gather_enabled = (
            self._is_glm_next
            and self.dcp_world_size > 1
            and envs.VLLM_B12X_MLA_CKV_GATHER
        )
""", """        self._ckv_gather_enabled = (
            (self._is_glm_next or _DSV32_CKV_FOR_DSA)
            and self.dcp_world_size > 1
            and envs.VLLM_B12X_MLA_CKV_GATHER
        )
"""),
]


def patch_b12x_backend() -> int:
    with open(B12X_TARGET, encoding="utf-8") as fh:
        src = fh.read()
    if "_DSV32_CKV_FOR_DSA" in src:
        print(f"{MARK} b12x backend already applied")
        return 0
    for i, (old, _new) in enumerate(B12X_EDITS):
        if src.count(old) != 1:
            print(f"{MARK} ERROR: b12x anchor {i} found {src.count(old)} times", file=sys.stderr)
            return 1
    for old, new in B12X_EDITS:
        src = src.replace(old, new, 1)
    with open(B12X_TARGET, "w", encoding="utf-8") as fh:
        fh.write(src)
    print(f"{MARK} applied to {B12X_TARGET}")
    return 0


# ---------------------------------------------------------------------------
# Part 4 (debug only, DSV32_DCP_DEBUG=1): dump the model-level inputs of the first
# real prefill (70..90 tokens): input_ids, positions, embedding output, attn_in.
# ---------------------------------------------------------------------------
MODEL_TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v32/nvidia/model.py"

MODEL_ANCHOR = """        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
"""

MODEL_PATCH = """        full_num_tokens = positions.shape[0]
        # [fix-dsv32-dcp-combine] debug dump of model inputs
        if _dsv32_os.getenv("DSV32_DCP_DEBUG", "0") == "1" and 70 <= full_num_tokens <= 90 and not _DSV32_MODEL_DUMPED:
            _DSV32_MODEL_DUMPED.append(True)
            try:
                from vllm.distributed import get_dcp_group, get_tensor_model_parallel_rank
                _g = get_dcp_group(); _r = _g.rank_in_group if _g.world_size > 1 else 0
                _d = "/root/.cache/vllm/dcpdbg"; _dsv32_os.makedirs(_d, exist_ok=True)
                torch.save({
                    "tp_rank": get_tensor_model_parallel_rank(), "dcp_rank": _r, "dcp_world": _g.world_size,
                    "input_ids": None if input_ids is None else input_ids[:full_num_tokens].detach().cpu(),
                    "positions": positions[:full_num_tokens].detach().cpu(),
                    "hidden": hidden_states[:full_num_tokens].detach().float().cpu(),
                    "attn_in": None if attn_in is None else attn_in[:full_num_tokens].detach().float().cpu(),
                    "replicated_embed": bool(self.replicated_embed),
                    "use_sequence_parallel": bool(self.use_sequence_parallel),
                }, f"{_d}/model_inputs_dcp{_g.world_size}_rank{_r}.pt")
            except Exception as _e:  # noqa: BLE001
                print("[fix-dsv32-dcp-combine] model dump failed:", _e)
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
"""

MODEL_HELPER_ANCHOR = """from itertools import islice
"""
MODEL_HELPER_PATCH = """from itertools import islice
import os as _dsv32_os  # [fix-dsv32-dcp-combine]
_DSV32_MODEL_DUMPED: list = []
"""


def patch_model_dump() -> int:
    with open(MODEL_TARGET, encoding="utf-8") as fh:
        src = fh.read()
    if "_DSV32_MODEL_DUMPED" in src:
        print(f"{MARK} model dump already applied")
        return 0
    for name, anchor in (("model", MODEL_ANCHOR), ("model-helper", MODEL_HELPER_ANCHOR)):
        if src.count(anchor) != 1:
            print(f"{MARK} ERROR: {name} anchor found {src.count(anchor)} times", file=sys.stderr)
            return 1
    src = src.replace(MODEL_ANCHOR, MODEL_PATCH, 1).replace(MODEL_HELPER_ANCHOR, MODEL_HELPER_PATCH, 1)
    with open(MODEL_TARGET, "w", encoding="utf-8") as fh:
        fh.write(src)
    print(f"{MARK} applied to {MODEL_TARGET}")
    return 0


# ---------------------------------------------------------------------------
# Part 5 (THE ROOT CAUSE): vllm/models/deepseek_v32/common/kernels.py
# fused_norm_rope treats a negative slot as "padding" and returns before doing any
# work for that row. Under plain DCP the slot mapping is -1 for every token this
# rank does not own, yet every rank still needs q_c / kv_c / k_pe for all rows
# (attention queries are gathered across DCP ranks). Result on GLM-5.3 mini:
# q_c rows for non-owned tokens are exactly zero on each rank => garbage output.
# Fix: a constexpr COMPUTE_UNOWNED (set by the attention layer only when
# dcp_world_size > 1 and not PCP) keeps the row computation and skips only the
# indexer-cache and MLA-cache stores for negative slots.
# ---------------------------------------------------------------------------
KERNEL_TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v32/common/kernels.py"

KERNEL_EDITS = [
    # 1) kernel signature: add the constexpr (anchored on the unique first-kernel prologue)
    ("""    INDEX_ROPE_INTERLEAVE: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    tok_idx = tl.program_id(0).to(tl.int64)
    pid = tl.program_id(1)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if pid == 3:
""", """    INDEX_ROPE_INTERLEAVE: tl.constexpr,
    USE_PDL: tl.constexpr,
    COMPUTE_UNOWNED: tl.constexpr = False,  # [fix-dsv32-dcp-combine]
):
    tok_idx = tl.program_id(0).to(tl.int64)
    pid = tl.program_id(1)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if pid == 3:
"""),
    # 2) early return only when not computing unowned rows
    ("""    elif tl.load(slot_mapping_ptr + tok_idx) < 0:
        # Padding
        return
""", """    else:
        # [fix-dsv32-dcp-combine] under plain DCP a negative slot means "not owned
        # by this rank", not padding: still compute the norms for the row.
        if not COMPUTE_UNOWNED:
            if tl.load(slot_mapping_ptr + tok_idx) < 0:
                # Padding
                return
"""),
    # 3) MLA cache store: skip for negative slots
    ("""            slot_idx = tl.load(slot_mapping_ptr + tok_idx)
            mla_block_idx = slot_idx // mla_cache_block_size
""", """            slot_idx = tl.load(slot_mapping_ptr + tok_idx)
            if COMPUTE_UNOWNED:
                if slot_idx < 0:
                    return  # [fix-dsv32-dcp-combine] not owned: no cache write
            mla_block_idx = slot_idx // mla_cache_block_size
"""),
    # 4) indexer cache store: skip for negative slots
    ("""        if indexer_cache_ptr is not None and slot_mapping_ptr is not None:
            slot_idx = tl.load(slot_mapping_ptr + tok_idx)
            _fp8_quant_and_cache_write(
""", """        if indexer_cache_ptr is not None and slot_mapping_ptr is not None:
            slot_idx = tl.load(slot_mapping_ptr + tok_idx)
            if COMPUTE_UNOWNED:
                if slot_idx < 0:
                    return  # [fix-dsv32-dcp-combine] not owned: no cache write
            _fp8_quant_and_cache_write(
"""),
    # 5) wrapper signature
    ("""    index_k_out: torch.Tensor | None = None,
) -> torch.Tensor:
    assert positions.ndim == 1
""", """    index_k_out: torch.Tensor | None = None,
    compute_unowned: bool = False,  # [fix-dsv32-dcp-combine]
) -> torch.Tensor:
    assert positions.ndim == 1
"""),
    # 6) launch
    ("""        USE_PDL=use_pdl,
        launch_pdl=use_pdl,
    )
    return q_c_out
""", """        USE_PDL=use_pdl,
        COMPUTE_UNOWNED=bool(compute_unowned),
        launch_pdl=use_pdl,
    )
    return q_c_out
"""),
]

ATTN_CALL_ANCHOR = """            index_k_out=index_k_out,
        )

        q = self.q_b_proj(q_c)[0]"""
ATTN_CALL_PATCH = """            index_k_out=index_k_out,
            compute_unowned=((not self.use_pcp) and self.impl.dcp_world_size > 1),  # [fix-dsv32-dcp-combine]
        )

        q = self.q_b_proj(q_c)[0]"""


def patch_kernel() -> int:
    with open(KERNEL_TARGET, encoding="utf-8") as fh:
        src = fh.read()
    if "COMPUTE_UNOWNED" in src:
        print(f"{MARK} kernel already applied")
    else:
        for i, (old, _new) in enumerate(KERNEL_EDITS):
            if src.count(old) != 1:
                print(f"{MARK} ERROR: kernel anchor {i} found {src.count(old)} times", file=sys.stderr)
                return 1
        for old, new in KERNEL_EDITS:
            src = src.replace(old, new, 1)
        with open(KERNEL_TARGET, "w", encoding="utf-8") as fh:
            fh.write(src)
        print(f"{MARK} applied to {KERNEL_TARGET}")
    with open(TARGET, encoding="utf-8") as fh:
        asrc = fh.read()
    if "compute_unowned=" in asrc:
        print(f"{MARK} attention call already applied")
        return 0
    if asrc.count(ATTN_CALL_ANCHOR) != 1:
        print(f"{MARK} ERROR: attention call anchor found {asrc.count(ATTN_CALL_ANCHOR)} times", file=sys.stderr)
        return 1
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(asrc.replace(ATTN_CALL_ANCHOR, ATTN_CALL_PATCH, 1))
    print(f"{MARK} attention call patched")
    return 0


# ---------------------------------------------------------------------------
# Part 6 (VLLM_ROCE_DCP_COLLECTIVES=1): route DCP-group gathers through the b12x
# RoCE one-shot runtime. The communicator only builds the RoCEnante adapter for
# groups named "tp"; the runtime only accepts dim-0 / last-dim gathers. So:
#  (a) cuda_communicator: treat a "dcp" group like "tp" when the env is set;
#  (b) DSA attention: query gather as dim-0 gather + reshape (in PATCH_GATHER above);
#  (c) b12x_indexer._merge_dcp_topk: candidate gather as dim-0 gather + reshape.
# Oversized shards (>VLLM_ROCE_ALLGATHER_MAX_SIZE) fall back to NCCL automatically.
# ---------------------------------------------------------------------------
COMM_TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/cuda_communicator.py"
COMM_EDITS = [
    ("""        if "tp" not in unique_name:
            # custom allreduce or torch symm mem can be used only by tp
""", """        _roce_dcp = os.getenv("VLLM_ROCE_DCP_COLLECTIVES", "0") == "1"  # [fix-dsv32-dcp-combine]
        if "tp" not in unique_name and not (_roce_dcp and "dcp" in unique_name):
            # custom allreduce or torch symm mem can be used only by tp
"""),
]
INDEXER_TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/b12x_indexer.py"
INDEXER_EDITS = [
    ("""    gathered = get_dcp_group().all_gather(packed, dim=1)
""", """    if _dsv32_os.getenv("VLLM_ROCE_DCP_COLLECTIVES", "0") == "1":  # [fix-dsv32-dcp-combine]
        _grp = get_dcp_group(); _n, _k, _c = packed.shape
        gathered = (
            _grp.all_gather(packed.contiguous(), dim=0)
            .view(_grp.world_size, _n, _k, _c).movedim(0, 1).reshape(_n, _grp.world_size * _k, _c)
        )
    else:
        gathered = get_dcp_group().all_gather(packed, dim=1)
"""),
]


def _apply_edits(target: str, edits, marker: str) -> int:
    with open(target, encoding="utf-8") as fh:
        src = fh.read()
    if marker in src:
        print(f"{MARK} {target.rsplit('/', 1)[-1]} already applied")
        return 0
    for i, (old, _new) in enumerate(edits):
        if src.count(old) != 1:
            print(f"{MARK} ERROR: {target.rsplit('/', 1)[-1]} anchor {i} found {src.count(old)} times", file=sys.stderr)
            return 1
    for old, new in edits:
        src = src.replace(old, new, 1)
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(src)
    print(f"{MARK} applied to {target}")
    return 0


def patch_roce_dcp() -> int:
    # cuda_communicator imports os?  ensure it.
    with open(COMM_TARGET, encoding="utf-8") as fh:
        csrc = fh.read()
    if "VLLM_ROCE_DCP_COLLECTIVES" not in csrc and "\nimport os\n" not in csrc:
        csrc = csrc.replace("import torch\n", "import os\nimport torch\n", 1)
        with open(COMM_TARGET, "w", encoding="utf-8") as fh:
            fh.write(csrc)
    rc = _apply_edits(COMM_TARGET, COMM_EDITS, "VLLM_ROCE_DCP_COLLECTIVES")
    with open(INDEXER_TARGET, encoding="utf-8") as fh:
        isrc = fh.read()
    if "_dsv32_os" not in isrc:
        isrc = isrc.replace("import torch\n", "import torch\nimport os as _dsv32_os  # [fix-dsv32-dcp-combine]\n", 1)
        with open(INDEXER_TARGET, "w", encoding="utf-8") as fh:
            fh.write(isrc)
    return rc or _apply_edits(INDEXER_TARGET, INDEXER_EDITS, "VLLM_ROCE_DCP_COLLECTIVES")


if __name__ == "__main__":
    rc = main()
    rc = rc or patch_b12x_backend()
    rc = rc or patch_model_dump()
    rc = rc or patch_kernel()
    sys.exit(rc or patch_roce_dcp())
