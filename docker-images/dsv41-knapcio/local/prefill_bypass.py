"""Admit short requests while a long prefill is mid-chunk (knapcio#9).

Stock behaviour: each scheduling pass, the in-flight chunked request claims the whole chunk budget
(`rem_chunk_tokens` -> 0) in `PrefillAdder.add_chunked_req`, so every waiting request is refused
before its own memory check runs. A new 20-token chat waits until a 200K-token prefill finishes
(measured on this fleet: TTFT 35.6 s, 8 s into a 199K prefill).

With `DSV41_PREFILL_BYPASS=<tokens>` (unset or 0 = off):

* when the chunked request is still unfinished after this pass's chunk, the adder gets an extra
  `<tokens>` of chunk budget for waiting requests; the long prefill keeps its full chunk;
* only requests that fit WHOLE are admitted during that pass. A request that would itself need
  chunking is refused, because the scheduler allows a single chunked request
  (`assert self.chunked_req is None` when a new one appears);
* memory admission (`_check_prefill_budget`, the KV/SWA budgets) is the stock code, unchanged.

The decision depends only on scheduler state that is identical on every TP rank, so all ranks
build the same batch. Prefill-result processing already handles a middle chunk next to finished
requests (it skips requests with `inflight_middle_chunks > 0`).
"""
import logging
import os
import time

logger = logging.getLogger(__name__)

_STATE = {"admitted": 0, "tokens": 0, "last_log": 0.0}


def bypass_tokens():
    raw = os.environ.get("DSV41_PREFILL_BYPASS", "").strip()
    if raw.lower() in ("", "0", "off", "false", "no"):
        return 0
    try:
        return max(int(raw), 0)
    except ValueError:
        logger.warning("DSV41_PREFILL_BYPASS=%r is not an int; bypass off", raw)
        return 0


def install(module):
    """Wrap PrefillAdder in sglang.srt.managers.schedule_policy."""
    limit = bypass_tokens()
    if limit <= 0:
        return
    adder_cls = module.PrefillAdder
    admission_cls = module._PrefillAdmission
    add_req_result = module.AddReqResult
    if getattr(adder_cls, "_dsv41_prefill_bypass", False):
        return

    orig_add_chunked = adder_cls.add_chunked_req
    orig_select = adder_cls._select_prefill_admission
    orig_add_one = adder_cls.add_one_req

    def add_chunked_req(self, req):
        ret = orig_add_chunked(self, req)
        self._dsv41_bypass = False
        if (
            ret is not None  # the long prefill continues after this pass
            and req in self.can_run_list  # and it did get a chunk this pass
            and getattr(self, "dllm_config", None) is None
            and self.rem_chunk_tokens is not None
        ):
            self._dsv41_bypass = True
            self.rem_chunk_tokens = max(self.rem_chunk_tokens, 0) + limit
        return ret

    def _select_prefill_admission(self, req, **kwargs):
        res = orig_select(self, req, **kwargs)
        if (
            getattr(self, "_dsv41_bypass", False)
            and isinstance(res, admission_cls)
            and res.is_chunked
        ):
            # Would become a second chunked request: refuse, it waits for the long prefill.
            return add_req_result.OTHER
        return res

    def add_one_req(self, req, *args, **kwargs):
        before = len(self.can_run_list)
        res = orig_add_one(self, req, *args, **kwargs)
        if getattr(self, "_dsv41_bypass", False) and len(self.can_run_list) > before:
            _STATE["admitted"] += 1
            _STATE["tokens"] += int(getattr(req.extend_range, "length", 0) or 0)
            now = time.monotonic()
            if now - _STATE["last_log"] >= 60.0:
                logger.info(
                    "DSV41 prefill bypass: %d request(s), %d tokens admitted beside a chunked prefill so far",
                    _STATE["admitted"], _STATE["tokens"],
                )
                _STATE["last_log"] = now
        return res

    adder_cls.add_chunked_req = add_chunked_req
    adder_cls._select_prefill_admission = _select_prefill_admission
    adder_cls.add_one_req = add_one_req
    adder_cls._dsv41_prefill_bypass = True
    logger.info(
        "DSV41 prefill bypass ARMED: up to %d extra prefill tokens per pass for requests that fit whole "
        "while a chunked prefill is in flight (knapcio#9)", limit,
    )
