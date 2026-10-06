"""CPU test for adapter/prefill_bypass.py (local, knapcio#9)."""
import enum, os, sys, types
from dataclasses import dataclass

os.environ["DSV41_PREFILL_BYPASS"] = "4096"
import prefill_bypass


def fake_module():
    m = types.ModuleType("fake_schedule_policy")

    class AddReqResult(enum.Enum):
        CONTINUE = 0
        NO_TOKEN = 1
        OTHER = 2

    @dataclass
    class _PrefillAdmission:
        prefix_len: int
        extend_len: int
        max_new_tokens: int
        is_chunked: bool

    class Range:
        def __init__(self, n): self.length = n

    class PrefillAdder:
        def __init__(self, chunk=4096):
            self.can_run_list, self.rem_chunk_tokens, self.dllm_config = [], chunk, None
            self.new_chunked_req = None

        def add_chunked_req(self, req):
            take = min(req.remaining, self.rem_chunk_tokens)
            req.remaining -= take
            self.rem_chunk_tokens -= take
            self.can_run_list.append(req)
            return req if req.remaining > 0 else None

        def _select_prefill_admission(self, req, *, total_tokens=0, host_hit_length=0,
                                      swa_host_hit_length=0, truncation_align_size=None):
            if self.rem_chunk_tokens <= 0:
                return AddReqResult.OTHER
            chunked = req.need > self.rem_chunk_tokens
            n = min(req.need, self.rem_chunk_tokens)
            return _PrefillAdmission(0, n, 0 if chunked else 8, chunked)

        def add_one_req(self, req, has_chunked_req, truncation_align_size):
            adm = self._select_prefill_admission(req, total_tokens=req.need)
            if isinstance(adm, AddReqResult):
                return adm
            req.extend_range = Range(adm.extend_len)
            self.can_run_list.append(req)
            self.rem_chunk_tokens -= adm.extend_len
            if adm.is_chunked:
                self.new_chunked_req = req
            return AddReqResult.CONTINUE if self.rem_chunk_tokens > 0 else AddReqResult.OTHER

    m.AddReqResult, m._PrefillAdmission, m.PrefillAdder = AddReqResult, _PrefillAdmission, PrefillAdder
    return m


class Req:
    def __init__(self, need=0, remaining=0):
        self.need, self.remaining, self.extend_range = need, remaining, None


def check(name, cond):
    print(("[OK]   " if cond else "[FAIL] ") + name)
    if not cond:
        sys.exit(1)


m = fake_module()
prefill_bypass.install(m)
R = m.AddReqResult

# 1. long prefill still chunking -> side budget opens, short request admitted whole
a = m.PrefillAdder()
long_req = Req(remaining=200_000)
check("long prefill keeps its full chunk", a.add_chunked_req(long_req) is long_req and a.rem_chunk_tokens == 4096)
short = Req(need=300)
check("short request admitted whole", a.add_one_req(short, True, None) == R.CONTINUE and short in a.can_run_list
      and a.new_chunked_req is None)
# 2. a request that would need chunking is refused (no second chunked request)
big = Req(need=10_000)
check("oversized request refused", a.add_one_req(big, True, None) == R.OTHER and big not in a.can_run_list
      and a.new_chunked_req is None)

# 3. last chunk of the long prefill -> no side budget (stock behaviour)
b = m.PrefillAdder()
tail = Req(remaining=1000)
check("finishing chunk: no bypass", b.add_chunked_req(tail) is None and not b._dsv41_bypass
      and b.rem_chunk_tokens == 3096)

# 4. off when unset
os.environ["DSV41_PREFILL_BYPASS"] = "0"
m2 = fake_module()
prefill_bypass.install(m2)
c = m2.PrefillAdder()
c.add_chunked_req(Req(remaining=200_000))
check("unset: stock budget (0 left)", c.rem_chunk_tokens == 0 and not hasattr(m2.PrefillAdder, "_dsv41_prefill_bypass"))

# 5. real SGLang class: the three hooks attach
os.environ["DSV41_PREFILL_BYPASS"] = "4096"
try:
    import sglang.srt.managers.schedule_policy as sp
except Exception as e:  # host without sglang: skip
    print("[SKIP] real schedule_policy import:", e)
else:
    for attr in ("add_chunked_req", "_select_prefill_admission", "add_one_req"):
        check(f"real PrefillAdder has {attr}", hasattr(sp.PrefillAdder, attr))
    check("real _PrefillAdmission has is_chunked", "is_chunked" in getattr(sp._PrefillAdmission, "__dataclass_fields__", {}))
    prefill_bypass.install(sp)
    check("real PrefillAdder wrapped", getattr(sp.PrefillAdder, "_dsv41_prefill_bypass", False))
print("prefill_bypass: all checks passed")
