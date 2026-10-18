import importlib.util, sys, difflib
from pathlib import Path
REF = Path("/ref")
spec = importlib.util.spec_from_file_location("ref_encoding", REF / "encoding.py")
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
import sglang.srt.entrypoints.openai.encoding_dsv41 as sgl
def run(label):
    ok = 0
    for i in range(1, 6):
        case = ref.load_cases(str(REF / "tests" / f"test_input_{i}.json"))[0]
        golden = (REF / "tests" / f"test_output_{i}.txt").read_text()
        ref_prompt, _ = ref.encode_case(case, thinking_mode="chat")
        sgl_prompt, _ = sgl.encode_messages(case["messages"], thinking_mode=case.get("thinking_mode") or "chat",
                                            context=case.get("context"), reasoning_effort=case.get("reasoning_effort"),
                                            return_multi_modal_data=True)
        same = sgl_prompt == golden
        ok += same
        feats = [k for k in ("tools", "thinking_mode", "context", "reasoning_effort") if case.get(k) or (k == "tools" and case["messages"][0].get("tools"))]
        print(f"  [{label}] case {i}: reference==golden {ref_prompt == golden}, SGLang==golden {same}  ({len(golden)} chars; features: {', '.join(feats) or 'plain chat'})")
        if not same:
            d = list(difflib.unified_diff(golden.splitlines(), sgl_prompt.splitlines(), "golden", "sglang", lineterm="", n=1))
            print("    " + "\n    ".join(d[:12]))
    return ok
a = run("stock SGLang")
sys.path.insert(0, "/opt/dsv41/adapter")
import encoding_compat; encoding_compat.install_encoder(sgl)
b = run("as served")
print(f"SGLang matches DeepSeek's reference prompts: stock {a}/5, as served {b}/5")
