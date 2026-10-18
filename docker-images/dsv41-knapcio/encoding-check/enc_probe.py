import json, urllib.request, logging, sys
logging.disable(logging.WARNING)
sys.path.insert(0, "/opt/dsv41/adapter")
from transformers import AutoTokenizer
import sglang.srt.entrypoints.openai.encoding_dsv41 as enc
import encoding_compat; encoding_compat.install_encoder(enc)
tok = AutoTokenizer.from_pretrained("/model")
def live(body):
    body = dict({"model": "deepseek-v4.1-flash", "max_tokens": 1}, **body)
    r = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=120))["usage"]
def ref(messages, mode, effort=None):
    return len(tok.encode(enc.encode_messages(messages, thinking_mode=mode, reasoning_effort=effort), add_special_tokens=False))
tool = {"type": "function", "function": {"name": "lookup", "description": "Look up a value", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}
m = [{"role": "user", "content": "hi there"}]
mt = [{"role": "system", "content": "", "tools": [tool]}, {"role": "user", "content": "hi there"}]
for label, body, r in [
    ("chat", {"messages": m, "chat_template_kwargs": {"thinking": False}}, ref(m, "chat")),
    ("thinking, default effort", {"messages": m, "chat_template_kwargs": {"thinking": True}}, ref(m, "thinking")),
    ("thinking, top-level reasoning_effort=high", {"messages": m, "chat_template_kwargs": {"thinking": True}, "reasoning_effort": "high"}, ref(m, "thinking", "high")),
    ("chat + tool", {"messages": m, "tools": [tool], "chat_template_kwargs": {"thinking": False}}, ref(mt, "chat")),
    ("thinking + tool", {"messages": m, "tools": [tool], "chat_template_kwargs": {"thinking": True}}, ref(mt, "thinking")),
]:
    u = live(body)
    print(f"{label:45s} reference {r:4d}  live {u['prompt_tokens']:4d}  diff {u['prompt_tokens'] - r:+d}   usage extras: {[k for k in u if k not in ('prompt_tokens','completion_tokens','total_tokens')]}")
print("reference thinking prompt:", repr(enc.encode_messages(m, thinking_mode="thinking"))[:220])
