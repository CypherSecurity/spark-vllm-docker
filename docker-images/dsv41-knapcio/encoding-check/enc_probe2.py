import json, urllib.request, logging, sys
logging.disable(logging.WARNING)
sys.path.insert(0, "/opt/dsv41/adapter")
from transformers import AutoTokenizer
import sglang.srt.entrypoints.openai.encoding_dsv41 as enc
import encoding_compat; encoding_compat.install_encoder(enc)
tok = AutoTokenizer.from_pretrained("/model")
def post(path, body):
    r = urllib.request.Request("http://127.0.0.1:8000" + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=120))["usage"]
hist = [{"role": "user", "content": "What is 19+23?"},
        {"role": "assistant", "content": "42", "reasoning_content": "19+23 = 42."},
        {"role": "user", "content": "And times 2?"}]
r = len(tok.encode(enc.encode_messages(hist, thinking_mode="thinking"), add_special_tokens=False))
u = post("/v1/chat/completions", {"model": "deepseek-v4.1-flash", "max_tokens": 1, "messages": hist, "chat_template_kwargs": {"thinking": True}})
print(f"OpenAI endpoint, thinking history : reference {r}, live {u['prompt_tokens']}, diff {u['prompt_tokens'] - r:+d}")
a = post("/v1/messages", {"model": "deepseek-v4.1-flash", "max_tokens": 1, "thinking": {"type": "enabled"},
    "messages": [{"role": "user", "content": "What is 19+23?"},
                 {"role": "assistant", "content": [{"type": "thinking", "thinking": "19+23 = 42.", "signature": "x"}, {"type": "text", "text": "42"}]},
                 {"role": "user", "content": "And times 2?"}]})
print(f"Anthropic endpoint (zcode style)  : reference {r}, live {a['input_tokens']}, diff {a['input_tokens'] - r:+d}")
