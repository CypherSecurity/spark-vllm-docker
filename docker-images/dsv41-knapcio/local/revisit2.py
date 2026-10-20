# Two long sessions back to back, then revisit each: does the first stay cached?
import json, random, time, urllib.request, uuid, sys
URL = "http://127.0.0.1:8000/v1/chat/completions"; MODEL = "deepseek-v4.1-flash"
SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
words = open("/usr/share/dict/words").read().split()
def ask(messages):
    body = {"model": MODEL, "messages": messages, "max_tokens": 16, "temperature": 0,
            "chat_template_kwargs": {"thinking": False}}
    t = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"}), timeout=3600))
    u = d["usage"]; cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return time.time() - t, u["prompt_tokens"], cached, d["choices"][0]["message"]["content"]
sessions = {}
for name in ("A", "B"):
    random.seed(name + str(time.time()))
    doc = f"[session {name} {uuid.uuid4()}] " + " ".join(random.choice(words) for _ in range(int(SIZE / 2.24)))
    msgs = [{"role": "user", "content": doc + "\n\nReply with the single word OK."}]
    wall, pt, cached, ans = ask(msgs)
    sessions[name] = msgs + [{"role": "assistant", "content": ans}]
    print(f"session {name} first load : {pt} prompt tokens, {cached} cached, {wall:.1f} s", flush=True)
for name in ("A", "B"):
    msgs = sessions[name] + [{"role": "user", "content": "Now reply with the single word DONE."}]
    wall, pt, cached, ans = ask(msgs)
    print(f"session {name} revisit    : {pt} prompt tokens, {cached} cached ({100*cached/pt:.1f}%), {wall:.1f} s", flush=True)
