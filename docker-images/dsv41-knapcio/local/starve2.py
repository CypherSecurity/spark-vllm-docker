# One long cold prefill vs (a) a generation already running and (b) a new short chat.
import json, random, threading, time, urllib.request, uuid, sys
URL="http://127.0.0.1:8000/v1/chat/completions"; MODEL="deepseek-v4.1-flash"
LONG=int(sys.argv[1]) if len(sys.argv)>1 else 200000
words=open("/usr/share/dict/words").read().split()
def post(body):
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    return urllib.request.urlopen(req,timeout=3600)
def stream(body, stamps):
    for line in post(dict(body, stream=True, stream_options={"include_usage": True})):
        line=line.decode().strip()
        if not line.startswith("data:") or line=="data: [DONE]": continue
        d=json.loads(line[5:])
        if d.get("usage"): stamps.append(("usage", time.time(), d["usage"]["completion_tokens"]))
        ch=d.get("choices") or []
        if ch and (ch[0].get("delta") or {}).get("content"): stamps.append(("tok", time.time(), 1))
gen=[]; t_gen=time.time()
ga=threading.Thread(target=stream, args=({"model":MODEL,"messages":[{"role":"user","content":f"[{uuid.uuid4()}] Write a long, detailed essay about the history of cryptography."}],"max_tokens":3000,"chat_template_kwargs":{"thinking":False}}, gen)); ga.start()
time.sleep(4)
long_t={}
def long_req():
    p=f"[{uuid.uuid4()}] "+" ".join(random.choice(words) for _ in range(int(LONG/2.24)))+"\nReply OK."
    t=time.time(); d=json.load(post({"model":MODEL,"messages":[{"role":"user","content":p}],"max_tokens":1,"chat_template_kwargs":{"thinking":False}}))
    long_t.update(start=t, end=time.time(), tokens=d["usage"]["prompt_tokens"])
lt=threading.Thread(target=long_req); lt.start(); time.sleep(8)
short=[]; t_s=time.time()
stream({"model":MODEL,"messages":[{"role":"user","content":f"[{uuid.uuid4()}] What is 19+23? Reply with the number."}],"max_tokens":8,"chat_template_kwargs":{"thinking":False}}, short)
lt.join(); ga.join()
toks=[t for k,t,_ in gen if k=="tok"]
inside=[t for t in toks if long_t["start"] < t < long_t["end"]]
before=[t for t in toks if t < long_t["start"]]
print(f"long prefill: {long_t['tokens']} tokens in {long_t['end']-long_t['start']:.1f} s")
print(f"running generation: {len(before)} chunks in the {long_t['start']-t_gen:.1f} s before, {len(inside)} chunks during the {long_t['end']-long_t['start']:.1f} s prefill"
      f" (longest gap {max((b-a for a,b in zip(toks,toks[1:]) if long_t['start']<b<long_t['end']), default=0):.1f} s)")
st=[t for k,t,_ in short if k=="tok"]
print(f"new short chat: TTFT {st[0]-t_s:.1f} s, finished {short[-1][1]-t_s:.1f} s after sending")
