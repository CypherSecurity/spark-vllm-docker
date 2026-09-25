# Prefill on real log text: two different journal slices (so the second is not a prefix-cache hit).
import sys, json, time, urllib.request, uuid
URL="http://localhost:8000"; MODEL="deepseek-v4.1-flash"
def run(path, label):
    text=open(path, errors="replace").read()[:190000]
    p=f"[{uuid.uuid4()}]\nBelow is a system log excerpt.\n\n{text}\n\nList the three most frequent error sources in one line."
    body={"model":MODEL,"messages":[{"role":"user","content":p}],"max_tokens":16,"temperature":0,"stream":True,
          "stream_options":{"include_usage":True},"chat_template_kwargs":{"thinking":False}}
    req=urllib.request.Request(URL+"/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t0=time.time(); tft=None; usage=None
    with urllib.request.urlopen(req,timeout=1800) as r:
        for line in r:
            line=line.decode().strip()
            if not line.startswith("data:") or line=="data: [DONE]": continue
            d=json.loads(line[5:])
            if d.get("usage"): usage=d["usage"]
            ch=d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content") and tft is None: tft=time.time()-t0
    pt=usage["prompt_tokens"]
    print(f"[{label}] prompt={pt} ttft={tft:.2f}s cold prefill={pt/tft:.0f} tok/s"); sys.stdout.flush()
for i,p in enumerate(sys.argv[1:]): run(p, f"log{i+1}")
