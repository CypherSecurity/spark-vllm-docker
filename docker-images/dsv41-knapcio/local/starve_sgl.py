# Reproduce knapcio#9: TTFT of a short chat while a long cold prefill is in flight.
import json, random, threading, time, urllib.request, uuid, sys, os
URL="http://127.0.0.1:8000/v1/chat/completions"; MODEL="deepseek-v4.1-flash"
LONG=int(sys.argv[1]) if len(sys.argv)>1 else 200000
words=open("/usr/share/dict/words").read().split()
def post(body):
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    return urllib.request.urlopen(req,timeout=3600)
res={}
def long_req():
    random.seed(time.time())
    p=f"[{uuid.uuid4()}] "+" ".join(random.choice(words) for _ in range(int(LONG/2.24)))+"\nReply OK."
    t=time.time(); d=json.load(post({"model":MODEL,"messages":[{"role":"user","content":p}],"max_tokens":1,"chat_template_kwargs":{"thinking":False}}))
    res["long"]=(d["usage"]["prompt_tokens"], time.time()-t)
th=threading.Thread(target=long_req); th.start(); time.sleep(8)
t=time.time(); ttft=None
r=post({"model":MODEL,"messages":[{"role":"user","content":f"[{uuid.uuid4()}] What is 19+23? Reply with the number."}],"max_tokens":8,"stream":True,"chat_template_kwargs":{"thinking":False}})
for line in r:
    line=line.decode().strip()
    if line.startswith("data:") and line!="data: [DONE]" and ttft is None:
        ch=json.loads(line[5:]).get("choices") or []
        if ch and (ch[0].get("delta") or {}).get("content"): ttft=time.time()-t
short_total=time.time()-t; th.join()
print(f"long prefill: {res['long'][0]} tokens in {res['long'][1]:.1f} s")
print(f"short chat sent 8 s into it: TTFT {ttft:.1f} s, done in {short_total:.1f} s")
