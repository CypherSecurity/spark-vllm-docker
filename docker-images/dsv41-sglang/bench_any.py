# Engine-agnostic bench (vLLM or SGLang): same prompts as ds_bench1.py / ds_bench_long.py,
# decode = usage.completion_tokens / (wall - ttft), no engine metrics needed.
import json, time, urllib.request, random, sys, os, threading
MODEL=os.environ.get("BENCH_MODEL","deepseek-v4.1-flash"); URL=os.environ.get("BENCH_URL","http://localhost:8000")
TKW={"thinking": False}
P={"code":"Write a Python function that merges two sorted lists, with a docstring and three test cases. Then write a second function that finds the k-th smallest element in the merged result without fully merging, with tests.",
   "prose":"Write a detailed 600-word essay about the history and engineering of the London Underground.",
   "count":"Count from 1 to 150. One number per line. Nothing else."}
def run(prompt, max_tokens, label, quiet=False):
    body={"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0,
          "stream":True,"stream_options":{"include_usage":True},"chat_template_kwargs":TKW}
    req=urllib.request.Request(URL+"/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t0=time.time(); tft=None; usage=None; text=[]
    with urllib.request.urlopen(req,timeout=3600) as r:
        for line in r:
            line=line.decode().strip()
            if not line.startswith("data:") or line=="data: [DONE]": continue
            d=json.loads(line[5:])
            if d.get("usage"): usage=d["usage"]
            ch=d.get("choices") or []
            c=(ch[0].get("delta") or {}).get("content") if ch else None
            if c:
                text.append(c)
                if tft is None: tft=time.time()-t0
    wall=time.time()-t0; ct=usage["completion_tokens"]; pt=usage["prompt_tokens"]; dec=wall-(tft or 0)
    res=dict(label=label,prompt=pt,out=ct,ttft=tft,wall=wall,decode=ct/dec if dec>0 else 0,text="".join(text))
    if not quiet:
        print(f"[{label}] prompt={pt} out={ct} ttft={tft:.2f}s wall={wall:.2f}s decode={res['decode']:.1f} tok/s e2e={ct/wall:.1f} tok/s"); sys.stdout.flush()
    return res
mode=sys.argv[1] if len(sys.argv)>1 else "short"
if mode=="short":
    for k in ["code","prose","count"]: run(P[k], 300 if k=="count" else 512, k)
elif mode=="long":
    n=int(sys.argv[2]) if len(sys.argv)>2 else 30000
    random.seed(int(time.time()))
    words=open("/usr/share/dict/words").read().split() if os.path.exists("/usr/share/dict/words") else [f"w{i}" for i in range(5000)]
    doc=" ".join(random.choice(words) for _ in range(n))
    p=f"Here is a document of random words:\n\n{doc}\n\nWrite a Python function that merges two sorted lists, with a docstring and three test cases."
    r=run(p,16,"cold-prefill"); print(f"   cold prefill {r['prompt']/r['ttft']:.0f} tok/s")
    run(p,400,"cached-decode")
elif mode=="conc":
    n=int(sys.argv[2]) if len(sys.argv)>2 else 8
    out=[None]*n
    def w(i): out[i]=run(P["code"]+f" (variant {i})",512,f"c{i}",quiet=True)
    t0=time.time(); ts=[threading.Thread(target=w,args=(i,)) for i in range(n)]
    [t.start() for t in ts]; [t.join() for t in ts]; wall=time.time()-t0
    tot=sum(o["out"] for o in out)
    print(f"[conc {n}] total out={tot} wall={wall:.1f}s aggregate={tot/wall:.1f} tok/s  per-stream decode mean={sum(o['decode'] for o in out)/n:.1f} tok/s")
elif mode=="sanity":
    r=run("What is 19 + 23? Reply only with the number.",16,"sanity"); print("   answer:",repr(r["text"]))
