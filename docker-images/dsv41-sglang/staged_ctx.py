# Staged long-context test for SGLang DeepSeek: needle retrieval at increasing prompt sizes,
# with a 4-node MemAvailable guard that aborts all requests if any node drops below GUARD_GB.
import json, os, random, subprocess, sys, threading, time, urllib.request, uuid
URL = "http://127.0.0.1:8000"; MODEL = "deepseek-v4.1-flash"
NODES = ["local", "192.168.177.13", "192.168.177.11", "192.168.177.14"]
GUARD_GB = float(os.environ.get("GUARD_GB", "3.0"))
TARGETS = [int(x) for x in (sys.argv[1:] or ["256000", "512000", "900000"])]
WORDS = open("/usr/share/dict/words").read().split() if os.path.exists("/usr/share/dict/words") else [f"w{i}" for i in range(20000)]
TOK_PER_WORD = 2.24  # measured: 24,000 dict words -> 53.6K tokens

mem = {n: [] for n in NODES}; stop = threading.Event(); aborted = threading.Event()
def watch(node):
    cmd = "while true; do awk '/MemAvailable/{print $2}' /proc/meminfo; sleep 1; done"
    p = subprocess.Popen(cmd if node == "local" else ["ssh", "-o", "BatchMode=yes", node, cmd],
                         shell=(node == "local"), stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        if stop.is_set(): break
        try: gb = int(line) / 1048576
        except ValueError: continue
        mem[node].append(gb)
        if gb < GUARD_GB and not aborted.is_set():
            aborted.set()
            print(f"  !! GUARD: {node} MemAvailable {gb:.2f} GB < {GUARD_GB} GB -> abort_all", flush=True)
            try: urllib.request.urlopen(urllib.request.Request(URL + "/abort_request", data=b'{"abort_all": true}',
                                        headers={"Content-Type": "application/json"}), timeout=10)
            except Exception as e: print("  abort_request failed:", e, flush=True)
    p.kill()
for n in NODES: threading.Thread(target=watch, args=(n,), daemon=True).start()
time.sleep(3)

def run(target):
    random.seed(target)
    n_words = int(target / TOK_PER_WORD)
    code = str(random.randint(100000, 999999))
    words = [random.choice(WORDS) for _ in range(n_words)]
    pos = n_words // 2
    words.insert(pos, f". IMPORTANT: the secret access code is {code}. Remember it. .")
    prompt = f"[{uuid.uuid4()}]\n" + " ".join(words) + "\n\nWhat is the secret access code mentioned in the text? Reply with the number only."
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 16, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": False}}
    for n in NODES: mem[n].clear()
    t0 = time.time(); tft = None; usage = None; text = []; err = None
    try:
        req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]": continue
                d = json.loads(line[5:])
                if d.get("usage"): usage = d["usage"]
                ch = d.get("choices") or []
                c = (ch[0].get("delta") or {}).get("content") if ch else None
                if c:
                    text.append(c)
                    if tft is None: tft = time.time() - t0
    except Exception as e:
        err = repr(e)[:200]
    wall = time.time() - t0
    mins = {n: (min(v) if v else float("nan")) for n, v in mem.items()}
    pt = usage["prompt_tokens"] if usage else None
    ans = "".join(text).strip()
    ok = code in ans
    print(f"[{target//1000}K] prompt_tokens={pt} ttft={tft if tft is None else round(tft,1)}s wall={wall:.0f}s "
          f"prefill={(pt/tft) if (pt and tft) else 0:.0f} tok/s needle={'PASS' if ok else 'FAIL'} answer={ans!r} err={err}", flush=True)
    print("   min MemAvailable GB: " + ", ".join(f"{('head' if n=='local' else n.split('.')[-1])}={v:.1f}" for n, v in mins.items()), flush=True)
    return ok and not err and not aborted.is_set()

for t in TARGETS:
    healthy = urllib.request.urlopen(URL + "/health", timeout=10).status == 200
    if not healthy: print("server unhealthy, stopping"); break
    if not run(t):
        print(f"stopping escalation after {t//1000}K"); break
    time.sleep(5)
stop.set()
