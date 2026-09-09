#!/usr/bin/env python3
"""DCP parity harness: dump prompt logprobs + greedy continuations, or compare two dumps.

  parity.py dump <label>          -> writes <label>.json from localhost:8000
  parity.py compare <a.json> <b.json>
"""
import json, sys, urllib.request, math

URL = "http://localhost:8000/v1/completions"
PROMPTS = [
    "The quick brown fox jumps over the lazy dog. " * 8,
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\nprint([fibonacci(i) for i in range(20)])\n" * 4,
    "In 1969, Apollo 11 landed on the Moon. Neil Armstrong and Buzz Aldrin walked on the surface while Michael Collins orbited above. " * 12,
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " * 40,
]


def call(prompt, max_tokens, prompt_logprobs):
    body = {"model": "glm5.3-mini", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "logprobs": 5, "prompt_logprobs": prompt_logprobs, "seed": 0}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def dump(label):
    out = []
    for i, p in enumerate(PROMPTS):
        r = call(p, 32, 1)["choices"][0]
        # per-position top-1 logprob of the actual prompt token (prefill path)
        plp = []
        for entry in (r.get("prompt_logprobs") or []):
            if not entry:
                plp.append(None); continue
            plp.append(max((v["logprob"] if isinstance(v, dict) else v) for v in entry.values()))
        out.append({"prompt_idx": i, "n_prompt_tokens": len(plp), "prompt_top1_logprob": plp,
                    "text": r["text"], "gen_tokens": r["logprobs"]["tokens"] if r.get("logprobs") else [],
                    "gen_logprobs": r["logprobs"]["token_logprobs"] if r.get("logprobs") else []})
        print(f"[{label}] prompt {i}: {len(plp)} prompt tokens, gen={r['text'][:60]!r}")
    json.dump(out, open(f"{label}.json", "w"), indent=1)


def compare(a, b):
    A = json.load(open(a)); B = json.load(open(b))
    ok = True
    for x, y in zip(A, B):
        n = min(len(x["prompt_top1_logprob"]), len(y["prompt_top1_logprob"]))
        diffs = [abs(u - v) for u, v in zip(x["prompt_top1_logprob"][1:n], y["prompt_top1_logprob"][1:n]) if u is not None and v is not None]
        mx = max(diffs) if diffs else float("nan"); mean = sum(diffs) / len(diffs) if diffs else float("nan")
        same_gen = x["gen_tokens"] == y["gen_tokens"]
        first_div = next((i for i, (s, t) in enumerate(zip(x["gen_tokens"], y["gen_tokens"])) if s != t), None)
        print(f"prompt {x['prompt_idx']}: prefill top1-logprob diff mean={mean:.4f} max={mx:.4f} over {len(diffs)} pos | greedy 32-token match={same_gen}" + ("" if same_gen else f" (diverges at token {first_div})"))
        if not same_gen or (mx == mx and mx > 0.5): ok = False
    print("PARITY:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    if sys.argv[1] == "dump": dump(sys.argv[2])
    else: compare(sys.argv[2], sys.argv[3])
