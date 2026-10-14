#!/usr/bin/env python3
"""Copy of /home/shared/opt-model/efficient_qwen_competition/latency_eval/bench_real_client.py
with a --model flag added (원본은 "model": "default"로 고정돼 있어서, served_model_name이
"Qwen/Qwen3.5-4B"인 3등팀 서버에서는 404가 남 — 1등팀 서버는 어차피 model 필드를 무시하고
자기 SERVED_MODEL_NAME으로 덮어써서 영향 없음). 공유 원본은 건드리지 않고 이 파일에만 반영.

Latency benchmark with REALISTIC, per-run-distinct prompts (no filler).

Matches the official protocol (greedy temp=0, sequential /v1/completions, median
of N runs) EXCEPT the prompt is a different natural-text passage every run, so
prefix caching can't hit across runs and MTP acceptance reflects realistic text.
Warmup uses a DISJOINT set of prompts from the measured ones.
Prints one JSON line: {"label":..., "results":{cat:{median_ms,...}}}.
"""
import sys, json, time, statistics, urllib.request, argparse

PROMPT_CFG = {  # match official max_new_tokens
    "short":  128, "medium": 256, "long": 256,
}

def invoke(base, prompt, max_tokens, model, domain=""):
    payload = {"model": model, "prompt": prompt,
               "max_tokens": max_tokens, "temperature": 0.0}
    if domain:
        payload["domain"] = domain  # 라우터(eqc_router.py)용 힌트, 일반 vLLM 서버는 무시함
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{base}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    ms = (time.perf_counter() - t0) * 1000
    ntok = d.get("usage", {}).get("completion_tokens", max_tokens)
    return ms, ntok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompts", default="workspace/real_latency_prompts.json")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--cat", default="")   # run only this category (short/medium/long); "" = all
    ap.add_argument("--max-tokens", type=int, default=0)  # >0 overrides PROMPT_CFG output len (prefill/decode decomp)
    ap.add_argument("--model", default="default")  # served_model_name; 1st팀 wrapper는 무시함, 3등팀은 정확히 맞아야 함
    ap.add_argument("--domain", default="")  # eqc_router.py용: "natural"/"math"/"code" (일반 서버 직접 호출 땐 비워둘 것)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    allp = json.load(open(a.prompts))

    results = {}
    for cat, max_new in PROMPT_CFG.items():
        if a.cat and cat != a.cat:
            continue
        if a.max_tokens > 0:
            max_new = a.max_tokens
        pl = allp[cat]
        measured = pl[:a.runs]                       # distinct measured prompts
        warm = pl[a.runs:a.runs + a.warmup]          # DISJOINT warmup prompts
        if len(measured) < a.runs:
            print(f"[{a.label}/{cat}] WARN only {len(measured)} prompts", file=sys.stderr)
        for p in warm:
            try: invoke(base, p, max_new, a.model, a.domain)
            except Exception as e: print(f"[{a.label}/{cat}] warmup err {e}", file=sys.stderr)
        lat, toks = [], []
        for p in measured:
            ms, n = invoke(base, p, max_new, a.model, a.domain)
            lat.append(ms); toks.append(n)
        med = statistics.median(lat)
        med_tok = statistics.median(toks)
        results[cat] = {"median_ms": round(med, 1), "gen_tokens": med_tok,
                        "tok_per_s": round(med_tok / (med/1000), 1)}
        print(f"[{a.label}/{cat}] median={med:.1f}ms gen={med_tok}tok "
              f"({results[cat]['tok_per_s']} tok/s)  [realistic, {len(measured)} distinct]",
              file=sys.stderr)
    print(json.dumps({"label": a.label, "results": results}))

if __name__ == "__main__":
    main()
