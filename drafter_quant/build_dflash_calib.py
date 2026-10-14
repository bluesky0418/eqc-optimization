#!/usr/bin/env python3
"""sample_nemotron.py가 뽑은 프롬프트에 1등팀 실제 모델(qwen35-1st-fast)로 응답을
생성시켜서, quant_dflash_gptq.py가 요구하는 최소 스키마
{"rendered_full": "...", "prompt_n_tokens": N} 형식으로 저장한다.

주의: quant_dflash_gptq.py는 이 두 필드만 읽음(rendered_full 전체 토큰화 +
prompt_n_tokens로 프롬프트/응답 경계 구분) — 그 이상은 필요 없음.

사용법:
  ~/eqc-3rd-team/.venv/bin/python ~/build_dflash_calib.py \
      --prompts ~/nemotron_calib_prompts.jsonl \
      --port 8080 --out ~/dflash_calib.jsonl \
      --chat-template ~/eqc-1st-team/model/chat_template.jinja \
      --tokenizer ~/qwen-weights \
      --max-tokens 512 --limit 360
"""
import argparse
import json
import sys
import time
import urllib.request


def render_prompt(tok, messages, chat_template):
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=chat_template,
    )


def generate(base, prompt, max_tokens, model="default", temperature=0.0):
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{base}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return d["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True,
                     help="sample_nemotron.py 출력 (prompts.jsonl)")
    ap.add_argument("--port", type=int, required=True,
                     help="1등팀 서버 포트 (qwen35-1st-fast)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chat-template", required=True,
                     help="1등팀 모델의 chat_template.jinja 경로")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B",
                     help="HF 토크나이저 경로/이름 (기본: 순정 원본)")
    ap.add_argument("--max-tokens", type=int, default=512,
                     help="캘리브레이션용 응답 생성 길이 (앵커 다양성 확보용)")
    ap.add_argument("--limit", type=int, default=0,
                     help="0이면 전체, >0이면 앞에서 N개만 처리")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    chat_template = open(a.chat_template).read()

    base = f"http://127.0.0.1:{a.port}"

    lines = [json.loads(l) for l in open(a.prompts)]
    if a.limit > 0:
        lines = lines[:a.limit]

    n_ok, n_err = 0, 0
    t0 = time.time()
    with open(a.out, "w") as fout:
        for i, rec in enumerate(lines):
            messages = rec["conversations"]
            try:
                rendered_prompt = render_prompt(tok, messages, chat_template)
                prompt_n_tokens = len(
                    tok(rendered_prompt, add_special_tokens=False).input_ids
                )
                completion = generate(base, rendered_prompt, a.max_tokens)
                rendered_full = rendered_prompt + completion
                fout.write(json.dumps({
                    "rendered_full": rendered_full,
                    "prompt_n_tokens": prompt_n_tokens,
                }) + "\n")
                fout.flush()
                n_ok += 1
            except Exception as e:
                print(f"[{i}] skip: {e}", file=sys.stderr)
                n_err += 1
            if (i + 1) % 20 == 0:
                dt = time.time() - t0
                print(f"  {i+1}/{len(lines)}  ok={n_ok} err={n_err}  "
                      f"dt={dt:.0f}s", file=sys.stderr, flush=True)

    print(f"done: {n_ok} ok, {n_err} err -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
