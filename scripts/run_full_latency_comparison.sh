#!/usr/bin/env bash
# 표3(1등 기본/split-KV만/양자화만/결합) + 표4(1등/2등/3등) 통합 자동측정.
# "1등 기본" 측정은 한 번만 하고 표3의 '원본' 행과 표4의 '1등' 행 양쪽에 씀
# (같은 세션 안에서 한 번에 다 재니 표4 내부 공정성도 그대로 유지됨).
#
# 설계 원칙은 run_team_comparison.sh와 동일: 한 구성이 실패해도 나머지는
# 계속 진행, trap으로 항상 정리, 재실행마다 타임스탬프로 파일명 분리.
#
# 사용법: ~/a6000_investigation/scripts/run_full_latency_comparison.sh
set -uo pipefail

TS=$(date +%Y%m%d_%H%M%S)
OUTDIR=~/a6000_investigation/team_comparison_results
mkdir -p "$OUTDIR"
LOG="$OUTDIR/full_${TS}.log"
RESULTS="$OUTDIR/full_${TS}_results.jsonl"

WIKITEXT=/home/shared/opt-model/efficient_qwen_competition/latency_eval/real_latency_prompts.json
MATH=~/math_gsm8k_concat_pool.json
CODE=~/code_humaneval_concat_pool.json

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

wait_ready() {
  local url=$1 timeout=${2:-300} waited=0
  while ! curl -sf "$url" > /dev/null 2>&1; do
    sleep 3
    waited=$((waited + 3))
    if [ "$waited" -ge "$timeout" ]; then
      return 1
    fi
  done
  return 0
}

run_client() {
  local label=$1 prompts=$2
  shift 2
  log "  traffic: $label"
  if python3 ~/a6000_investigation/scripts/bench_real_client_patched.py \
      --port 8080 --label "$label" --prompts "$prompts" \
      --warmup 5 --runs 50 "$@" >> "$RESULTS" 2>> "$LOG"; then
    log "  ok: $label"
  else
    log "  FAILED: $label (계속 진행)"
  fi
}

run_domains() {
  # $1 = 라벨 접두어, 나머지 = run_client에 그대로 넘길 추가 인자(--model 등)
  local prefix=$1
  shift
  run_client "${prefix}_wikitext_${TS}" "$WIKITEXT" "$@"
  run_client "${prefix}_math_${TS}" "$MATH" "$@"
  run_client "${prefix}_code_${TS}" "$CODE" "$@"
}

cleanup_all() {
  for name in qwen1st_cmp qwen1st_splitkv_cmp qwen1st_quant_cmp qwen1st_combined_cmp; do
    docker kill -s INT "$name" > /dev/null 2>&1
    docker wait "$name" > /dev/null 2>&1
    docker rm "$name" > /dev/null 2>&1
  done
  docker stop eq2_cmp > /dev/null 2>&1
  docker rm eq2_cmp > /dev/null 2>&1
  if [ -f ~/eqc-3rd-team/cmp_server.pid ]; then
    kill "$(cat ~/eqc-3rd-team/cmp_server.pid)" > /dev/null 2>&1
    rm -f ~/eqc-3rd-team/cmp_server.pid
  fi
  return 0
}
trap cleanup_all EXIT

log "=== 시작 (결과: $RESULTS) ==="
log "사전 정리 중..."
cleanup_all
sleep 2

MY_GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
log "GPU UUID: $MY_GPU_UUID"

# --- [표3 '원본' + 표4 '1등' 겸용] 1등 기본 ---
log "=== 1등 기본(원본) — 표3 '원본' + 표4 '1등' 겸용 ==="
if docker run -d --gpus "\"device=${MY_GPU_UUID}\"" -p 8080:8080 \
    -v ~/eqc-1st-team/model:/opt/ml/model \
    -v ~/eqc-1st-team/draft_model:/opt/ml/draft_model \
    --name qwen1st_cmp qwen35-1st-fast:latest >> "$LOG" 2>&1
then
  if wait_ready "http://localhost:8080/ping" 300; then
    run_domains "1st_baseline"
  else
    log "  1등 기본 서버 준비 타임아웃 — 건너뜀"
  fi
else
  log "  1등 기본 컨테이너 실행 자체가 실패 — 건너뜀"
fi
docker kill -s INT qwen1st_cmp > /dev/null 2>&1
docker wait qwen1st_cmp > /dev/null 2>&1
docker rm qwen1st_cmp > /dev/null 2>&1
sleep 3

# --- [표3] split-KV만 ---
log "=== split-KV만 (표3) ==="
if docker run -d --gpus "\"device=${MY_GPU_UUID}\"" -p 8080:8080 \
    -v ~/eqc-1st-team/model:/opt/ml/model \
    -v ~/eqc-1st-team/draft_model:/opt/ml/draft_model \
    -v ~/dflash_splitkv_patch/vllm-flash-attn-src/vllm_flash_attn/_vllm_fa2_C.abi3.so:/opt/vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so:ro \
    -e EQC_SPLIT_KV_MAX_SEQLEN_Q=16 \
    --name qwen1st_splitkv_cmp qwen35-1st-fast:latest >> "$LOG" 2>&1
then
  if wait_ready "http://localhost:8080/ping" 300; then
    run_domains "splitkv"
  else
    log "  split-KV만 서버 준비 타임아웃 — 건너뜀"
  fi
else
  log "  split-KV만 컨테이너 실행 자체가 실패 — 건너뜀"
fi
docker kill -s INT qwen1st_splitkv_cmp > /dev/null 2>&1
docker wait qwen1st_splitkv_cmp > /dev/null 2>&1
docker rm qwen1st_splitkv_cmp > /dev/null 2>&1
sleep 3

# --- [표3] 양자화만 ---
log "=== 양자화만 (표3) ==="
if docker run -d --gpus "\"device=${MY_GPU_UUID}\"" -p 8080:8080 \
    -v ~/eqc-1st-team/model:/opt/ml/model \
    -v ~/dflash_draft_quant/dflash_w4_gptq:/opt/ml/draft_model \
    -v ~/eqc-3rd-team/src/vllm_plugin:/opt/eqc_vllm_plugin_src \
    -e EQC_DFLASH_QUANT_PATCH=1 \
    --entrypoint bash --name qwen1st_quant_cmp qwen35-1st-fast:latest \
    -c "/opt/venv/bin/pip install -e /opt/eqc_vllm_plugin_src --no-deps && exec python3 /opt/program/serve_vllm.py" >> "$LOG" 2>&1
then
  if wait_ready "http://localhost:8080/ping" 300; then
    run_domains "quant"
  else
    log "  양자화만 서버 준비 타임아웃 — 건너뜀"
  fi
else
  log "  양자화만 컨테이너 실행 자체가 실패 — 건너뜀"
fi
docker kill -s INT qwen1st_quant_cmp > /dev/null 2>&1
docker wait qwen1st_quant_cmp > /dev/null 2>&1
docker rm qwen1st_quant_cmp > /dev/null 2>&1
sleep 3

# --- [표3] 결합(양자화+split-KV) ---
log "=== 결합 (표3) ==="
if docker run -d --gpus "\"device=${MY_GPU_UUID}\"" -p 8080:8080 \
    -v ~/eqc-1st-team/model:/opt/ml/model \
    -v ~/dflash_draft_quant/dflash_w4_gptq:/opt/ml/draft_model \
    -v ~/eqc-3rd-team/src/vllm_plugin:/opt/eqc_vllm_plugin_src \
    -v ~/dflash_splitkv_patch/vllm-flash-attn-src/vllm_flash_attn/_vllm_fa2_C.abi3.so:/opt/vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so:ro \
    -e EQC_DFLASH_QUANT_PATCH=1 -e EQC_SPLIT_KV_MAX_SEQLEN_Q=16 \
    --entrypoint bash --name qwen1st_combined_cmp qwen35-1st-fast:latest \
    -c "/opt/venv/bin/pip install -e /opt/eqc_vllm_plugin_src --no-deps && exec python3 /opt/program/serve_vllm.py" >> "$LOG" 2>&1
then
  if wait_ready "http://localhost:8080/ping" 300; then
    run_domains "combined"
  else
    log "  결합 서버 준비 타임아웃 — 건너뜀"
  fi
else
  log "  결합 컨테이너 실행 자체가 실패 — 건너뜀"
fi
docker kill -s INT qwen1st_combined_cmp > /dev/null 2>&1
docker wait qwen1st_combined_cmp > /dev/null 2>&1
docker rm qwen1st_combined_cmp > /dev/null 2>&1
sleep 3

# --- [표4] 2등팀 ---
log "=== 2등팀 (표4) ==="
if docker run -d --gpus "\"device=${MY_GPU_UUID}\"" -p 8080:8080 \
    --name eq2_cmp efficient-qwen-2nd:latest >> "$LOG" 2>&1
then
  if wait_ready "http://localhost:8080/ping" 300; then
    run_domains "2nd"
  else
    log "  2등팀 서버 준비 타임아웃 — 건너뜀"
  fi
else
  log "  2등팀 컨테이너 실행 자체가 실패 — 건너뜀"
fi
docker stop eq2_cmp > /dev/null 2>&1
docker rm eq2_cmp > /dev/null 2>&1
sleep 3

# --- [표4] 3등팀 (native 프로세스) ---
log "=== 3등팀 (표4) ==="
(
  cd ~/eqc-3rd-team
  export PATH="$PWD/.venv/bin:$PATH"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src EQC_CONFIG_PATH=src/configs/serve_dflash_gptq.yaml \
    EQC_PORT=8080 EQC_VLLM_INTERNAL_PORT=28080 \
    EQC_DFLASH_QUANT_PATCH=1 EQC_DFLASH_SWA_WINDOW=1024 \
    .venv/bin/python -m serving > ~/eqc-3rd-team/cmp_server.log 2>&1 &
  echo $! > ~/eqc-3rd-team/cmp_server.pid
)
if wait_ready "http://localhost:8080/ping" 300; then
  run_domains "3rd" --model "Qwen/Qwen3.5-4B"
else
  log "  3등팀 서버 준비 타임아웃 — 건너뜀 (로그: ~/eqc-3rd-team/cmp_server.log 확인)"
fi
if [ -f ~/eqc-3rd-team/cmp_server.pid ]; then
  kill "$(cat ~/eqc-3rd-team/cmp_server.pid)" > /dev/null 2>&1
  rm -f ~/eqc-3rd-team/cmp_server.pid
fi

log "=== 완료 ==="
log "결과: $RESULTS"
log "로그: $LOG"
