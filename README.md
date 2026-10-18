# EQC (Efficient Qwen Challenge) 최적화 기록

AdaptFM Efficient Qwen Challenge(ICML'26)의 1등팀 솔루션을 재현하고, 그 위에 추가
최적화(split-KV 그리드 병렬화, 드래프터 양자화 이식)를 직접 만들어 붙인 기록입니다.

대회는 A10G(24GB)에서 공식 hidden-set 기준으로 채점됐고 1등팀의 공식 발표 가속배수는
7.745x입니다. 이 레포의 수치는 그것과 직접 비교 가능한 게 아니라, **로컬 RTX 3090에서
1등팀 솔루션을 직접 재현한 뒤 그 재현 결과를 베이스라인 삼아 측정한 것**입니다(GPU가
다르고, 평가 콘텐츠도 대회가 참가자 편의용으로 제공한 filler 기반 테스트 스크립트 대신
실제 텍스트로 측정했습니다 — `WRITEUP.md` 2절 참고). 이 기준으로 1등팀 재현 베이스
3.667x를 **4.097x**(약 11.7% 개선)까지 끌어올렸습니다. 대회 hidden-set 자체를 그대로
재현할 수 없는 상황에서, 적어도 로컬 안에서는 일관된 기준으로 공정하게 비교할 수 있는
평가 방법론을 직접 세운 것도 이 작업의 일부입니다.

이 과정에서 찾은 버그는 실제로 FlashAttention2 업스트림 라이브러리 자체의 버그였다는 걸
확인해 [vllm-project/flash-attention#201](https://github.com/vllm-project/flash-attention/pull/201)로
제출했습니다. 이후 같은 버그를 고치는 PR(#176, #190)이 먼저 올라와 있음을 확인해
중복으로 닫았고, 해당 수정은 현재 그쪽에서 논의 중입니다.

**전체 과정(동기 → 방법론 → 두 최적화 → 디버깅 → 업스트림 PR 제출)은 [WRITEUP.md](./WRITEUP.md)에
정리돼 있습니다.** 이 README는 결과/파일 구성 요약입니다.

---

## 폴더 구성

```
├── README.md                          (이 파일)
├── WRITEUP.md                          전체 과정 기록
├── scripts/
│   ├── bench_real_client_patched.py    레이턴시 측정 도구
│   ├── run_full_latency_comparison.sh  레이턴시 측정 오케스트레이터
│   └── run_eval_local_official.py      품질평가(MMLU-Pro/IFEval/GPQA) 하니스
├── patch/
│   ├── flash_api.cpp.diff              split-KV 활성화 조건 확장(로컬 패치 버전)
│   └── flash_fwd_kernel.h.diff         batch>1 정확성 버그 근본 수정
└── drafter_quant/
    ├── build_dflash_calib.py           캘리브레이션 데이터 생성 스크립트
    ├── dflash_calib.jsonl              위 스크립트로 만든 캘리브레이션 데이터
    ├── config.json                     실제 적용된 양자화 설정
    └── gptq_calib_meta.json            캘리브레이션 메타데이터
```

## 파일별 설명

### `scripts/`
- **`bench_real_client_patched.py`**: 대회 공식 레이턴시 측정 스크립트를 그대로 가져오되,
  `--model` 플래그 한 줄만 추가(원본은 `"model":"default"` 고정이라 served_model_name이
  다른 서버에서 404가 남). 매 요청마다 실제 텍스트(반복 filler 아님)로 5 warmup + 50
  measurement, median 집계 — 대회 공식 프로토콜 그대로.
- **`run_full_latency_comparison.sh`**: 위 도구를 여러 구성(1등 기본/split-KV만/양자화만/
  결합/2등/3등)×3도메인(자연어/수학/코드)에 대해 자동으로 순회 실행. 순수 bash+docker라
  SLURM 등 특정 환경 없이 어디서든 실행 가능.
- **`run_eval_local_official.py`**: 대회 공식 로컬 품질평가 하니스(MMLU-Pro/IFEval/
  GPQA-Diamond). `EVAL_MODE=quality|latency|full` 환경변수로 제어, 로컬 docker
  컨테이너만 있으면 실행 가능.

### `patch/` — 이번 프로젝트의 핵심 코드 산출물
FlashAttention2(`vllm-flash-attn`)에 적용한 로컬 패치. 원본 대비 주석 제외 22줄
(`flash_api.cpp` 18줄 + `flash_fwd_kernel.h` 4줄)로 long 버킷 기준 어텐션 커널시간
-74.8% 개선. 이 버그를 업스트림(`vllm-project/flash-attention`)에도 맞게 다듬어 제출한
버전은 [PR #201](https://github.com/vllm-project/flash-attention/pull/201) 참고. 먼저
올라온 #176·#190과 중복되어 닫았습니다(자세한 경위는 `WRITEUP.md` 8절).
- **`flash_api.cpp.diff`**: split-KV 활성화 조건을 DFlash 패턴(`max_seqlen_q≤16`)까지
  확장(환경변수 `EQC_SPLIT_KV_MAX_SEQLEN_Q`로 게이팅, 기본값이면 원본과 100% 동일 동작).
- **`flash_fwd_kernel.h.diff`**: split-KV 활성화 직후 발견된 batch>1 정확성 버그(varlen
  API에서 배치별 출력 오프셋 계산에 `cu_seqlens_q` 대신 옛날 고정shape 공식을 쓰고
  있던 버그) 근본 수정.

### `drafter_quant/`
1등팀은 드래프터를 양자화하지 않았는데, 3등팀이 만든 GPTQ 도구(드래프터가 표준 토큰
입력이 아니라 타깃 모델의 중간 히든스테이트를 입력으로 받는 특수 구조라 표준 양자화
도구를 못 쓰고 3등팀이 직접 만든 것)를 1등팀 드래프터에 적용. **캘리브레이션 데이터는
3등팀 것을 재사용하지 않고, 1등팀의 실제 서버를 직접 호출해서 저희가 새로 만들었습니다**
(`build_dflash_calib.py`). 라운드당 순수 계산시간 -41.63%(3306.63us→1929.92us) 개선.

---

## 결과 요약

| 구성 | 3도메인 평균 가속배수 |
|---|---:|
| 1등팀 기본(원본) | 3.667x |
| + split-KV만 | 3.918x |
| + 드래프터 양자화만 | 3.923x |
| **+ 둘 다(최종)** | **4.097x** |

품질 게이트(MMLU-Pro/GPQA-Diamond/IFEval) 전부 로컬 재보정 기준 PASS.

---

## Acknowledgments

이 프로젝트는 아래 두 공개 저장소(둘 다 Apache-2.0)의 솔루션을 재현하고 그 위에
최적화를 더했습니다. 코드는 재배포하지 않고, 설명/링크로 출처를 표기합니다.

- 1등팀: [sai-samarth/qwen35-4b-fast-inference](https://github.com/sai-samarth/qwen35-4b-fast-inference) — 재현 베이스(GPTQ, untied lm_head, DFlash 통합)
- 3등팀: [nota-github/adaptfm-quant-dflash](https://github.com/nota-github/adaptfm-quant-dflash) — 드래프터 GPTQ 캘리브레이션 도구 참고

`patch/`의 split-KV 수정은 전부 독자 발견/구현이며, 이후 동일 버그가
[Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)과
[vllm-project/flash-attention](https://github.com/vllm-project/flash-attention)에도
있음을 확인해 [#201](https://github.com/vllm-project/flash-attention/pull/201)로 제출했습니다
(중복으로 닫힘, #176·#190 참고 — 경위는 `WRITEUP.md` 참고).
