# Blocked-snippet diagnosis (Tier 1)

생성: 2026-04-30 (8 OLMo 2 모델 × standard config, Phase 2 결과 9555 spans / 195 records)
API call 사용: **54 / 300 budget**

## TL;DR

기존에 "blocked"로 추정했던 silent-drop 버그의 **진짜 원인은 저작권 정책이 아니라 API의 spans 배열 quirk** — `get_doc_by_rank`가 `spans=[[<str>, None]]` 형태를 반환할 때 `"".join(...)`이 TypeError로 죽고 outer `except Exception: continue`가 흡수. probe 20 spans 통틀어 **`blocked=true` 단 한 건도 없음**. None-spans doc은 dclm/pes2o (Common Crawl/S2) 등 일반 문서.

**failure mode 두 가지**:
- (A) 자연 rare span — 1B/7B/13B(±instruct)에서 보이는 undershoot의 거의 전부 (probe 12개 모두 `true_count==stored`).
- (B) None-spans silent drop — 32B/32B-instruct의 ~99% zero-snippet의 원인. probe 4 spans × 5 docs = 19개 docs **전원 `blocked=False, none_count>0`**.

**escalation**: Tier 2/3 불필요. None-safe patch + 32B 두 모델만 재실행하면 됨.

---

## 1. 기존 결과 통계 (zero API)

### 1.1 모델별 undershoot (`num_snippets < 10`) — lower bound

| Model | Records (Phase-2) | Total spans | Undershoot | Undershoot % | Zero | Zero % |
|---|---:|---:|---:|---:|---:|---:|
| olmo2_1b | 25/25 | 1364 | 745 | 54.62% | 0 | 0.00% |
| olmo2_7b | 38/38 | 2038 | 1366 | 67.03% | 0 | 0.00% |
| olmo2_13b | 45/45 | 2168 | 1266 | 58.39% | 0 | 0.00% |
| **olmo2_32b** | **39/39** | **1981** | **1981** | **100.00%** | **1977** | **99.80%** |
| olmo2_1b_instruct | 16/16 | 666 | 452 | 67.87% | 0 | 0.00% |
| olmo2_7b_instruct | 6/6 | 274 | 132 | 48.18% | 0 | 0.00% |
| olmo2_13b_instruct | 8/8 | 343 | 197 | 57.43% | 0 | 0.00% |
| **olmo2_32b_instruct** | **18/18** | **721** | **720** | **99.86%** | **718** | **99.58%** |
| **AGGREGATE** |  | 9555 | 6859 | **71.78%** | 2695 | **28.21%** |
| **AGGREGATE \\ 32B** |  | 6853 | 4158 | **60.67%** | 0 | **0.00%** |

**관찰**: zero-snippet의 99.96% (2695/2696)가 32B 두 모델에 집중. 다른 6 모델은 zero-snippet 0건. 이건 32B에서 별개의 실패 모드가 있었음을 의미.

### 1.2 num_snippets 히스토그램 (aggregate)

| n | count | 비고 |
|---:|---:|---|
| 0 | 2695 | 거의 전부 32B |
| 1 | 1348 | 자연 rare (대부분 1B/7B/13B) |
| 2 | 667 |  |
| 3 | 503 |  |
| 4 | 428 |  |
| 5 | 321 |  |
| 6 | 234 |  |
| 7 | 265 |  |
| 8 | 210 |  |
| 9 | 188 |  |
| **10** | **2696** | full retrieval (성공) |

이중-모드 분포 (peaks at 0 and 10) — 0은 32B 실패 모드, 10은 corpus 풍부 매치 성공.

### 1.3 영향받은 records — 모델별 분포

| Model | Total | Phase-2 | 영향 없음 | 부분 undershoot | 전체 undershoot | 영향 % |
|---|---:|---:|---:|---:|---:|---:|
| olmo2_1b | 25 | 25 | 2 | 17 | 6 | 92.0% |
| olmo2_7b | 38 | 38 | 2 | 29 | 7 | 94.7% |
| olmo2_13b | 45 | 45 | 2 | 39 | 4 | 95.6% |
| **olmo2_32b** | **39** | **39** | **0** | **0** | **39** | **100.0%** |
| olmo2_1b_instruct | 16 | 16 | 0 | 16 | 0 | 100.0% |
| olmo2_7b_instruct | 6 | 6 | 0 | 6 | 0 | 100.0% |
| olmo2_13b_instruct | 8 | 8 | 0 | 8 | 0 | 100.0% |
| **olmo2_32b_instruct** | **18** | **18** | **0** | **1** | **17** | **100.0%** |
| AGGREGATE | 195 | 195 | 6 | 116 | 73 | **96.9%** |

대부분 records가 영향받지만, **base 6 모델에서는 record당 평균 1-2개 span만 undershoot이고 stored>0인 spans가 다수**. 즉 영향이 결과적으로 거의 없음.

---

## 2. Probe 결과 (54 API calls, 20 spans, ≤5 docs/span)

### 2.1 모델별 probe 결과

| # | Model | rec_id | span | gap | true_cnt | stored | n_probed | n_blocked | n_None |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | olmo2_1b | 11 | 0 | 9 | 1 | 1 | 0 | 0 | 0 |
| 2 | olmo2_7b | 3 | 1 | 9 | 1 | 1 | 0 | 0 | 0 |
| 3 | olmo2_13b | 0 | 5 | 9 | 1 | 1 | 0 | 0 | 0 |
| **4** | **olmo2_32b** | **0** | **1** | **10** | **2** | **0** | **2** | **0** | **2** |
| 5 | olmo2_1b_instruct | 11 | 4 | 9 | 1 | 1 | 0 | 0 | 0 |
| 6 | olmo2_7b_instruct | 30 | 2 | 9 | 1 | 1 | 0 | 0 | 0 |
| 7 | olmo2_13b_instruct | 30 | 10 | 9 | 1 | 1 | 0 | 0 | 0 |
| **8** | **olmo2_32b_instruct** | **30** | **0** | **10** | **13** | **0** | **5** | **0** | **5** |
| 9 | olmo2_1b | 11 | 1 | 9 | 1 | 1 | 0 | 0 | 0 |
| 10 | olmo2_7b | 3 | 11 | 9 | 1 | 1 | 0 | 0 | 0 |
| 11 | olmo2_13b | 0 | 6 | 9 | 1 | 1 | 0 | 0 | 0 |
| **12** | **olmo2_32b** | **0** | **2** | **10** | **2** | **0** | **2** | **0** | **2** |
| 13 | olmo2_1b_instruct | 11 | 7 | 9 | 1 | 1 | 0 | 0 | 0 |
| 14 | olmo2_7b_instruct | 30 | 9 | 9 | 1 | 1 | 0 | 0 | 0 |
| 15 | olmo2_13b_instruct | 30 | 15 | 9 | 1 | 1 | 0 | 0 | 0 |
| **16** | **olmo2_32b_instruct** | **30** | **1** | **10** | **9** | **0** | **5** | **0** | **5** |
| 17 | olmo2_1b | 11 | 7 | 9 | 1 | 1 | 0 | 0 | 0 |
| 18 | olmo2_7b | 3 | 15 | 9 | 1 | 1 | 0 | 0 | 0 |
| 19 | olmo2_13b | 0 | 9 | 9 | 1 | 1 | 0 | 0 | 0 |
| **20** | **olmo2_32b** | **0** | **3** | **10** | **27** | **0** | **5** | **0** | **5** |

aggregate: probed=19 docs, **blocked=0 (0%)**, **None>0=19 (100%)**.

### 2.2 패턴 해석

- **non-32B 12개 spans**: 모두 `true_count == stored` (1=1). 즉 corpus에 정말로 1개 매치 → 1개 retrieve 성공. **gap=9는 자연 현상이지 실패가 아님**. 이미 저장된 doc은 정상 (probe 안 해도 됨).
- **32B 4개 spans**: corpus에 매치 존재 (`true_count={2, 2, 9, 13, 27}`)인데 stored=0. 신선하게 retrieve 시도한 19개 doc **모두 `blocked=False`이고 `spans` 배열에 None entry 보유**. 즉 None-spans 문제로 인한 silent drop이 32B에서만 빈번 발생.

### 2.3 None-spans doc의 metadata 패턴

probe된 19개 None-spans doc의 source/path:

| path prefix | source | count |
|---|---|---:|
| `dclm-*.json.zst` | `dclm-hero-run-fasttext_for_HF` (Common Crawl) | 18 |
| `pes2o-*.json.gz` | `s2` (Semantic Scholar) | 1 |

**모두 일반 공개 문서 (Common Crawl, S2 academic)** — 저작권 dataset (`books-`, `lyrics-` 등) 마커 전무. **blocking 정책으로 설명되지 않음**. 모든 None-spans doc은 `spans_total=2, none_count=1` — 즉 `spans[0] = [<matched_text>, None]` 형태. API가 짧은 doc (예: `doc_len=219`) 또는 `max_disp_len=80`보다 가까운 boundary에서 두 번째 segment를 None으로 반환하는 동작 quirk로 추정.

### 2.4 왜 32B만 영향받았나?

직접 확인 못 했지만 정황 추론:
- 32B 두 모델은 39+18=57 records × 평균 ~50 spans ≈ ~2700 retrieve 시도. 다른 모델보다 큰 단일 batch.
- 32B 응답이 더 짧고 dense하게 verbatim span을 만들기 쉬워서 (Methanol/Mercury chloride 같이 짧고 정확한 named entity), 매칭 doc도 짧은 게 많이 잡혀 None-spans quirk 발생률이 높음.
- 1B/7B/13B는 spans 자체가 더 길고 다양해서 None-spans 케이스를 적게 만나 noticeable한 손실 없이 stored>0 유지 — 그래서 zero-snippet 0건.

---

## 3. 사용자 4개 deliverable

### (1) `num_snippets < 10`인 span의 % (lower-bound blocked rate)

- **Aggregate**: 71.78% (6859/9555)
- **Excl. 32B**: 60.67% (4158/6853) — 자연 rare 압도적
- **32B만**: ~100%
- **`num_snippets == 0`**: aggregate 28.21% — 99.96% (2695/2696)이 32B에 집중
- **올바른 lower-bound 해석**: 저작권/blocked rate ≈ **0%** (probe로 확인). undershoot의 절대다수는 (A) 자연 rare 또는 (B) None-spans quirk.

### (2) 영향받은 record 수와 모델별 분포

| Model class | Affected records | 영향 |
|---|---:|---|
| 1B/7B/13B base | 17 (전체 undershoot) + 85 (부분) | partial이 dominant — 실질 영향 미미 |
| **32B base** | **39/39 (전체)** | **데이터 거의 전체 미사용 가능** |
| 1B/7B/13B instruct | 0 (전체) + 30 (부분) | partial only |
| **32B-instruct** | **17/18 (전체)** | **데이터 거의 전체 미사용 가능** |

**실질적으로 재처리가 필요한 모델**: **olmo2_32b, olmo2_32b_instruct 두 모델만**. 다른 6 모델은 patch 적용 후 자연스레 (None-safe join으로) 거의 동일한 결과 유지.

### (3) 20개 probe span의 raw API response에서 발견된 blocked metadata 패턴

- `blocked=true` 0건. 즉 **저작권 dataset 트리거 패턴 발견 안 됨**.
- 19개 None-spans doc 출처: dclm-*.json.zst 18개, pes2o-*.json.gz 1개.  source: `dclm-hero-run-fasttext_for_HF` 18, `s2` 1. 모두 일반 공개 문서.
- 결론: **API의 `blocked` 정책은 본 데이터에서 거의 트리거되지 않음**. 우리가 본 silent drop은 blocked가 아니라 **None-spans quirk**.

### (4) Tier 2/3 escalation 판단

- **escalation 불필요**. 진단된 두 가지 failure mode (A: 자연 rare, B: None-spans)는 모두 Tier 1 데이터로 충분히 분리됨.
- **재실행 필요한 모델은 olmo2_32b / olmo2_32b_instruct 두 개뿐**. 다른 6개 모델은 그대로 둬도 됨.
- patch 검증: prompt-level `--test --record_id 30`이 None-spans 케이스를 안정적으로 재현하므로 patch 효과를 즉시 확인 가능.

---

## 4. 패치 plan (확정)

### 4.1 변경 위치

`e1_verbatim_trace.py:retrieve_snippets_for_span` (lines 370-443) 의 API mode 분기 (lines 416-422). `e1_retrieve_snippets.py`는 동일 helper를 import하므로 자동 수혜.

### 4.2 변경 내용 (우선순위 순)

1. **None-safe join + warning** (dominant 케이스):
   - `parts = spans_data[0]`에 `None`이 섞여 있어도 `"".join(t if t is not None else "" for t in parts)`로 sanitize.
   - non-blocked doc의 경우 `logger.warning("doc_ix=%s has %d None entries in spans[0]", ...)` 로깅.
2. **`blocked` 필드 unconditional 보존** (defensive, 향후 분석 가능):
   - `snippet_info["blocked"] = bool(doc.get("blocked", False))` 추가.
3. **blocked-aware path** (defensive, 미래 대비):
   - `is_blocked == True`이면 marker entry로 보존 (`snippet_text=""`, `snippet_token_ids=[]`) — silent drop 금지.
   - 본 진단에서 이 경로는 0건 트리거됐지만 defense in depth.
4. **logger 접근**:
   - 시그니처 변경 없이 module-level `logger = logging.getLogger(__name__)` 사용 (호출처 변경 불필요).

### 4.3 backward compatibility

- `blocked` 키는 missing 시 False로 간주 — 기존 결과 파일 그대로 유효.
- `e1_retrieve_snippets.py:206-209` resume 조건 (`len(filled) >= len(top_k_spans)`)은 그대로. 새로 retrieve된 doc만 blocked 마커가 들어가므로 6개 모델 결과는 기존과 동일.
- `analysis/nb_utils.py`는 `snip.get('snippet_text', '')`로 안전 접근하므로 문제 없음.

### 4.4 패치 후 재실행

- **olmo2_32b, olmo2_32b_instruct**: standard config Phase 2 재실행. 기존 결과의 `num_snippets=0` span만 재시도되도록 resume 로직 활용 (e1_retrieve_snippets.py:206-209). API call 추정: zero spans 약 2695 × 평균 5 = ~13K calls (rate-limit 회복 가정). 별도 작업으로 분리 권장.
- **6개 다른 모델**: 재실행 불필요.

### 4.5 검증

```
python3 e1_verbatim_trace_prompt.py --config standard \
    --api_index v4_olmo-mix-1124_llama \
    --test --record_id 30 --retrieve_snippets
```

기대: 두 top-K span 모두 num_snippets > 0 (정확히는 corpus count up to 10, max 5). warning log에 "None entries in spans[0]" 메시지 출현. blocked entry는 0건이지만 패치 정상 동작 확인.

---

## Appendix: data 위치

- 자세한 stats: `analysis/blocked_diagnosis/analysis.json`
- probe raw responses: `analysis/blocked_diagnosis/probe_raw_responses.json`
- log: `analysis/blocked_diagnosis/diagnose_blocked_snippets.log`
