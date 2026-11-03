# CURRENT — adhan · 2026-11-03

> Reset every Monday. Facts only; each number cites its source. The template and rules are in [`README.md`](README.md).

## 1. Headline

**🟡 Execution reset — base decided, now prove it.** On 3 Nov the base question closed:
Kutty v1 is a fine-tune of **`Qwen/Qwen3-1.7B-Base`** (Apache-2.0), with
**`sarvamai/sarvam-m` 24B** (Apache-2.0) as the server-side teacher/reference.
`Sarvam-1` 2B is rejected for product use because its licence is non-commercial/research
only. `adhan-nano` + Swaram continue as the research artifact, not the product base.
Full record: [`docs/reports/base_decision_2026-11-03.md`](../docs/reports/base_decision_2026-11-03.md).

The critical risk is now **Qwen3's Tamil tokenizer tax**: 2.016 tok/akshara vs 0.538 for
Sarvam-1 on the same 300-doc sample. The 48-hour bake-off (`K1`) decides whether Qwen3-1.7B
is good enough on Tamil quality, or whether we move to Qwen3-4B / a negotiated Sarvam-1
licence. The other hard blocker is **GPU access** for SFT, distillation, and the teacher.

## 2. Health

| Workstream | RAG | Fact (source) | Next checkpoint |
|---|---|---|---|
| Base decision | 🟢 | **Qwen3-1.7B-Base student · Sarvam-M 24B teacher · Sarvam-1 rejected** (`docs/reports/base_decision_2026-11-03.md`) | Bake-off result K1 |
| Base bake-off | 🟡 | Preliminary CPU smoke: Qwen3-1.7B XNLI **42.0%** (n=50), IndicQA **EM 0.140/F1 0.192** (n=50); 4B/full-GPU run pending (`docs/reports/base_bakeoff_qwen3.md`) | K1 full run, due 5 Nov |
| SFT v0 | ⚪ | Data ready: Sangam 9,961 train / 1,303 val / 1,146 test, zero poem overlap; grounded QA 971/124/182. No Qwen3 SFT run yet | K2, due 10 Nov |
| Distillation | ⚪ | Sarvam-M 24B teacher not yet run; no Tamil prompt/verifier pipeline | K3, due 14 Nov |
| Export / phone | 🔴 | No Qwen3 INT4 GGUF/ONNX build; reference phone not chosen | K4/K5, due 21 Nov |
| Tanglish sanitization | 🟡 | Source scoped: `vishnu-n/Tanglish-Corpus-185k` @ `927f5b4` (CC-BY-4.0); 5.5% keyword-political, 171k neutral candidates. Gazetteer-first plan | K6, due 10 Nov |
| Safety (Aram) | 🟡 | Stage 1b baseline 0.13, 7 CRITICAL/9 (`docs/reports/aram_report_stage1b.json`); Qwen3 untested | K7, due 21 Nov |
| Serving | 🟢 | Adhan FastAPI validated against Stage 1b: `/generate` p50 1,757 ms / p95 2,540 ms CPU (`docs/reports/serving_stage1b.json`) | Re-test with Qwen3 endpoint |
| Research track | 🟡 | Swaram A/B: 0.527 vs Tamil BPE 0.346 → compression thesis unsupported; morphology 5.6–24.2%, gate 30% | Keep as paper, not product gate |
| CI | 🟢 | 127 passed / 2 skipped (3 Nov); structure-invariant + remat tests added | Keep green |

## 3. Key metrics

| Metric | Now | Target | Source |
|---|---|---|---|
| Kutty v1 product base | **Qwen3-1.7B-Base** (Apache-2.0) | bake-off confirms Tamil quality | `docs/reports/base_decision_2026-11-03.md` |
| Teacher / reference | **Sarvam-M 24B** (Apache-2.0) | distillation data + eval ceiling | same |
| Tokenizer fertility, Qwen3 | **2.016 tok/akshara** | measure after SFT; no regression | 300-doc sample, 35,986 aksharas |
| Tokenizer fertility, Sarvam-1 | 0.538 | — (product-rejected) | same sample |
| Tokenizer fertility, Sarvam-M | 0.804 | teacher efficiency | same sample |
| Tokenizer fertility, Swaram v2 | 0.527 | research artifact | `docs/reports/ab_swaram_vs_bpe.json` |
| Tamil-trained BPE fertility | 0.346 | research baseline | same |
| SFT examples | 11,232 (9,961 + 1,271) | 20–50k | `data/final/sft_*/stats.json` |
| bits/akshara (Adhan Stage 1b) | 1.820 | research only | `docs/reports/eval_report_nano_stage1b.json` |
| Serving `/generate` latency | p50 1,757 ms · p95 2,540 ms (Adhan CPU) | Qwen3 INT4 on phone | `docs/reports/serving_stage1b.json` |
| Aram Harness score | 0.13 · 7 CRITICAL (Adhan Stage 1b) | >0.90, 0 CRITICAL for Qwen3 | `docs/reports/aram_report_stage1b.json` |
| Tanglish neutral candidates | 171,449 (after keyword filter) | freeze 500 | `docs/reports/n3_tanglish_source_scope.md` |

## 4. In flight

| Change | What | Age | State | Needed to land |
|---|---|---|---|---|
| K1 base bake-off | Qwen3-1.7B vs Qwen3-4B on tamil-bench + 20-prompt read | new | partial CPU smoke only | GPU/full n + Qwen3-4B + generation read |
| K2 SFT v0 | Qwen3-1.7B + Sangam/QA SFT | new | not started | GPU window + regression gate |
| K3 distillation | Sarvam-M teacher prompts/filter | new | not started | GPU + Tamil verifier |
| K6 Tanglish freeze | gazetteer sanitizer + 500 sentences | new | not started | filter code + native review |

## 5. Blockers (oldest first)

| # | Blocker | Since | Blocks | Ask / who |
|---|---|---|---|---|
| B1 | **No GPU.** SFT, distillation, and Qwen3-4B/24B teacher runs need one | ongoing | K1–K3, 26 Dec | Book a T4/A10 weekend or platform GPU node (`K1`) |
| B2 | **Qwen3 Tamil quality unknown** and its tokenizer is 3.75× less efficient than Sarvam-1 | 3 Nov | product base confirmation | Run K1 bake-off now |
| B3 | **Export path for Qwen3 → INT4 GGUF/ONNX** is untested | 3 Nov | phone demo K4/K5 | Spike export on Qwen3-1.7B |
| B4 | **Tanglish political filter** has no human-labeled audit | 3 Nov | public eval table K6 | Gazetteer pass + 100-sentence native review |
| B5 | **Sarvam-M teacher access** (24B) not booked | 3 Nov | distillation K3 | GPU with ≥48 GB or quantised teacher |

## 6. Decisions needed

| # | Decision | Options | Recommendation | Decider | By |
|---|---|---|---|---|---|
| D1 | Kutty v1 base | Adhan-nano · Sarvam-1 · Qwen3-1.7B · Qwen3-4B | **Qwen3-1.7B-Base** chosen 3 Nov; revisit only if K1 fails | Founder | **recorded** |
| D2 | If Qwen3-1.7B Tamil quality fails K1 | Qwen3-4B · Gemma-3-4B-IT · negotiated Sarvam-1 commercial licence | Prefer Qwen3-4B first; do not silently change licence | Founder | 5 Nov |
| D3 | Teacher data policy | Sarvam-M outputs + human filter · Qwen3-4B outputs · no distillation | Sarvam-M + verifier/human sample; record licence/terms | Adhan DRI | 7 Nov |
| D4 | Reference phone + runtime | Android ₹10–15k · emulator · server demo | Phone + llama.cpp/ONNX; fall back to desktop demo only via cut list | Adhan DRI | 7 Nov |

## 7. Done last iteration

- **3 Nov — base decision recorded.** Qwen3-1.7B student, Sarvam-M teacher, Sarvam-1 rejected; tokenizer fertility evidence and fallback documented (`docs/reports/base_decision_2026-11-03.md`).
- **3 Nov — MuRIL assessed.** Encoder-only, not a base; frozen XNLI 42.77%, Tanglish political weak-label CV 67.75%/AUC 0.754 → gazetteer-first sanitization (`docs/reports/muril_base_assessment.md`).
- **3 Nov — factory reset** around the new execution plan; old N1–N17 backlog removed.
- Earlier: Stage 1b eval 1.820 bits/akshara; packing fix 99.9997%; SFT v0 Sangam set; serving validated; first Aram baseline; structure-invariant CI.
