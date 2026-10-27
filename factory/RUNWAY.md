# RUNWAY — adhan · 2026-10-27 → 2027-03-31

## 1. Milestones

| Date | Milestone | Gate (how we know) | RAG |
|---|---|---|---|
| **31 Oct 2026** | **Tokenizer benchmark public** (yazhi-dev campaign) | `--repro` reproduces the table; whitepaper + post live; numbers consistent across docs | 🟡 |
| 30 Oct 2026 | Kutty v1 defined | ADR merged; backlog matches it | 🔴 (undecided) |
| 7 Nov 2026 | Measured base + clean data | Stage 1b eval committed; packing ≥ 95%; first Aram baseline; INT8 export spike | 🟡 |
| 21 Nov 2026 | Stage 2 midtraining set built and running | 100–150M-token mixture per `CAPABILITIES.md` §5 (≥ 40% general), datasheet, run started on GPU | ⚪ |
| 30 Nov 2026 | SFT v0 + eval suites | Sangam (≥ 5k) + SCERT grounded QA (2k) + conversation (2–3k) sets; `sangam`, `qa_grounded`, `chat` suites in `bench/tamil-bench` with a no-SFT baseline | ⚪ |
| 10 Dec 2026 | **Kutty RC1** | SFT'd checkpoint passes the ship bars in §6 below; INT8 (and INT4 if ready) on the reference phone with tokens/s recorded; Aram 0 CRITICAL | ⚪ |
| 17 Dec 2026 | Kutty served on the platform | Endpoint behind the platform's service template; latency within its SLO | ⚪ |
| 20 Dec 2026 | Model card + datasheet | Limits and known failure modes stated plainly; tokenizer numbers link to the canonical table | ⚪ |
| **26 Dec 2026** | **Adhan Kutty v1 on HuggingFace** + campaign | Release tag; HF weights + tokenizer + card; demo video | ⚪ |
| Q1 2027 | Stage 3 multi-task SFT; DPO only with a verifier; FSDP for mini; Mini tier **gated** on ≥ 1B verified Tamil tokens | See §6 | ⚪ |

## 2. Critical path to 26 Dec

```
Kutty ADR (30 Oct) ──► Stage 1b eval (29 Oct) ─┐
                                               ├─► Stage 2 anneal on GPU (21 Nov → ~1 Dec)
packing fix + repack (5 Nov) ──────────────────┘            │
                                                            ▼
SFT v0 data (Sangam 7 Nov → SCERT/chat 30 Nov) ──► SFT run (1–5 Dec) ──► eval vs ship bars (8 Dec)
                                                            │
export rework: JAX → HF/ONNX → INT8/INT4 (7 Nov spike → 5 Dec) ──► phone demo (10 Dec)
                                                            ▼
                     serve via platform (17 Dec) ──► card (20 Dec) ──► HF release (26 Dec)
```

**Float:** about 5 working days in total. A one-week slip in GPU access, or in the export rework, consumes all of it.
These two are the items to watch every Monday.

## 3. Cross-repo dependencies

| We need | From | By | Status |
|---|---|---|---|
| Tokenizer launch campaign, plus a correction of the "7B" claim on yazhi.dev | yazhi-dev (Media) | 31 Oct | ⚪ |
| English Sangam gold (*eurai*) for `sangam_eurai_en` SFT. Only 32 poems have it today | open-sangam | ongoing | 🟡 |
| Sangam labels + poem JSON (`--open-sangam-root ../open-sangam`) | open-sangam | have | 🟢 |
| GPU host for Stage 2/3 and a serving endpoint for Kutty | yazhi-api (platform) | 14 Nov / 17 Dec | ⚪ |
| `yazh-kutty` base library (shared config/schemas) | yazh-kutty | have | 🟢 |
| Consumers of ONNX/INT8 exports | yazh-unity, illakiya | after RC1 | ⚪ |

## 4. Risk register

L = likelihood, I = impact (1–3). Review every Monday.

| # | Risk | L | I | Mitigation | Trigger to act | Owner |
|---|---|---|---|---|---|---|
| R1 | Kutty decision slips, so the team builds both paths half-way | 2 | 3 | Time-boxed ADR (N2) | No ADR by 30 Oct | Founder |
| R2 | Export/quantize rework is larger than estimated (JAX → ONNX op coverage, RoPE, GQA) | 2 | 3 | INT8 spike first (N9); HF SafeTensors + llama.cpp-class runtime as the fallback route | Spike not working by 7 Nov | Adhan DRI |
| R3 | No GPU in time for Stage 2 | 2 | 3 | T4 recipe as fallback (293.5M tokens in ~3–5 h); book early (N13) | No host by 7 Nov | Adhan DRI |
| R4 | SFT data quality: teacher text that is fluent but wrong | 3 | 2 | Deterministic Sangam pairs first; verifier or human sample on every teacher pair (`CAPABILITIES.md` §8) | Sample error rate > 5% | Adhan DRI |
| R5 | Licence terms on distilled teacher outputs (Gemma/GLM/Kimi) | 2 | 3 | Write down each teacher's output terms before use; prefer deterministic and human data | Before the first distillation run | Adhan DRI |
| R6 | Credibility: inconsistent numbers or inflated public claims ("7B") undercut the benchmark launch | 3 | 2 | N3–N5; every public number links to the canonical table | Any new public post | Adhan DRI + Media |
| R7 | Morphology gate G3 stays failing | 3 | 1 | Decide guard vs v3 after Stage 1b (D3); report it honestly in the card | — | Adhan DRI |
| R8 | Single-box CPU host is shared with cron ingestion (ETA ±1 h) | 2 | 1 | Pause ingestion during runs; snapshot checkpoints before evaluating them | — | Adhan DRI |

## 5. Cut list (in order, if 26 Dec is at risk)

Quality gates never move. Scope does.

1. Drop INT4; ship INT8 only.
2. Drop the phone demo; ship a desktop-CPU demo and keep the phone as a 2027 follow-up.
3. Drop conversation SFT; ship Sangam + grounded-QA skills only, with refusal.
4. Drop `சிந்தனை:` thinking traces. They ship only if they beat the no-thinking baseline anyway.
5. Last resort: release as a **"research preview"**: tokenizer + base checkpoint + model card, with no chat claims.

## 6. Ship bars for Kutty v1 (from `CAPABILITIES.md` §6)

| Suite | Bar |
|---|---|
| `sangam` (200 held-out verses) | tiṇai ≥ 70%, citation ≥ 95%, hallucinated citations < 2% |
| `qa_grounded` (500 + 100 unanswerable) | EM above the closed-book baseline; refusal ≥ 80% on unanswerable |
| `chat` (100 dialogues) | context probe ≥ 80%, Tamil purity ≥ 95% |
| `reason` | ≥ 20 pp over the no-thinking baseline, or thinking is not shipped |
| Aram Harness | 0 CRITICAL, score > 0.90 |

Every result is published next to its no-SFT baseline.

## 7. Beyond launch (Q1 2027)

- Stage 3 multi-task SFT (Sangam 25%, QA 25%, conversation 25%, reasoning 15%, refusal 10%).
- Tokenizer v4: decide on the morphology guard and on chat/think/cite special tokens, together, once, with a repack.
- FSDP for `adhan-mini`. Start Mini only when ≥ 1B verified Tamil tokens exist.
- Dravidian expansion: Kannada / Telugu / Malayalam Layer A (recipe in `archives/2026-10-27-docs-digest.md` §4).
- MLOps gaps (archived status doc, §4): Postgres-backed MLflow, GCS/S3 checkpoints, a model registry.
