# RUNWAY — adhan · 2026-11-03 → 2027-03-31

> **Plan reset, 20 Nov 2026:** we are replanning from scratch, so every launch and due date in this doc is TBD (replanning).

## 1. Milestones

| Date | Milestone | Gate (how we know) | RAG |
|---|---|---|---|
| **TBD (replanning)** | **Base bake-off decided** | Qwen3-1.7B vs 4B on `xnli`/`milu`/`indicqa` + 20-prompt read; go/no-go recorded | 🟡 |
| **TBD (replanning)** | **SFT v0 + neutral Tanglish eval** | Qwen3-1.7B SFT beats no-SFT on Sangam/QA; 500 neutral Tanglish sentences frozen | ⚪ |
| **TBD (replanning)** | **Distillation data v0** | ≥5k Sarvam-M teacher examples, Tamil-purity filtered + verifier/human sample | ⚪ |
| **TBD (replanning)** | **INT4 phone build + safety gate** | Qwen3-1.7B INT4 GGUF/ONNX runs on reference phone/emulator; Aram 0 CRITICAL | ⚪ |
| **TBD (replanning)** | **SFT v1 / RC candidate** | Sangam tiṇai ≥70%, citation ≥95%, hallucinated citations <2%, grounded QA above baseline | ⚪ |
| **TBD (replanning)** | **Kutty RC1** | Ship bars in §6; INT4 tokens/s + RSS recorded; no CRITICAL | ⚪ |
| **TBD (replanning)** | **Kutty served** | OpenAI-compatible endpoint behind yazhi-api; p50/p95 within SLO | ⚪ |
| **TBD (replanning)** | **Model card + datasheet** | Limits, known failures, tokenizer numbers, teacher/SFT provenance stated | ⚪ |
| **TBD (replanning)** | **Adhan Kutty v1 on HuggingFace** | Release tag; weights + tokenizer + card; demo video | ⚪ |
| TBD (replanning) | Distilled Swaram/Qwen student, larger Tamil data, DPO only with verifier | Separate research gate; not a TBD (replanning) dependency | ⚪ |

## 2. Critical path (launch date TBD (replanning))

```
Base decision (3 Nov) ──► K1 bake-off (TBD (replanning)) ──► K2 SFT v0 (TBD (replanning))
                                                   │
K6 neutral Tanglish eval (TBD (replanning)) ─────────────────┤
                                                   ▼
K3 Sarvam-M distillation (TBD (replanning)) ──► SFT v1 (TBD (replanning)) ──► K7 safety/eval (TBD (replanning))
                                                   │
K5 phone/runtime (TBD (replanning)) ──► K4 INT4 export (TBD (replanning)) ──► phone demo (TBD (replanning))
                                                   ▼
K9 serve via yazhi-api (TBD (replanning)) ──► model card (TBD (replanning)) ──► HF release (TBD (replanning))
```

**Float:** about 5 working days. GPU access and Qwen3 Tamil quality are the two items
that can consume it.

## 3. Cross-repo dependencies

| We need | From | By | Status |
|---|---|---|---|
| GPU host for SFT, distillation, and teacher runs | yazhi-api / infra | TBD (replanning) | ⚪ |
| Reference phone procurement | Yazhi hardware | TBD (replanning) | ⚪ |
| Sangam labels + verse text | open-sangam | have | 🟢 |
| English eurai gold for translation SFT | open-sangam | ongoing | 🟡 |
| Political/abuse review of Tanglish sample | native reviewer | TBD (replanning) | ⚪ |
| Endpoint template for Qwen3 serving | yazhi-api | TBD (replanning) | ⚪ |
| Model card / public claim correction on yazhi.dev | yazhi-dev | TBD (replanning) | 🔴 |

## 4. Risk register

| # | Risk | L | I | Mitigation | Trigger to act | Owner |
|---|---|---|---|---|---|---|
| R1 | **Qwen3-1.7B Tamil quality is not good enough** | 2 | 3 | K1 bake-off; fallback Qwen3-4B, then Gemma-3-4B-IT or negotiated Sarvam-1 | K1 misses the 20-prompt bar | Adhan DRI |
| R2 | **Qwen3 tokenizer tax** — 2.016 tok/akshara vs 0.538 Sarvam-1 | 3 | 2 | Keep answers short; measure INT4 tokens/s; plan a Swaram-distilled student for v2 | Phone tokens/s below target or context thrash | Adhan DRI |
| R3 | **No GPU in time** | 2 | 3 | Rent a T4/A10 weekend; quantised teacher; platform node | No host by TBD (replanning) | Adhan DRI + infra |
| R4 | **Sarvam-M teacher licence/data terms** | 1 | 3 | Apache-2.0 model, but write down output terms; verify + human sample every teacher batch | Before first distillation run | Adhan DRI |
| R5 | **Tanglish source provenance** | 2 | 3 | Raw text local-only; manifest only; gazetteer filter; native review; legal check before public release | Any public post | Adhan DRI |
| R6 | **INT4 export fails on Qwen3** | 2 | 3 | Spike GGUF first; ONNX Runtime Mobile fallback; desktop demo only via cut list | No working INT4 by TBD (replanning) | Adhan DRI |
| R7 | **Safety gate fails** | 2 | 3 | Aram critical-only before release; refusal/citation SFT; cut conversation if needed | Any CRITICAL | Adhan DRI |
| R8 | **Public claim still says 7B/22+** | 2 | 3 | Fix About page and all public copy; link every number to a source | Any new public post | Founder + yazhi-dev |

## 5. Cut list (in order, if the launch date is at risk)

Quality gates never move. Scope does.

1. Drop Qwen3-4B as a second candidate; ship 1.7B only.
2. Drop Sarvam-M distillation; ship SFT-on-Sangam/QA only.
3. Drop INT4 phone demo; ship desktop/server demo and keep phone as 2027.
4. Drop conversation SFT; ship Sangam + grounded-QA only.
5. Drop `சிந்தனை:` traces.
6. Last resort: release as a **research preview** (Qwen3-1.7B SFT + card + eval), no chat claim.

## 6. Ship bars for Kutty v1

| Suite | Bar |
|---|---|
| `sangam` (200 held-out verses) | tiṇai ≥70%, citation ≥95%, hallucinated citations <2% |
| `qa_grounded` (500 + 100 unanswerable) | EM above no-SFT baseline; refusal ≥80% on unanswerable |
| `chat` (100 dialogues) | context probe ≥80%, Tamil purity ≥95% |
| `reason` | ≥20 pp over no-thinking baseline, or thinking is not shipped |
| Aram Harness | 0 CRITICAL, score >0.90 |
| Phone INT4 | tokens/s + RSS recorded on the reference device; no OOM |

Every result is published next to its no-SFT baseline.

## 7. Beyond launch (Q1 2027)

- Distil Sarvam-M into a smaller Swaram-tokenized student for on-device efficiency.
- Revisit Sarvam-1 only with a commercial licence.
- Dravidian expansion only after Tamil ships and ≥1B verified pure-Tamil tokens exist.
- DPO only with a verifier; no preference data without a licence review.
- Keep `adhan-nano` + Swaram as the research paper/ablation line.
