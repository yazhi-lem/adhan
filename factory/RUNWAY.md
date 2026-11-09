# RUNWAY — adhan · 2026-11-03 → 2027-03-31

## 1. Milestones

| Date | Milestone | Gate (how we know) | RAG |
|---|---|---|---|
| **5 Nov 2026** | **Base bake-off decided** | Qwen3-1.7B vs 4B on `xnli`/`milu`/`indicqa` + 20-prompt read; go/no-go recorded | 🟡 |
| **10 Nov 2026** | **SFT v0 + neutral Tanglish eval** | Qwen3-1.7B SFT beats no-SFT on Sangam/QA; 500 neutral Tanglish sentences frozen; **the SFT/pretraining mix carries a recorded leak-removal count** (tooling: `adhan-decontaminate`, landed 5 Nov) | ⚪ |
| **14 Nov 2026** | **Distillation data v0** | ≥5k Sarvam-M teacher examples, Tamil-purity filtered + verifier/human sample | ⚪ |
| **21 Nov 2026** | **INT4 phone build + safety gate** | Qwen3-1.7B INT4 GGUF/ONNX runs on reference phone/emulator; Aram 0 CRITICAL | ⚪ |
| **1 Dec 2026** | **SFT v1 / RC candidate** | Sangam tiṇai ≥70%, citation ≥95%, hallucinated citations <2%, grounded QA above baseline | ⚪ |
| **10 Dec 2026** | **Kutty RC1** | Ship bars in §6; INT4 tokens/s + RSS recorded; no CRITICAL | ⚪ |
| **17 Dec 2026** | **Kutty served** | OpenAI-compatible endpoint behind yazhi-api; p50/p95 within SLO | ⚪ |
| **20 Dec 2026** | **Model card + datasheet** | Limits, known failures, tokenizer numbers, teacher/SFT provenance stated | ⚪ |
| **26 Dec 2026** | **Adhan Kutty v1 on HuggingFace** | Release tag; weights + tokenizer + card; demo video | ⚪ |
| Q1 2027 | Distilled Swaram/Qwen student, larger Tamil data, DPO only with verifier | Separate research gate; not a 26 Dec dependency | ⚪ |

## 2. Critical path to 26 Dec

```
Base decision (3 Nov) ──► K1 bake-off (5 Nov) ──► K2 SFT v0 (10 Nov)
                                                   │
K6 neutral Tanglish eval (10 Nov) ─────────────────┤
                                                   ▼
K3 Sarvam-M distillation (14 Nov) ──► SFT v1 (1 Dec) ──► K7 safety/eval (10 Dec)
                                                   │
K5 phone/runtime (7 Nov) ──► K4 INT4 export (21 Nov) ──► phone demo (10 Dec)
                                                   ▼
K9 serve via yazhi-api (17 Dec) ──► model card (20 Dec) ──► HF release (26 Dec)
```

**Float:** about 5 working days. GPU access and Qwen3 Tamil quality are the two items
that can consume it.

## 3. Cross-repo dependencies

| We need | From | By | Status |
|---|---|---|---|
| GPU host for SFT, distillation, and teacher runs | yazhi-api / infra | 7 Nov | ⚪ |
| Reference phone procurement | Yazhi hardware | 7 Nov | ⚪ |
| Sangam labels + verse text | open-sangam | have | 🟢 |
| English eurai gold for translation SFT | open-sangam | ongoing | 🟡 |
| Political/abuse review of Tanglish sample | native reviewer | 10 Nov | ⚪ |
| Endpoint template for Qwen3 serving | yazhi-api | 17 Dec | ⚪ |
| Model card / public claim correction on yazhi.dev | yazhi-dev | 4 Nov | 🔴 |

## 4. Risk register

| # | Risk | L | I | Mitigation | Trigger to act | Owner |
|---|---|---|---|---|---|---|
| R1 | **Qwen3-1.7B Tamil quality is not good enough** | 2 | 3 | K1 bake-off; fallback Qwen3-4B, then Gemma-3-4B-IT or negotiated Sarvam-1 | K1 misses the 20-prompt bar | Adhan DRI |
| R2 | **Qwen3 tokenizer tax** — 2.027 tok/akshara vs 0.584 Sarvam-1 and 0.418 Agaram | 3 | 2 | Keep answers short; measure INT4 tokens/s; plan a Tamil-heavy-BPE student for v2. **Partially retired 5 Nov:** a licence-clean tokenizer at 0.418 exists (Agaram, Apache-2.0), so the tax is a data choice, not a licensing dead end | Phone tokens/s below target or context thrash | Adhan DRI |
| R3 | **No GPU in time** | 2 | 3 | Rent a T4/A10 weekend; quantised teacher; platform node | No host by 7 Nov | Adhan DRI + infra |
| R4 | **Sarvam-M teacher licence/data terms** | 1 | 3 | Apache-2.0 model, but write down output terms; verify + human sample every teacher batch | Before first distillation run | Adhan DRI |
| R5 | **Tanglish source provenance** | 2 | 3 | Raw text local-only; manifest only; gazetteer filter; native review; legal check before public release | Any public post | Adhan DRI |
| R6 | **INT4 export fails on Qwen3** | 2 | 3 | Spike GGUF first; ONNX Runtime Mobile fallback; desktop demo only via cut list | No working INT4 by 21 Nov | Adhan DRI |
| R7 | **Safety gate fails** | 2 | 3 | Aram critical-only before release; refusal/citation SFT; cut conversation if needed | Any CRITICAL | Adhan DRI |
| R8 | **Public claim still says 7B/22+** | 2 | 3 | Fix About page and all public copy; link every number to a source | Any new public post | Founder + yazhi-dev |
| R9 | **Reported eval numbers rest on unrerunnable rules** | 2 | 2 | Every headline number names a committed script + sample rule. 5 Nov closed the worst case (Qwen3 fertility 2.016 had no script → now `bases_tokenizer_bench.json`, 2.027, rerunnable) | Any metric in `CURRENT` with no script behind it | Adhan DRI |

## 5. Cut list (in order, if 26 Dec is at risk)

Quality gates never move. Scope does.

1. Drop Qwen3-4B as a second candidate; ship 1.7B only.
2. Drop Sarvam-M distillation; ship SFT-on-Sangam/QA only.
3. Drop INT4 phone demo; ship desktop/server demo and keep phone as 2027.
4. Drop conversation SFT; ship Sangam + grounded-QA only.
5. Drop `சிந்தனை:` traces.
6. Last resort: release as a **research preview** (Qwen3-1.7B SFT + card + eval), no chat claim.

## 6. Ship bars for Kutty v1

Data-hygiene bar (applies to any mix we train on, added 5 Nov): the datasheet states
documents in, duplicates removed (MinHash), and benchmark-leak documents removed with the
benchmark list and thresholds named. A mix without those three numbers does not ship.

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

- Distil Sarvam-M into a smaller student whose tokenizer is trained on a Tamil-heavy diet
  (Agaram's 64k byte BPE reached 0.418 tok/akshara on our sample, so the target for Kutty v2
  is Sarvam-class fertility with a permissive licence — not Swaram by default).
- Revisit Sarvam-1 only with a commercial licence.
- Dravidian expansion only after Tamil ships and ≥1B verified pure-Tamil tokens exist.
- DPO only with a verifier; no preference data without a licence review.
- Keep `adhan-nano` + Swaram as the research paper/ablation line.
