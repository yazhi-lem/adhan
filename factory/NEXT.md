# NEXT — adhan · iteration 3 · 2026-11-03 → 2026-11-17

> **Execution reset.** The base question is closed: Qwen3-1.7B-Base is the product
> student, Sarvam-M 24B is the teacher, Sarvam-1 is rejected for its non-commercial
> licence, and Adhan-nano stays a research artifact. This iteration is only the path to
> Kutty v1 on 26 Dec.

**Iteration goal:** prove Qwen3-1.7B is good enough on Tamil, fine-tune it on Adhan's
Sangam/QA data with a regression gate, start distillation from Sarvam-M, produce an INT4
phone build, and freeze a neutral Tamil/Tanglish eval set.

**Capacity:** one full-time DRI ≈ 8 working days. K1–K4 are the committed set; K5–K8 are
P1 and land only if compute and review are available; K11–K15 in the stretch table are the
overflow and are **not** committed.

**Carried in from 5 Nov (code already landed, wiring not):** corpus hygiene (MinHash dedup +
benchmark-leak filter) and training diagnostics exist as modules with tests but touch no
pipeline yet. Their wiring is K11/K12 below, deliberately not committed this iteration —
K1–K4 own the critical path.

| ID | P | Task | DRI | Size | Exit criterion | Depends on | Due |
|---|---|---|---|---|---|---|---|
| K1 | P0 | 🟡 **Base bake-off.** Run Qwen3-1.7B-Base vs Qwen3-4B-Base on tamil-bench `xnli`/`milu`/`indicqa` (same `n`, seed 42) + a 20-prompt Tamil generation read. Preliminary CPU smoke done: 1.7B XNLI 42.0% n=50, IndicQA EM 0.140/F1 0.192 n=50 | Adhan DRI | M (2d) | `docs/reports/base_bakeoff_qwen3.{json,md}`; full GPU run + 4B + generation read still needed | GPU or local Ollama runner | **5 Nov** |
| K2 | P0 | **SFT v0 on Qwen3-1.7B.** Fine-tune on `data/final/sft_v0_sangam` + grounded QA; assistant-turn loss; 2–3 epochs; no-SFT regression gate | Adhan DRI | L (4d) | SFT checkpoint + `docs/reports/sft_v0_qwen3.json`; Sangam/QA scores beat the no-SFT baseline; Aram 0 CRITICAL on the run | K1, GPU | 10 Nov |
| K3 | P0 | **Distillation from Sarvam-M.** Generate Tamil prompts/answers with the 24B teacher, filter for Tamil purity + verifier/human sample, add to SFT v1 | Adhan DRI | L (4d) | `data/final/distill_sarvam_m/` + datasheet; ≥5k verified examples; licence/terms recorded | K1, GPU ≥48 GB | 14 Nov |
| K4 | P0 | **INT4 export + phone path.** Export Qwen3-1.7B SFT to GGUF/ONNX INT4; measure size, RSS, tokens/s on the reference device or emulator | Adhan DRI | M (3d) | `docs/reports/qwen3_int4_phone.json`; working INT4 artifact; latency recorded | K2, K5 | 21 Nov |
| K5 | P1 | **Reference phone + runtime.** Pick the ₹10–15k Android device and runtime (llama.cpp / ONNX Runtime Mobile); start procurement | Adhan DRI | S (1d) | Device + runtime in `CURRENT`; procurement started | — | 7 Nov |
| K6 | P0 | **Tanglish eval freeze.** Add the gazetteer political/sensitive filter, select 500 neutral sentences from `vishnu-n/Tanglish-Corpus-185k` @ `927f5b4`, native-review them, commit the manifest | Adhan DRI | M (2d) | `eval_manifests/tokenizer_bench.sources.json` + manifest + 500 reviewed hashes; no raw text committed | K1 not required | 10 Nov |
| K7 | P1 | **Aram + tamil-bench safety gate on Qwen3.** Run critical-only Aram and the tamil-bench `chat`/`sangam` suites on the SFT checkpoint; publish next to the no-SFT baseline | Adhan DRI | M (2d) | `docs/reports/aram_qwen3_sft.json`; 0 CRITICAL required before any release | K2 | 21 Nov |
| K8 | P1 | **Model card + datasheet + release.** Qwen3-1.7B-Kutty-v1 model card, tokenizer numbers, limits, SFT datasheet, HF release tag | Adhan DRI | M (2d) | Card + datasheet merged; release checklist complete | K2, K4, K7 | 26 Dec |
| K9 | P1 | **Serve via yazhi-api.** Wrap the Qwen3 SFT checkpoint behind an OpenAI-compatible endpoint; measure p50/p95 | Adhan DRI | S (1d) | Endpoint + latency in `CURRENT` | K2 | 17 Dec |
| K10 | P2 | **Research track (non-blocking).** Keep `adhan-nano` + Swaram paper/ablation alive; no product dependency | Research DRI | S (1d/wk) | Swaram A/B + morphology report updated; no product gate | — | ongoing |

## Stretch (only if committed work is done)

| ID | P | Task | DRI | Size | Exit criterion | Depends on |
|---|---|---|---|---|---|---|
| K11 | P1 | **Wire corpus hygiene into the corpus build.** Call `deduplicate_records` + `detect_contamination` from `scripts/prepare_slm_corpus.py` (modules and tests already landed 5 Nov); write the removed-doc counts and the benchmark list into `datasheet.json` | Adhan DRI | S (1d) | A rebuilt shard dir whose `datasheet.json` states docs-in, duplicates-removed, leaks-removed, and the benchmark files used | — |
| K12 | P1 | **Diagnostics on a real run.** Set `train.diagnostics_every` on the next continued-pretraining/SFT run and triage the first spike report | Adhan DRI | S (1d) | `logs/diagnostics/` from a real run + one paragraph naming any layer that spiked and what was done | K2 or any GPU run |
| K13 | P2 | **Adhan GGUF spike (research only).** Implement the JAX→HF weight-copy mapping documented in `scripts/export_gguf.py` and a token-parity check against the JAX tokenizer | Research DRI | M (2d) | A GGUF that answers the same 10 prompts token-for-token as the JAX checkpoint, parity report committed | — |
| K14 | P2 | 500–1,000 human labels for a fine-tuned MuRIL political classifier (if gazetteer recall is insufficient) | Adhan DRI | M (2d) | Label file + inter-annotator check | K6 |
| K15 | P2 | Qwen3-4B as a second SFT candidate if 1.7B quality is marginal | Adhan DRI | M (2d) | 4B SFT row in the bake-off table | K1, K2 |

## Explicitly not this iteration

- Building the product on `adhan-nano`, or fine-tuning it for the 26 Dec launch.
- Using Sarvam-1 without a commercial licence.
- A new tokenizer, chat special tokens, or Swaram swap into Qwen3.
- DPO, mini/1B+ scaling, or multilingual Kannada/Telugu.
- Re-running MinHash dedup over the already-packed shards; dedup applies to the *next* corpus build, not to shipped artifacts.
- Publishing raw vishnu Tanglish text or any unlabelled political content.
