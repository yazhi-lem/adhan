# K1 — Qwen3-1.7B Tamil bake-off (partial, 3 Nov 2026)

**Status:** preliminary CPU smoke; not the final K1 decision.
**Raw:** [`base_bakeoff_qwen3.json`](base_bakeoff_qwen3.json)

## Setup

- `qwen3:1.7b` via Ollama 0.32.14 on CPU, native `/api/chat` with `think=false`.
- tamil-bench runner, seed 42, 0-shot.
- No GPU; MILU not run because the dataset is gated.

## Results

| Task | n | Result |
|---|---:|---|
| IndicXNLI-Tamil | 50 | **42.0%** accuracy (3-way chance 33.3%) |
| IndicQA-Tamil | 50 | **EM 0.140 · F1 0.192**, 0 errors |
| IndicXNLI smoke | 20 | 35.0% accuracy |

For context, the frozen MuRIL linear probe on the full 5,010-row XNLI test split was
42.77% (`docs/reports/muril_probe.json`); these are different experiments and not a
like-for-like comparison.

## Reading

Qwen3-1.7B is **at chance-plus on this small XNLI sample** and weak on extractive QA
before any Tamil SFT. That is expected for a base model on a hard Tamil task, but it is
not yet evidence that 1.7B is the right student. The decisive result is the post-SFT
comparison (K2), not this zero-shot smoke.

## What is still needed

1. Full `xnli` n=200, `indicqa` n=100, `milu` n=199 on GPU.
2. Qwen3-4B with identical prompts/seed.
3. 20-prompt Tamil generation read.
4. INT4 size + tokens/s on the reference phone.

Until then, K1 stays open and the base remains `Qwen3-1.7B-Base` as a working choice,
not a proven one.
