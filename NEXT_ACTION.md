# Founder Refresher & Next Actions — Adhan SLM

> **Milestone Expectations:**
> - 🎯 **October 2026:** Pilot Release — 12k Swaram Akshara Tokenizer validation & Nano CPU/GPU baseline model evaluation
> - 🚀 **December 2026:** Public Launch — Pretrained Tamil-first Foundation SLM with 3-stage annealing (SCERT -> Synthetic Q&A -> Sangam Gold Set)

---

## 1. Executive Summary & Architecture

Adhan is a from-scratch, pure-Tamil small language model designed for high linguistic efficiency, native agglutinative root handling, and zero-cloud edge deployment.
- **Atomic Unit:** Swaram (உயிர்–மெய்) tokens (~12k vocabulary).
- **Core Stacks:** JAX/Flax for distributed training, PyTorch trainer for hybrid compute pipelines, and MLflow for experiment tracking.

---

## 2. October 2026 Pilot Scope

- [ ] **Tokenizer Benchmark & Finalization:**
  - Complete Swaram 12k vocabulary compression benchmarks against BPE/SentencePiece on Sangam + Modern Tamil corpora.
- [ ] **Nano Baseline Convergence:**
  - Verify stable loss curves across CPU and CUDA training targets (`src/adhan_slm/training/train_jax.py` and `train_torch.py`).
- [ ] **Corpus Pipeline Harmonization:**
  - Ingest Stage 1 (SCERT + textbooks) and Stage 2 (Curriculum Synthetic Q&A) data with full deduplication and quality filtering.

---

## 3. December 2026 Launch Scope

- [ ] **Full 3-Stage Annealed Pretraining:**
  - Execute Stage 3 high-weight annealing on Sangam Gold Set literature corpus.
- [ ] **Edge & ONNX Optimization:**
  - Export quantized GGUF / ONNX artifacts ready for edge execution in `yazh-unity` and mobile devices.
- [ ] **Model Evaluation & Leaderboard:**
  - Automated evaluation harness on Tamil grammatical, cultural, and reasoning benchmarks.

---

## 4. Immediate Next Actions

1. Verify MLflow logging metrics on local test runs.
2. Ensure training script CLI parity across both JAX and Torch runners.
3. Update corpus validation scripts with rigorous Unicode normalization and Tamil glyph integrity checks.
