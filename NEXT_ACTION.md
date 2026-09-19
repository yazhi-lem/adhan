# Founder Refresher & Next Actions — Adhan SLM

> **Milestone Expectations:**
> - 🎯 **October 2026:** Pilot Release — 12k Swaram Akshara Tokenizer validation & Nano CPU/GPU baseline model evaluation
> - 🚀 **December 2026:** Public Launch — Pretrained Tamil-first Foundation SLM with 3-stage annealing (SCERT -> Synthetic Q&A -> Sangam Gold Set)

---

## 1. Executive Summary & Architecture

Adhan is a from-scratch, pure-Tamil small language model designed for high linguistic efficiency, native agglutinative root handling, and zero-cloud edge deployment.
- **Atomic Unit:** Swaram (உயிர்–மெய்) tokens (~12k vocabulary).
- **Core Stacks:** JAX/Flax for distributed training, PyTorch trainer for hybrid compute pipelines, and MLflow for experiment tracking.
- **Tiṇai & Open-Sangam Grounding:** Organizing the neural network conditioning around the **Five Tiṇai (ஐந்திணை)** and Sangam **Mupporul (முதல், கரு, உரிப் பொருள்)** with persona routing (Avvaiyar, Tholkappiyar, Kapilar, Paranar, Nakkirar).

---

## 2. October 2026 Pilot Scope

- [ ] **Tokenizer Benchmark & Finalization:**
  - Complete Swaram 12k vocabulary compression benchmarks against BPE/SentencePiece on Sangam + Modern Tamil corpora.
- [ ] **Nano Baseline Convergence:**
  - Verify stable loss curves across CPU and CUDA training targets (`src/adhan_slm/training/train_jax.py` and `train_torch.py`).
- [ ] **Corpus Pipeline Harmonization:**
  - Ingest Stage 1 (SCERT + textbooks) and Stage 2 (Curriculum Synthetic Q&A) data with full deduplication and quality filtering.
- [ ] **Tiṇai Routing & Conditioning Layer:**
  - Integrate Tiṇai landscape embeddings (`TinaiRouter`) into JAX/Flax transformer architecture.

---

## 3. December 2026 Launch Scope

- [ ] **Full 3-Stage Annealed Pretraining:**
  - Execute Stage 3 high-weight annealing on Sangam Gold Set literature corpus.
- [ ] **vLLM & Transformers Bridge Export:**
  - Convert JAX Orbax checkpoints to SafeTensors + HuggingFace FastTokenizer format for vLLM high-throughput serving.
- [ ] **Edge & ONNX Optimization:**
  - Export quantized INT8 / INT4 ONNX artifacts ready for edge execution in `yazh-unity` and `illakiya` mobile keyboard.
- [ ] **Model Evaluation & Leaderboard:**
  - Automated evaluation harness on Tamil grammatical, cultural, and reasoning benchmarks.

---

## 4. Immediate Next Actions

1. **Verify Tokenizer Sharding & Fertilty:** Monitor `prepare_slm_corpus.py` completion; confirm fertility strictly `< 1.15` and check packed `train.bin` / `val.bin`.
2. **Run Sanity Overfit-Batch Gate:** Execute `--overfit-batch` on `adhan_slm_nano_cpu.yaml` to confirm loss collapse before pretraining.
3. **Deploy Yazhi-One Cron Ingestion:** Set up scheduled nightly execution of `scripts/cron_yazhi_one_ingest.sh` on `yazhi-one`.
4. **Implement Tiṇai Neural Conditioning:** Build `TinaiRouter` and Sangam Avai persona conditioning in `src/adhan_slm/model/transformer.py`.
5. **Execute Quality Tamil Data Strategy:** Curate SCERT textbooks, Project Madurai, and synthetic conversational distillation from Gemma 2 / GLM-4 / Kimi.

---

## 5. Data Availability Audit & Quality Strategy

### A. Data Availability Audit
| Tier | Source / Repository | Volume / Status | Quality Rating | Use in Pipeline |
|---|---|---|---|---|
| **Tier 1 (Gold)** | **Open-Sangam** (`repos/open-sangam`) | 18 poems, 2,552 verses, word glossaries, modern urai | ⭐⭐⭐⭐⭐ (0.95) | Anchor for classical syntax, Sandhi, & annealing stage |
| **Tier 1 (Gold)** | **SCERT Textbooks** (Grades 1–12) | Science, history, Tamil language reader txt files | ⭐⭐⭐⭐⭐ (0.95) | Foundational general world knowledge & grammar |
| **Tier 2 (Silver)**| **Tamil Wikipedia** (HF stream) | 150,000 articles | ⭐⭐⭐⭐ (0.90) | Fact/entity richness, modern prose style |
| **Tier 2 (Silver)**| **Project Madurai** | 700+ classical & medieval public domain texts | ⭐⭐⭐⭐ (0.90) | Literary vocabulary expansion & morphology depth |
| **Tier 3 (Bronze)**| **AI4Bharat IndicCorp v2 / Sangraha**| 250,000+ Tamil web documents | ⭐⭐⭐ (0.80) | Broad modern vocabulary & syntax scale |
| **Stage 2 (SFT)** | **Synthetic Conversational Distillation** | Gemma 2 27B / GLM-4 / Kimi distilled dialogues | ⭐⭐⭐⭐⭐ (0.95) | Multi-turn conversational flow & instruction-following |

### B. Strategy to Improve Tamil Content Quality
1. **Aggressive Near-Deduplication:** Enforce 64-hash MinHash LSH shingle filtering to eliminate web boilerplate (removing 90%+ redundancy).
2. **Strict Tamil Density Thresholds:** Enforce `min_tamil_ratio >= 0.80` to reject Tanglish noise, code snippets, and untranslated English fragments.
3. **Unicode NFC & Pulli Normalization:** Standardize all decomposed Tamil characters and fix isolated vowel/consonant markers (`Unicode NFC`).
4. **Dual-Register Synthetic Distillation:** Use frontier teacher models (Gemma 2, GLM, Kimi) to generate paired classical-to-colloquial Tamil explanations, idioms, and multi-turn educational dialogues.
5. **Sandhi & Grammar Integrity Gate:** Filter documents through morphology checkers (`solthiruthi` / `open-tamil`) to eliminate ungrammatical machine-translated web content.
