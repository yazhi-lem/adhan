# Adhan SLM — Master Progress & Training Architecture Report

> **Adhan (ஆதன்)** — Native Tamil Small Language Model (SLM) trained from scratch with *Swaram* akshara-morpheme tokenization.

---

## 1. Hardware Audit & Execution Profile

| Hardware Component | Detected Specification | Optimization Strategy in Adhan |
|---|---|---|
| **CPU** | **AMD Ryzen 5 5600X (6 Cores / 12 Threads)** | Thread-pinning to 6 physical workers (`torch.set_num_threads(6)`), AVX2 vectorization, bfloat16/fp32 mixed precision. |
| **GPU** | **NVIDIA Quadro GPU (Pascal/Turing/RTX)** | Zero-OOM micro-batching ($B=8$) + Gradient Accumulation ($K=16\rightarrow32$, effective batch $128\rightarrow256$), FP16 tensor core acceleration, memory-mapped I/O. |
| **System RAM** | **16 GB (15 GiB usable)** | Zero-RAM-overhead virtual memory streaming via `np.memmap` binary shards (`train.bin`, `val.bin`). |
| **Storage & I/O** | NVMe / Fast SSD | Flat binary sequence layout (uint16 token IDs, $512$ seq length). |

---

## 2. Comprehensive Tamil Data Audit & Prioritization

The dataset is partitioned into three prioritized tiers to ensure **grammatical correctness, kid-level conversational register, encyclopedic breadth, and high syntactic diversity**:

```
                                  ┌───────────────────────────────────────────────┐
                                  │      ADHAN CURATED CORPUS (300M - 1B TOK)     │
                                  └───────────────────────┬───────────────────────┘
                                                          │
         ┌────────────────────────────────────────────────┼────────────────────────────────────────────────┐
         ▼                                                ▼                                                ▼
┌───────────────────────────────────┐    ┌───────────────────────────────────┐    ┌───────────────────────────────────┐
│     TIER 1: FOUNDATION & SCERT    │    │      TIER 2: ENCYCLOPEDIC & NEWS  │    │     TIER 3: LARGE-SCALE INDIC     │
│             (35% Weight)          │    │             (30% Weight)          │    │             (35% Weight)          │
├───────────────────────────────────┤    ├───────────────────────────────────┤    ├───────────────────────────────────┤
│ • TN SCERT School Books (1 to 12) │    │ • Tamil Wikipedia (160k Articles) │    │ • AI4Bharat IndicCorp v2 (Tamil)  │
│ • Tamil Virtual Academy (TVA)     │    │ • Tamil Wiktionary (400k Lemmas)  │    │ • AI4Bharat Sangraha (Tamil)      │
│ • Open-Sangam (2,552 Verses + Urai│    │ • Press Information Bureau (PIB)  │    │ • Samanantar Parallel (Tamil)     │
│ • Project Madurai Classical Works │    │ • All India Radio News Bulletins  │    │ • Swecha Open Tamil Corpus        │
└───────────────────────────────────┘    └───────────────────────────────────┘    └───────────────────────────────────┘
```

### 2.1 Detailed Resource Audit Matrix

| Tier | Source / Dataset | Domain | Est. Tokens | Ingestion Connector | Quality Score | Role in Model |
|:---:|---|---|---|---|:---:|---|
| **1** | **TN SCERT School Books** | Education (Classes 1–12) | 45M | `ingest_scert_textbooks()` | **0.95** | **Gold standard.** Provides foundational vocabulary and grammar for kid-level fluency. |
| **1** | **Open-Sangam** | Classical Poetry & Urai | 2.5M | `ingest_open_sangam()` | **0.95** | Deep classical grounding, 18 Sangam poems + line-by-line glossary. |
| **1** | **Tamil Virtual Academy** | Children's Stories & Dictionaries | 50M | `TVA_Extractor` | **0.90** | Folk tales, moral stories, standard children's literature. |
| **1** | **Project Madurai** | Digitized Literature | 15M | `PMWorks_Extractor` | **0.90** | 600+ literary works (Bharathiyar, Kalki, Sangam). |
| **2** | **Tamil Wikipedia** | Encyclopedic Knowledge | 45M | `ingest_wikipedia_ta()` | **0.90** | World facts, science, geography, history in formal Tamil. |
| **2** | **Tamil Wiktionary** | Lexicons & Etymology | 20M | `Wiktionary_Extractor` | **0.88** | Morpheme boundaries, lemma definitions, root words. |
| **2** | **Press Information Bureau** | Governance & Policy | 30M | `PIB_Extractor` | **0.85** | Official state releases, modern formal sentence structure. |
| **3** | **AI4Bharat IndicCorp v2** | Filtered Web Crawl | 300M+ | `ingest_indiccorp_sample()`| **0.80** | Pretraining scale, modern conversational language. |
| **3** | **AI4Bharat Sangraha** | Curated Web & Synthetic | 200M+ | `Sangraha_Extractor` | **0.82** | High-density linguistic reasoning and dialogue. |
| **3** | **Samanantar (Tamil side)**| High-Quality Parallel | 80M | `Samanantar_Extractor` | **0.85** | High grammatical precision from professional translation. |

---

## 3. Pipeline Architecture & Execution Commands

### Step 1: Ingest, Deduplicate & Prioritize Data
Run the unified data connector to pull from Open-Sangam, SCERT, Wikipedia, and IndicCorp with **MinHash LSH deduplication**:

```bash
cd repos/adhan

# Ingest all prioritized datasets into unified JSONL
python scripts/ingest_all_prioritized_data.py \
    --output-dir data/raw/unified_prioritized \
    --sangam-path ../open-sangam \
    --limit-wiki 100000 \
    --limit-indic 200000
```

### Step 2: Freeze Swaram Tokenizer v1 & Pack Binary Shards
Train the *Swaram* akshara-morpheme tokenizer (12,000 vocabulary) and pack fixed-length 512-token binary sequences:

```bash
python scripts/prepare_slm_corpus.py \
    --corpus data/raw/unified_prioritized/unified_corpus.jsonl \
    --out data/final/tamil_slm \
    --vocab-size 12000 \
    --seq-len 512 \
    --val-frac 0.02
```

### Step 3: Run Overfit Sanity Gate
Sanity-check model and loss wiring by forcing the loss to collapse on a single repeated batch:

```bash
python scripts/train_efficient.py \
    --data-dir data/final/tamil_slm \
    --overfit-batch
```

### Step 4: Launch Hardware-Optimized Pretraining
Train `adhan-nano` (~8.2M parameters) with gradient accumulation on your NVIDIA Quadro or Ryzen 5 CPU:

```bash
python scripts/train_efficient.py \
    --data-dir data/final/tamil_slm \
    --size nano \
    --batch-size 8 \
    --grad-accum 16 \
    --max-steps 15000 \
    --lr 1.5e-3 \
    --checkpoint-dir checkpoints/adhan-nano
```

*(Note: For JAX training loop, run `python -m adhan_slm.training.train_jax --config src/adhan_slm/configs/adhan_slm_nano_cpu.yaml`)*

### Step 5: Export to ONNX for Yazh Unity
Export the trained weights directly into the Unity 3D game project for on-device XR inference:

```bash
python scripts/train_efficient.py \
    --export-onnx \
    --checkpoint checkpoints/adhan-nano/adhan_nano_best.pt \
    --output-onnx ../yazh-unity/Assets/StreamingAssets/MLModels/yazh-30k.onnx
```

---

## 4. Current Status Checklist

- [x] Swaram Tokenizer Layer A (Akshara) + Layer B (Morpheme BPE) with fertility $< 1.15$
- [x] Transformer Architecture: RoPE, SwiGLU, RMSNorm, Weight-tied embeddings
- [x] Unified Data Ingestion Engine (`scripts/ingest_all_prioritized_data.py`)
- [x] MinHash LSH near-deduplication & Unicode NFC sanitization
- [x] Dual Training Engine:
  - JAX/Flax Trainer with Orbax + MLflow (`src/adhan_slm/training/train_jax.py`)
  - Fast PyTorch / Quadro / Ryzen Trainer (`scripts/train_efficient.py`)
- [x] Overfit-a-batch sanity test gate
- [x] Direct ONNX Exporter for `yazh-unity`
- [ ] Complete full 300M token pretraining run on unified dataset
- [ ] Run kid-level SFT alignment (50 prompt benchmarks)
- [ ] Quantize checkpoint to INT4 (target $\le 25\text{MB}$)

---

*Last Updated: 2026-08-17*  
*Adhan — Sovereign Native Tamil Intelligence.*
