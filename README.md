# Adhan

Tamil-first SLM data, tokenization, training & evaluation pipeline.

See [NEXT_ACTION.md](./NEXT_ACTION.md) for the roadmap, **October 2026 Pilot**, and **December 2026 Launch** deliverables.

---

## 🆕 Native Tamil SLM (Swaram Tokens + JAX)

We are building a **from-scratch, pure-Tamil small language model** — akshara (உயிர்–மெய் / *swaram*) as the atomic token, agglutination-aware modeling, trained in **JAX/Flax** and tracked with **MLflow**, targeting a light, edge-deployable launch.

- **Roadmap:** [`ROADMAP_JAX_SLM.md`](ROADMAP_JAX_SLM.md)
- **Architecture:** [`docs/ARCHITECTURE_SWARAM_SLM.md`](docs/ARCHITECTURE_SWARAM_SLM.md)
- **CPU Training:** [`docs/CPU_TRAINING.md`](docs/CPU_TRAINING.md) — **no GPU required**
- **Multi-Language Spec:** [`lang/README.md`](lang/README.md) — Tamil flagship (`lang/tamil/`) + English bridge (`lang/english/`)

---

## ⚡ Unified CLI: `adhan`

Adhan comes with a sovereign unified CLI for end-to-end development, training, inference, and mutation:

```bash
# 1. Start environment & launch MLflow UI dashboard
adhan start --mlflow

# 2. Check system status, data records & tokenizer readiness
adhan status

# 3. Run the 1-minute overfit sanity gate before training
adhan train --overfit-batch

# 4. Train adhan-nano on CPU (or adhan-tiny on GPU) with live tracing
adhan train --model nano --device cpu --trace

# 5. Launch high-performance REST/Streaming inference server
adhan serve --port 8000 --model adhan-nano

# 6. Mutate datasets (dedup, distill) or models (quantize, export)
adhan mutate quantize --checkpoint checkpoints/adhan-nano --format int8
adhan mutate distill --teacher gemma2-27b

# 7. Run evaluation suite (Thirukkural, Sandhi, Morphology, Kid prompts)
adhan eval

# 8. Interactive terminal session with live tokenization & fertility tracing
adhan interact
```

---

## 🔍 Logs & Interactive Tracing

Adhan maintains structured logging and interactive tracing by default:
- **Persistent Log Files:** Automatically rotated under `logs/adhan_cli_YYYYMMDD_HHMMSS.log` with a symlink to `logs/adhan_cli_latest.log`.
- **Interactive Tracing:** Pass `--trace` to any command for live micro-step timing, token fertility metrics, and JAX memory profiling.
- **MLflow Tracing:** Integrated experiment tracking in `mlflow.db` (`http://localhost:5000`).

---

## Installation

### Option 1: Development Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/yazhi-lem/adhan.git
cd adhan

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with CLI and all development extras
pip install -e ".[dev,jax,tamil-nlp]"
```

On a machine with an NVIDIA GPU, add the `jax-cuda` extra:
```bash
pip install -e ".[dev,jax-cuda,tamil-nlp]"
```

---

## Multi-Language Architecture (`lang/`)

| Language | Directory | Script / Family | Priority | Status |
|---|---|---|---|---|
| **Tamil** | `lang/tamil/` | Dravidian (Tamil script) | **P1 (Flagship)** | **Active / Ingestion & Evaluation Ready** |
| **English** | `lang/english/` | Germanic (Latin script) | **P1 (Secondary Bridge)**| **Active / Cross-lingual Alignment** |
| **Telugu** | `lang/telugu/` | Dravidian (Telugu script) | P2 | Planned (Brahmi Akshara base) |
| **Malayalam** | `lang/malayalam/` | Dravidian (Malayalam script)| P2 | Planned (Chillu & Sandhi base) |
| **Kannada** | `lang/kannada/` | Dravidian (Kannada script) | P2 | Planned (Vattu conjunct base) |
| **Odia** | `lang/odia/` | Indo-Aryan (Odia script) | P3 | Planned |
| **Marathi** | `lang/marathi/` | Indo-Aryan (Devanagari) | P3 | Planned |
| **Hindi** | `lang/hindi/` | Indo-Aryan (Devanagari) | P3 | Planned (Aksharam prototype) |

---

## Evaluation Harnesses

- **Classical Thirukkural Harness (`src/adhan_slm/eval/thirukkural_eval.py`):** 1,330 couplets benchmarked on fertility (`0.598` tokens/akshara) and pure Tamil purity (`75.8%`).
- **Morphology & Sandhi Probes (`src/adhan_slm/eval/morphology.py`):** Word-junction correctness (*புணர்ச்சி*) against *Tholkaappiyam* rules.
- **50 Kid-Level Conversational Prompts (`src/adhan_slm/eval/kid_level_prompts.py`):** Real-world prompt generation tests.

---

## License

MIT
