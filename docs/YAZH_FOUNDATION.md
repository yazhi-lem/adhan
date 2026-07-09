# Adhan as the Foundational Model of the Yazh Ecosystem

Adhan is not a standalone project — it is the **foundational language model** of the
**Yazh** ecosystem. Yazh provides the shared base (infrastructure, data schemas,
serving contracts); Adhan provides the model. This document defines that relationship
and the dependency direction.

```
                         ┌──────────────────────────────┐
                         │        Yazh ecosystem        │
                         │                              │
  yazh-kutty ──────────► │  base: config, schemas,      │
  (foundation / "little  │  shared utils, contracts     │
   one" base library)    │                              │
                         │            ▲                 │
                         │            │ builds on       │
  Adhan ─────────────────┼──── FOUNDATIONAL MODEL       │
  (Tamil/Dravidian SLM)  │   swaram + aksharam tokens,  │
                         │   JAX/Flax native SLM        │
                         │            │                 │
                         │            ▼ serves via       │
  yazhi-api ─────────────┼──► deployment: REST + SDK    │
  (private serving)      │                              │
                         └──────────────────────────────┘
```

## 1. Dependency direction
- **`yazh-kutty`** (base requirement): the Yazh foundation package. Adhan depends on
  it for shared configuration, data schemas, and cross-project contracts. Declared in
  `requirements-jax.txt` as a git dependency.
- **Adhan** (this repo): the foundational **model** — swaram/aksharam tokenizers +
  native JAX/Flax SLM. Reusable by every downstream Yazh product.
- **`yazhi-api`** (private): production serving. Adhan checkpoints deploy here as a
  REST endpoint + Python SDK (`from yazhi_api import AdhanClient`).

## 2. Why "foundational"
A foundational model is trained once, broadly, then reused across tasks. Adhan is
foundational for Yazh because:
- **Native tokenization** (swaram/aksharam) is script-family infrastructure every Yazh
  Indic product needs — not Adhan-specific.
- **The pretrained base** (`adhan-nano/tiny/mini`) is fine-tuned downstream (sentiment,
  NER, chat, kid tutor) rather than retrained.
- **The training + eval harness** (JAX loop, MLflow contract, Tamil-first probes) is a
  reusable Yazh capability.

## 3. Tokenizer family (JAX-accelerated)
Both tokenizers share the two-layer design and a JAX batch-encoding fast path
(`src/adhan_slm/tokenizer/jax_encode.py`), so corpus tokenization and batched
inference run vectorized on GPU/TPU inside the JAX pipeline.

| Tokenizer | Script family | Prototype language | Status |
|---|---|---|---|
| **Swaram** | Dravidian | Tamil (தமிழ்) | Phase 0 ✅ |
| **Aksharam** | Indic / Devanagari | Hindi (हिन्दी) | Phase 0 ✅ |

Both are lossless (round-trip verified) and closed-vocabulary at the base layer.

## 4. Foundational-model launch target
`adhan-nano` v0.1 — **kid-level Tamil** (basic reading, speaking, comprehension),
shipped light (INT4 ≤ 30 MB, on-device) and served via `yazhi-api`. See
`ROADMAP_JAX_SLM.md`.

## 5. Setup
```bash
python3 -m venv .venv-jax && source .venv-jax/bin/activate
pip install -r requirements-jax.txt
# Yazh base (once packaged):
# pip install "yazh-kutty @ git+https://github.com/yazhi-lem/yazh-kutty.git"

# Swaram (Tamil / Dravidian)
python -m adhan_slm.tokenizer.swaram_tokenizer "படித்துக்கொண்டிருந்தேன்"
# Aksharam (Hindi / Indic)
python -m adhan_slm.tokenizer.aksharam_tokenizer "पढ़ रहा था"
```
