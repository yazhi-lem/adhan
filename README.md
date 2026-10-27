# Adhan

Tamil-first SLM: data, tokenization, training and evaluation.

Plan and status live in [`factory/`](./factory): [`CURRENT`](./factory/CURRENT.md) · [`NEXT`](./factory/NEXT.md) · [`RUNWAY`](./factory/RUNWAY.md). Retired docs are summarised in [`archives/`](./archives).

Adhan is a **from-scratch, pure-Tamil small language model**. The akshara (உயிர்–மெய் / *swaram*) is the atomic
token, the modelling is agglutination-aware, training runs on JAX/Flax with MLflow tracking, and the target is a
light, edge-deployable model. The Swaram tokenizer removes the "Tamil token tax": English-centric GPT-4 tokenization
spends **4.21×** the tokens on the same Tamil text ([`docs/tokenizer/TOKEN_TAX_BENCHMARK_V3.md`](docs/tokenizer/TOKEN_TAX_BENCHMARK_V3.md),
[`docs/tokenizer/WHITEPAPER_SWARAM.md`](docs/tokenizer/WHITEPAPER_SWARAM.md)).

- **Roadmap:** [`docs/ROADMAP_JAX_SLM.md`](docs/ROADMAP_JAX_SLM.md) · **Capabilities and ship bars:** [`docs/behaviour/CAPABILITIES.md`](docs/behaviour/CAPABILITIES.md)
- **Architecture:** [`docs/architecture/ARCHITECTURE_SWARAM_SLM.md`](docs/architecture/ARCHITECTURE_SWARAM_SLM.md) · **Commands:** [`docs/DEV.md`](docs/DEV.md)
- **No GPU required:** [`docs/training/CPU_TRAINING.md`](docs/training/CPU_TRAINING.md) · free T4: [`docs/training/COLAB_T4.md`](docs/training/COLAB_T4.md)
- **Languages:** [`lang/README.md`](lang/README.md) (Tamil flagship, English bridge, Dravidian and Indo-Aryan next)

## Install

```bash
git clone https://github.com/yazhi-lem/adhan.git && cd adhan
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,jax,tamil-nlp]"          # CPU JAX wheels: works anywhere
pip install -e ".[dev,jax-cuda,tamil-nlp]"     # or: NVIDIA GPU (CUDA 12, Linux)
pip install -e ".[serving]"                    # for `adhan serve`
pip install -e ".[torch]"                      # optional PyTorch trainer (scripts/train_efficient.py)
```

## The `adhan` CLI

```bash
adhan start --mlflow                              # environment + MLflow UI (http://localhost:5000)
adhan status                                      # data, tokenizer and checkpoint readiness
adhan train --overfit-batch                       # 1-minute wiring gate before any long run
adhan train --model nano --device cpu --trace     # train with live tracing
adhan serve --port 8000 --model adhan-nano        # REST/streaming API; 503 until a checkpoint is loaded
adhan eval                                        # Thirukkural, sandhi, morphology, kid prompts
adhan interact                                    # terminal session with live tokenization
adhan mutate quantize --checkpoint checkpoints/adhan-nano --format int8   # see factory: export rework
```

Logs rotate under `logs/adhan_cli_*.log` (symlink: `logs/adhan_cli_latest.log`). `--trace` adds per-step timing,
fertility and JAX memory profiling.

## Evaluation

`src/adhan_slm/eval/`: the Thirukkural harness (1,330 couplets), morphology + sandhi probes (புணர்ச்சி, per
Tholkāppiyam), 50 kid-level prompts, the classical unigram floor and model perplexity, all combined into one
`run_eval` JSON report ([`docs/behaviour/EVAL_TAMIL.md`](docs/behaviour/EVAL_TAMIL.md)). Safety: [`docs/behaviour/ARAM_HARNESS.md`](docs/behaviour/ARAM_HARNESS.md).
Serving: [`docs/serving/DEPLOYMENT.md`](docs/serving/DEPLOYMENT.md).

## License

MIT
