# Adhan

Tamil-first SLM: data, tokenization, training and evaluation.

Plan and status live in [`factory/`](./factory): [`CURRENT`](./factory/CURRENT.md) · [`NEXT`](./factory/NEXT.md) · [`RUNWAY`](./factory/RUNWAY.md). Retired docs are summarised in [`archives/`](./archives).

Adhan has **two tracks**, and they are not the same product.

**Product — Kutty.** Since 3 Nov 2026 the product base is a fine-tune of
**`Qwen/Qwen3-1.7B-Base`** (Apache-2.0), with **`sarvamai/sarvam-m` 24B** (Apache-2.0) as the
distillation teacher. The first product release is due 26 Dec 2026. The record is
[`docs/reports/base_decision_2026-11-03.md`](docs/reports/base_decision_2026-11-03.md) and
[`docs/adr/0001-kutty-v1.md`](docs/adr/0001-kutty-v1.md).

**Research — Swaram and `adhan-nano`.** The from-scratch, pure-Tamil model where the akshara
(உயிர்–மெய் / *swaram*) is the atomic token and the modelling is agglutination-aware;
training is JAX/Flax with MLflow tracking. This is the tokenizer and ablation study, **not the
product base**: `adhan-nano` is 14.5M params at 1.820 bits/akshara, and its generations are not
yet coherent Tamil.

The research track is real and measured, but read it honestly. English-centric GPT-4
tokenization spends **4.21×** the tokens on the same Tamil text
([`docs/tokenizer/TOKEN_TAX_BENCHMARK_V3.md`](docs/tokenizer/TOKEN_TAX_BENCHMARK_V3.md),
[`docs/tokenizer/WHITEPAPER_SWARAM.md`](docs/tokenizer/WHITEPAPER_SWARAM.md)) — that is the
token tax. The claim that Swaram *solves* it did not survive its first controlled test: a
Tamil-trained SentencePiece BPE is 1.52× more compressed than Swaram (0.346 vs 0.527
tok/akshara), and Swaram's 0.527 is essentially tied with Sarvam-1's 0.538
([`docs/reports/ab_swaram_vs_bpe.md`](docs/reports/ab_swaram_vs_bpe.md)).

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
- **Tamil Tokenizer Benchmark (`src/adhan_slm/eval/tokenizer_bench.py`):** fertility, grapheme splits and cost per 1k sentences on a frozen eval corpus. Reproduce with `scripts/reproduce_benchmark.sh` — see [`docs/REPRODUCE_BENCHMARK.md`](docs/REPRODUCE_BENCHMARK.md).

---

## License

MIT
