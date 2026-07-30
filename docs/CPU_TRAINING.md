# CPU Training — Adhan SLM

Adhan is designed to be trainable **without a GPU**. `adhan-nano` is small enough
(~6–8M params at a 4k swaram vocab) that a laptop or a CI runner can pretrain it,
which matters for three reasons the roadmap depends on:

- **The sanity gates are free.** Overfit-a-batch, config validation, resume, and the
  eval harness all run on CPU, so no GPU hour is spent discovering a wiring bug.
- **Contributors need no accelerator.** The whole pipeline — corpus → tokenizer →
  packed shards → training → generation — runs on the machine you already have.
- **Edge is the launch target.** Phase 6 ships an INT4 nano running on a Raspberry
  Pi 5. A model that cannot be *trained* on CPU-class hardware is a model whose
  inference cost nobody has felt.

---

## 1. Install (no CUDA)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[jax,dev]"
```

`.[jax]` installs the **CPU** JAX wheels. GPU is opt-in: `pip install -e ".[jax-cuda]"`.

Verify the backend:

```bash
python -c "import jax; print(jax.default_backend(), jax.devices())"
# cpu [CpuDevice(id=0)]
```

## 2. Build a packed corpus

Pack at the same sequence length your model config declares — attention and
activation memory both scale with it, so 256 tokens is the CPU-sane default.

```bash
python scripts/prepare_slm_corpus.py \
    --corpus data/raw/tamil/ \
    --out data/final/tamil_slm_cpu \
    --vocab-size 4000 --seq-len 256
```

This freezes `vocab.json` + `merges.txt`, writes `train.bin` / `val.bin` (+ manifests)
and a `datasheet.json`. The trainer reads the frozen vocab size from `vocab.json` and
overrides the config, so the embedding rows always line up with the ids in the shards.

## 3. Sanity-gate the wiring before any long run

```bash
python -m adhan_slm.training.train_jax \
    --config src/adhan_slm/configs/adhan_slm_nano_cpu.yaml \
    --device cpu --overfit-batch
```

This repeats **one** batch until the loss collapses. A correctly wired causal LM
memorises a single batch almost completely; if the loss will not fall below half its
starting value the trainer exits non-zero and the bug is in the model, the optimizer
wiring or the token ids — *not* in the learning rate. Takes 2–3 minutes at the nano
config's `batch_size: 8` / `max_seq_len: 256`, and ends with:

```
overfit-a-batch PASSED: loss 6.4791 -> 0.0183 (0% of start, target <50%)
```

The gate also shortens `warmup_steps` to `max_steps // 10` for itself: a production
warmup of 200+ over a 200-update gate would keep the LR near zero for the whole run
and fail for the one reason the gate is not meant to detect — too little training.

There is also `--smoke`, which trains a few steps on synthetic tokens and needs no
corpus at all — the fastest possible "does the loop run here" check.

## 4. Train

```bash
python -m adhan_slm.training.train_jax \
    --config src/adhan_slm/configs/adhan_slm_nano_cpu.yaml --device cpu
```

Every run opens with a pre-flight banner so the cost is visible *before* it is paid:

```
backend=cpu x1 (cpu) jax=0.10.2 dtype=bfloat16 cores=4
~6.4M params · 16,384 tok/step · 3,000 steps · rough ETA 4h 33m
```

If the estimate exceeds 6 hours on CPU the trainer warns and points at the nano config
— a multi-day CPU run is nearly always a GPU config aimed at the wrong machine, and
the failure mode is the worst kind: it looks like it is working.

## 5. What actually makes CPU training viable

### Mixed precision: keep bfloat16

Counter to the usual assumption, bf16 is **not** a pessimisation on CPU. Measured on a
4-core x86 runner (jax 0.10, `adhan-nano`, batch 8 × 256 tok):

| compute dtype | throughput | ms/step |
|---|---|---|
| `bfloat16` | **~3.0k tok/s** | 680 |
| `float32`  | ~2.0k tok/s | 1014 |

XLA:CPU lowers bf16 fine and the halved memory traffic wins. So `dtype: bfloat16`
stays the default on CPU. `float16` is *rejected* on CPU rather than silently upcast —
a run whose logged precision is wrong is a run that cannot be reproduced
(ROADMAP_JAX_SLM §5).

### Gradient accumulation, not a bigger batch

A CPU cannot hold a batch of 32 × 1024 tokens at a useful speed, but small batches
make noisy gradients. `grad_accum_steps` decouples the two: **effective batch =
`batch_size` × `grad_accum_steps`**, at the peak memory of `batch_size` alone.

```yaml
train:
  batch_size: 8         # what one CPU step holds
  grad_accum_steps: 8   # effective batch 64
  max_steps: 3000       # counts OPTIMIZER UPDATES, not micro-batches
```

`max_steps` counts optimizer updates, so `max_steps: 3000` with `grad_accum_steps: 8`
runs 24,000 micro-batches over ~49M tokens. The LR schedule and Adam moments advance per *update*
(via `optax.MultiSteps`), so the schedule means the same thing on CPU and GPU.

### Shorter context beats a smaller model

Attention cost grows with sequence length; parameter count does not. Halving
`max_seq_len` from 512 to 256 buys more throughput than halving `d_model`, and costs
less quality for kid-level Tamil, where sentences are short anyway.

### Resume is expected, not exceptional

Long CPU runs get interrupted. Orbax checkpoints are written every
`checkpoint_every` updates, and a restart picks up the **global** step budget:

```
resumed from checkpoint step 2199 (2200 of 3000 updates already done)
```

so the remaining 800 updates run — not another full 3000. Optimizer momentum and the
LR schedule position are restored with the state.

### Thread count is not a knob you have

XLA:CPU parallelises each op across cores through its own thread pool;
`OMP_NUM_THREADS` has no measurable effect (verified: 3,100 vs 3,130 tok/s at 1 vs 4).
`--cpu-devices N` exists only to expose N *addressable* devices for exercising
sharded/`pmap` code paths on a CPU box — it does not make a single-device run faster.

## 6. Reference: nano on CPU

| Setting | Value | Why |
|---|---|---|
| tier | `nano` (6 layers, d_model 256) | the tier Phase 6 ships |
| `vocab_size` | 4000 | tied output head is a full vocab matmul every step |
| `max_seq_len` | 256 | attention + activations scale with this |
| `dtype` | `bfloat16` | ~1.5× float32 on XLA:CPU (measured) |
| `batch_size` | 8 | fits comfortably in a few GB |
| `grad_accum_steps` | 8 | effective batch 64 |
| `max_steps` | 3000 | ~49M tokens, ~4.5h — an overnight run |
| throughput | ~3.0–3.2k tok/s | measured, 4-core x86, jax 0.10 |

At ~3k tok/s a 100M-token dry run is ~9 hours of CPU — fine overnight, and the honest
reason the full `adhan-nano` pretrain still wants a GPU (Phase 3). CPU's job is to
make every step *before* that one free.

## 7. Tests

The CPU path is covered end-to-end and runs in about a minute:

```bash
pytest tests/integration/test_train_cpu.py -v        # corpus -> shards -> train
python -m adhan_slm.training.test_device             # dtype/backend resolution
```

`tests/integration/test_train_cpu.py` asserts the backend really is CPU, that
overfit-a-batch collapses the loss, that gradient accumulation scales the effective
batch, that resume continues the step budget, and that a shard packed longer than
`model.max_seq_len` is rejected instead of silently mis-training.

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ConfigError: model.dtype='float16' is not supported on cpu` | Use `bfloat16` (mixed precision) or `float32`. |
| `ConfigError: packed shard sequence length N exceeds model.max_seq_len` | Repack with `--seq-len <max_seq_len>`, or raise `model.max_seq_len`. |
| `FileNotFoundError: packed shard not found` | Run `scripts/prepare_slm_corpus.py` first; check `data.shards` in the config. |
| `configure_backend() ran after jax was already imported` | Something imported `jax` before `main()`. `--device`/`--cpu-devices` cannot take effect; import the trainer's `train()` directly with the env already set. |
| ETA warning about a multi-hour run | Intended. Lower `max_steps` / `batch_size` / `max_seq_len`, or use the nano CPU config. |
| `mlflow not installed — metrics will not be tracked` | Optional; `pip install mlflow` and set `mlflow_uri: file:./mlruns`. |

## See also

- [`ROADMAP_JAX_SLM.md`](../ROADMAP_JAX_SLM.md) — Phase 3 (pretrain) and Phase 6 (ship)
- [`docs/TRAINING_GUIDE.md`](TRAINING_GUIDE.md) — the PyTorch baseline pipeline
- [`docs/PERFORMANCE.md`](PERFORMANCE.md) — single-GPU throughput work
- [`docs/EVAL_TAMIL.md`](EVAL_TAMIL.md) — the eval harness these runs feed
