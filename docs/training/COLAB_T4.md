# Pretraining `adhan-nano` on a free Colab T4

The free T4 tier is the cheapest way to get this model trained properly: 16 GB of
VRAM, fp16 tensor cores, and enough session time to cover the **whole 293.5M-token
archive in one go** (~3.3–5.4 h at an expected 15–25k tok/s). This page is the
recipe; the notebook is `notebooks/adhan_nano_t4_pretrain.ipynb`.

Config: `src/adhan_slm/configs/adhan_slm_nano_t4.yaml`
Model: **9 layers × 256 d_model · 8 query heads / 2 KV heads (GQA) · dropout 0.1 ·
pure RoPE · 16,740,096 params**

## Why this configuration is T4-shaped

| Choice | Reason |
|---|---|
| `dtype: float16` | The T4 is Turing (sm_75): fp16 tensor cores, **no bf16**. bf16 starts at Ampere. Master weights stay fp32 and the head accumulates logits in fp32, which is what makes fp16 training here stable. NaN in the first ~200 updates → switch to `float32` (≈half speed, no instability). |
| `n_kv_heads: 2` of 8 query heads | K/V projections shrink 4× in KV memory. On a 16 GB card with 512-token sequences this is what lets `batch 32` fit and keeps attention cheap. |
| 9 layers (from 6) | Free capacity at 16.7M params; the T4 is idle-bound, not FLOP-bound, at this size. |
| 8 heads, head_dim 32 (from 4×64) | Smaller heads + GQA share more computation per parameter; 4:1 query:KV grouping is the standard sweet spot. |
| `dropout: 0.1` | One epoch over ~293M tokens is enough to overfit a 16.7M model; dropout is applied to embeddings, attention output and MLP output, and is switched off at eval. |
| `batch 32 × accum 2` | 32,768 tokens per optimizer update at 32 × 512 per micro-batch — fits comfortably, keeps the update count at 8,957 for one epoch. |
| `lr 5e-4`, warmup 200 | Slightly below the CPU config's 6e-4 because the effective batch is wider. |
| `eval_every` / `checkpoint_every: 200` | 200 updates ≈ 6.55M tokens ≈ 4–7 min on a T4. Preemption then costs minutes, not a session. |

## Packed shards (the only data you need)

The tokenizer and packed shards are small enough to move around:

| Artifact | Size | Contents |
|---|---|---|
| `data/final/tamil_slm_v2_s512/train.bin` | ~587 MB | 293.5M tokens, seq_len 512 |
| `data/final/tamil_slm_v2_s512/val.bin` | ~12 MB | 2% held-out |
| `vocab.json` / `merges.txt` / `tokenizer.json` | ~3 MB | the frozen v2 tokenizer |

Build them on the dev box (2–3 min with the Rust encoder):

```bash
python scripts/freeze_fast_tokenizer.py \
  --corpus data/raw/expanded \
  --out data/final/tamil_slm_v2_s512 \
  --reuse-tokenizer data/final/tamil_slm_v2 \
  --seq-len 512 --max-tokens 0
```

Publish to the Hugging Face Hub once (Colab pulls it back in seconds):

```bash
huggingface-cli upload <user>/adhan-tamil-shards data/final/tamil_slm_v2_s512 . --repo-type dataset
```

Then in Colab: `snapshot_download("<user>/adhan-tamil-shards", repo_type="dataset", local_dir="data/final/tamil_slm_v2_s512")`.
Google Drive works too, but Hub is faster and versioned.

## Colab cells

```python
# 1) JAX with CUDA 12 — the preinstalled jax is CPU-only
!pip -q install "jax[cuda12]" flax optax orbax-checkpoint mlflow "tokenizers>=0.15"
import jax; print(jax.devices())          # expect [CudaDevice(id=0)] (Tesla T4)

# 2) repo + data
!git clone https://github.com/yazhi-lem/adhan.git && cd adhan
!pip -q install -e .
from huggingface_hub import snapshot_download
snapshot_download("<user>/adhan-tamil-shards", repo_type="dataset",
                  local_dir="data/final/tamil_slm_v2_s512")

# 3) wiring gate first — 3 minutes, loss must collapse (<50% of start)
!python -m adhan_slm.training.train_jax \
    --config src/adhan_slm/configs/adhan_slm_nano_t4.yaml --overfit-batch --device gpu

# 4) the run (resumable: rerun this exact cell after a preemption)
!python -m adhan_slm.training.train_jax \
    --config src/adhan_slm/configs/adhan_slm_nano_t4.yaml --device gpu
```

Checkpoints land in `checkpoints/adhan-nano-t4/` **on the Colab VM**, which is
wiped when the session ends. Mount Drive *before* starting and symlink, or copy
the best checkpoint out at the end:

```python
from google.colab import drive; drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/adhan && ln -sfn /content/drive/MyDrive/adhan checkpoints
```

## Preemption playbook

Free sessions die without warning. The run is built for it:

- `checkpoint_every: 200` → at most ~5 min of work lost.
- `--resume` is automatic: on restart the trainer restores the **global** step,
  optimizer state and LR schedule, and continues to `max_steps`.
- The val split is a separate `val.bin`, so a resumed run cannot leak training
  data into validation.

If the session dies mid-checkpoint, Orbax's `commit_success.txt` marks the last
complete one; the next run picks that up.

## Expected numbers (to sanity-check the run, not to promise)

| Quantity | Expected on T4 |
|---|---|
| throughput | 15–25k tok/s (16.7M params, fp16) |
| 1 episode | 8,957 updates ≈ 293.5M tokens ≈ 3.3–5.4 h |
| first val loss at step 200 | ~5.5–6.5 (see the CPU run's 5.60 at step 200) |
| params | 16,740,096 |
| checkpoint size | ~200 MB (fp32 master + Adam moments) |

If throughput lands well below 15k tok/s, check in order: `jax.devices()` reports
the GPU, dtype is `float16` (not the config default `bfloat16`), `XLA_FLAGS` is
not forcing CPU, and `batch_size` is 32 (not 8 — a small batch on a T4 is
launch-overhead-bound).

## Post-run, on the dev box

```bash
python -m adhan_slm.eval.run_eval \
  --tokenizer-dir data/final/tamil_slm_v2 \
  --eval-text data/raw/expanded \
  --config src/adhan_slm/configs/adhan_slm_nano_t4.yaml \
  --checkpoint checkpoints/adhan-nano-t4 \
  --out docs/eval_report_nano_t4.json
```

Then quantize (`scripts/quantize_model.py --format int8`) and serve through
`scripts/run_api_server.py`. See `docs/training/CONTINUED_PRETRAIN.md` §5 for the gates and
`docs/strategy/MULTILINGUAL_SCOPE.md` for what to claim afterwards.

---

**Next in the stack:** after the T4 run — the eval gates in [`CONTINUED_PRETRAIN.md`](CONTINUED_PRETRAIN.md) §5 (bits/akshara vs the classical floor, boundary agreement, sandhi, kid prompts), then `scripts/quantize_model.py` → `run_api_server.py`.
