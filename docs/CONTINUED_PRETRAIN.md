# Stage 1b — continued pretraining of `adhan-nano`

Status: **running** (started 2026-10-24 22:31 IST, CPU).
Config: [`adhan_slm_nano_ta_cont.yaml`](../src/adhan_slm/configs/adhan_slm_nano_ta_cont.yaml).
MLflow: experiment `adhan-nano-ta`, run `0de93ec8` (`amusing-newt-112`).
Log / PID: `logs/pretrain_nano_ta_cont.log`, `logs/pretrain_nano_ta_cont.pid`.

---

## 1. Why this run exists

Stage 1 trained 1,830 updates (~60M tokens over the old 30M-token shard set) and
finished at train loss 3.789 / val loss 3.890 (old val split). The loss curve had
flattened, and the eval showed the model only just beating the classical unigram
floor (see `docs/eval_report_nano.json`). The bottleneck was **data budget**, not
architecture: 60M tokens is ~4 tokens/param for this 14.5M-param model, 5× below the
Chinchilla ~20:1 point. The full archive is ~293M tokens, so the correct move was to
continue this model rather than restart it.

What changed:

| | Stage 1 | Stage 1b (this run) |
|---|---|---|
| shards | `nano_30m_s512` (15.3M train tokens) | `nano_300m_s512` (133.6M train tokens) |
| budget | 1,830 updates / ~60M tokens | +3,052 updates / ~100M tokens (1830 → 4,882) |
| weights | from scratch | resumed from step 1829 |
| optimizer | fresh | **reset** (warm restart), LR peak 3.0e-4 |
| checkpoint dir | `checkpoints/adhan-nano-ta` | same (continues step numbering) |

Stage-1 checkpoints were copied to `checkpoints/adhan-nano-ta-step1829-backup/`
before the run started (Orbax `keep_checkpoints: 4` eventually rotates the older ones).

---

## 2. Repacking (step 1)

```bash
python scripts/freeze_fast_tokenizer.py \
    --corpus data/raw/expanded \
    --out data/final/nano_300m_s512 \
    --reuse-tokenizer data/final/tamil_slm_v2 \
    --seq-len 512 --val-frac 0.02 --max-tokens 300000000
```

Result (`nano_300m_s512/datasheet.json`, `*.manifest.json`):

| | |
|---|---|
| corpus | 1,334,114 docs · 1,055.8M chars · 293.5M corpus tokens |
| tokenizer | reused v2: 31,964 vocab · fertility 0.349 · 100% lossless |
| train | 261,003 seqs × 512 = **133,633,536 tokens** |
| val | 5,350 seqs × 512 = **2,739,200 tokens** |
| time | 120 s (265 MB on disk) |

> **Known defect — per-document packing throws away half the corpus.**
> `pack_stream()` drops each document's sub-512-token remainder, and the average
> document is ~220 tokens, so 293.5M corpus tokens pack to only 133.6M usable
> tokens (~54% discarded). The budget flag counts *fed* tokens, not *kept* ones.
> Fixing this means packing all documents through one continuous stream (carry the
> remainder across document boundaries) in `freeze_fast_tokenizer.py` /
> `prepare_slm_corpus.py`. Until then, "300M-token shard set" means ~134M packed.

---

## 3. Warm restart semantics (step 2)

`train_jax.py` gained `train.warm_restart_on_resume` (default `false`). It matters:

- **Exact resume** restores the optimizer, including its schedule count. That is only
  correct if `max_steps` did not change. Applied to a 10× longer cosine it would snap
  the LR from its end value back to near peak for a step that is only 37% into the new
  schedule.
- **Warm restart** keeps `params` from the checkpoint but creates a fresh
  `TrainState` (Adam moments + schedule count start at 0), so the new LR schedule is
  followed honestly: warmup 0 → 3.0e-4, then cosine decay to 3.0e-5 at step 4,882.
- The `learning_rate` metric is indexed by that reset schedule count
  (`s = step - completed`), not by the global step.

Verified before launch with a 2-step run on a copy of the checkpoint: params restored
at 1829, warm-restart banner, first logged LR 0.0, then 3e-6, 6e-6, … (correct ramp).
The real run started with `train_loss 3.8723` at step 1839.

`train.warm_restart_on_resume` is a **planned-extension** switch. For crash recovery
(see §6) leave it `false` so the optimizer state and LR position are restored exactly.

---

## 4. Schedule (measured, not the pre-flight hint)

- throughput: **1,900–2,100 tok/s** (17–18 s/update) on this 12-core CPU
- tokens/update: 32,768 (batch 8 × accum 8 × seq 512)
- budget: 3,052 updates ≈ 100M tokens → **~14.5–15 h**
- expected finish: **~13:00–14:00 IST, 2026-10-25**

Checkpoint / val cadence is every 100 updates (`keep_checkpoints: 4`), plus the last
step. Approximate milestones from the 22:31 start:

| progress | global step | wall clock (IST) |
|---|---|---|
| 25% | ~2593 | ~02:00–02:30 Oct 25 |
| 50% | ~3356 | ~05:30–06:15 |
| 75% | ~4119 | ~09:15–10:00 |
| final | 4881 | ~13:00–14:00 |

Treat these as ±1 h; the per-update rate varies with cron ingest and system load.

---

## 5. Benchmarking protocol — when and how

### 5.1 When to look

| moment | what to run | why |
|---|---|---|
| **now → 25%** | nothing (watch `val_loss` only) | LR is still near peak; batch-level train loss is intentionally noisy (±0.4) and says nothing |
| **mid-run: checkpoints 3399 and 4199** | G1–G4 + model ppl (matching val, §5.2) | early warning only. Expect a small win over the unigram floor and still-bad generations. If bits/akshara is *worse* than the 1829 baseline, stop and fix instead of burning 10 more hours |
| **final checkpoint 4881 (and 4799 / best-val)** | full harness, publish numbers | LR has decayed to 10% of peak; this is the first honest pretraining result. This is when `docs/eval_report_nano.json` should be regenerated |
| **after kid-level SFT (Stage 3)** | G5 (50 prompts), human read-through | generation/instruction quality is a *post-training* claim. Judging an autocomplete model on Q&A prompts at the pretraining gate produces the wrong conclusion |

So: **the right time for the real benchmark is after the run finishes, ~13:00–14:00
IST on Oct 25**, using the final/best-val checkpoint. The only benchmark worth running
before that is one cheap mid-run sanity pass at ~step 3399 (≈06:00 IST).

Also re-baseline **checkpoint 1829 on the new val set** before comparing anything.
The old 83.0 ppl in `docs/eval_report_nano.json` was measured against
`tamil_slm_v2/val.bin` (seq_len 1024, different split) and is not comparable to this
run's numbers. On the new val, 1829 scores ≈ **50.4 ppl** (nll 3.92; measured by the
2-step probe). For reference, the run had already reached val nll 3.799 / ppl 44.6 by
step ~2400 — real progress, and the reason the final number must use one fixed val set.

### 5.2 Critical: evaluate against the same packed val set

`run_eval`'s model probe reads `val.bin` from `--tokenizer-dir`, so pass the **shard
dir**, not the old tokenizer dir:

```bash
python -m adhan_slm.eval.run_eval \
    --tokenizer-dir data/final/nano_300m_s512 \
    --eval-text data/raw/expanded \
    --config src/adhan_slm/configs/adhan_slm_nano_ta_cont.yaml \
    --checkpoint <checkpoint-snapshot-dir> \
    --out docs/eval_report_nano_stage1b.json
```

Eval while training is running competes for the same cores. Snapshot the step first
(a single step dir is a valid Orbax manager dir), then evaluate it:

```bash
mkdir -p /tmp/adhan-eval-3399
cp -a checkpoints/adhan-nano-ta/3399 /tmp/adhan-eval-3399/
nice -n 19 python -m adhan_slm.eval.run_eval \
    --tokenizer-dir data/final/nano_300m_s512 \
    --eval-text data/raw/expanded \
    --config src/adhan_slm/configs/adhan_slm_nano_ta_cont.yaml \
    --checkpoint /tmp/adhan-eval-3399 \
    --out /tmp/eval_3399.json
```

At the end (training idle) point `--checkpoint checkpoints/adhan-nano-ta` directly —
it loads the latest step.

### 5.3 Gates and the one number that matters

| gate | metric | pass | baseline |
|---|---|---|---|
| G1 fertility | tokens/akshara | < 1.15 | 0.357 ✅ (tokenizer, model-independent) |
| G2 model | **bits/akshara** below unigram floor | < 2.322 | stage 1: 2.274 |
| G3 morphology | boundary agreement | ≥ 30% | 24.2% ❌ (tokenizer property; unchanged by pretraining) |
| G4 sandhi | sandhi-correct rate | ≥ 0.90 | 91.5% ✅ |
| G5 kid prompts | grammatical Tamil, 50 prompts | human read-through | only meaningful after SFT |

`bits/akshara = (nll_per_token / ln 2) × fertility` is the only cross-tokenizer
comparison worth quoting. With fertility 0.3567: unigram floor nll 4.510 → **2.322**;
stage-1 model nll 4.419 → **2.274**; step ~2400 training val nll 3.799 → **1.955**.

Note G3 is a tokenizer/morphology property — this continuation will not move it.
Do not read a flat 24.2% as "pretraining failed".

---

## 6. Operating it / if interrupted

```bash
tail -f logs/pretrain_nano_ta_cont.log
.venv/bin/python scripts/mlflow_status.py --live --plot
```

If the process dies, **resume exactly** (keeps Adam + LR position):

1. set `train.warm_restart_on_resume: false` in the config,
2. rerun the same command; it restores the latest step and continues to 4,882.

Do *not* re-enable warm restart for a crash: that would reset Adam moments and the
LR schedule mid-run. The flag is only for deliberately extending a finished run.

Normal finish: `done (pretrain).` in the log, Orbax checkpoint at 4881, MLflow status
`FINISHED`.

---

## 7. Follow-ups this run does not cover

1. **Cross-document packing fix** (§2) — the cheapest data win; would make the
   archive's ~293M tokens actually trainable.
2. **Tiṇai routing / boundary embeddings** — still off (`use_tinai: false`).
3. **Kid-level SFT** — required before judging generation quality; without it the
   50-prompt gate is measuring the wrong thing.
4. **Morphology gate (G3)** — needs a tokenizer/aux-objective change, not more tokens.
