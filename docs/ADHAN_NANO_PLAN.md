# `adhan-nano` — pretraining to production plan

Status: **Stage 1 running (CPU).** Every number marked *(measured)* came from this
machine (12-core CPU, no GPU, 15 GB RAM) during this plan; numbers marked *(est.)*
are derived and must be re-measured per host.

> **Corrections to the first draft of this plan** (kept here rather than silently
> edited, because two of them were wrong in ways worth remembering):
>
> 1. **The ETA was wrong by ~1.6×.** `train_jax`'s pre-flight estimate comes from a
>    hard-coded per-platform throughput hint (3,000 tok/s), not from the machine. The
>    banner said 5 h 33 m; the run measured **1,950 tok/s = 16.7 s/update → ~8.5 h**.
>    Fixed in the trainer (it now logs a measured ETA after the first flush) — see
>    PR #94.
> 2. **This CPU profile is not the shipping profile.** It is the 6-layer / MHA /
>    14.47M-param config. The T4 profile supersedes it for any GPU run: **9 layers,
>    8 query / 2 KV heads (GQA), dropout 0.1, fp16, peak LR 5e-4, 16,740,096 params**
>    — `src/adhan_slm/configs/adhan_slm_nano_t4.yaml`, documented in
>    [`COLAB_T4.md`](COLAB_T4.md) (PR #94).
> 3. **`model.dropout` was dead config until #94**: the field existed and was never
>    applied. Any earlier run in this plan therefore trained with dropout *off*,
>    regardless of what a config said.

```
corpus 1.33M docs / 1.06G chars / ~293M tokens   (frozen, dedup'd, 0 exact dupes)
        │
        ▼  tokenizer v2: 31,964 vocab, byte-lossless, 0.349 tok/akshara
30M-token shard set @ seq_len 512                (Stage 0, done)
        │
        ▼  adhan-nano: 6L x 256d, tied embeddings, 14.47M params
pretrain 1,830 updates / ~60M tokens             (Stage 1, running, ~8.5h measured)
        │
        ▼  eval gates (bits/akshara, sandhi, boundary agreement, tamil-bench)
SFT on intended behaviour → quantize → serve      (Stages 2–4)
```

---

## Stage 0 — artifacts the run depends on (done)

| Artifact | Value | Where |
|---|---|---|
| Corpus | 1,334,035 docs · 1,055,782,057 chars · ~293M swaram tokens | `data/raw/expanded` (70 shards) |
| Tokenizer | vocab 31,964 · 28,087 merges · byte fallback · fertility 0.349 · 100% lossless *(measured)* | `data/final/tamil_slm_v2` |
| Shards (full) | 293.5M tokens, 85,857 train / 1,768 val seqs @ 1024 | `data/final/tamil_slm_v2` |
| Shards (nano, seq 512) | **15.33M train + 0.32M val tokens** packed from a 30M-token budget | `data/final/nano_30m_s512` |
| Morph-guarded variant | fertility 0.374, boundary agreement 37.5% vs 13.6% | `data/final/tamil_slm_30m_morph` |

**Param math** (this is the decision that shapes everything else):

| tied vocab | params | embedding share |
|---|---|---|
| 4,000 (old cpu config) | 7.32M | 14% |
| 8,000 | 8.34M | 25% |
| 12,000 | 9.36M | 33% |
| **31,964 (v2)** | **14.47M** | **57%** |

We run the 32k vocabulary because it is the one that is byte-lossless and 3–4× more
compressed than BPE. The price is that 57% of the model is an embedding table and
throughput drops: **2.04k tok/s** *(measured)* versus ~3.0k tok/s for the 4k config on
4 cores. Stage 3 revisits this with an 8k-vocab variant for the shippable CPU artifact.

**Wiring gate — `--overfit-batch`** *(measured, passed)*: loss 10.3656 → 0.6739
(7% of start, target <50%) in 200 updates, ~2,100 tok/s. No point spending hours on a
run whose loss cannot collapse.

---

## Stage 1 — pretraining (RUNNING)

```bash
python -m adhan_slm.training.train_jax \
    --config src/adhan_slm/configs/adhan_slm_nano_ta.yaml --device cpu
```

| Setting | Value | Why |
|---|---|---|
| model | 6L × 256d, 4 heads, d_ff 1024, tied embeddings | the nano tier |
| seq_len | 512 | matches the packed shard manifest exactly |
| batch / accum | 8 × 8 = 64 seqs = **32,768 tokens/update** | CPU memory vs gradient noise |
| lr | 6.0e-4, 200 warmup | small model tolerates a high LR |
| updates | 1,830 | 2 epochs over the 30M-token set (~60M tokens) |
| throughput | **1,950–2,040 tok/s** *(measured)* | |
| ETA (pre-flight) | 5 h 33 m | from a hard-coded 3,000 tok/s platform hint — **not** a measurement |
| ETA (measured) | **~8.5 h** (16.7 s/update × 1,830) | what the box actually needs; the trainer now logs this after the first flush |
| checkpoint | every 100 updates, keep 3, best-by-val | resumable; `--resume` picks up global step |

**What 60M tokens means for a 14.5M-param model:** 4.1 tokens/param, i.e. **5× below
the Chinchilla ~20:1 heuristic.** This run is a *plumbing + trend* milestone, not a
quality milestone. The full archive (293M tokens) is 20 tokens/param — exactly the
compute-optimal point for this size, and therefore the natural Stage 2 target.

**Operating it**

```bash
tail -f logs/pretrain_nano_ta.log                     # progress, tok/s, val loss
python scripts/monitor_training.py                    # MLflow view
# resume after interruption (nothing is lost by Ctrl-C):
python -m adhan_slm.training.train_jax --config ... --resume
```

Gate to leave Stage 1: **val loss trending down, no divergence, checkpoints
loadable.** Stop-and-continue decisions are made on the val curve, not the step count.

---

## Stage 2 — evaluation gates (before any "quality" claim)

Run the existing harness; it degrades gracefully, so every section that cannot run
says so instead of crashing:

```bash
python -m adhan_slm.eval.run_eval \
    --tokenizer-dir data/final/tamil_slm_v2 \
    --eval-text data/raw/expanded \
    --config src/adhan_slm/configs/adhan_slm_nano_ta.yaml \
    --checkpoint checkpoints/adhan-nano-ta \
    --out docs/eval_report_nano.json
```

| Gate | Metric | Threshold to pass | Baseline we must beat |
|---|---|---|---|
| G1 fertility | tokens/akshara on held-out | < 1.15 | 0.349 already *(measured)* |
| G2 model | **bits/akshara** | below the classical unigram floor | floor ppl 98.9 *(measured)* |
| G3 morphology | boundary agreement | ≥ 30% | 13.6–37.5% depending on guard |
| G4 sandhi | sandhi-correct rate | ≥ 0.90 | 0.9223 on corpus text *(measured)* |
| G5 kid prompts | grammatical Tamil on 50 prompts | human read-through | — |
| G6 external | MILU-Tamil / IndicQA / XNLI | report, no threshold yet | frontier: 0.945 / 0.200 EM / — |

**`bits/akshara = (nll_per_token / ln 2) × (tokens / akshara)`** is the only
cross-tokenizer comparison that doesn't flatter whoever has more merges. Publish it
next to fertility so the tokenizer advantage is visible instead of rhetorical.

Decision at the end of Stage 2: (a) extend to the 293M full archive, (b) shrink the
vocabulary to 8k and re-run for the CPU-shippable artifact, or (c) stop and fix data.

---

## Stage 3 — post-training (make it useful, cheaply)

Pretraining teaches Tamil orthography and morphology; it does not teach answering.

| Step | Data | Method | Cost |
|---|---|---|---|
| SFT (kid register) | 50 kid prompts + Thirukkural harness (1,330 pairs, already generated by `raw_extractors/thirukkural_extractor.py`) | supervised fine-tune, same trainer, 1–2 epochs | < 1h |
| SFT (task) | curated instruction set (QA from `vazhi`, syllabus text) | 2 epochs, lr 1e-4 | hours |
| Preference (optional) | sandhi/morphology-correct vs broken generations | DPO | skip until SFT is stable |

Guardrails for a 14.5M model: it will not follow open-ended instructions. Scope the
product to **completion, correction, and curriculum-shaped tasks**, and refuse
out-of-scope chat explicitly rather than emitting plausible garbage.

---

## Stage 4 — packaging, serving, production

**Package**

```bash
python scripts/quantize_model.py --checkpoint checkpoints/adhan-nano-ta \
    --output models/adhan-nano-ta-int8.onnx --format int8
python scripts/export_onnx.py --checkpoint checkpoints/adhan-nano-ta
```

| Artifact | Size (est.) | Target |
|---|---|---|
| fp32 checkpoint | ~58 MB | reference / server |
| int8 ONNX | ~15 MB | edge, laptop, phone-adjacent |
| tokenizer.json | 2.8 MB | ships with the model |

**Serve** — the server is already OpenAI-compatible, which means `bench/tamil-bench`
can point at it with `--endpoint` and nothing bespoke is needed:

```bash
python scripts/run_api_server.py --model adhan-nano-ta --host 0.0.0.0 --port 8000
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"adhan-nano-ta","messages":[{"role":"user","content":"திருக்குறள்:"}]}'
```

| Production concern | Plan |
|---|---|
| Latency | measure p50/p95 tokens/s on the target host; publish it, do not guess |
| Concurrency | uvicorn workers = cores; the model is tiny enough to keep several hot |
| Health | `/health` must report model hash + shard version, not just "ok" |
| Monitoring | log prompt length + token throughput + refusal rate (no raw user text by default) |
| Drift / regression | nightly `run_eval` on a fixed held-out slice; alert on bits/akshara regression >5% |
| Rollout | internal canary → bounded beta → public; keep the previous artifact served |
| Rollback | checkpoints are content-addressed by step; serving config pins an artifact dir |
| Governance | model card + datasheet (`code_sha`, corpus hash, tokenizer hash, eval numbers), PII policy, licence |

**Reproducibility checklist** (all already emitted by the pipeline): `code_sha`,
`datasheet.json` for corpus and tokenizer, shard manifests with dtype/seq_len,
training seed, MLflow run id, and the exact command line in the model card.

---

## Stage 5 — the honest capability statement

At 14.5M parameters and ~60M tokens this model is a **Tamil text engine**, not an
assistant. What it can plausibly do well: next-token completion in classical and
modern Tamil, orthographic/sandhi-consistent output, curriculum-shaped exercises,
on-device inference at a few megabytes. What it cannot do: reason, follow multi-step
instructions, answer general knowledge questions reliably, or compete with frontier
multilingual models on MILU/IndicQA.

That claim is defensible precisely because it is narrow — see
`docs/MULTILINGUAL_SCOPE.md` for why the akshara tokenizer, not scale, is the durable
advantage, and why we publish bits/akshara instead of perplexity.

---

## Timeline (measured where possible)

| When | What | Wall clock |
|---|---|---|
| now | Stage 0 artifacts + overfit gate | done |
| +8.5 h | Stage 1 pretrain (1,830 updates, 60M tokens) | ~8.5 h *(measured)*, not the 5 h 33 m hint |
| +6 h | Stage 2 eval gates + report | ~30 min |
| +8 h | Stage 3 SFT (kid prompts + Thirukkural) | 1–2 h |
| +9 h | Stage 4 quantize + serve + latency measurement | ~1 h |
| next | Stage 2b: extend to 293M tokens (20 tok/param) | ~40 h *(est.)* |
| next | 8k-vocab variant for the shippable CPU artifact | ~4 h *(est.)* |

## Risks

| Risk | Mitigation |
|---|---|
| 32k vocab is 57% of params on CPU → slow, embedding-dominated | 8k variant is a 15-min freeze + repack; measure both |
| 60M tokens ≪ Chinchilla → undertrained, misleading samples | label Stage 1 as plumbing; extend to 293M before any quality claim |
| Pre-flight ETA silently 1.6× optimistic | fixed in #94: measured ETA logged after the first flush, and the hint it replaced is named in the log |
| Loss looks fine but generations are garbage | G5 human read-through + sandhi probe are gates, not optional |
| Tokenizer advantage overstated as model quality | publish bits/akshara **and** MILU/IndicQA, including the losses |
| CPU-only host dies mid-run | checkpoint every 100 updates; `--resume` restores the global step |

---

**Next in the stack:** the GPU profile — [`COLAB_T4.md`](COLAB_T4.md): 9 layers, 8 query / 2 KV heads (GQA), dropout 0.1, fp16, 16,740,096 params.
