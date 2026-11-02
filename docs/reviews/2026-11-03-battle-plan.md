# Adhan — Senior ML Review & Battle Plan

**Date:** 2026-11-03 · **Scope:** history, current state, next work · **Author:** Adhan DRI (review)

> This is a deliberately blunt review. It is written to be acted on, not admired.
> Every claim below is checkable against a file, a run, or a command in this repo.

---

## 1. The roast

### 1.1 You have a tokenizer paper and a model-shaped placeholder

The actual artifact is a **14.5M-parameter, 6-layer, d=256** transformer trained on a
**12-core CPU** for roughly 100M tokens. That is not a foundation model; it is a
distilgpt2-class toy. The Stage 1b generations ("Rails Satival", "பெண் பெண் பெண் பெண்")
confirm it: the model is not producing Tamil, it is producing statistically plausible
akshara noise.

Meanwhile the public site says:

> *"Adhan — a **7B-parameter** open AI model for **22+ Indian languages**"*

That is a ~3-orders-of-magnitude overstatement and a language-count claim the repo
contradicts on line one. **This is the most urgent fix — not a PR problem, but evidence
the team is not reading its own numbers.**

### 1.2 The weight-sharing bug is the tell

`Block(c)` was hoisted out of the layer loop, so the "14.5M" model was silently **9.2M**
for the entire v3-long run. It trained. Loss went down. Checkpoints were written. Nobody
noticed until the checkpoints *wouldn't load*. That is a **process** story: there is no
correctness harness asserting the model trained is the model configured. `remat` (a perf
optimization) was added before that harness existed.

### 1.3 The tokenizer thesis is real but unvalidated — and the whitepaper says so

The whitepaper is the best thing in the repo, and it honestly admits the fatal gap:

> *"no Tamil-trained BPE baseline yet."*

The central claim — **akshara tokenization beats subword tokenization for Tamil** — has
never been tested against the one baseline that matters. Comparing Swaram to GPT-4's
cl100k is comparing to a bad, English-fitted tokenizer. Beating cl100k on Tamil is table
stakes, not a result. The required experiment is **Swaram vs a Tamil-trained
SentencePiece at the same vocab, same corpus, equal compute**.

Also: **seven different fertility values** across docs (0.306, 0.334, 0.339, 0.349,
0.357, 0.506, 0.554). Morphology agreement is **5.6%** in the whitepaper and **24.2%** in
the eval — the tokenizer cuts through morphemes almost always, and the guard that fixes
it is off by default.

### 1.4 The compute reality is being politely ignored

- 293M-token archive, one 12-core no-GPU box, ~1,800 tok/s.
- The "12h long run" is **82M tokens = 0.28 epoch**. A warm-up, not pretraining.
- Chinchilla-optimal for 14.5M params is ~290M tokens. Trained on maybe a third, once, on CPU.
- Stage 2 is "21 Nov" and no GPU is booked.

A from-scratch LM research program at ~1/1000th the compute of the field is a schedule
risk, not a virtue.

### 1.5 Process gaps

- **Aram Harness**: built, 12 principles, 27 prompts, **never run**. Zero safety baseline.
- **Serving**: never validated against a real checkpoint.
- **Export/ONNX**: targets an HF pipeline that was removed.
- **Kutty v1 decision** (from scratch vs fine-tune): open 6+ weeks, blocking the backlog.
- **29 remote branches**, 6 stacked duplicates of work already on `main`.
- 2 OOM kills before a cgroup unit existed.

### 1.6 What is actually good

- The **test suite is real**: 124 passing tests, round-trip, resume, KV-cache, sharding.
- The **tokenizer engineering is genuinely nice**: 100% lossless, byte fallback, Rust
  backend, 2.5 Mchar/s.
- The **docs-as-engineering** discipline (`factory/`, ADRs, gates) is mature beyond the
  code's maturity.
- The **honesty in the whitepaper** about limitations. Keep that instinct.

---

## 2. The strategic fork

**Is Adhan a research project or a product?** Right now it is pretending to be both and
excelling at neither. Decide, then execute.

Recommended: **ship a fine-tuned open base as the product, keep the from-scratch nano +
Swaram tokenizer as the public research artifact.** Purity is not a feature.

---

## 3. Battle plan

### Phase 0 — Stop the bleeding (this week, ~2 days)

| # | Action | Why | Done when |
|---|---|---|---|
| 0.1 | **Fix the public claim.** Replace "7B · 22+ languages" with "14.5M research-preview · Tamil". | Credibility is the only durable asset. | yazhi.dev About matches the model card |
| 0.2 | **Structure-invariant test in CI**: assert `n_layers` distinct blocks, param count == `approx_params()`, init→save→load round-trip. | The bug that ate v3-long must be impossible to reintroduce. | Test fuzzed over all configs |
| 0.3 | **One canonical tokenizer table.** Delete/repoint the other six fertility values. | Stop publishing contradictions. | `grep -r fertility docs/` points at one table |
| 0.4 | **Run Aram critical-only on Stage 1b.** | Cannot launch without a safety baseline. | Report committed |
| 0.5 | Close the 20 stale branches. 3 open max. | Every branch is a deferred decision. | `git branch -r \| wc -l` < 10 |

### Phase 1 — Prove or kill the thesis (2 weeks) — the most important work

**Run the A/B the whitepaper says does not exist.**

```
Fixed: same 293M-token corpus, same model config (14.5M), same compute budget,
       same held-out val set (deduped, frozen).
A: Swaram 32k     B: SentencePiece BPE 32k trained on the same corpus.
Report: bits/akshara on held-out, tokens/word, and a 20-prompt human coherence score.
```

**Decision rule:** if Swaram does not beat Tamil-BPE by a meaningful margin (say ≥10%
bits/akshara) at equal vocab and equal compute, the from-scratch akshara-SLM thesis is
**not supported**, and Kutty v1 pivots to a fine-tune of an open base while Swaram
remains the tokenizer contribution.

Also:
- **Reconcile the corpus numbers.** 1.05B chars, "~328M swaram tokens," but CURRENT says
  609.7M across v1+v2, and 38.5% of tokens are `▁` markers. Measure pure-Tamil tokens,
  post-dedup docs, and actual tokens in the 293M shards.
- **Fix the 38.5% `▁`-marker tax.** It is silently eating the context window.

### Phase 2 — Fix the compute reality (parallel, this month)

- **Book a GPU now.** One rented T4/A10 for a weekend: 293M tokens at ~15k tok/s ≈
  **5–6 hours**, ~$10–20. Sovereignty is about ownership of weights and data, not about
  starving yourself of rented FLOPs.
- Move Stage 2 to the GPU or accept that nano stays a toy.
- **Do not build `tiny`/`mini` until nano is coherent and ≥1B verified pure-Tamil tokens
  exist.** Scaling a broken pipeline makes broken output faster.
- Run the existing T4 config and record tokens/s.

### Phase 3 — Make evaluation real (2–3 weeks, overlaps Phase 1)

| Probe | Current | Target |
|---|---|---|
| Held-out bits/akshara vs Tamil-BPE | not measured | the Phase-1 A/B |
| Coherence (human rubric, 50 prompts) | eyeballed | scored 1–5, published |
| Sandhi correctness | 91.5% | ≥ 90% (tighten to 95%) |
| Morphology agreement | 5.6–24.2% | ≥ 30% or kill the claim |
| Aram | never run | 0 CRITICAL |
| SFT-vs-base regression | none | required gate before any SFT run |

- SFT'ing a 14.5M model trained on 100M tokens will **overfit and destroy the base**.
  The regression gate is mandatory before touching the ~10k Sangam + ~1.3k QA examples.
- Re-baseline the 1,829 checkpoint on the *same* val set so the delta is meaningful.

### Phase 4 — The strategic decision (this week, one page)

Write the ADR the roadmap has been dodging:

> **Kutty v1 = ?**
> - (a) from-scratch nano + SFT (the thesis, but 60 days and no GPU)
> - (b) LoRA/continued-pretrain of an open 0.5–2B base (ships, but not "from scratch")
> - (c) **(recommended)** ship (b) as the product, keep (a) as the public research
>   artifact + Swaram tokenizer paper.

### Phase 5 — Process hygiene (continuous)

- **One branch per PR**, delete on merge. 29 → <10.
- **Single source of truth**: `factory/CURRENT.md`. Retire everything else.
- **One decision log.** Any decision open > 1 week gets a default and a named decider.
- **Every perf change ships with a correctness test first.** `remat` is the cautionary tale.

---

## 4. The 30/60/90

| Window | Outcome |
|---|---|
| **30 days** | Public claim fixed · structure test in CI · canonical tokenizer table · Aram baseline · **Swaram-vs-BPE A/B result** · GPU booked · Kutty ADR merged |
| **60 days** | Nano or fine-tune coherent on 50-prompt human eval · 1B verified tokens · SFT-v0 with regression gate · INT8 export spike working |
| **90 days** | Kutty RC1 passing ship bars · INT8 on a reference phone · model card + datasheet · tokenizer paper submitted |

---

## 5. The one-sentence version

> **You have a genuinely interesting tokenizer and a model that does not work yet, and
> you are spending your scarcest resource (time) on docs and process instead of the one
> experiment — Swaram vs Tamil-BPE at equal compute — that determines whether the whole
> thing is real.**

Fix the public claim today, run the A/B this week, rent a GPU this month, and make the
Kutty from-scratch-vs-fine-tune call by Friday.
