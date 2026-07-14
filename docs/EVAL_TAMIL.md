# Adhan SLM — Tamil evaluation (`docs/EVAL_TAMIL.md`)

The evaluation harness lives in `src/adhan_slm/eval/` and is driven by one entrypoint,
`adhan_slm.eval.run_eval`. It produces a single JSON report combining every
Tamil-specific probe the roadmap (ROADMAP_JAX_SLM.md §Phase 3–4) defines. Each probe
**degrades gracefully**: sections that need open-tamil or a trained checkpoint are
reported as `skipped` with a reason rather than crashing, so the same command works in
a minimal environment and gets richer as dependencies and artifacts appear.

## Run it

```bash
# Tokenizer + corpus probes only (needs only the frozen tokenizer + eval text):
python -m adhan_slm.eval.run_eval \
    --tokenizer-dir data/final/tamil_slm \
    --eval-text     data/final/tamil_slm

# Add trained-model perplexity + sample generations:
python -m adhan_slm.eval.run_eval \
    --tokenizer-dir data/final/tamil_slm \
    --eval-text     data/final/tamil_slm \
    --config        src/adhan_slm/configs/adhan_slm_tiny.yaml \
    --checkpoint    checkpoints/adhan-tiny \
    --out           docs/eval_report.json
```

## Probes

| Probe | Module | What it measures | Roadmap |
|---|---|---|---|
| **fertility** | `tokenizer` | Mean tokens/akshara of the frozen tokenizer on held-out text. Target **< 1.15**. | §Phase 1 |
| **classical** | `eval/ngram_baseline.py` | Add-one smoothed unigram-over-aksharas perplexity — a near-zero-cost **floor** the trained model must beat. | §Phase 4 |
| **morphology** | `eval/morphology.py` | Fraction of inflected words whose tokenizer merge boundary matches open-tamil's stemmer split; sandhi (புணர்ச்சி) correctness rate. | §Phase 4 |
| **model** | `eval/perplexity.py` | Trained-model **per-token validation perplexity** on `val.bin`, plus sample generations. Directly comparable to the classical floor and (later) a distilgpt2 baseline. | §Phase 3 |
| **kid_prompts** | `eval/kid_level_prompts.py` | The 50-prompt kid-level set (5–7-year-old register) from open-tamil lexicons, with model generations when a checkpoint is given. | §Phase 4 |

## Milestone criteria (first launch, `adhan-nano` v0.1)

1. **Fertility** `< 1.15` tokens/akshara, lossless round-trip (fertility probe).
2. **Model per-token val perplexity** beats the classical unigram floor comfortably,
   then beats a distilgpt2-Tamil baseline (model probe vs classical probe).
3. **Kid-level generations** are grammatical Tamil on the 50-prompt set (kid_prompts
   probe + human read-through rubric).

## Interpreting `perplexity`

The model probe reports mean per-token negative log-likelihood (`nll`) and its exp
(`perplexity`) over up to `max_batches` validation batches, using the same next-token
objective as training — so a run's reported `val_perplexity` and this offline number
should agree (they do in the CPU verification run: ~8.5 on the toy corpus). Lower is
better; the classical `unigram_akshara_ppl` is the floor to clear first.

> Note: perplexities are only comparable across models that share the **same
> tokenizer/vocab**. When comparing to distilgpt2, compare **per-akshara** bits so the
> different segmentations are normalized to the same unit.
