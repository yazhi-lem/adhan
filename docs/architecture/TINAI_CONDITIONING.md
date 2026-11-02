# Tiṇai conditioning (திணை)

Classical Tamil poetics sets every poem in one of five landscapes. Each fixes the
*karupporul* (the natural setting) and the *uripporul* (the human feeling that belongs
to it). Adhan can condition on this structure: an opt-in layer inside the transformer.

| # | Tiṇai | Tamil | Landscape | Feeling |
|---|---|---|---|---|
| 0 | kurinji | குறிஞ்சி | mountains | union (புணர்தல்) |
| 1 | mullai | முல்லை | forest / pastoral | patient waiting (இருத்தல்) |
| 2 | marutham | மருதம் | farmland | lovers' quarrel (ஊடல்) |
| 3 | neithal | நெய்தல் | seashore | anxious pining (இரங்கல்) |
| 4 | paalai | பாலை | arid wasteland | separation (பிரிதல்) |

## What it adds

Set `use_tinai: true` in the `model:` section of a training config (or
`AdhanConfig(use_tinai=True)`). Default is **off**, and with it off the model has no extra
parameters at all, so existing checkpoints and configs are unaffected.

When on:

1. **Causal router.** A linear layer over the running mean of the token embeddings
   predicts a distribution over the five tiṇai at every position. It only sees tokens up
   to that position, so it is safe for autoregressive generation.
2. **Conditioning vector.** `weights @ tinai_emb`, where `weights` is the router's softmax,
   or a one-hot when the caller names a tiṇai (`tinai=1`, `-1` means "let the router decide").
3. **Injection.** The vector is scaled by a learned gate and added to the input embeddings,
   and it drives a FiLM (scale and shift) on both pre-norms of every block.

The gate and every FiLM kernel start at zero, so **enabling the layer starts from exactly
the unconditioned model** (tested). Training then moves it only as far as it helps.

## Using it

```python
from adhan_slm.model import AdhanConfig, AdhanSLM, generate, tinai_aux_loss

model = AdhanSLM(AdhanConfig.nano(vocab_size=8000))  # use_tinai=True to enable

# Steer generation toward a landscape (index 0-4 or key / Tamil name via tinai_index):
ids = generate(model, params, prompt_ids, tinai="neithal")

# Supervise the router with weak labels:
logits, router_logits = model.apply({"params": params}, tokens, return_router=True)
loss = lm_loss + 0.1 * tinai_aux_loss(router_logits, labels)  # labels: [B], -1 = ignore
```

`adhan_slm.model.label_text(text)` gives a *weak* label from a small, conservative keyword
lexicon (`TINAI_KEYWORDS`), or `-1` when the text has no clear cue or ties. It is for
supervising the router, not ground truth, and the lexicon should be extended carefully:
a wrong keyword teaches the router a wrong rule.

## Status and limits

- This adds the **mechanism** and tests that it is correct (identity at init, causal,
  steerable, the router can learn labels). It does **not** yet claim a quality gain: that
  needs a trained model, labelled data and an evaluation, which are follow-ups.
- With no labels, the router trains only through the language-model loss.
- Not built yet: corpus labelling in `prepare_slm_corpus.py` (sidecar label files), the
  auxiliary loss inside `train_jax.py`, and Sangam persona routing (Avvaiyar, Kapilar,
  Paranar, Nakkirar, ...), which can reuse the same conditioning vector.
