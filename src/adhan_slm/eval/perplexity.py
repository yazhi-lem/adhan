"""Model perplexity on a held-out packed shard — the Phase 3/4 milestone metric.

`model_token_perplexity` computes mean per-token cross-entropy (and its exp,
perplexity) over a validation shard, so a trained `adhan-nano` can be compared to
the classical n-gram floor (`eval/ngram_baseline.py`) and, later, a distilgpt2
baseline — the "val perplexity beats distilgpt2" milestone the roadmap sets but had
no code to actually measure. JAX-only; imported lazily.
"""
from __future__ import annotations

import math
from typing import Optional


def model_token_perplexity(model, params, val_loader, max_batches: int = 500):
    """Mean per-token NLL + perplexity over up to `max_batches` batches.

    `val_loader` yields ``(batch, seq_len)`` int arrays (a finite `PackedDataset`).
    Uses the same next-token objective as training (predict token t+1 from ≤t).
    """
    import jax
    import jax.numpy as jnp
    import optax

    @jax.jit
    def batch_nll(p, batch):
        logits = model.apply({"params": p}, batch[:, :-1])
        targets = batch[:, 1:]
        ll = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return ll.sum(), ll.size

    total_nll, total_tok = 0.0, 0
    for i, b in enumerate(val_loader):
        if i >= max_batches:
            break
        s, n = batch_nll(params, jnp.asarray(b))
        total_nll += float(s)
        total_tok += int(n)
    if total_tok == 0:
        return {"n_tokens": 0, "nll": float("nan"), "perplexity": float("nan")}
    mean_nll = total_nll / total_tok
    return {
        "n_tokens": total_tok,
        "nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 20.0)),
    }
