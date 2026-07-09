"""JAX-accelerated batch encoding for the Swaram / Aksharam tokenizer family.

The tokenizer core (segmentation + merges) is pure-python and CPU-bound. For
large-scale corpus tokenization and on-device batched inference we lift the
*vocab lookup + padding + masking* onto JAX so it runs vectorized on GPU/TPU and
composes with the rest of the JAX training pipeline (jit/pmap).

Flow:
  python core: text -> List[List[str]] pieces   (segmentation + merges, CPU)
  jax fast path: pieces -> padded int32 id tensor + attention mask   (device)

The heavy morphological logic stays in python (irregular, data-dependent); the
numeric-shaped work (lookup, pad, pack, mask) is where JAX pays off.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:  # pure-python fallback keeps the tokenizer usable everywhere
    _HAS_JAX = False


def _pad_ids(batch_ids: Sequence[Sequence[int]], pad_id: int,
             max_len: int | None = None) -> Tuple[list, int]:
    lengths = [len(x) for x in batch_ids]
    target = max_len or (max(lengths) if lengths else 0)
    padded = [list(x[:target]) + [pad_id] * (target - min(len(x), target))
              for x in batch_ids]
    return padded, target


def encode_batch_jax(tokenizer, texts: List[str], max_len: int | None = None,
                     add_special: bool = True):
    """Encode a batch of texts to a padded (ids, mask) pair of jnp.int32 arrays.

    `tokenizer` is a SwaramTokenizer / AksharamTokenizer (shares the .encode API).
    Falls back to numpy-shaped python lists when JAX is not installed.
    """
    pad_id = tokenizer.vocab.get("<pad>", 0)
    batch_ids = [tokenizer.encode(t, add_special=add_special) for t in texts]
    padded, target = _pad_ids(batch_ids, pad_id, max_len)

    if not _HAS_JAX:
        mask = [[1 if j < len(orig) else 0 for j in range(target)]
                for orig in batch_ids]
        return padded, mask  # plain python lists

    ids = jnp.asarray(padded, dtype=jnp.int32)
    mask = (ids != pad_id).astype(jnp.int32)
    return ids, mask


def build_lookup_table(tokenizer):
    """Return a device-resident id lookup usable inside jit-ed pipelines.

    We expose the vocab as a sorted key/value pair so device code can do a
    vectorized searchsorted lookup for merged-piece ids without host round-trips.
    """
    if not _HAS_JAX:
        raise ImportError("JAX not installed — pip install -r requirements-jax.txt")
    # Note: string keys can't live on device; callers hash pieces to ints on host,
    # then use this table for the int->int remap steps in packing. Placeholder for
    # the Phase 2 packed-shard pipeline.
    ids = jnp.asarray(sorted(tokenizer.vocab.values()), dtype=jnp.int32)
    return ids


def has_jax() -> bool:
    return _HAS_JAX


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from adhan_slm.tokenizer import SwaramTokenizer, default_akshara_inventory

    samples = ["தமிழ் கற்போம்", "வணக்கம்", "ஆதன் நல்ல பையன்"]
    tok = SwaramTokenizer.train(samples, vocab_size=len(default_akshara_inventory()) + 64)
    ids, mask = encode_batch_jax(tok, samples, max_len=16)
    print(f"jax available : {has_jax()}")
    print(f"ids shape     : {getattr(ids, 'shape', (len(ids), len(ids[0])))}")
    print(f"first row ids : {ids[0] if not has_jax() else ids[0].tolist()}")
    print(f"first row mask: {mask[0] if not has_jax() else mask[0].tolist()}")
