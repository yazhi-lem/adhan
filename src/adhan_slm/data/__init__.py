"""Adhan SLM data pipeline: corpus → swaram tokens → packed fixed-length shards.

This is the Phase 2/3 data layer the training loop needs. It closes the gap where
`train_jax.data_iterator()` used to raise `NotImplementedError`: there is now a real
path from plain-text/JSONL corpus files to packed, batched, causal-LM training
sequences.

Design goals:
  * **Pure-python core** (no hard numpy/JAX dependency) so packing/splitting is
    testable and runnable on-device, matching the tokenizer's philosophy. numpy is
    used only to hand off contiguous batches to JAX when it is installed.
  * **Sequence packing** to `max_seq_len` with document separators, so no compute is
    wasted on padding — the roadmap's "packed fixed-length sequences" (§Phase 2).
  * **Deterministic** train/val split and shuffling from a seed, for reproducibility
    (roadmap §5: "reproducible from MLflow / data version + code SHA").
"""
from .corpus import iter_documents, read_corpus
from .packing import pack_documents, PackedShard
from .loader import PackedDataset, batches_from_shard

__all__ = [
    "iter_documents",
    "read_corpus",
    "pack_documents",
    "PackedShard",
    "PackedDataset",
    "batches_from_shard",
]
