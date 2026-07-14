"""Batch iteration over packed shards for the JAX training loop.

`PackedDataset` turns a shard (numpy array or list-of-lists from `read_shard`) into an
epoch/step iterator of ``(batch, seq_len)`` int32 batches, with deterministic
seed-driven shuffling. It is the concrete object `train_jax.data_iterator()` returns.

numpy is used when present (fast contiguous batches for XLA); a pure-python fallback
keeps the loader importable and unit-testable in a minimal environment.
"""
from __future__ import annotations

import random
from typing import Iterator, List, Optional

from .packing import PackedShard, read_shard, load_manifest


def _to_batch(rows, use_numpy):
    if use_numpy:
        import numpy as np

        return np.asarray(rows, dtype="int32")
    return [list(r) for r in rows]


class PackedDataset:
    """Iterate ``(batch_size, seq_len)`` batches over a packed shard.

    Parameters
    ----------
    data : numpy array (N, T) or list-of-lists — from `read_shard`.
    batch_size : rows per batch.
    shuffle : shuffle sequence order each epoch (seed-deterministic).
    seed : base RNG seed; epoch index is mixed in so each epoch differs but the whole
        run is reproducible.
    drop_last : drop a final partial batch (keeps every batch full for jit).
    infinite : loop forever (training); False yields exactly one pass (eval).
    """

    def __init__(
        self,
        data,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        infinite: bool = True,
    ):
        self._data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.infinite = infinite
        try:
            import numpy as np  # noqa: F401

            self._use_numpy = not isinstance(data, list)
            self.n = data.shape[0] if self._use_numpy else len(data)
        except ImportError:
            self._use_numpy = False
            self.n = len(data)
        if self.n < batch_size and drop_last:
            raise ValueError(
                f"shard has {self.n} sequences < batch_size {batch_size}; "
                "use a smaller batch or drop_last=False"
            )

    @property
    def seq_len(self) -> int:
        """Token length of each packed sequence."""
        return self._data.shape[1] if self._use_numpy else len(self._data[0])

    @classmethod
    def from_shard(cls, bin_path, batch_size, manifest: Optional[PackedShard] = None, **kw):
        manifest = manifest or load_manifest(bin_path)
        return cls(read_shard(bin_path, manifest), batch_size, **kw)

    def _row(self, i):
        return self._data[i]

    def __iter__(self) -> Iterator:
        epoch = 0
        while True:
            order = list(range(self.n))
            if self.shuffle:
                random.Random(self.seed + epoch).shuffle(order)
            last = self.n - (self.n % self.batch_size) if self.drop_last else self.n
            for start in range(0, last, self.batch_size):
                idx = order[start:start + self.batch_size]
                yield _to_batch([self._row(i) for i in idx], self._use_numpy)
            epoch += 1
            if not self.infinite:
                return


def batches_from_shard(bin_path, batch_size, **kw) -> PackedDataset:
    """Shorthand for `PackedDataset.from_shard`."""
    return PackedDataset.from_shard(bin_path, batch_size, **kw)
