"""Tests for batch iteration over packed shards (`adhan_slm.data.loader`).

Run: PYTHONPATH=src python -m adhan_slm.data.loader_tests
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adhan_slm.core.selftest import run_module_tests  # noqa: E402
from adhan_slm.data import loader, packing  # noqa: E402
from adhan_slm.data._test_fixtures import TAMIL_LINES, sample_tokenizer  # noqa: E402

SEQ_LEN = 16
BATCH_SIZE = 4


def _sequences():
    return packing.pack_documents(TAMIL_LINES * 50, sample_tokenizer(), seq_len=SEQ_LEN)


def test_same_seed_gives_identical_batch_order() -> None:
    """Reproducibility (roadmap §5) starts with a deterministic data order."""
    seqs = _sequences()
    kwargs = dict(batch_size=BATCH_SIZE, shuffle=True, seed=7, infinite=False)

    first = list(loader.PackedDataset(seqs, **kwargs))
    second = list(loader.PackedDataset(seqs, **kwargs))
    assert first == second, "same seed must give identical batch order"


def test_drop_last_keeps_every_batch_full() -> None:
    """A short final batch would retrigger XLA compilation for a new shape."""
    seqs = _sequences()
    batches = list(
        loader.PackedDataset(seqs, batch_size=BATCH_SIZE, shuffle=True, seed=7, infinite=False)
    )

    assert batches, "loader produced no batches"
    assert all(len(b) == BATCH_SIZE for b in batches), "drop_last must keep batches full"
    for batch in batches:
        for row in batch:
            assert len(row) == SEQ_LEN


def test_infinite_loader_wraps_past_one_epoch() -> None:
    """Training runs to a step budget, not an epoch count, so the loader must not stop."""
    seqs = _sequences()
    per_epoch = len(
        list(
            loader.PackedDataset(seqs, batch_size=BATCH_SIZE, shuffle=True, seed=7, infinite=False)
        )
    )

    it = iter(loader.PackedDataset(seqs, batch_size=BATCH_SIZE, seed=1))
    got = [next(it) for _ in range(per_epoch + 3)]
    assert len(got) == per_epoch + 3, "infinite loader must wrap past one epoch"


def test_batch_larger_than_shard_is_rejected() -> None:
    """Better a clear error than a silently empty iterator that looks like convergence."""
    seqs = _sequences()
    try:
        loader.PackedDataset(seqs, batch_size=len(seqs) + 1, drop_last=True)
    except ValueError as e:
        assert "batch_size" in str(e)
    else:  # pragma: no cover
        raise AssertionError("batch_size > shard size should raise ValueError")


if __name__ == "__main__":
    run_module_tests(globals(), "data loader")
