"""Tests for sequence packing + shard I/O (`adhan_slm.data.packing`).

Run: PYTHONPATH=src python -m adhan_slm.data.packing_tests
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adhan_slm.core.selftest import run_module_tests  # noqa: E402
from adhan_slm.data import packing  # noqa: E402
from adhan_slm.data._test_fixtures import TAMIL_LINES, sample_tokenizer  # noqa: E402

SEQ_LEN = 16


def test_every_packed_sequence_is_full_length() -> None:
    """No padding is the whole point: fertility, not padding, drives the token budget."""
    tok = sample_tokenizer()
    seqs = packing.pack_documents(TAMIL_LINES * 20, tok, seq_len=SEQ_LEN)

    assert seqs, "packing produced no sequences"
    assert all(len(s) == SEQ_LEN for s in seqs), "every packed seq must be full length"


def test_packed_stream_is_a_lossless_prefix_of_the_raw_stream() -> None:
    """Packing may only drop the trailing partial window — never reorder or lose ids."""
    tok = sample_tokenizer()
    docs = TAMIL_LINES * 20
    seqs = packing.pack_documents(docs, tok, seq_len=SEQ_LEN)

    raw = list(packing.tokens_from_documents(docs, tok))
    flat = [t for s in seqs for t in s]
    assert flat == raw[: len(flat)], "packed tokens must match the raw stream prefix"

    dropped = len(raw) - len(flat)
    assert 0 <= dropped < SEQ_LEN, f"drop remainder must be < seq_len, got {dropped}"


def test_shard_write_read_round_trip_is_lossless() -> None:
    tok = sample_tokenizer()
    seqs = packing.pack_documents(TAMIL_LINES * 30, tok, seq_len=SEQ_LEN)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "train.bin"
        shard = packing.write_shard(seqs, path, seq_len=SEQ_LEN, vocab_size=len(tok))

        assert shard.n_sequences == len(seqs)
        assert shard.n_tokens == len(seqs) * SEQ_LEN

        reloaded = packing.read_shard(path)
        rows = reloaded.tolist() if hasattr(reloaded, "tolist") else reloaded
        assert rows == seqs, "shard round-trip must be lossless"


def test_shard_write_emits_a_manifest() -> None:
    """The trainer derives seq_len / dtype / count from the manifest, not the blob."""
    tok = sample_tokenizer()
    seqs = packing.pack_documents(TAMIL_LINES * 30, tok, seq_len=SEQ_LEN)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "train.bin"
        written = packing.write_shard(seqs, path, seq_len=SEQ_LEN, vocab_size=len(tok))
        assert Path(str(path) + ".manifest.json").exists(), "manifest must be written"

        manifest = packing.load_manifest(path)
        assert manifest.seq_len == SEQ_LEN
        assert manifest.n_sequences == written.n_sequences
        assert manifest.dtype == written.dtype


if __name__ == "__main__":
    run_module_tests(globals(), "data packing")
