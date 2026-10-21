"""Tests for sequence packing + shard I/O (`adhan_slm.data.packing`).

Run: PYTHONPATH=src python -m adhan_slm.data.packing_tests
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adhan_slm.core.selftest import run_module_tests, skip_unless  # noqa: E402
from adhan_slm.data import packing  # noqa: E402
from adhan_slm.data._test_fixtures import TAMIL_LINES, sample_tokenizer  # noqa: E402

SEQ_LEN = 16

try:
    import numpy  # noqa: F401

    HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy is a core dependency
    HAS_NUMPY = False

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_REPO_ROOT = Path(__file__).resolve().parents[3]


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


def _write_sample_shard(root: Path, vocab_size: int | None = None):
    tok = sample_tokenizer()
    seqs = packing.pack_documents(TAMIL_LINES * 30, tok, seq_len=SEQ_LEN)
    path = root / "train.bin"
    packing.write_shard(seqs, path, seq_len=SEQ_LEN, vocab_size=vocab_size or len(tok))
    return tok, seqs, path


@skip_unless(HAS_NUMPY, "numpy not installed")
def test_memmap_reader_round_trips_pack_load_decode() -> None:
    """LEM-25: pack -> memmap load -> identical token ids -> decodes back to text."""
    with tempfile.TemporaryDirectory() as d:
        tok, seqs, path = _write_sample_shard(Path(d))
        mm = packing.open_shard_memmap(path, expected_seq_len=SEQ_LEN)

        assert mm.shape == (len(seqs), SEQ_LEN)
        assert mm.tolist() == seqs, "memmap reader must see exactly what the writer wrote"
        text = tok.decode([int(t) for t in mm[0]])
        assert any(ch in text for line in TAMIL_LINES for ch in line), text
        del mm


@skip_unless(HAS_NUMPY, "numpy not installed")
def test_memmap_reader_handles_uint32_shards() -> None:
    """Vocab > 65,535 switches the writer to uint32; the reader must follow the manifest."""
    with tempfile.TemporaryDirectory() as d:
        _, seqs, path = _write_sample_shard(Path(d), vocab_size=70_000)
        assert packing.load_manifest(path).dtype == "<u4"
        mm = packing.open_shard_memmap(path)
        assert mm.tolist() == seqs
        del mm


@skip_unless(HAS_NUMPY, "numpy not installed")
def test_memmap_reader_rejects_seq_len_mismatch() -> None:
    with tempfile.TemporaryDirectory() as d:
        _, _, path = _write_sample_shard(Path(d))
        try:
            packing.open_shard_memmap(path, expected_seq_len=SEQ_LEN + 1)
        except packing.ShardFormatError as e:
            assert f"seq_len={SEQ_LEN}" in str(e) and f"--seq-len {SEQ_LEN}" in str(e)
        else:
            raise AssertionError("expected ShardFormatError for seq_len mismatch")


@skip_unless(HAS_NUMPY, "numpy not installed")
def test_memmap_reader_rejects_truncated_shard_and_missing_files() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _, _, path = _write_sample_shard(root)
        with path.open("r+b") as fh:
            fh.truncate(path.stat().st_size - 2)
        try:
            packing.open_shard_memmap(path)
        except packing.ShardFormatError as e:
            assert "truncated or stale" in str(e)
        else:
            raise AssertionError("expected ShardFormatError for truncated shard")

        for missing in (root / "nope.bin", path):
            if missing == path:
                Path(str(path) + ".manifest.json").unlink()
            try:
                packing.open_shard_memmap(missing)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError(f"expected FileNotFoundError for {missing}")


@skip_unless(HAS_NUMPY, "numpy not installed")
def test_memmap_reader_rejects_unknown_dtype_and_allows_empty_shard() -> None:
    import json

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        empty = root / "empty.bin"
        packing.write_shard([], empty, seq_len=SEQ_LEN, vocab_size=100)
        assert packing.open_shard_memmap(empty).shape == (0, SEQ_LEN)

        manifest_path = Path(str(empty) + ".manifest.json")
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        meta["dtype"] = "<f4"
        manifest_path.write_text(json.dumps(meta), encoding="utf-8")
        try:
            packing.open_shard_memmap(empty)
        except packing.ShardFormatError as e:
            assert "unknown dtype" in str(e)
        else:
            raise AssertionError("expected ShardFormatError for unknown dtype")


@skip_unless(HAS_TORCH, "torch not installed (pip install -e '.[torch]')")
def test_train_efficient_dataset_reads_packing_shards() -> None:
    """The PyTorch trainer's dataset yields (row[:-1], row[1:]) from real shard rows."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_efficient", _REPO_ROOT / "scripts" / "train_efficient.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as d:
        _, seqs, path = _write_sample_shard(Path(d))
        ds = module.MemmapShardDataset(path, seq_len=SEQ_LEN)
        assert len(ds) == len(seqs)
        x, y = ds.get_batch(4, torch.device("cpu"))
        assert tuple(x.shape) == (4, SEQ_LEN - 1) and tuple(y.shape) == (4, SEQ_LEN - 1)
        rows = {tuple(s) for s in seqs}
        for xi, yi in zip(x.tolist(), y.tolist()):
            assert tuple(xi + yi[-1:]) in rows, "batch rows must be real shard rows"
        try:
            module.MemmapShardDataset(path, seq_len=SEQ_LEN * 2)
        except packing.ShardFormatError:
            pass
        else:
            raise AssertionError("expected ShardFormatError for seq_len mismatch")
        del ds


if __name__ == "__main__":
    run_module_tests(globals(), "data packing")
