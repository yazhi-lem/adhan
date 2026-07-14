"""Pure-python tests for the SLM data pipeline (no numpy/JAX required).

Run: PYTHONPATH=src python -m adhan_slm.data.test_data_pipeline
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adhan_slm.data import corpus, packing, loader
from adhan_slm.tokenizer import SwaramTokenizer, default_akshara_inventory

TAMIL_LINES = [
    "நான் பள்ளிக்கு போகிறேன்",
    "அவன் புத்தகம் படித்தான்",
    "நாய் வேகமாக ஓடியது",
    "அம்மா சாதம் சமைத்தார்",
]


def _tokenizer():
    return SwaramTokenizer.train(TAMIL_LINES, vocab_size=len(default_akshara_inventory()) + 64)


def test_corpus_txt_and_jsonl():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "a.txt").write_text("\n".join(TAMIL_LINES) + "\n", encoding="utf-8")
        (d / "b.jsonl").write_text(
            "\n".join(json.dumps({"text": t}) for t in TAMIL_LINES) + "\n",
            encoding="utf-8",
        )
        (d / "empty.txt").write_text("\n\n  \n", encoding="utf-8")
        docs = corpus.read_corpus(d)
    assert len(docs) == len(TAMIL_LINES) * 2, f"expected 8 docs, got {len(docs)}"
    assert all(docs), "no empty docs should survive"
    print(f"  corpus: read {len(docs)} docs from txt + jsonl (empties dropped) OK")


def test_packing_is_lossless_and_full():
    tok = _tokenizer()
    seq_len = 16
    seqs = packing.pack_documents(TAMIL_LINES * 20, tok, seq_len=seq_len)
    assert seqs, "packing produced no sequences"
    assert all(len(s) == seq_len for s in seqs), "every packed seq must be full length"
    # Reconstruct: packed stream must be a prefix of the raw token stream.
    raw = list(packing.tokens_from_documents(TAMIL_LINES * 20, tok))
    flat = [t for s in seqs for t in s]
    assert flat == raw[: len(flat)], "packed tokens must match the raw stream prefix"
    dropped = len(raw) - len(flat)
    assert 0 <= dropped < seq_len, f"drop remainder must be < seq_len, got {dropped}"
    print(f"  packing: {len(seqs)} full seqs of {seq_len}, {dropped} trailing tokens dropped OK")


def test_shard_roundtrip():
    tok = _tokenizer()
    seq_len = 16
    seqs = packing.pack_documents(TAMIL_LINES * 30, tok, seq_len=seq_len)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "train.bin"
        shard = packing.write_shard(seqs, p, seq_len=seq_len, vocab_size=len(tok))
        assert shard.n_sequences == len(seqs)
        assert shard.n_tokens == len(seqs) * seq_len
        reloaded = packing.read_shard(p)
        rows = reloaded.tolist() if hasattr(reloaded, "tolist") else reloaded
        assert rows == seqs, "shard round-trip must be lossless"
        assert (Path(str(p) + ".manifest.json")).exists(), "manifest must be written"
    print(f"  shard: wrote+reloaded {shard.n_sequences} seqs, manifest OK ({shard.dtype})")


def test_loader_batches_deterministic():
    tok = _tokenizer()
    seq_len, bs = 16, 4
    seqs = packing.pack_documents(TAMIL_LINES * 50, tok, seq_len=seq_len)
    ds = loader.PackedDataset(seqs, batch_size=bs, shuffle=True, seed=7, infinite=False)
    b1 = [b for b in ds]
    b2 = [b for b in loader.PackedDataset(seqs, batch_size=bs, shuffle=True, seed=7, infinite=False)]
    assert b1 == b2, "same seed must give identical batch order"
    assert all(len(batch) == bs for batch in b1), "drop_last must keep batches full"
    for batch in b1:
        for row in batch:
            assert len(row) == seq_len
    # infinite loader keeps producing
    it = iter(loader.PackedDataset(seqs, batch_size=bs, seed=1))
    got = [next(it) for _ in range(len(b1) + 3)]
    assert len(got) == len(b1) + 3, "infinite loader must wrap past one epoch"
    print(f"  loader: {len(b1)} batches/epoch, deterministic, infinite-wrap OK")


def main():
    print("data pipeline tests:")
    test_corpus_txt_and_jsonl()
    test_packing_is_lossless_and_full()
    test_shard_roundtrip()
    test_loader_batches_deterministic()
    print("ALL PASSED")


if __name__ == "__main__":
    main()
