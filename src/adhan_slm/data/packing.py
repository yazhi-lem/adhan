"""Sequence packing: token id streams → contiguous fixed-length training sequences.

Causal-LM pretraining wants long, dense sequences with no padding. We tokenize each
document, wrap it with ``<bos> … <eos>``, concatenate the whole corpus into one token
stream, then slice it into ``seq_len``-token windows. The final partial window is
dropped (a few hundred tokens out of millions) so every sequence is full — this is the
"packed fixed-length sequences" the roadmap (§Phase 2) calls for and is why fertility,
not padding, drives the token budget.

Persistence is a flat little-endian ``uint32``/``uint16`` blob (``.bin``) plus a JSON
manifest — the classic nanoGPT-style shard layout. It needs no numpy to *write* (pure
``array`` + struct), and reads back with or without numpy. Keeping it dependency-free
means corpus prep runs in the same minimal environment as the tokenizer.
"""

from __future__ import annotations

import array
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

# A packed shard stores raw token ids; width is chosen from vocab size at write time.
_DTYPE_FOR_WIDTH = {2: "H", 4: "I"}  # uint16, uint32 (python `array` typecodes)
_NUMPY_DTYPE = {2: "<u2", 4: "<u4"}


def _typecode(vocab_size: int) -> str:
    return "H" if vocab_size <= 0xFFFF else "I"


@dataclass
class PackedShard:
    """Metadata for a written shard (mirrors the on-disk ``*.manifest.json``)."""

    path: str
    seq_len: int
    n_sequences: int
    n_tokens: int
    vocab_size: int
    dtype: str  # numpy-style, e.g. "<u2"

    def as_dict(self) -> dict:
        return asdict(self)


def tokens_from_documents(
    documents: Iterable[str],
    tokenizer,
    add_special: bool = True,
) -> Iterator[int]:
    """Stream token ids for a corpus, one ``<bos> doc <eos>`` group per document.

    ``tokenizer`` needs ``.encode(text, add_special=...)`` (SwaramTokenizer /
    AksharamTokenizer both satisfy this).
    """
    for doc in documents:
        ids = tokenizer.encode(doc, add_special=add_special)
        for tid in ids:
            yield tid


def pack_stream(token_ids: Iterable[int], seq_len: int) -> Iterator[List[int]]:
    """Slice a flat id stream into full ``seq_len``-length windows (drop remainder)."""
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2 (need at least one input/target pair)")
    buf: List[int] = []
    for tid in token_ids:
        buf.append(tid)
        if len(buf) == seq_len:
            yield buf
            buf = []
    # trailing < seq_len tokens are intentionally dropped


def pack_documents(
    documents: Iterable[str],
    tokenizer,
    seq_len: int,
    add_special: bool = True,
) -> List[List[int]]:
    """Convenience: tokenize + pack a (small) corpus into a list of sequences."""
    stream = tokens_from_documents(documents, tokenizer, add_special=add_special)
    return list(pack_stream(stream, seq_len))


def write_shard(
    sequences: Sequence[Sequence[int]],
    out_path: str | Path,
    seq_len: int,
    vocab_size: int,
) -> PackedShard:
    """Write packed sequences to ``out_path`` (.bin) + ``out_path + '.manifest.json'``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    typecode = _typecode(vocab_size)
    flat = array.array(typecode)
    n_seq = 0
    for seq in sequences:
        if len(seq) != seq_len:
            raise ValueError(f"sequence length {len(seq)} != seq_len {seq_len}")
        flat.extend(seq)
        n_seq += 1
    with out_path.open("wb") as fh:
        flat.tofile(fh)

    width = flat.itemsize
    shard = PackedShard(
        path=str(out_path),
        seq_len=seq_len,
        n_sequences=n_seq,
        n_tokens=n_seq * seq_len,
        vocab_size=vocab_size,
        dtype=_NUMPY_DTYPE[width],
    )
    Path(str(out_path) + ".manifest.json").write_text(
        json.dumps(shard.as_dict(), indent=2), encoding="utf-8"
    )
    return shard


def load_manifest(bin_path: str | Path) -> PackedShard:
    meta = json.loads(Path(str(bin_path) + ".manifest.json").read_text(encoding="utf-8"))
    return PackedShard(**meta)


def read_shard(bin_path: str | Path, manifest: Optional[PackedShard] = None):
    """Load a shard as a 2-D array of shape (n_sequences, seq_len).

    Returns a numpy array when numpy is available (zero-copy-ish, memmap-friendly),
    otherwise a list-of-lists so the pure-python path still works.
    """
    manifest = manifest or load_manifest(bin_path)
    width = 2 if manifest.dtype == "<u2" else 4
    try:
        import numpy as np

        arr = np.fromfile(bin_path, dtype=manifest.dtype)
        return arr.reshape(manifest.n_sequences, manifest.seq_len)
    except ImportError:
        typecode = _DTYPE_FOR_WIDTH[width]
        flat = array.array(typecode)
        with Path(bin_path).open("rb") as fh:
            flat.frombytes(fh.read())
        sl = manifest.seq_len
        return [list(flat[i * sl : (i + 1) * sl]) for i in range(manifest.n_sequences)]


class ShardFormatError(ValueError):
    """A shard on disk does not match its manifest or the caller's expectations."""


def open_shard_memmap(bin_path: str | Path, expected_seq_len: Optional[int] = None):
    """Memory-map a shard written by :func:`write_shard`, shape ``(n_sequences, seq_len)``.

    This is the one reader every trainer should use, so the on-disk format has a
    single source of truth: row width, dtype and row count all come from the
    ``*.manifest.json`` sidecar, never from guessing at the file size. Each row is
    exactly ``seq_len`` tokens; a causal-LM step uses ``row[:-1]`` as input and
    ``row[1:]`` as target.

    Raises
    ------
    FileNotFoundError
        if the ``.bin`` or its manifest is missing.
    ShardFormatError
        if ``expected_seq_len`` differs from the manifest, the manifest dtype is
        unknown, or the file size does not equal ``n_sequences * seq_len`` tokens.
    """
    import numpy as np

    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f"shard not found: {bin_path}")
    manifest_path = Path(str(bin_path) + ".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest not found: {manifest_path} (write shards with "
            "adhan_slm.data.packing.write_shard / scripts/prepare_slm_corpus.py)"
        )
    manifest = load_manifest(bin_path)
    if manifest.dtype not in _NUMPY_DTYPE.values():
        raise ShardFormatError(f"{bin_path}: unknown dtype {manifest.dtype!r} in manifest")
    if expected_seq_len is not None and expected_seq_len != manifest.seq_len:
        raise ShardFormatError(
            f"{bin_path}: shard was packed with seq_len={manifest.seq_len} but "
            f"seq_len={expected_seq_len} was requested; pass --seq-len {manifest.seq_len} "
            "or re-pack the corpus"
        )
    itemsize = np.dtype(manifest.dtype).itemsize
    expected_bytes = manifest.n_sequences * manifest.seq_len * itemsize
    actual_bytes = bin_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ShardFormatError(
            f"{bin_path}: file is {actual_bytes} bytes but the manifest describes "
            f"{manifest.n_sequences} x {manifest.seq_len} {manifest.dtype} tokens "
            f"({expected_bytes} bytes); the shard is truncated or stale"
        )
    if manifest.n_sequences == 0:
        return np.zeros((0, manifest.seq_len), dtype=manifest.dtype)
    return np.memmap(
        bin_path,
        dtype=manifest.dtype,
        mode="r",
        shape=(manifest.n_sequences, manifest.seq_len),
    )
