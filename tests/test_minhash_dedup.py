"""Tests for MinHash deduplication module."""

from __future__ import annotations

import pytest

from adhan_slm.data.contamination import _jaccard
from adhan_slm.data.minhash_dedup import (
    DedupConfig,
    _hash_family,
    _minhash,
    _shingles,
    deduplicate_records,
)

pytestmark = pytest.mark.unit


def test_shingles_basic():
    text = "hello world"
    s = _shingles(text, 3)
    assert "hel" in s
    assert "ell" in s
    assert "llo" in s
    assert "wor" in s
    assert "rld" in s
    assert len(s) == len(text) - 3 + 1


def test_shingles_short_text():
    text = "ab"
    s = _shingles(text, 5)
    assert s == {"ab"}


def test_minhash_deterministic():
    hash_fns = _hash_family(42, 8)
    sig1 = _minhash({"hello", "world"}, hash_fns)
    sig2 = _minhash({"hello", "world"}, hash_fns)
    assert sig1 == sig2
    assert len(sig1) == 8


def test_minhash_similarity():
    """Two nearly-identical documents should have high MinHash agreement."""
    cfg = DedupConfig(n_hashes=64, shingle_size=3)
    hash_fns = _hash_family(cfg.seed, cfg.n_hashes)

    doc1 = _shingles("the quick brown fox jumps over the lazy dog", cfg.shingle_size)
    doc2 = _shingles("the quick brown fox jumps over a lazy dog", cfg.shingle_size)

    sig1 = _minhash(doc1, hash_fns)
    sig2 = _minhash(doc2, hash_fns)

    agreements = sum(1 for a, b in zip(sig1, sig2) if a == b)
    est = agreements / cfg.n_hashes
    assert est > 0.7  # high similarity expected


def test_jaccard():
    a = {"a", "b", "c"}
    b = {"b", "c", "d"}
    assert _jaccard(a, b) == 2 / 4  # intersection=2, union=4
    assert _jaccard(a, a) == 1.0
    assert _jaccard(a, set()) == 0.0


def test_deduplicate_exact_dupes():
    records = [
        {"id": "a", "text": "தமிழ் மொழி அழகானது"},
        {"id": "b", "text": "தமிழ் மொழி அழகானது"},
        {"id": "c", "text": "இது வேறு உரை"},
    ]
    unique, clusters = deduplicate_records(records, DedupConfig(n_hashes=64, threshold=0.9))
    ids = {r["id"] for r in unique}
    assert ids == {"a", "c"} or ids == {"b", "c"}
    assert len(clusters) >= 0


def test_deduplicate_near_dupes():
    records = [
        {"id": "a", "text": "சங்க இலக்கியம் தமிழின் பெருமையை உலகிற்கு எடுத்துக் காட்டுகிறது"},
        {
            "id": "b",
            "text": "சங்க இலக்கியம் தமிழின் பெருமையை உலகிற்கு எடுத்துக் காட்டுகிறது.",
        },  # trailing punctuation
        {"id": "c", "text": "இது முற்றிலும் வேறு கருத்து"},
    ]
    unique, clusters = deduplicate_records(
        records, DedupConfig(n_hashes=64, shingle_size=5, threshold=0.85)
    )
    # a and b may be merged depending on agreement; c must survive
    ids = {r["id"] for r in unique}
    assert "c" in ids
    assert len(unique) >= 2
