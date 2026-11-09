"""Tests for benchmark contamination detection module."""

from __future__ import annotations

import pytest

from adhan_slm.data.contamination import (
    BenchmarkIndex,
    ContaminationConfig,
    _char_ngrams,
    _jaccard,
    _normalise,
    _word_ngrams,
    detect_contamination,
)

pytestmark = pytest.mark.unit


def test_normalise():
    assert _normalise("  Hello   World  ") == "hello world"
    assert _normalise("Tamil\nLanguage") == "tamil language"


def test_word_ngrams():
    text = "the quick brown fox"
    grams = _word_ngrams(text, 2)
    assert "the quick" in grams
    assert "quick brown" in grams
    assert "brown fox" in grams
    assert len(grams) == 3


def test_char_ngrams():
    text = "hello"
    grams = _char_ngrams(text, 3)
    assert "hel" in grams
    assert "ell" in grams
    assert "llo" in grams
    assert len(grams) == 3


def test_jaccard_same():
    a = {"x", "y", "z"}
    b = {"x", "y", "z"}
    assert _jaccard(a, b) == 1.0


def test_benchmark_index_exact_match():
    entries = [
        {"id": "b1", "text": "தமிழ் மொழி அழகானது"},
        {"id": "b2", "text": "சங்க இலக்கியம் பழமையானது"},
    ]
    index = BenchmarkIndex(entries)
    contaminated, match_type, _ = index.check("தமிழ் மொழி அழகானது")
    assert contaminated is True
    assert match_type == "exact"


def test_benchmark_index_no_match():
    entries = [
        {"id": "b1", "text": "தமிழ் மொழி அழகானது"},
    ]
    index = BenchmarkIndex(entries, cfg=ContaminationConfig(exact_match=False))
    contaminated, _, _ = index.check("இது முற்றிலும் வேறு உரை")
    assert contaminated is False


def test_detect_contamination_basic():
    corpus = [
        {"id": "c1", "text": "தமிழ் மொழி அழகானது"},
        {"id": "c2", "text": "புதிய உரை இங்கே"},
        {"id": "c3", "text": "சங்க இலக்கியம் பழமையானது"},
    ]
    index = BenchmarkIndex(
        [
            {"id": "b1", "text": "தமிழ் மொழி அழகானது"},
        ]
    )
    clean, report = detect_contamination(corpus, index)
    assert len(clean) == 2
    assert report.total_docs == 3
    assert report.exact_matches == 1
    assert report.total_contaminated == 1


def test_detect_contamination_word_ngram():
    # Use identical sentences: Jaccard of n-gram sets should be 1.0
    cfg = ContaminationConfig(
        exact_match=False,
        word_ngram_size=3,
        word_ngram_threshold=0.8,
        substring_match=False,
        char_ngram_threshold=1.0,  # disable
    )
    corpus = [
        {"id": "c1", "text": "the quick brown fox jumps over the lazy dog"},
        {"id": "c2", "text": "completely different text here"},
    ]
    index = BenchmarkIndex(
        [
            {"id": "b1", "text": "the quick brown fox jumps over the lazy dog"},
        ],
        cfg=cfg,
    )
    clean, report = detect_contamination(corpus, index, text_key="text")
    # c1 is identical to b1 so its word n-gram Jaccard is 1.0
    assert len(clean) == 1
    assert any(r["id"] == "c2" for r in clean)
