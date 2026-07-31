"""Regression tests for TextDeduplicator.deduplicate() stats reporting.

Guards against a bug where deduplicate() always returned an empty stats
dict (via a hardcoded `_dedup_stats_placeholder()`), silently discarding
the real counts/rates computed inside the inner generator.
"""

from adhan_slm.data.deduplicator import TextDeduplicator


def _docs():
    return [
        {"id": "a", "text": "இது ஒரு சோதனை வாக்கியம்", "source": "wiki"},
        {"id": "b", "text": "இது ஒரு சோதனை வாக்கியம்", "source": "wiki"},  # exact dup of a
        {"id": "c", "text": "முற்றிலும் வேறு உள்ளடக்கம்", "source": "news"},
    ]


def test_deduplicate_returns_real_stats_not_empty_dict():
    dedup = TextDeduplicator()
    gen, stats = dedup.deduplicate(iter(_docs()))

    # Before the generator is exhausted, stats is only partially populated —
    # that's expected, since counts accrue as documents are streamed through.
    kept = list(gen)

    assert [d["id"] for d in kept] == ["a", "c"]

    # This is the crux of the regression: stats must NOT be `{}`.
    assert stats != {}
    assert stats["total_seen"] == 3
    assert stats["kept"] == 2
    assert stats["removed"] == 1
    assert stats["exact_duplicates"] == 1
    assert stats["near_duplicates"] == 0
    assert stats["removal_rate"] == 1 / 3
    assert stats["per_source"]["wiki"] == {"seen": 2, "duplicates": 1}
    assert stats["per_source"]["news"] == {"seen": 1, "duplicates": 0}


def test_deduplicate_stats_on_empty_input():
    dedup = TextDeduplicator()
    gen, stats = dedup.deduplicate(iter([]))
    assert list(gen) == []
    assert stats["total_seen"] == 0
    assert stats["removal_rate"] == 0.0
