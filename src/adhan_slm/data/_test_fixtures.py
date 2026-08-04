"""Shared corpus + tokenizer fixtures for the data-pipeline test modules.

Leading underscore so pytest never collects this as a test module. Kept as a plain
importable module rather than a `conftest.py` because the data tests must also run
standalone (`python -m adhan_slm.data.packing_tests`), where conftest fixtures
do not exist.
"""

from __future__ import annotations

from typing import List

from adhan_slm.tokenizer import SwaramTokenizer, default_akshara_inventory

#: Four short, grammatical Tamil sentences — enough distinct aksharas to train a
#: merge layer, small enough that packing/loader assertions stay hand-checkable.
TAMIL_LINES: List[str] = [
    "நான் பள்ளிக்கு போகிறேன்",
    "அவன் புத்தகம் படித்தான்",
    "நாய் வேகமாக ஓடியது",
    "அம்மா சாதம் சமைத்தார்",
]


def sample_tokenizer() -> SwaramTokenizer:
    """A tiny frozen swaram tokenizer over `TAMIL_LINES` (akshara inventory + 64 merges)."""
    return SwaramTokenizer.train(TAMIL_LINES, vocab_size=len(default_akshara_inventory()) + 64)
