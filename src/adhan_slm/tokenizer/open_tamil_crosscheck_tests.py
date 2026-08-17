"""Differential test: our from-scratch Layer A akshara segmentation vs
open-tamil's `tamil.utf8.get_letters` reference implementation.

open-tamil is optional (see requirements-jax.txt) — the swaram/aksharam
tokenizer cores stay pure-python and dependency-free by design (see their
module docstrings), so these tests skip cleanly rather than fail when
open-tamil isn't installed, instead of gating the whole test suite on it.

Run: PYTHONPATH=src python -m adhan_slm.tokenizer.open_tamil_crosscheck_tests
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on path

from adhan_slm.core.selftest import run_module_tests, skip_unless  # noqa: E402
from adhan_slm.external.open_tamil_bridge import (  # noqa: E402
    HAS_OPEN_TAMIL,
    reference_segment_aksharas,
)
from adhan_slm.tokenizer import segment_aksharas  # noqa: E402

SAMPLES = [
    "தமிழ்",
    "கற்போம்",
    "படித்துக்கொண்டிருந்தேன்",
    "வணக்கம்",
    "அப்பா",
    "செய்தார்",
    "நடந்தது",
    "பார்த்தேன்",
    "கேட்டான்",
    "மரம் வளர்ந்தது",
]


@skip_unless(HAS_OPEN_TAMIL, "open-tamil not installed")
def test_agrees_with_open_tamil_reference_segmenter():
    for s in SAMPLES:
        ours = segment_aksharas(s)
        reference = reference_segment_aksharas(s)
        assert (
            ours == reference
        ), f"segmentation mismatch for {s!r}: ours={ours} open-tamil={reference}"


if __name__ == "__main__":
    run_module_tests(globals(), "open-tamil segmentation crosscheck")
