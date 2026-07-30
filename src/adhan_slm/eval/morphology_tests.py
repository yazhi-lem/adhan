"""Tests for the Tamil morphology probes (`adhan_slm.eval.morphology`).

Roadmap Phase 4: stemmer-boundary agreement between the tokenizer's Layer-B merges
and open-tamil's `TamilStemmer`, plus the sandhi (புணர்ச்சி) correctness rate.

Run: PYTHONPATH=src python -m adhan_slm.eval.morphology_tests
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adhan_slm.core.selftest import run_module_tests  # noqa: E402
from adhan_slm.eval._test_fixtures import (  # noqa: E402
    GRAMMATICAL_PHRASES,
    INFLECTED_WORDS,
    requires_open_tamil,
)


@requires_open_tamil
def test_stemmer_boundary_agreement_on_trained_tokenizer() -> None:
    from adhan_slm.eval.morphology import stemmer_boundary_agreement
    from adhan_slm.tokenizer import SwaramTokenizer, default_akshara_inventory

    tok = SwaramTokenizer.train(INFLECTED_WORDS, vocab_size=len(default_akshara_inventory()) + 20)
    report = stemmer_boundary_agreement(tok, INFLECTED_WORDS)

    assert report.n_with_suffix > 0, "expected the stemmer to find suffixes in inflected words"
    assert 0.0 <= report.agreement_rate <= 1.0


@requires_open_tamil
def test_sandhi_probe_counts_every_word() -> None:
    from adhan_slm.eval.morphology import sandhi_correctness_rate

    report = sandhi_correctness_rate(GRAMMATICAL_PHRASES)
    assert report.n_words == sum(len(p.split()) for p in GRAMMATICAL_PHRASES)


@requires_open_tamil
def test_sandhi_probe_finds_nothing_to_fix_in_correct_phrases() -> None:
    """The probe's floor: grammatical input must score 1.0, or it flags false positives."""
    from adhan_slm.eval.morphology import sandhi_correctness_rate

    report = sandhi_correctness_rate(GRAMMATICAL_PHRASES)
    assert report.word_correctness_rate == 1.0, report.corrections


if __name__ == "__main__":
    run_module_tests(globals(), "eval morphology probes")
