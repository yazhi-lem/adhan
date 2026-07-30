"""Tests for the classical n-gram perplexity floor (`adhan_slm.eval.ngram_baseline`).

Roadmap Phase 4: an add-one smoothed unigram-over-aksharas baseline that `adhan-nano`
must clear before the distilgpt2 comparison is worth running.

Run: PYTHONPATH=src python -m adhan_slm.eval.ngram_baseline_tests
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adhan_slm.core.selftest import run_module_tests  # noqa: E402
from adhan_slm.eval._test_fixtures import CORPUS, requires_open_tamil  # noqa: E402


@requires_open_tamil
def test_perplexity_is_finite_and_positive() -> None:
    """An infinite or NaN floor is useless as a comparison target."""
    from adhan_slm.eval.ngram_baseline import AksharaUnigramBaseline

    ppl = AksharaUnigramBaseline(CORPUS).perplexity("நான் தமிழில் படிக்கிறேன்")
    assert ppl > 0, f"expected positive perplexity, got {ppl}"
    assert ppl == ppl, "perplexity must not be NaN"  # ppl == ppl is False for NaN
    assert ppl != float("inf"), "add-one smoothing must keep perplexity finite"


@requires_open_tamil
def test_add_one_smoothing_keeps_unseen_aksharas_possible_but_unlikely() -> None:
    """Laplace smoothing must give an unseen akshara a positive but smaller mass —
    zero would make any text containing it infinitely perplexing."""
    from adhan_slm.eval.ngram_baseline import AksharaUnigramBaseline

    baseline = AksharaUnigramBaseline(CORPUS)
    seen = baseline.probability("த")  # appears in "தமிழ்"
    unseen = baseline.probability("ஃ")  # aytham, absent from CORPUS

    assert 0.0 < unseen < seen, f"expected 0 < p(unseen)={unseen} < p(seen)={seen}"


@requires_open_tamil
def test_non_tamil_text_yields_nan_rather_than_a_bogus_score() -> None:
    """This is a Tamil-script-only floor. `run_eval` relies on the NaN to report the
    probe as inapplicable instead of publishing a meaningless number."""
    from adhan_slm.eval.ngram_baseline import AksharaUnigramBaseline

    ppl = AksharaUnigramBaseline(CORPUS).perplexity("hello")
    assert ppl != ppl, f"expected NaN for non-Tamil text, got {ppl}"


if __name__ == "__main__":
    run_module_tests(globals(), "eval n-gram baseline")
