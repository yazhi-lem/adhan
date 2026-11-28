"""Purity metric tests.

``compute_pure_tamil_purity`` is reported as a headline number in the Thirukkural
eval report. Two bugs made it overstate purity: the Grantha set omitted ஶ, and the
denominator counted only Tamil-block characters, so Latin-script code-switching
was invisible. These tests pin both fixes and the reporting helper.
"""

from __future__ import annotations

import pytest

from adhan_slm.eval.thirukkural_eval import (
    GRANTHA_CHARS,
    compute_pure_tamil_purity,
    compute_purity_breakdown,
)


class TestGranthaSet:
    def test_matches_documented_inventory(self):
        """The architecture doc lists ஜ ஷ ஸ ஹ ஶ; the code must agree."""
        for char in "ஜஷஸஹஶ":
            assert char in GRANTHA_CHARS, f"{char} missing from GRANTHA_CHARS"

    def test_underscore_form_is_not_grantha(self):
        """ழ is a native Tamil letter, not a Grantha consonant."""
        assert "ழ" not in GRANTHA_CHARS

    def test_common_letters_are_not_grantha(self):
        """க must not be in the set: the old set("ஜஷஸஹக்ஷ") captured it from க்ஷ."""
        for char in "கசடதபற":
            assert char not in GRANTHA_CHARS

    def test_pulli_is_not_grantha(self):
        """The virama came in with க்ஷ and is not a Grantha consonant."""
        assert "்" not in GRANTHA_CHARS

    def test_set_is_exactly_five_consonants(self):
        assert GRANTHA_CHARS == set("ஜஷஸஹஶ")


class TestPureTamilPurity:
    def test_pure_tamil_scores_one(self):
        assert compute_pure_tamil_purity("அறம் வெல்லும்") == pytest.approx(1.0)

    def test_whole_kural_scores_one(self):
        couplet = "அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு."
        assert compute_pure_tamil_purity(couplet) == pytest.approx(1.0)

    def test_ha_form_word_scores_below_one(self):
        """ஶ was absent from the set, so this scored 1.0 before the fix."""
        assert compute_pure_tamil_purity("ஶ்ரீ") < 1.0

    def test_grantha_word_scores_below_one(self):
        assert compute_pure_tamil_purity("ஜனம்") < 1.0

    def test_english_code_switch_scores_below_one(self):
        """Latin letters now reach the denominator, so code-switching costs purity."""
        assert compute_pure_tamil_purity("கூல் AI") < 1.0

    def test_english_only_scores_zero(self):
        assert compute_pure_tamil_purity("hello world") == 0.0

    def test_empty_scores_one(self):
        assert compute_pure_tamil_purity("") == 1.0

    def test_whitespace_only_scores_one(self):
        assert compute_pure_tamil_purity("  \n\t ") == 1.0

    def test_punctuation_is_ignored(self):
        """Digits and punctuation are not letters; they must not depress the score."""
        assert compute_pure_tamil_purity("அறம், 123!") == pytest.approx(1.0)

    def test_purity_is_bounded(self):
        for text in ["அறம்", "கூல் AI", "hello", "", "ஶ்ரீ AI 123"]:
            value = compute_pure_tamil_purity(text)
            assert 0.0 <= value <= 1.0, f"{text!r} produced {value}"


class TestPurityBreakdown:
    def test_counts_are_consistent(self):
        # அறம் = அ ற ம் (3 letters); AI = 2 letters.
        breakdown = compute_purity_breakdown("அறம் AI")
        assert breakdown["tamil_letters"] == 3
        assert breakdown["grantha_letters"] == 0
        assert breakdown["other_letters"] == 2
        assert breakdown["purity"] == pytest.approx(3 / 5)

    def test_grantha_counted_separately(self):
        # ஜனம் = ஜ ன ம் (3 letters), one of them Grantha.
        breakdown = compute_purity_breakdown("ஜனம்")
        assert breakdown["grantha_letters"] == 1
        assert breakdown["tamil_letters"] == 3
        assert breakdown["purity"] == pytest.approx(2 / 3)

    def test_empty_breakdown(self):
        assert compute_purity_breakdown("") == {
            "purity": 1.0,
            "tamil_letters": 0,
            "grantha_letters": 0,
            "other_letters": 0,
        }
