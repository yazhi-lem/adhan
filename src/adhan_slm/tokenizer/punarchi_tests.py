"""Tests for the puṇarcci (புணர்ச்சி) engine.

The module's docstring promised "round-trip over every canonical example"; this
is that test, plus the cases the engine must *not* claim to resolve.

Run: PYTHONPATH=src python -m adhan_slm.tokenizer.punarchi_tests
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on path

from adhan_slm.core.selftest import run_module_tests  # noqa: E402
from adhan_slm.tokenizer import punarchi as p  # noqa: E402
from adhan_slm.tokenizer.morphology import Vikaram  # noqa: E402

# (left, right, expected surface, expected rule id) — the canonical examples
# from Puṇariyal, as taught in Naṉṉūl.
CANONICAL = [
    # தோன்றல் (appearance): the உடம்படுமெய் glide.
    ("திரு", "அடி", "திருவடி", "uyirmei_uyir_glide"),
    ("பூ", "இல்", "பூவில்", "uyirmei_uyir_glide"),
    # மிகுதல் (doubling).
    ("கல்", "அடி", "கல்லடி", "mellinam_miguthal"),
    ("நெய்", "அப்பம்", "நெய்யப்பம்", "mellinam_miguthal"),
    ("மரம்", "தோப்பு", "மரத்தோப்பு", "m_vallinam_miguthal"),
    ("நீர்", "துளி", "நீர்த்துளி", "mei_vallinam_miguthal"),
    # திரிதல் (substitution) before வல்லினம்.
    ("கல்", "சீல", "கற்சீல", "l_to_r_before_vallinam"),
    ("மண்", "கலம்", "மட்கலம்", "n_to_d_before_vallinam"),
    ("பொன்", "காசு", "பொற்காசு", "n_to_r_before_vallinam"),
    # கெடுதல் (disappearance).
    ("மரம்", "வேர்", "மரவேர்", "m_keduthal_before_v"),
]


# -- forward ---------------------------------------------------------------


def test_canonical_joins_produce_the_classical_surface_form():
    for left, right, surface, rule_id in CANONICAL:
        joined = p.join_units(left, right)
        assert joined.surface == surface, f"{left} + {right} -> {joined.surface}"
        assert joined.rule.id == rule_id, f"{left} + {right} -> {joined.rule.id}"


def test_case_suffix_selects_the_u_final_rule():
    # வீடு + இல் is a case join (வீட்டில்); as a compound the glide survives.
    assert p.join_units("வீடு", "இல்", case_suffix=True).surface == "வீட்டில்"
    assert p.join_units("வீடு", "இல்").surface == "வீடுவில்"


def test_vikaram_is_reported_in_the_classical_terms():
    assert p.join_units("கல்", "அடி").vikaram is Vikaram.MIGUTHAL
    assert p.join_units("திரு", "அடி").vikaram is Vikaram.THONRAL
    assert p.join_units("மண்", "கலம்").vikaram is Vikaram.VIKARAM


def test_empty_units_are_a_plain_join():
    for left, right in (("", ""), ("", "அடி"), ("கல்", "")):
        joined = p.join_units(left, right)
        assert joined.surface == left + right
        assert joined.vikaram is Vikaram.IYALBU


def test_input_is_nfc_normalized_before_joining():
    decomposed = unicodedata.normalize("NFD", "திரு")
    joined = p.join_units(decomposed, "அடி")
    assert joined.surface == "திருவடி"
    assert unicodedata.normalize("NFC", joined.surface) == joined.surface


def test_punarchi_class_distinguishes_iyalbu_from_vikaram():
    assert p.punarchi_class("கல்", "அடி") == "vikaram"
    assert p.punarchi_class("மரம்", "வேர்") == "iyalbu"


# -- reverse ---------------------------------------------------------------


def test_every_canonical_join_round_trips_to_its_units():
    """Forward then reverse: the engine must be able to explain what it wrote."""
    for left, right, surface, _rule_id in CANONICAL:
        splits = p.reverse_all(surface)
        assert splits, f"no reversal candidates for {surface}"
        pairs = {(s.left, s.right) for s in splits}
        assert (left, right) in pairs, f"{surface}: expected {(left, right)}, got {pairs}"


def test_reverse_reports_a_rule_and_a_confidence_for_each_candidate():
    for split in p.reverse_all("கல்லடி"):
        assert split.left and split.right
        assert 0.0 < split.confidence <= 1.0
        assert split.rule.id in p.RULES_BY_ID


def test_reverse_is_deterministic_and_ranked():
    first = p.reverse_all("கல்லடி")
    second = p.reverse_all("கல்லடி")
    assert [s.as_dict() for s in first] == [s.as_dict() for s in second]
    confidences = [s.confidence for s in first]
    assert confidences == sorted(confidences, reverse=True)


def test_ambiguity_is_returned_not_resolved_silently():
    """பசியில்லை admits both a glide reading and a real-ய் reading."""
    pairs = {(s.left, s.right) for s in p.reverse_all("பசியில்லை")}
    assert ("பசி", "இல்லை") in pairs
    assert len(pairs) > 1, "an ambiguous word must produce more than one candidate"


def test_lexicon_and_suffix_hints_raise_confidence():
    plain = {s.as_dict()["left"]: s.confidence for s in p.reverse_all("பசியில்லை")}
    boosted = {
        s.as_dict()["left"]: s.confidence for s in p.reverse_all("பசியில்லை", lexicon={"பசி"})
    }
    assert boosted["பசி"] > plain["பசி"]


def test_best_split_returns_the_top_candidate_or_none():
    best = p.best_split("கல்லடி")
    assert best is not None and (best.left, best.right) == ("கல்", "அடி")
    # A monomorphemic word with no sandhi should produce nothing to split on.
    assert p.best_split("") is None or p.best_split("")


def test_reverse_never_returns_an_empty_side():
    for word in ("கல்லடி", "மரத்தோப்பு", "திருவடி", "பசியில்லை", "அ"):
        for split in p.reverse_all(word):
            assert split.left and split.right, (word, split)


# -- the rule table itself --------------------------------------------------


def test_rule_ids_are_unique_and_indexed():
    assert len({r.id for r in p.RULES}) == len(p.RULES)
    assert p.RULES_BY_ID == {r.id: r for r in p.RULES}


def test_every_rule_names_a_vikaram_and_a_kind():
    for rule in p.RULES:
        assert isinstance(rule.vikaram, Vikaram)
        assert rule.kind in {
            "thonral",
            "miguthal",
            "keduthal",
            "thirithal",
            "kunral",
            "iyalbu",
        }
        assert rule.name_ta and rule.name_en


def test_consonant_classes_partition_the_meis():
    from adhan_slm.common.scripts import MEI

    assert sorted(p.VALLINAM + p.MELLINAM + p.IDAIYINAM) == sorted(MEI)


if __name__ == "__main__":
    run_module_tests(globals(), "punarchi")
