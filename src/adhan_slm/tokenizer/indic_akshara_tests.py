"""Tests for script-generic akshara segmentation (indic_akshara)."""

from __future__ import annotations

import unicodedata

import pytest

from adhan_slm.tokenizer import indic_akshara as ia
from adhan_slm.tokenizer.aksharam_tokenizer import segment_devanagari
from adhan_slm.tokenizer.swaram_tokenizer import segment_aksharas

# (text, script) pairs: every Indic script the tokenizer family targets, plus
# Latin/digits/emoji, which must pass through untouched.
SAMPLES = [
    ("தமிழ்நாடு மொழி க்ஷ ஸ்ரீ ஃ", ia.TAMIL),
    ("हिन्दी भाषा क्ष त्र ज्ञ पढ़ र्ह", ia.DEVANAGARI),
    ("తెలుగు భాష", ia.TELUGU),
    ("ಕನ್ನಡ ಭಾಷೆ", ia.KANNADA),
    ("മലയാളം ഭാഷ", ia.MALAYALAM),
    ("বাংলা ভাষা র্য", ia.BENGALI),
    ("ਪੰਜਾਬੀ ਭਾਸ਼ਾ", ia.GURMUKHI),
    ("ગુજરાતી ભાષા", ia.GUJARATI),
    ("ଓଡ଼ିଆ ଭାଷା", ia.ORIYA),
    ("සිංහල භාෂාව", ia.SINHALA),
    ("தமிழ் + हिन्दी + English 123 😀 ₹", None),
]


@pytest.mark.parametrize("text,script", SAMPLES)
def test_segmentation_is_lossless(text: str, script) -> None:
    pieces = ia.segment(text, [script]) if script else ia.segment(text)
    assert "".join(pieces) == unicodedata.normalize("NFC", text)


@pytest.mark.parametrize("text,script", SAMPLES)
def test_segmentation_emits_no_empty_pieces(text: str, script) -> None:
    pieces = ia.segment(text, [script]) if script else ia.segment(text)
    assert pieces and all(p for p in pieces)


def test_tamil_matches_reference_segmenter() -> None:
    text = "தமிழ்நாடு வாழ்க க்ஷ ஸ்ரீ ஃ கல்லான் வெகுளும்"
    assert ia.segment(text, [ia.TAMIL]) == segment_aksharas(text)


def test_devanagari_matches_reference_segmenter() -> None:
    text = "हिन्दी क्ष त्र ज्ञ पढ़ र्ह"
    assert ia.segment(text, [ia.DEVANAGARI]) == segment_devanagari(text)


def test_tamil_aytham_is_a_letter_not_a_matra() -> None:
    """0B83 aytham (ஃ) starts its own akshara; it does not attach to the previous one."""
    assert ia.segment("எஃகு", [ia.TAMIL]) == ["எ", "ஃ", "கு"]


def test_tamil_does_not_join_conjuncts_but_devanagari_does() -> None:
    # Tamil pulli ends an akshara; the next consonant starts a new one.
    assert ia.segment("க்ஷ", [ia.TAMIL]) == ["க்", "ஷ"]
    # Devanagari halant glues the next consonant into one conjunct.
    assert ia.segment("क्ष", [ia.DEVANAGARI]) == ["क्ष"]


def test_zero_width_joiner_attaches() -> None:
    assert ia.segment("র‍্য", [ia.BENGALI]) == ["র‍্য"]


def test_script_lookup() -> None:
    assert ia.script_of(ord("த")) is ia.TAMIL
    assert ia.script_of(ord("क")) is ia.DEVANAGARI
    assert ia.script_of(ord("A")) is None


def test_derived_inventory_is_closed_over_matra_combinations() -> None:
    inv = ia.base_inventory(ia.DEVANAGARI)
    assert "क" in inv  # inherent vowel
    assert "क्" in inv  # virama form
    for matra in ia.DEVANAGARI.matras():
        assert "क" + matra in inv
    assert len(inv) == len(set(inv))


def test_script_alphabet_covers_block() -> None:
    alpha = ia.script_alphabet(ia.TAMIL)
    assert "க" in alpha and chr(ia.TAMIL.virama) in alpha
    assert len(alpha) == ia.TAMIL.block_end - ia.TAMIL.block_start + 1
