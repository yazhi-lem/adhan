"""Hardening tests for Swaram (Tamil) and Aksharam (Devanagari).

These pin the invariants that make the tokenizer safe on real corpus text:
losslessness under *any* input, NFC idempotence, no empty aksharas, agreement
between the two tokenizers and the generic segmenter, correct handling of the
characters that used to break it (Tamil anusvara, the ௗ length mark, ZWJ/ZWNJ,
Devanagari composition exclusions, Vedic accents), and no unassigned codepoints
in any inventory.

Run: ``python -m pytest tests/test_tokenizer_hardening.py``
"""

from __future__ import annotations

import random
import time
import unicodedata

import pytest

from adhan_slm.common.scripts import (
    DEVANAGARI,
    TAMIL,
    akshara_inventory,
    segment,
)
from adhan_slm.tokenizer import indic_akshara as ia
from adhan_slm.tokenizer import morphology as m
from adhan_slm.tokenizer.aksharam_tokenizer import (
    AksharamTokenizer,
    default_aksharam_inventory,
    segment_devanagari,
)
from adhan_slm.tokenizer.swaram_tokenizer import (
    SwaramTokenizer,
    default_akshara_inventory,
    segment_aksharas,
)

# -- fixtures ----------------------------------------------------------------

# Hand-picked nasties: anusvara, āytam, the au length mark, a vintage ZWNJ,
# combining marks with no base, Devanagari composition exclusions, Vedic
# accents, prishthamatra, digits, emoji, and an unpaired surrogate-free pile of
# characters that are *not* Indic at all.
HARD_TAMIL = [
    "தமிழ்நாடு",
    "கற்போம்",
    "அஂ",  # anusvara ஂ used to become its own akshara
    "எஃகு",  # āytam is a letter, not a matra
    "கொ",  # NFC-composed ொ
    "கெ\u0bbe",  # NFD-ish ெ + ா (must compose to ொ by NFC)
    "தமிழ்\u200cநாடு",  # ZWNJ must not split the akshara
    "தமிழ்\u200dநாடு",  # ZWJ
    "௦௧௨௩௪௫௬௭௮௯",
    "க்ஷ ஸ்ரீ",
    "\u0bbe",  # a matra with no base consonant
    "",  # empty
    "Tamil 123 😀 ₹",
]

HARD_DEVANAGARI = [
    "हिन्दी",
    "क्षत्रिय",
    "क़िला",  # NFC: क + ़ + ि
    "\u0958\u093f",  # precomposed क़ + ि, before normalization
    "क्‍ष",  # ZWJ half-form control
    "हूँ",
    "संस्कृतम्",
    "कॎ",  # prishthamatra e
    "क᳐",  # Vedic accent
    "०१२३४५६७८९",
    "\u093f",  # a vowel sign with no base
    "",
    "Hindi 123 😀 ₹",
]

# Random codepoints from each block, including the unassigned holes: whatever
# they are, segmentation must stay lossless and never emit an empty piece.
random.seed(20261109)


def _random_block_text(script, n: int = 300) -> str:
    cps = [random.randrange(script.block_start, script.block_end + 1) for _ in range(n)]
    return "".join(chr(cp) for cp in cps)


# -- losslessness and idempotence --------------------------------------------


@pytest.mark.parametrize("text", HARD_TAMIL)
def test_swaram_lossless_and_nfc_idempotent(text: str) -> None:
    pieces = segment_aksharas(text)
    assert "".join(pieces) == unicodedata.normalize("NFC", text)
    assert all(pieces), "segmentation must never emit an empty akshara"
    assert segment_aksharas("".join(pieces)) == pieces


@pytest.mark.parametrize("text", HARD_DEVANAGARI)
def test_aksharam_lossless_and_nfc_idempotent(text: str) -> None:
    pieces = segment_devanagari(text)
    assert "".join(pieces) == unicodedata.normalize("NFC", text)
    assert all(pieces)
    assert segment_devanagari("".join(pieces)) == pieces


@pytest.mark.parametrize("script", [TAMIL, DEVANAGARI], ids=lambda s: s.name)
def test_random_block_text_is_lossless(script) -> None:
    text = _random_block_text(script)
    pieces = segment(text, (script,))
    assert "".join(pieces) == unicodedata.normalize("NFC", text)
    assert all(pieces)


@pytest.mark.parametrize("script", [TAMIL, DEVANAGARI], ids=lambda s: s.name)
def test_random_ascii_and_emoji_pass_through(script) -> None:
    text = "abc 😀 123\t\n,,, ൠ"
    pieces = segment(text, (script,))
    assert "".join(pieces) == text
    assert all(len(p) == 1 for p in pieces)


def test_joined_pieces_always_re_segment_to_themselves() -> None:
    """A tokenizer that is not idempotent silently corrupts its own output."""
    for text in HARD_TAMIL:
        pieces = segment_aksharas(text)
        for piece in pieces:
            assert segment_aksharas(piece) == [piece] or len(piece) == 1


# -- the specific bugs this hardening fixed ----------------------------------


def test_tamil_anusvara_attaches_to_the_previous_letter() -> None:
    # U+0B82 was missing from the combining set, so அஂ became அ + ஂ.
    assert segment_aksharas("அஂ") == ["அஂ"]


def test_tamil_au_length_mark_attaches() -> None:
    # U+0BD7 (ௗ) is the length mark of ௌ; it must never start an akshara.
    assert segment_aksharas("கௗ") == ["கௗ"]


def test_zero_width_joiners_do_not_split_aksharas() -> None:
    for joiner in ("\u200c", "\u200d"):
        assert segment_aksharas("க" + joiner) == ["க" + joiner]
        assert segment_devanagari("क" + joiner) == ["क" + joiner]
    # Devanagari half-form: क + virama + ZWJ + ष is one akshara.
    assert segment_devanagari("क्\u200dष") == ["क्\u200dष"]


def test_devanagari_composition_exclusions_are_normalized_not_invented() -> None:
    """U+0958–U+095F are composition exclusions: NFC spells them क + ़."""
    assert segment_devanagari("\u0958\u093f") == ["क़ि"]
    assert segment_devanagari("क़िला") == ["क़ि", "ला"]
    # The precomposed spelling is not what NFC text contains.
    assert "\u0958" not in "".join(segment_devanagari("\u0958"))


def test_devanagari_vedic_and_prishthamatra_signs_attach() -> None:
    assert segment_devanagari("क\u1cd0") == ["क\u1cd0"]
    assert segment_devanagari("कॎ") == ["कॎ"]
    assert segment_devanagari("क\u0962") == ["क\u0962"]


def test_a_mark_never_attaches_to_whitespace_or_punctuation() -> None:
    """An orphaned vowel sign must not invent composite tokens like " ா".

    Real corpora contain these (a sign stranded by a line break): 51 distinct
    forms over 160k documents before this was fixed.
    """
    assert segment_aksharas("போ  ா வா") == ["போ", " ", " ", "ா", " ", "வா"]
    assert segment_devanagari("क \u093f") == ["क", " ", "ि"]
    # A mark that does follow a letter still attaches, and later marks chain.
    assert segment_aksharas("பொ") == ["பொ"]
    assert segment_aksharas("\u0bbe\u0bbf") == ["\u0bbe\u0bbf"]


def test_leading_matra_does_not_crash_or_vanish() -> None:
    for text in ("\u0bbe", "\u093f", "\u0b82"):
        assert "".join(segment_aksharas(text)) == unicodedata.normalize("NFC", text)
        assert "".join(segment_devanagari(text)) == unicodedata.normalize("NFC", text)


# -- one implementation, no drift --------------------------------------------


@pytest.mark.parametrize("text", HARD_TAMIL)
def test_swaram_agrees_with_the_generic_segmenter(text: str) -> None:
    assert segment_aksharas(text) == segment(text, (TAMIL,))
    assert segment_aksharas(text) == ia.segment(text, [TAMIL])


@pytest.mark.parametrize("text", HARD_DEVANAGARI)
def test_aksharam_agrees_with_the_generic_segmenter(text: str) -> None:
    assert segment_devanagari(text) == segment(text, (DEVANAGARI,))
    assert segment_devanagari(text) == ia.segment(text, [DEVANAGARI])


def test_script_selection_cache_does_not_leak_between_scripts() -> None:
    """``_owner_index`` is lru_cached by script set; a stale cache would be silent."""
    tamil_first = segment("क्ष", (TAMIL,))
    deva = segment("क्ष", (DEVANAGARI,))
    tamil_again = segment("क्ष", (TAMIL,))
    assert tamil_first == tamil_again == ["क", "्", "ष"]
    assert deva == ["क्ष"]


# -- inventories -------------------------------------------------------------


@pytest.mark.parametrize("inventory", [default_akshara_inventory, default_aksharam_inventory])
def test_inventories_are_closed_deduplicated_and_assigned(inventory) -> None:
    inv = inventory()
    assert inv and len(inv) == len(set(inv))
    for akshara in inv:
        assert unicodedata.normalize("NFC", akshara) == akshara, "inventory must be NFC"
        for ch in akshara:
            assert unicodedata.category(ch) != "Cn", f"unassigned codepoint in {akshara!r}"


def test_tamil_inventory_covers_every_tamil_letter() -> None:
    inv = set(default_akshara_inventory())
    for akshara in akshara_inventory(TAMIL):
        assert akshara in inv
    # Every consonant × every vowel sign is representable.
    for consonant in TAMIL.consonants():
        assert consonant in inv
        for sign in TAMIL.vowel_signs():
            assert consonant + sign in inv


def test_devanagari_inventory_keeps_nukta_and_digits() -> None:
    inv = set(default_aksharam_inventory())
    assert "क़" in inv and "क़ि" in inv
    assert {"०", "५", "९"} <= inv


# -- letter classes and morphology now agree --------------------------------


def test_grantha_letters_are_classified() -> None:
    """Grantha meis used to fall through to OTHER, silently killing sandhi."""
    for letter in ("ஜ", "ஶ", "ஷ", "ஸ", "ஹ"):
        assert m.letter_class(letter) is m.LetterClass.MEI
        assert m.letter_class(letter + "ா") is m.LetterClass.UYIRMEI
        assert m.letter_class(letter + "்") is m.LetterClass.MEI


def test_morphology_counts_match_the_registry() -> None:
    assert len(m.UYIR) == 12
    assert len(m.MEI) == 18
    assert len(m.GRANTHA) == 5
    assert len(m.MEI_ALL) == 23
    assert len(m.MATRAS) == 11
    assert all(unicodedata.category(sign) in {"Mn", "Mc"} for sign in m.MATRAS)


def test_final_akshara_handles_grantha_and_vowel_signs() -> None:
    assert m.final_akshara("ராஜா") == "ஜா"
    # The final akshara of a cluster is the last consonant + its vowel sign.
    assert m.final_akshara("ஸ்ரீ") == "ரீ"
    assert m.final_akshara("கண்") == "ண்"
    assert m.final_akshara("") == ""


# -- end-to-end: the tokenizers survive the hard cases -----------------------


def _swaram_tokenizer() -> SwaramTokenizer:
    corpus = [
        " ".join(HARD_TAMIL) * 2,
        "தமிழ்நாடு மொழி உலகின் மிகச் சிறந்த மொழிகளில் ஒன்று",
        "அஂ எஃகு கொ ௧௨௩",
    ] * 5
    return SwaramTokenizer.train(corpus, vocab_size=len(default_akshara_inventory()) + 128)


def test_swaram_round_trip_on_hard_tamil() -> None:
    tok = SwaramTokenizer.train(
        [" ".join(HARD_TAMIL)] * 5,
        vocab_size=len(default_akshara_inventory()) + 128,
        byte_fallback=True,
    )
    for text in HARD_TAMIL:
        if not text:
            continue
        ids = tok.encode(text)
        assert tok.decode(ids) == unicodedata.normalize("NFC", text), text


def test_aksharam_round_trip_on_hard_devanagari() -> None:
    tok = AksharamTokenizer.train(
        [" ".join(HARD_DEVANAGARI)] * 5,
        vocab_size=len(default_aksharam_inventory()) + 128,
        byte_fallback=True,
    )
    for text in HARD_DEVANAGARI:
        if not text:
            continue
        ids = tok.encode(text)
        assert tok.decode(ids) == unicodedata.normalize("NFC", text), text


# -- performance guard -------------------------------------------------------


def test_segmentation_is_linear_enough_for_corpus_work() -> None:
    text = "தமிழ்நாடு மொழி " * 2000  # ~28k aksharas
    start = time.perf_counter()
    pieces = segment_aksharas(text)
    elapsed = time.perf_counter() - start
    assert len(pieces) > 15000
    assert elapsed < 1.0, f"segmentation took {elapsed:.2f}s for {len(text)} chars"
