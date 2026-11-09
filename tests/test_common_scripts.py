"""Tests for the shared script registry (``adhan_slm.common.scripts``).

These are the tests that make "one safe home for all the scripts" a fact rather
than an intention: the registry must validate itself, refuse mutation, hold the
classical Tamil counts, and match the Dravidian matrix column for column.
"""

from __future__ import annotations

import unicodedata
from dataclasses import FrozenInstanceError

import pytest

from adhan_slm.common import scripts as s

# -- the registry holds together ---------------------------------------------


def test_blocks_are_aligned_and_non_overlapping() -> None:
    ordered = sorted(s.SCRIPTS, key=lambda sc: sc.block_start)
    for script in ordered:
        assert script.block_start & 0x7F == 0
        assert script.block_end & 0x7F == 0x7F
    for prev, nxt in zip(ordered, ordered[1:]):
        assert prev.block_end < nxt.block_start, f"{prev.name} overlaps {nxt.name}"


def test_script_of_is_total_over_every_registered_block() -> None:
    for script in s.SCRIPTS:
        for cp in (script.block_start, script.block_start + 0x4D, script.block_end):
            assert s.script_of(cp) is script, f"{cp:#06x} not owned by {script.name}"


def test_script_of_rejects_non_indic_codepoints() -> None:
    for ch in ("A", "0", "😀", "₹", "Ω", "\u200d"):
        assert s.script_of(ord(ch)) is None


def test_script_of_neighbours_are_not_confused() -> None:
    """Blocks are adjacent (Tamil 0B80–0BFF, Telugu 0C00…): off-by-one would be silent."""
    assert s.script_for_char("\u0bff") is s.TAMIL
    assert s.script_for_char("\u0c00") is s.TELUGU
    assert s.script_for_char("\u0bff") is not s.script_for_char("\u0c00")


def test_validate_registry_is_idempotent_and_public() -> None:
    s.validate_registry()
    s.validate_registry()


# -- it is safe: frozen, read-only, validated --------------------------------


def test_script_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        s.TAMIL.block_start = 0  # type: ignore[misc]


def test_script_name_map_is_read_only() -> None:
    with pytest.raises(TypeError):
        s.SCRIPT_BY_NAME["klingon"] = s.TAMIL  # type: ignore[index]
    assert s.SCRIPT_BY_NAME["tamil"] is s.TAMIL


def test_matrix_glyphs_are_read_only() -> None:
    column = s.COLUMNS_BY_ROMAN["k"]
    with pytest.raises(TypeError):
        column.glyphs["tamil"] = "X"  # type: ignore[index]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "Bad", "block_start": 0x1200},  # name not lowercase ascii
        {"name": "bad", "block_start": 0x1201},  # block not 128-aligned
        {"name": "bad", "block_start": 0x12FF},  # start after end
        {"name": "bad", "block_start": 0x1200, "extra_marks": (0x0B82,)},
        {"name": "bad", "block_start": 0x1200, "extra_consonants": (0x0000,)},
        {"name": "bad", "block_start": 0x1200, "consonant_offsets": (0x30, 0x10)},
        {"name": "bad", "block_start": 0x1200, "sign_offsets": (0x80,)},
    ],
)
def test_malformed_script_is_rejected_at_definition(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        s.Script(block_end=kwargs.pop("block_end", 0x127F), **kwargs)


# -- Tamil eluttu counts are a contract --------------------------------------


def test_tamil_letter_counts_are_classical() -> None:
    assert len(s.UYIR) == 12
    assert len(s.MEI) == 18
    assert len(s.GRANTHA) == 5
    assert len(s.MEI_ALL) == 23
    assert len(s.TAMIL_VOWEL_SIGNS) == 11  # அ is the consonant's inherent vowel
    assert len(s.TAMIL_DIGITS) == 10
    assert s.AYTHAM == "ஃ"


def test_vowel_signs_are_the_eleven_real_signs() -> None:
    assert list(s.TAMIL_VOWEL_SIGNS) == list("ாிீுூெேைொோௌ")
    for sign in s.TAMIL_VOWEL_SIGNS:
        assert unicodedata.category(sign) in {"Mn", "Mc"}


def test_consonant_classes_partition_the_meis() -> None:
    assert sorted(s.VALLINAM + s.MELLINAM + s.IDAIYINAM) == sorted(s.MEI)
    assert len(s.VALLINAM) == len(s.MELLINAM) == len(s.IDAIYINAM) == 6


def test_tamil_akshara_inventory_size_and_contents() -> None:
    inv = s.akshara_inventory(s.TAMIL)
    assert len(inv) == 322, "12 uyir + āytam + 10 digits + 23 × 13 forms"
    assert len(inv) == len(set(inv))
    for expected in ("அ", "ஃ", "௦", "௯", "க", "க்", "கா", "ஜ", "ஹ", "கௌ"):
        assert expected in inv, expected


def test_inventory_contains_no_unassigned_codepoints() -> None:
    """Regression: U+0BC3–U+0BC5 and U+0BC9 are holes in the Tamil block.

    The old inventory was built from ``range(0x0BBE, 0x0BCD)`` and shipped 69
    reserved codepoints as tokens (''.join of a consonant with U+0BC3).
    """
    holes = {0x0BC3, 0x0BC4, 0x0BC5, 0x0BC9}
    for script in s.SCRIPTS:
        for akshara in s.akshara_inventory(script):
            assert not any(unicodedata.category(ch) == "Cn" for ch in akshara), repr(akshara)
            assert not any(ord(ch) in holes for ch in akshara), repr(akshara)


def test_derived_inventories_cover_the_script_specials() -> None:
    tamil = s.akshara_inventory(s.TAMIL)
    deva = s.akshara_inventory(s.DEVANAGARI)
    assert "ஃ" in tamil and "௧" in tamil  # āytam, Tamil digit one
    assert "क़" in deva and "क़ि" in deva  # precomposed nukta consonant + sign
    assert "कॎ" in deva  # prishthamatra e: a vowel sign outside the common range
    assert "अ" in deva and "ऄ" in deva  # block-defined vowels, not a hand list


def test_every_derived_akshara_starts_with_its_own_script() -> None:
    for script in s.SCRIPTS:
        for akshara in s.akshara_inventory(script):
            assert s.script_of(ord(akshara[0])) is script, akshara


# -- the Dravidian phonetic matrix -------------------------------------------


def test_matrix_has_the_48_numbered_columns() -> None:
    assert len(s.DRAVIDIAN_MATRIX) == 48
    assert [c.number for c in s.DRAVIDIAN_MATRIX] == list(range(1, 49))
    assert {name for c in s.DRAVIDIAN_MATRIX for name in c.glyphs} == set(s.DRAVIDIAN_SCRIPTS)


def test_matrix_gaps_are_exactly_the_tamil_voiced_and_aspirated_series() -> None:
    """Tamil writes neither voicing nor aspiration: 14 cells must stay empty.

    Filling them with the voiceless letter would silently claim that Tamil
    distinguishes /k/ and /g/, which it does not.
    """
    gaps = [c.roman for c in s.DRAVIDIAN_MATRIX if "tamil" not in c.glyphs]
    assert gaps == [
        "kh",
        "g",
        "gh",
        "ch",
        "jh",
        "ṭh",
        "ḍ",
        "ḍh",
        "th",
        "d",
        "dh",
        "ph",
        "b",
        "bh",
    ]


@pytest.mark.parametrize(
    "roman,tamil,telugu,kannada,malayalam",
    [
        ("a", "அ", "అ", "ಅ", "അ"),
        ("k", "க", "క", "ಕ", "ക"),
        ("ḻ", "ழ", "ఴ", "ೞ", "ഴ"),  # the three ways to write the retroflex approximant
        ("ḷ", "ள", "ళ", "ಳ", "ള"),
        ("ṟ", "ற", "ఱ", "ಱ", "റ"),
        ("ś", "ஶ", "శ", "ಶ", "ശ"),  # Grantha in Tamil
        ("h", "ஹ", "హ", "ಹ", "ഹ"),
    ],
)
def test_matrix_specials_match_the_periodic_table(
    roman: str, tamil: str, telugu: str, kannada: str, malayalam: str
) -> None:
    column = s.COLUMNS_BY_ROMAN[roman]
    assert column.get("tamil") == tamil
    assert column.get("telugu") == telugu
    assert column.get("kannada") == kannada
    assert column.get("malayalam") == malayalam


def test_every_matrix_glyph_is_a_letter_of_its_own_block() -> None:
    for column in s.DRAVIDIAN_MATRIX:
        for name, glyph in column.glyphs.items():
            assert len(glyph) == 1
            assert unicodedata.category(glyph) == "Lo", (column.roman, name)
            assert s.script_of(ord(glyph)).name == name, (column.roman, name, glyph)


def test_cognates_report_both_directions_and_respect_gaps() -> None:
    assert s.cognates("க") == {
        "tamil": "க",
        "telugu": "క",
        "kannada": "ಕ",
        "malayalam": "ക",
    }
    # Telugu ఖ is aspirated; Tamil has no such letter, so the key is absent.
    assert s.cognates("ఖ") == {"telugu": "ఖ", "kannada": "ಖ", "malayalam": "ഖ"}
    assert s.cognates("A") == {}


def test_column_lookup_rejects_multi_character_input() -> None:
    with pytest.raises(ValueError):
        s.column_of("கா")
    assert s.column_of("க") is s.COLUMNS_BY_ROMAN["k"]


# -- the one segmenter -------------------------------------------------------


@pytest.mark.parametrize("script", s.SCRIPTS, ids=lambda sc: sc.name)
def test_segment_is_lossless_on_its_own_inventory(script: s.Script) -> None:
    text = "".join(s.akshara_inventory(script))
    pieces = s.segment(text, (script,))
    assert "".join(pieces) == unicodedata.normalize("NFC", text)
    assert all(pieces)


def test_segment_uses_only_the_scripts_it_was_given() -> None:
    # Asked for Tamil only, Devanagari is "other text" and stands alone.
    assert s.segment("क्ष", (s.TAMIL,)) == ["क", "्", "ष"]
    # Asked for Devanagari only, the Tamil akshara is not joined by the halant.
    assert s.segment("क्த", (s.DEVANAGARI,)) == ["क्", "த"]
