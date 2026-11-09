"""Script-generic Brahmic (Indic) akshara segmentation — compatibility facade.

The tables and the segmenter moved to :mod:`adhan_slm.common.scripts` so that
Swaram (Tamil), Aksharam (Devanagari) and this generic path cannot disagree
about the same Unicode block. This module keeps the historical import surface
(``from adhan_slm.tokenizer.indic_akshara import TAMIL, segment``) working.

Why one generic segmenter
-------------------------
Every Indic script here is an abugida: a consonant carries an inherent vowel, a
trailing *vowel sign* changes that vowel, and a *pulli* / halant (virama)
suppresses it. Two per-script facts differ and live in
:class:`adhan_slm.common.scripts.Script`:

* ``join_conjuncts`` — whether ``virama + consonant`` forms one akshara
  (Devanagari ``क्ष``, ``त्र``) or splits (Tamil ``க்`` + ``ஷ``). Tamil pulli
  does not create conjunct graphemes; Devanagari halant does.
* which offsets inside the block are combining marks. This is *not* uniform:
  Tamil ``ஃ`` (U+0B83, āytam) is a standalone letter while Devanagari
  U+0900–U+0903 are all combining signs.

Everything is lossless: ``"".join(segment(text)) == NFC(text)``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from adhan_slm.common.scripts import (  # noqa: F401  (re-exported API)
    AYTHAM,
    BENGALI,
    DEVANAGARI,
    GRANTHA,
    GUJARATI,
    GURMUKHI,
    KANNADA,
    MALAYALAM,
    MEI,
    MEI_ALL,
    ORIYA,
    PULLI,
    SCRIPT_BY_NAME,
    SCRIPTS,
    SINHALA,
    TAMIL,
    TAMIL_AYTHAM,
    TAMIL_DIGITS,
    TAMIL_GRANTHA_MEI,
    TAMIL_MEI,
    TAMIL_UYIR,
    TELUGU,
    UYIR,
    UYIRMEI_PER_CONSONANT,
    ZERO_WIDTH_JOINERS,
    Script,
    akshara_inventory,
    base_inventory,
    script_alphabet,
    script_for_char,
    script_of,
    segment,
    validate_registry,
    vowel_signs,
)

__all__ = [
    "AYTHAM",
    "BENGALI",
    "DEVANAGARI",
    "GRANTHA",
    "GUJARATI",
    "GURMUKHI",
    "KANNADA",
    "MALAYALAM",
    "MEI",
    "MEI_ALL",
    "ORIYA",
    "PULLI",
    "SCRIPTS",
    "SCRIPT_BY_NAME",
    "SINHALA",
    "Script",
    "TAMIL",
    "TAMIL_AYTHAM",
    "TAMIL_DIGITS",
    "TAMIL_GRANTHA_MEI",
    "TAMIL_MEI",
    "TAMIL_UYIR",
    "TELUGU",
    "UYIR",
    "UYIRMEI_PER_CONSONANT",
    "ZERO_WIDTH_JOINERS",
    "akshara_inventory",
    "base_inventory",
    "script_alphabet",
    "script_for_char",
    "script_of",
    "segment",
    "validate_registry",
    "vowel_signs",
]


def segment_with(text: str, scripts: Optional[Sequence[Script]] = None) -> List[str]:
    """Segment ``text`` using only ``scripts`` (``None`` means every script)."""
    return segment(text, tuple(scripts) if scripts is not None else SCRIPTS)
