"""Script-generic Brahmic (Indic) akshara segmentation.

One rule set for every Indic script the tokenizer family targets — Dravidian
(Tamil, Telugu, Kannada, Malayalam) and Indo-Aryan (Devanagari/Hindi, Bengali,
Gujarati, Gurmukhi, Odia) plus Sinhala. Each script is an abugida: a consonant
carries an inherent vowel, a trailing *matra* changes that vowel, and a *virama*
(halant / pulli) suppresses it.

Layer A of Swaram/Aksharam is "split text into aksharas". The Tamil version of
that rule lives in ``swaram_tokenizer.segment_aksharas``; this module expresses
the same rule generically so a new script is a table entry, not a new tokenizer.

Two per-script facts differ and are captured in :class:`Script`:

* ``join_conjuncts`` — whether ``virama + consonant`` forms one akshara
  (Devanagari ``क्ष``, ``त्र``) or splits (Tamil ``க்`` + ``ஷ``). Tamil pulli does
  not create conjunct graphemes; Devanagari halant does.
* which offsets inside the block are combining marks. This is *not* uniform:
  Tamil ``ஃ`` (0B83, aytham) is a standalone letter while Devanagari 0900-0903
  are all combining signs, so the mark set is per script.

Everything here is lossless: ``"".join(segment(text)) == NFC(text)``.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence

# Zero-width joiner / non-joiner participate in real Indic grapheme clusters
# (Bengali র্য, Malayalam chillu forms), so they attach to the previous unit.
_ZW_JOINERS = {0x200C, 0x200D}

# Offsets shared by every Brahmic block: nukta (0x3C), matras (0x3E-0x4C),
# virama (0x4D).
_COMMON_MARK_OFFSETS: tuple[int, ...] = (0x3C,) + tuple(range(0x3E, 0x4E))
_DEVANAGARI_SIGNS = (0x00, 0x01, 0x02, 0x03, 0x51, 0x52, 0x53, 0x54, 0x62, 0x63)
_TAMIL_SIGNS = (0x02,)  # anusvara only; 0B83 aytham is a letter, not a matra


@dataclass(frozen=True)
class Script:
    """Unicode facts needed to segment one Brahmic script into aksharas."""

    name: str
    block_start: int
    block_end: int
    join_conjuncts: bool = True
    sign_offsets: tuple[int, ...] = (0x01, 0x02, 0x03)
    consonant_offsets: tuple[int, int] = (0x15, 0x39)

    def is_combining(self, cp: int) -> bool:
        """True if ``cp`` attaches to the preceding unit (matra, virama, sign...)."""
        if cp in _ZW_JOINERS:
            return True
        if not (self.block_start <= cp <= self.block_end):
            return False
        off = cp - self.block_start
        return off in _COMMON_MARK_OFFSETS or off in self.sign_offsets

    @property
    def virama(self) -> int:
        """The script's virama / halant / pulli codepoint."""
        return self.block_start + 0x4D

    def is_consonant(self, cp: int) -> bool:
        lo, hi = self.consonant_offsets
        return self.block_start + lo <= cp <= self.block_start + hi

    def matras(self) -> List[str]:
        return [chr(self.block_start + o) for o in range(0x3E, 0x4D)]

    def consonants(self) -> List[str]:
        lo, hi = self.consonant_offsets
        return [chr(cp) for cp in range(self.block_start + lo, self.block_start + hi + 1)]


# --- Script tables ------------------------------------------------------------
DEVANAGARI = Script("devanagari", 0x0900, 0x097F, True, _DEVANAGARI_SIGNS)
BENGALI = Script("bengali", 0x0980, 0x09FF)
GURMUKHI = Script("gurmukhi", 0x0A00, 0x0A7F)
GUJARATI = Script("gujarati", 0x0A80, 0x0AFF)
ORIYA = Script("oriya", 0x0B00, 0x0B7F)
TAMIL = Script("tamil", 0x0B80, 0x0BFF, False, _TAMIL_SIGNS)
TELUGU = Script("telugu", 0x0C00, 0x0C7F)
KANNADA = Script("kannada", 0x0C80, 0x0CFF)
MALAYALAM = Script("malayalam", 0x0D00, 0x0D7F)
SINHALA = Script("sinhala", 0x0D80, 0x0DFF, True, (0x01, 0x02, 0x03), (0x1A, 0x46))

SCRIPTS: Sequence[Script] = (
    DEVANAGARI,
    BENGALI,
    GURMUKHI,
    GUJARATI,
    ORIYA,
    TAMIL,
    TELUGU,
    KANNADA,
    MALAYALAM,
    SINHALA,
)


def script_of(cp: int) -> Script | None:
    """Return the Indic script owning this codepoint, if any."""
    for s in SCRIPTS:
        if s.block_start <= cp <= s.block_end:
            return s
    return None


@lru_cache(maxsize=8)
def _owner_index(names: tuple[str, ...]) -> Dict[int, Script]:
    """Codepoint -> owning Script, built once per script selection."""
    by_name = {s.name: s for s in SCRIPTS}
    idx: Dict[int, Script] = {}
    for name in names:
        s = by_name[name]
        for cp in range(s.block_start, s.block_end + 1):
            idx[cp] = s
    return idx


def segment(text: str, scripts: Sequence[Script] = SCRIPTS) -> List[str]:
    """Split text into aksharas for the given scripts; other characters stand alone.

    Lossless: ``"".join(segment(t)) == unicodedata.normalize("NFC", t)``.
    """
    text = unicodedata.normalize("NFC", text)
    owner_by_cp = _owner_index(tuple(s.name for s in scripts))

    out: List[str] = []
    prev_virama: Script | None = None
    for ch in text:
        cp = ord(ch)
        owner = owner_by_cp.get(cp)

        if out:
            if cp in _ZW_JOINERS:
                out[-1] += ch
                continue
            if owner is not None and owner.is_combining(cp):
                out[-1] += ch
                prev_virama = owner if cp == owner.virama else None
                continue
            # virama + consonant => one conjunct akshara (Devanagari family only)
            if prev_virama is not None and prev_virama is owner and owner.join_conjuncts:
                if owner.is_consonant(cp):
                    out[-1] += ch
                    prev_virama = None
                    continue

        out.append(ch)
        prev_virama = None
    return out


def base_inventory(script: Script, matras: Sequence[str] | None = None) -> List[str]:
    """Closed akshara inventory derived mechanically from the script table.

    Tamil uses the hand-curated inventory in ``swaram_tokenizer`` (it adds the
    Grantha letters and the aytham); this derivation exists for scripts with no
    curated table — ``consonant``, ``consonant+virama``, and every
    ``consonant+matra`` form, which is the closed set an abugida guarantees.
    """
    matras = matras if matras is not None else script.matras()
    inv: List[str] = []
    for c in script.consonants():
        inv.append(c)
        inv.append(c + chr(script.virama))
        for m in matras:
            inv.append(c + m)
    seen, out = set(), []
    for a in inv:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def script_alphabet(script: Script) -> List[str]:
    """Single-codepoint symbols a script uses, so every akshara can be composed."""
    return [chr(cp) for cp in range(script.block_start, script.block_end + 1)]
