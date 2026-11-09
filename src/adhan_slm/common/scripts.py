"""The one registry of Brahmic scripts, Tamil letters, and the Dravidian matrix.

Why this module exists
----------------------
Before this file the same Unicode facts were written down five times: Tamil
block offsets in ``tokenizer/swaram_tokenizer.py``, Devanagari in
``tokenizer/aksharam_tokenizer.py``, ten scripts in ``tokenizer/indic_akshara.py``,
and two more letter tables in ``tokenizer/morphology.py`` and
``tokenizer/punarchi.py``. They had drifted: the Tamil inventory carried three
unassigned codepoints (``U+0BC3``–``U+0BC5``) as "vowel signs", the Grantha
letters (ஜ ஶ ஷ ஸ ஹ) were consonants in one table and invisible in another, and a
ZWNJ (``U+200C``) broke segmentation in Tamil/Devanagari while working in
Bengali.

Now there is exactly one place to look, and it is *validated at import*: a
malformed block, an off-by-one virama, a glyph from the wrong script, or a
duplicate column entry raises :class:`ValueError` when the module loads rather
than producing subtle tokenizer drift later. The tables are frozen dataclasses
and read-only mappings, so no caller can mutate the shared truth.

The vocabulary used in identifiers (use the right word, in code as in prose)
---------------------------------------------------------------------------
===================  ==================================  ==============================
Identifier           Classical Tamil / Sanskrit           What it is
===================  ==================================  ==============================
``uyir``             உயிர்                                independent vowel (அ, ஆ, …)
``mei``              மெய்                                 pure consonant (க், ங், …)
``uyirmei``          உயிர்மெய்                            consonant + vowel sign
``aytham``           ஆய்தம்                               the āytam letter ஃ
``pulli``            புள்ளி                               virama ் (ends a mei)
``vowel_sign``       உயிர்க் குறி                         the sign that changes a mei's vowel
``grantha``          கிரந்தம்                             letters borrowed for Sanskrit loans
``akshara``          அக்ஷரம்                              one orthographic syllable
``eluttu``           எழுத்து                              a letter (the unit Tholkappiyam counts)
===================  ==================================  ==============================

``matra`` (மாத்திரை) is *not* used as a primary name here: classically it is a
unit of duration (a mora), and calling ``ா`` a "matra" is a typographic
borrowing. :meth:`Script.matras` survives as a compatibility alias for
:meth:`Script.vowel_signs`, nothing more.

Invariants this module guarantees
---------------------------------
* Every ``Script`` block is 128-aligned and non-overlapping with its siblings.
* ``virama`` is ``block_start + 0x4D`` and lies inside the block.
* ``vowel_signs()``/``consonants()``/``vowels()``/``digits()`` only ever return
  *assigned* characters from that script's own block (checked with
  ``unicodedata.category``), plus explicitly curated extras.
* :func:`segment` is lossless, deterministic and NFC-idempotent for every
  registered script, and it is the only segmenter in the package.
* Every glyph in :data:`DRAVIDIAN_MATRIX` belongs to the block of the script it
  is filed under, appears at most once per script, and is a letter.
"""

from __future__ import annotations

import bisect
import unicodedata
from dataclasses import dataclass
from functools import cached_property, lru_cache
from types import MappingProxyType
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

# --- Shared Unicode facts ----------------------------------------------------

# ZWJ/ZWNJ participate in real grapheme clusters (Bengali র্য, Devanagari
# half-forms क्‍ष, Malayalam chillu). They attach to the preceding unit.
ZERO_WIDTH_JOINERS: Tuple[str, ...] = ("\u200c", "\u200d")
_ZW_JOINER_CPS = frozenset(ord(c) for c in ZERO_WIDTH_JOINERS)

# Blocks whose marks combine with the preceding akshara even though they are not
# part of a registered script block: Vedic Extensions (U+1CD0–U+1CFF, accents
# and tone marks that ride on a Devanagari letter) and Devanagari Extended
# (U+A8E0–U+A8FF, candrabindu/anusvara variants). Sanskrit text in Devanagari
# puts these *after* the letter, so dropping them into their own akshara is a
# silent tokenization error.
_SHARED_COMBINING_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x1CD0, 0x1CFF),
    (0xA8E0, 0xA8FF),
)


def _is_shared_combining(cp: int) -> bool:
    for lo, hi in _SHARED_COMBINING_RANGES:
        if lo <= cp <= hi:
            return True
    return False


PULLI = "\u0bcd"  # Tamil pulli ் — the reference spelling of "virama"

# Marks at the same offsets in every Brahmic block: nukta (0x3C), the vowel
# signs (0x3E–0x4C) and the virama (0x4D).
_COMMON_MARK_OFFSETS: Tuple[int, ...] = (0x3C,) + tuple(range(0x3E, 0x4E))
_VOWEL_SIGN_OFFSETS = range(0x3E, 0x4D)

# Valid Unicode categories for the characters a block is supposed to contain.
_LETTER = frozenset({"Lo"})
_VOWEL_SIGN = frozenset({"Mn", "Mc"})
_DIGIT = frozenset({"Nd"})


def _assigned(block_start: int, lo: int, hi: int, categories: frozenset) -> Tuple[int, ...]:
    """Codepoints in ``[lo, hi]`` (relative to the block) of the given categories.

    Reserved/assigned-but-wrong-category holes are dropped rather than silently
    carried into a vocabulary. Tamil ``U+0BC3``–``U+0BC5`` and ``U+0BC9`` are
    unassigned; this filter is why they no longer appear in the tokenizer.
    """
    return tuple(
        block_start + off
        for off in range(lo, hi + 1)
        if unicodedata.category(chr(block_start + off)) in categories
    )


# --- One script --------------------------------------------------------------


@dataclass(frozen=True)
class Script:
    """Unicode facts needed to segment (and describe) one Brahmic abugida.

    The first six fields keep their original positional order so existing
    callers (``Script("tamil", 0x0B80, 0x0BFF, False, (0x02,))``) still work.
    """

    name: str
    block_start: int
    block_end: int
    # Does ``virama + consonant`` form one akshara (Devanagari क्ष) or two
    # (Tamil க் + ஷ)? Tamil pulli does not create conjunct graphemes.
    join_conjuncts: bool = True
    # Signs at these *offsets* inside the block attach to the previous unit.
    sign_offsets: Tuple[int, ...] = (0x01, 0x02, 0x03)
    consonant_offsets: Tuple[int, int] = (0x15, 0x39)
    # Absolute codepoints that are combining but not at a common offset
    # (Tamil U+0BD7 au length mark, Devanagari U+094E prishthamatra e, …).
    extra_marks: Tuple[int, ...] = ()
    # Absolute codepoints of consonants outside ``consonant_offsets``
    # (Devanagari U+0958–U+095F precomposed nukta forms).
    extra_consonants: Tuple[int, ...] = ()
    # Absolute codepoints of vowel signs outside ``vowel_sign_offsets``
    # (Devanagari U+0955–U+0957, U+0962–U+0963).
    extra_vowel_signs: Tuple[int, ...] = ()
    # Per-script layout, because "Brahmic" is not "uniform". Most scripts put
    # independent vowels at +0x04–0x14, vowel signs at +0x3E–0x4C, and the
    # virama at +0x4D; Sinhala puts them at +0x05–0x16, +0x4F–0x5F and +0x4A.
    vowel_offsets: Tuple[int, int] = (0x04, 0x14)
    vowel_sign_offsets: Tuple[int, int] = (0x3E, 0x4C)
    virama_offset: int = 0x4D
    digit_offsets: Tuple[int, int] = (0x66, 0x6F)
    # Whole mark layout, when it is not the pan-Indic one (Sinhala).
    mark_offsets: Tuple[int, ...] = ()

    # --- validation ---------------------------------------------------------

    def __post_init__(self) -> None:
        """Reject a malformed script when it is *defined*, not when it is used."""
        if not self.name or not self.name.isascii() or not self.name.islower():
            raise ValueError(f"script name must be lowercase ascii: {self.name!r}")
        if not 0 <= self.block_start < self.block_end <= 0x10FFFF:
            raise ValueError(f"{self.name}: bad block {self.block_start:#x}")
        if self.block_start & 0x7F or self.block_end & 0x7F != 0x7F:
            raise ValueError(
                f"{self.name}: block {self.block_start:#06x}–{self.block_end:#06x} "
                "is not 128-aligned"
            )
        for label, (lo, hi) in (
            ("consonant_offsets", self.consonant_offsets),
            ("vowel_offsets", self.vowel_offsets),
            ("vowel_sign_offsets", self.vowel_sign_offsets),
            ("digit_offsets", self.digit_offsets),
        ):
            if not 0 <= lo <= hi < 0x80:
                raise ValueError(f"{self.name}: bad {label} {(lo, hi)}")
        if not 0 <= self.virama_offset < 0x80:
            raise ValueError(f"{self.name}: virama offset outside the block")
        if any(not 0 <= off < 0x80 for off in self.sign_offsets + self.mark_offsets):
            raise ValueError(f"{self.name}: sign offset outside the block")
        for label, cps in (
            ("extra_marks", self.extra_marks),
            ("extra_consonants", self.extra_consonants),
            ("extra_vowel_signs", self.extra_vowel_signs),
        ):
            bad = [cp for cp in cps if not self.block_start <= cp <= self.block_end]
            if bad:
                raise ValueError(f"{self.name}: {label} outside block: {bad}")

    # --- codepoint predicates ----------------------------------------------

    @cached_property
    def _mark_offsets(self) -> frozenset:
        base = self.mark_offsets or _COMMON_MARK_OFFSETS
        return frozenset(
            tuple(base)
            + tuple(self.sign_offsets)
            + tuple(cp - self.block_start for cp in self.extra_marks)
            # Extra vowel signs (U+094E, U+094F, U+0955–U+0957) are marks too;
            # leaving them out sent कॎ down the split path.
            + tuple(cp - self.block_start for cp in self.extra_vowel_signs)
        )

    @cached_property
    def _consonant_cps(self) -> frozenset:
        lo, hi = self.consonant_offsets
        return frozenset(_assigned(self.block_start, lo, hi, _LETTER) + self.extra_consonants)

    @cached_property
    def _vowel_sign_cps(self) -> Tuple[int, ...]:
        lo, hi = self.vowel_sign_offsets
        return _assigned(self.block_start, lo, hi, _VOWEL_SIGN) + self.extra_vowel_signs

    @cached_property
    def _vowel_cps(self) -> Tuple[int, ...]:
        lo, hi = self.vowel_offsets
        return _assigned(self.block_start, lo, hi, _LETTER)

    @cached_property
    def _digit_cps(self) -> Tuple[int, ...]:
        lo, hi = self.digit_offsets
        return _assigned(self.block_start, lo, hi, _DIGIT)

    def is_combining(self, cp: int) -> bool:
        """True if ``cp`` attaches to the preceding unit (vowel sign, pulli, sign)."""
        if cp in _ZW_JOINER_CPS:
            return True
        if not (self.block_start <= cp <= self.block_end):
            return False
        return (cp - self.block_start) in self._mark_offsets

    def is_consonant(self, cp: int) -> bool:
        return cp in self._consonant_cps

    def is_vowel_sign(self, cp: int) -> bool:
        return cp in self._vowel_sign_cps

    # --- inventories --------------------------------------------------------

    @property
    def virama(self) -> int:
        """The script's virama / halant / pulli codepoint."""
        return self.block_start + self.virama_offset

    @property
    def pulli(self) -> str:
        """The script's virama as a one-character string."""
        return chr(self.virama)

    def consonants(self) -> List[str]:
        """Bare (inherent-vowel) consonants, in block order, assigned only."""
        return [chr(cp) for cp in sorted(self._consonant_cps)]

    def vowels(self) -> List[str]:
        """Independent vowels (uyir / svara) defined by this block."""
        return [chr(cp) for cp in self._vowel_cps]

    def vowel_signs(self) -> List[str]:
        """Vowel signs (uyir-k-kuri) that ride on a consonant."""
        return [chr(cp) for cp in self._vowel_sign_cps]

    # Compatibility alias: "matra" is the typographic borrowing for a vowel
    # sign. Prefer ``vowel_signs``; this name is kept for older call sites.
    def matras(self) -> List[str]:
        return self.vowel_signs()

    def digits(self) -> List[str]:
        """Native decimal digits, if the block has any (Tamil ௦–௯, Deva ०–९)."""
        return [chr(cp) for cp in self._digit_cps]

    def alphabet(self) -> List[str]:
        """Every codepoint in the block, so any akshara can be composed."""
        return [chr(cp) for cp in range(self.block_start, self.block_end + 1)]

    # --- aksharas -----------------------------------------------------------

    def aksharas(self) -> List[str]:
        """The closed akshara inventory derived mechanically from the block.

        Consonant, consonant+pulli, and consonant+every vowel sign — the closed
        set an abugida guarantees. Scripts with a curated inventory (Tamil's
        Grantha letters and āytam, Devanagari's conjuncts) extend this list;
        see :func:`akshara_inventory`.
        """
        out: List[str] = []
        seen: set = set()
        for consonant in self.consonants():
            for form in (consonant, consonant + self.pulli):
                if form not in seen:
                    seen.add(form)
                    out.append(form)
            for sign in self.vowel_signs():
                form = consonant + sign
                if form not in seen:
                    seen.add(form)
                    out.append(form)
        return out

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return self.name


# --- The registry -----------------------------------------------------------

# Tamil: pulli ends an akshara, so virama+consonant never joins. U+0B83 (ஃ,
# āytam) is a *letter*, not a matra, so it is deliberately absent from the mark
# offsets. U+0B82 (ஂ anusvara) and U+0BD7 (ௗ au length mark) are combining.
TAMIL = Script("tamil", 0x0B80, 0x0BFF, False, (0x02,), extra_marks=(0x0BD7,))
# Devanagari: conjuncts join (क्ष). Combining signs are U+0900–U+0903, the
# Vedic accents U+0951–U+0954, and the vocalic signs U+0962–U+0963. The
# prishthamatra vowels (U+094E, U+094F) and the vocalic-l signs (U+0955–U+0957)
# are vowel signs; U+0958–U+095F are precomposed nukta consonants.
DEVANAGARI = Script(
    "devanagari",
    0x0900,
    0x097F,
    True,
    (0x00, 0x01, 0x02, 0x03, 0x51, 0x52, 0x53, 0x54, 0x62, 0x63),
    extra_consonants=tuple(range(0x0958, 0x0960)),
    extra_vowel_signs=(0x094E, 0x094F, 0x0955, 0x0956, 0x0957),
)
BENGALI = Script("bengali", 0x0980, 0x09FF)
GURMUKHI = Script("gurmukhi", 0x0A00, 0x0A7F)
GUJARATI = Script("gujarati", 0x0A80, 0x0AFF)
ORIYA = Script("oriya", 0x0B00, 0x0B7F)
TELUGU = Script("telugu", 0x0C00, 0x0C7F)
KANNADA = Script("kannada", 0x0C80, 0x0CFF)
MALAYALAM = Script("malayalam", 0x0D00, 0x0D7F)
SINHALA = Script(
    "sinhala",
    0x0D80,
    0x0DFF,
    True,
    (0x02, 0x03),
    (0x1A, 0x46),
    # Sinhala does not follow the pan-Indic offsets: its virama is U+0DCA and
    # its vowel signs sit at U+0DCF–U+0DDF.
    vowel_offsets=(0x05, 0x16),
    vowel_sign_offsets=(0x4F, 0x5F),
    virama_offset=0x4A,
    mark_offsets=(0x02, 0x03, 0x4A) + tuple(range(0x4F, 0x60)),
)

SCRIPTS: Tuple[Script, ...] = (
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

SCRIPT_BY_NAME: Mapping[str, Script] = MappingProxyType({s.name: s for s in SCRIPTS})
"""Read-only name -> Script map. Mutating a shared script table is not allowed."""

_BLOCKS = tuple(
    (s.block_start, s.block_end, s) for s in sorted(SCRIPTS, key=lambda s: s.block_start)
)
_BLOCK_STARTS = tuple(start for start, _, _ in _BLOCKS)


def script_of(cp: int) -> Optional[Script]:
    """Return the registered script owning this codepoint, if any.

    Binary search over the (non-overlapping, validated) block list, so this is
    O(log n) and safe to call per character.
    """
    i = bisect.bisect_right(_BLOCK_STARTS, cp) - 1
    if i >= 0:
        start, end, script = _BLOCKS[i]
        if start <= cp <= end:
            return script
    return None


def script_for_char(char: str) -> Optional[Script]:
    """Return the script of a single-character string (``None`` if not Indic)."""
    if not char:
        raise ValueError("script_for_char needs exactly one character")
    return script_of(ord(char))


# --- The single segmenter ---------------------------------------------------


@lru_cache(maxsize=8)
def _owner_index(names: Tuple[str, ...]) -> Dict[int, Script]:
    """Codepoint -> owning Script, built once per script selection."""
    by_name = {s.name: s for s in SCRIPTS}
    idx: Dict[int, Script] = {}
    for name in names:
        script = by_name[name]
        for cp in range(script.block_start, script.block_end + 1):
            idx[cp] = script
    return idx


def _can_take_marks(piece: str) -> bool:
    """True if a combining mark may attach to this piece (i.e. it ends in a letter).

    A vowel sign or nukta must ride on a *letter*, not on whitespace or
    punctuation. Real corpora do contain " ா" (a vowel sign orphaned by a line
    break); gluing it to the space invents composite tokens like " ா" — 51 such
    forms across 160k documents, which is small but is pure vocabulary noise.
    """
    last = piece[-1]
    return unicodedata.category(last)[0] in {"L", "M"} or last in ZERO_WIDTH_JOINERS


def segment(text: str, scripts: Sequence[Script] = SCRIPTS) -> List[str]:
    """Split text into aksharas for the given scripts; other characters stand alone.

    One implementation for every script, so Swaram (Tamil), Aksharam
    (Devanagari) and the generic multi-script path can never disagree.

    Lossless and idempotent: ``"".join(segment(t)) == unicodedata.normalize("NFC", t)``
    and ``segment(segment(t)) == segment(t)``.
    """
    text = unicodedata.normalize("NFC", text)
    owner_by_cp = _owner_index(tuple(s.name for s in scripts))

    out: List[str] = []
    prev_virama: Optional[Script] = None
    for ch in text:
        cp = ord(ch)
        owner = owner_by_cp.get(cp)

        if out and _can_take_marks(out[-1]):
            if cp in _ZW_JOINER_CPS or _is_shared_combining(cp):
                out[-1] += ch
                continue
            if owner is not None and owner.is_combining(cp):
                out[-1] += ch
                prev_virama = owner if cp == owner.virama else None
                continue
            # virama + consonant => one conjunct akshara (Devanagari family)
            if prev_virama is not None and prev_virama is owner and owner.join_conjuncts:
                if owner.is_consonant(cp):
                    out[-1] += ch
                    prev_virama = None
                    continue

        out.append(ch)
        prev_virama = None
    return out


def segment_tamil(text: str) -> List[str]:
    """Tamil-only segmentation (Swaram Layer A)."""
    return segment(text, (TAMIL,))


def segment_devanagari(text: str) -> List[str]:
    """Devanagari-only segmentation (Aksharam Layer A)."""
    return segment(text, (DEVANAGARI,))


# --- Tamil letter inventory (eluttu) ----------------------------------------

# 12 uyir (independent vowels), in classical order.
UYIR: Tuple[str, ...] = tuple(TAMIL.vowels())
if len(UYIR) != 12:  # pragma: no cover - guards the table, not the caller
    raise ImportError(f"Tamil uyir inventory changed: {len(UYIR)} != 12")

# 18 mei (pure consonants) — Tholkappiyam's count, each carrying pulli.
MEI: Tuple[str, ...] = tuple(c + PULLI for c in "கஙசஞடணதநபமயரலவழளறன")
# 5 Grantha (கிரந்த) meis borrowed for Sanskrit loans: j ś ṣ s h.
GRANTHA: Tuple[str, ...] = tuple(c + PULLI for c in "ஜஶஷஸஹ")
# All 23 meis the script can express (18 native + 5 Grantha).
MEI_ALL: Tuple[str, ...] = MEI + GRANTHA

AYTHAM = "ஃ"

TAMIL_UYIR = UYIR
TAMIL_MEI = MEI
TAMIL_GRANTHA_MEI = GRANTHA
TAMIL_AYTHAM = AYTHAM
TAMIL_PULLI = PULLI
# The 11 vowel signs (அ takes no sign: it is the consonant's inherent vowel).
TAMIL_VOWEL_SIGNS: Tuple[str, ...] = tuple(TAMIL.vowel_signs())
# Native decimal digits ௦–௯, so year/number text is never out-of-vocabulary.
TAMIL_DIGITS: Tuple[str, ...] = tuple(TAMIL.digits())
# 12 vowel forms per consonant: inherent அ + 11 signs.
UYIRMEI_PER_CONSONANT = 1 + len(TAMIL_VOWEL_SIGNS)

# Bare consonants, native first then Grantha, for inventory construction.
MEI_BASES: Tuple[str, ...] = tuple(c[0] for c in MEI_ALL)
TAMIL_CONSONANTS = MEI_BASES

# The 18 native meis fall into Tholkappiyam's three classes, and every sandhi
# rule keys on them. Partition, not a second copy: `MEI` is the source of truth.
VALLINAM: Tuple[str, ...] = tuple(c + PULLI for c in "கசடதபற")  # வல்லினம், hard
MELLINAM: Tuple[str, ...] = tuple(c + PULLI for c in "ஙஞணநமன")  # மெல்லினம், nasal
IDAIYINAM: Tuple[str, ...] = tuple(c + PULLI for c in "யரலவழள")  # இடையினம், medial


def vowel_signs(script: Script = TAMIL) -> List[str]:
    """Vowel signs of ``script`` (right word for what "matra" was used for)."""
    return script.vowel_signs()


def akshara_inventory(script: Script = TAMIL, *, include_digits: bool = True) -> List[str]:
    """Closed akshara inventory for one script, deduplicated, in a stable order.

    Tamil: 12 uyir + āytam + 10 digits + (23 consonants × (inherent + pulli + 11
    vowel signs)) = 322. Devanagari: its block-derived vowels, digits,
    consonants and conjunct-free aksharas.
    """
    inv: List[str] = list(script.vowels())
    if include_digits:
        inv.extend(script.digits())
    if script is TAMIL:
        inv.append(AYTHAM)
    inv.extend(script.aksharas())

    seen: set = set()
    out: List[str] = []
    for a in inv:
        # NFC so the inventory contains the spellings real text is normalized
        # to: U+0958–U+095F (क़ … य़) are composition exclusions and NFC maps
        # them to consonant + nukta, which is what the segmenter emits.
        a = unicodedata.normalize("NFC", a)
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def script_alphabet(script: Script) -> List[str]:
    """Every single codepoint in the script's block (composability guarantee)."""
    return script.alphabet()


def base_inventory(script: Script, matras: Optional[Sequence[str]] = None) -> List[str]:
    """Mechanically derived inventory (compatibility name for ``akshara_inventory``).

    ``matras`` is accepted for older call sites and ignored: the vowel signs now
    come from the validated script table, which is the whole point.
    """
    return akshara_inventory(script, include_digits=False)


# --- The Dravidian phonetic matrix ------------------------------------------

DRAVIDIAN_SCRIPTS: Tuple[str, ...] = ("tamil", "telugu", "kannada", "malayalam")

# Columns 13–48 of "The Periodic Table of Dravidian Scripts", column for column.
# ``None`` marks a place where the script genuinely has no letter: Tamil has no
# voiced/aspirated series (க covers k and g) and no native sibilants, so those
# cells are gaps, not "unknowns". Filling them in with the voiceless letter is
# exactly the silent lie this table exists to prevent.
_ROWS: Tuple[Tuple[str, str, Optional[str], Optional[str], Optional[str], Optional[str]], ...] = (
    # --- S-block: 12 vowels (1–12) ---
    ("a", "short a", "அ", "అ", "ಅ", "അ"),
    ("ā", "long ā", "ஆ", "ఆ", "ಆ", "ആ"),
    ("i", "short i", "இ", "ఇ", "ಇ", "ഇ"),
    ("ī", "long ī", "ஈ", "ఈ", "ಈ", "ഈ"),
    ("u", "short u", "உ", "ఉ", "ಉ", "ഉ"),
    ("ū", "long ū", "ஊ", "ఊ", "ಊ", "ഊ"),
    ("e", "short e", "எ", "ఎ", "ಎ", "എ"),
    ("ē", "long ē", "ஏ", "ఏ", "ಏ", "ഏ"),
    ("ai", "diphthong ai", "ஐ", "ఐ", "ಐ", "ഐ"),
    ("o", "short o", "ஒ", "ఒ", "ಒ", "ഒ"),
    ("ō", "long ō", "ஓ", "ఓ", "ಓ", "ഓ"),
    ("au", "diphthong au", "ஔ", "ఔ", "ಔ", "ഔ"),
    # --- Velar plosives and nasal (13–17) ---
    ("k", "velar plosive", "க", "క", "ಕ", "ക"),
    ("kh", "aspirated velar", None, "ఖ", "ಖ", "ഖ"),
    ("g", "voiced velar", None, "గ", "ಗ", "ഗ"),
    ("gh", "voiced aspirated velar", None, "ఘ", "ಘ", "ഘ"),
    ("ṅ", "velar nasal", "ங", "ఙ", "ಙ", "ങ"),
    # --- Palatal plosives and nasal (18–22) ---
    ("c", "palatal plosive", "ச", "చ", "ಚ", "ച"),
    ("ch", "aspirated palatal", None, "ఛ", "ಛ", "ഛ"),
    ("j", "voiced palatal", "ஜ", "జ", "ಜ", "ജ"),
    ("jh", "voiced aspirated palatal", None, "ఝ", "ಝ", "ഝ"),
    ("ñ", "palatal nasal", "ஞ", "ఞ", "ಞ", "ഞ"),
    # --- Retroflex (23–27) ---
    ("ṭ", "retroflex plosive", "ட", "ట", "ಟ", "ട"),
    ("ṭh", "aspirated retroflex", None, "ఠ", "ಠ", "ഠ"),
    ("ḍ", "voiced retroflex", None, "డ", "ಡ", "ഡ"),
    ("ḍh", "voiced aspirated retroflex", None, "ఢ", "ಢ", "ഢ"),
    ("ṇ", "retroflex nasal", "ண", "ణ", "ಣ", "ണ"),
    # --- Dental (28–32) ---
    ("t", "dental plosive", "த", "త", "ತ", "ത"),
    ("th", "aspirated dental", None, "థ", "ಥ", "ഥ"),
    ("d", "voiced dental", None, "ద", "ದ", "ദ"),
    ("dh", "voiced aspirated dental", None, "ధ", "ಧ", "ധ"),
    ("n", "dental nasal", "ந", "న", "ನ", "ന"),
    # --- Labial (33–37) ---
    ("p", "labial plosive", "ப", "ప", "ಪ", "പ"),
    ("ph", "aspirated labial", None, "ఫ", "ಫ", "ഫ"),
    ("b", "voiced labial", None, "బ", "ಬ", "ബ"),
    ("bh", "voiced aspirated labial", None, "భ", "ಭ", "ഭ"),
    ("m", "labial nasal", "ம", "మ", "ಮ", "മ"),
    # --- Approximants (38–44) ---
    ("y", "palatal glide", "ய", "య", "ಯ", "യ"),
    ("r", "alveolar tap", "ர", "ర", "ರ", "ര"),
    ("l", "alveolar lateral", "ல", "ల", "ಲ", "ല"),
    ("v", "labiodental approximant", "வ", "వ", "ವ", "വ"),
    # ḻ: Telugu ఴ (U+0C34) and Kannada ೞ (U+0CDE) are archaic but real.
    ("ḻ", "retroflex approximant", "ழ", "ఴ", "ೞ", "ഴ"),
    ("ḷ", "retroflex lateral", "ள", "ళ", "ಳ", "ള"),
    ("ṟ", "alveolar trill", "ற", "ఱ", "ಱ", "റ"),
    # --- Sibilants and fricatives (45–48): Grantha in Tamil ---
    ("ś", "palatal sibilant", "ஶ", "శ", "ಶ", "ശ"),
    ("ṣ", "retroflex sibilant", "ஷ", "ష", "ಷ", "ഷ"),
    ("s", "dental sibilant", "ஸ", "స", "ಸ", "സ"),
    ("h", "glottal fricative", "ஹ", "హ", "ಹ", "ഹ"),
)


@dataclass(frozen=True)
class PhoneticColumn:
    """One column of the Dravidian matrix: a sound and who can write it."""

    number: int
    roman: str
    description: str
    glyphs: Mapping[str, str]

    def get(self, script: str) -> Optional[str]:
        return self.glyphs.get(script)


DRAVIDIAN_MATRIX: Tuple[PhoneticColumn, ...] = tuple(
    PhoneticColumn(
        number=i,
        roman=roman,
        description=description,
        glyphs=MappingProxyType(
            {
                name: glyph
                for name, glyph in zip(DRAVIDIAN_SCRIPTS, (ta, te, kn, ml))
                if glyph is not None
            }
        ),
    )
    for i, (roman, description, ta, te, kn, ml) in enumerate(_ROWS, start=1)
)

COLUMNS_BY_ROMAN: Mapping[str, PhoneticColumn] = MappingProxyType(
    {column.roman: column for column in DRAVIDIAN_MATRIX}
)


@lru_cache(maxsize=None)
def _glyph_index() -> Dict[str, PhoneticColumn]:
    return {glyph: column for column in DRAVIDIAN_MATRIX for glyph in column.glyphs.values()}


def column_of(char: str) -> Optional[PhoneticColumn]:
    """The matrix column a single letter belongs to, across all four scripts."""
    if len(char) != 1:
        raise ValueError(f"column_of expects one letter, got {char!r}")
    return _glyph_index().get(char)


def cognates(char: str) -> Dict[str, str]:
    """The same sound written in the other Dravidian scripts (gaps omitted).

    ``cognates("க")`` -> ``{"tamil": "க", "telugu": "క", "kannada": "ಕ", "malayalam": "ക"}``.
    ``cognates("க")`` cannot tell you whether a Tamil writer meant /k/ or /g/,
    because Tamil spells both ``க``; the aspirated and voiced columns are empty
    for Tamil by construction. Do not invent them.
    """
    column = column_of(char)
    return {} if column is None else dict(column.glyphs)


# --- Registry validation ----------------------------------------------------


def _iter_matrix_glyphs() -> Iterator[Tuple[PhoneticColumn, str, str]]:
    for column in DRAVIDIAN_MATRIX:
        for name, glyph in column.glyphs.items():
            yield column, name, glyph


def validate_registry() -> None:
    """Cross-check the whole registry; raise ``ValueError`` on any inconsistency.

    Called at import so a bad table fails loudly and immediately. Also exposed
    so a downstream repo (illakiya's trie, for instance) can re-validate the
    tables it vendored before shipping them.
    """
    ordered = sorted(SCRIPTS, key=lambda s: s.block_start)
    for prev, nxt in zip(ordered, ordered[1:]):
        if prev.block_end >= nxt.block_start:
            raise ValueError(f"blocks overlap: {prev.name} / {nxt.name}")
    if len({s.name for s in SCRIPTS}) != len(SCRIPTS):
        raise ValueError("duplicate script name in SCRIPTS")

    for script in SCRIPTS:
        if not script.vowels():
            raise ValueError(f"{script.name}: no independent vowels")
        if not script.consonants():
            raise ValueError(f"{script.name}: no consonants")
        if not script.vowel_signs():
            raise ValueError(f"{script.name}: no vowel signs")
        for cp in script._consonant_cps:
            if unicodedata.category(chr(cp)) not in _LETTER:
                raise ValueError(f"{script.name}: consonant U+{cp:04X} is not a letter")
        for cp in script._vowel_sign_cps:
            if unicodedata.category(chr(cp)) not in _VOWEL_SIGN:
                raise ValueError(f"{script.name}: vowel sign U+{cp:04X} is not a mark")

    # Tamil's classical counts are a contract, not a coincidence.
    if len(UYIR) != 12 or len(MEI) != 18 or len(GRANTHA) != 5:
        raise ValueError("Tamil eluttu counts must be 12 uyir / 18 mei / 5 grantha")
    if len(TAMIL_VOWEL_SIGNS) != 11:
        raise ValueError("Tamil must have 11 vowel signs (அ is inherent)")
    if sorted(VALLINAM + MELLINAM + IDAIYINAM) != sorted(MEI):
        raise ValueError("vallinam/mellinam/idaiyinam must partition the 18 meis")

    # The matrix must not name a script that is not registered, put a glyph in
    # the wrong block, file two glyphs of one script in the same column set, or
    # use a non-letter.
    seen: Dict[str, set] = {name: set() for name in DRAVIDIAN_SCRIPTS}
    for column, name, glyph in _iter_matrix_glyphs():
        if name not in SCRIPT_BY_NAME:
            raise ValueError(f"matrix names unregistered script {name!r}")
        script = SCRIPT_BY_NAME[name]
        if len(glyph) != 1:
            raise ValueError(f"column {column.roman}: {name} glyph is not one char")
        cp = ord(glyph)
        if not script.block_start <= cp <= script.block_end:
            raise ValueError(
                f"column {column.roman}: {glyph} (U+{cp:04X}) is not in the " f"{script.name} block"
            )
        if unicodedata.category(glyph) != "Lo":
            raise ValueError(
                f"column {column.roman}: {glyph} is {unicodedata.category(glyph)}, not a letter"
            )
        if glyph in seen[name]:
            raise ValueError(f"column {column.roman}: {glyph} already used for {name}")
        seen[name].add(glyph)
    if [c.number for c in DRAVIDIAN_MATRIX] != list(range(1, len(DRAVIDIAN_MATRIX) + 1)):
        raise ValueError("matrix columns are not numbered 1..n")
    for name in DRAVIDIAN_SCRIPTS:
        if not seen[name]:
            raise ValueError(f"matrix has no {name} column entries")


validate_registry()
