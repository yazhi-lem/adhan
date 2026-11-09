"""Tamil morphology for the tokenizer: letter classes, sandhi (புணர்ச்சி), suffixes.

Why the tokenizer needs a morphology module
-------------------------------------------
An akshara-native tokenizer is only as good as its morpheme boundaries. Layer A
splits text into aksharas — linguistically atomic, but morphology-blind. Layer B
(BPE merges) is what turns aksharas into *words and morphemes*: ``ப + டி + த் +
து`` becomes ``படித்து``. Whether those merges land on real morpheme boundaries
is exactly what the Phase 4 "morphology probe" measures, and it is the difference
between a tokenizer that learns Tamil's agglutinative endings and one that
shatters them.

This module supplies the two rule systems that define a Tamil morpheme boundary,
and does it with no third-party dependency (the tokenizer cores stay pure-python;
``open_tamil_bridge`` remains the optional cross-check oracle in ``eval/``):

1. **Sandhi (புணர்ச்சி)** — how a word boundary *disappears* when two units join.
   Tholkappiyam's Sollatikāram → Puṇariyal classifies this by the final letter of
   the first unit (uyir / mei / uyirmei) and names the resulting change with one
   of five vikārams: தோன்றல் (appearance), மிகுதல் (doubling), தொகுதல்
   (coalescence), குன்றல் (shortening), விகாரம் (mutation).
2. **Inflectional suffixes** — the closed set of case/number/tense endings Tamil
   glues onto a stem (கள், இல், உக்கு, ஆல், உடைய …). A suffix boundary is a
   boundary the tokenizer should be able to stop at.

Both are *descriptive*, not prescriptive: Tamil permits இயல்பு (no change), and
real corpora contain dialect and loanword patterns the classical rules do not
cover. Callers should treat these functions as signals for scoring and
alignment, never as a rewrite pass over training text — the corpus must keep the
surface form the language actually writes.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence, Set, Tuple

from adhan_slm.common.scripts import (
    MEI_BASES,
    PULLI,
    TAMIL_AYTHAM,
    TAMIL_GRANTHA_MEI,
    TAMIL_MEI,
    TAMIL_UYIR,
    TAMIL_VOWEL_SIGNS,
)

# --- 1. Letter classes (Eḻuttatikāram: the inventory before any joining rule) --
#
# Tholkappiyam fixes the alphabet first: 12 vowels (uyir), 18 consonants (mei),
# 216 consonant+vowel forms (uyirmei), and āytam. Every sandhi rule below is
# keyed on which of those four classes a word *ends* with.

# The letter inventory lives in ``adhan_slm.common.scripts`` so that the
# tokenizer, this sandhi layer and the puṇarcci engine share one validated table:
#   UYIR / MEI / GRANTHA / AYTHAM, and the 11 vowel signs (அ is inherent).
UYIR = TAMIL_UYIR  # 12 உயிர்
MEI = TAMIL_MEI  # 18 மெய் (with pulli) — Tholkappiyam's count
GRANTHA = TAMIL_GRANTHA_MEI  # 5 கிரந்த meis: ஜ் ஶ் ஷ் ஸ் ஹ்
MEI_ALL = MEI + GRANTHA  # all 23 meis the script can write
AYTHAM = TAMIL_AYTHAM  # ஃ

# Vowel signs (உயிர்க் குறிகள்) that ride on a consonant to make an uyirmei.
# "Matra" is kept as the historical name for this tuple; strictly a matra is a
# unit of duration, and the sign is an uyir-k-kuri.
MATRAS = TAMIL_VOWEL_SIGNS  # ா ி ீ ு ூ ெ ே ை ொ ோ ௌ (11; அ is inherent)
_PULLI = PULLI  # ்  (virama: turns a consonant into a bare mei)
_FRONT_MATRAS = ("ி", "ீ", "ை")  # front vowels: these select the ய் glide


class LetterClass(str, Enum):
    """The four classes a Tamil word can end with (plus anything else)."""

    UYIR = "uyir"
    MEI = "mei"
    UYIRMEI = "uyirmei"
    AYTHAM = "aytham"
    OTHER = "other"


def final_akshara(word: str) -> str:
    """The last akshara of ``word`` — i.e. consonant + (matra | virama), if any.

    Tamil words end in an akshara, never in a bare codepoint: ``திரு`` ends in
    ``ரு`` (not ``ு``), ``மரம்`` ends in ``ம்`` (not ``்``). Getting this right is
    what makes the sandhi rules below fire at all.
    """
    if not word:
        return ""
    i = len(word) - 1
    while i > 0 and (word[i] in MATRAS or word[i] == _PULLI):
        i -= 1
    return word[i:]


def letter_class(char: str) -> LetterClass:
    """Classify a single akshara (not a codepoint) as uyir/mei/uyirmei/āytam."""
    if not char:
        return LetterClass.OTHER
    if char == AYTHAM:
        return LetterClass.AYTHAM
    if char in UYIR:
        return LetterClass.UYIR
    if char in MEI_ALL or char in MEI_BASES:
        return LetterClass.MEI
    if char[0] in MEI_BASES and char.endswith(_PULLI):
        # A consonant cluster written with an explicit virama is still a mei.
        return LetterClass.MEI
    if char[0] in MEI_BASES and len(char) >= 2:
        # Consonant + vowel sign (or two signs, e.g. ொ = ெ + ா).
        return LetterClass.UYIRMEI
    return LetterClass.OTHER


class Vikaram(str, Enum):
    """The five surface operations Tholkappiyam/Naṉṉūl allow when units join."""

    IYALBU = "iyalbu"  # இயல்பு — nothing changes, the units simply concatenate
    THONRAL = "thonral"  # தோன்றல் — a letter appears (the உடம்படுமெய் glide ய்/வ்)
    MIGUTHAL = "miguthal"  # மிகுதல் — a consonant doubles
    THOKUTHAL = "thokuthal"  # தொகுதல் — vowels coalesce into one
    KUNRAL = "kunral"  # குன்றல் — a long vowel shortens
    VIKARAM = "vikaram"  # விகாரம் — a vowel changes quality


@dataclass(frozen=True)
class SandhiResult:
    """Outcome of joining two units, with the rule that produced it."""

    surface: str
    vikaram: Vikaram
    rule: str
    inserted: str = ""

    def __str__(self) -> str:  # pragma: no cover - convenience for logs
        return f"{self.surface} [{self.vikaram.value}: {self.rule}]"


# --- 2. Sandhi (புணர்ச்சி) ----------------------------------------------------
#
# The rules below are the uyirīṟṟu / mey-yīṟṟu / uyirmey-yīṟṟu sections of
# Puṇariyal, reduced to the cases that actually occur in modern written Tamil.
# Order matters: the first matching rule wins, mirroring the classical ordering
# (specific final-letter rules before the general "no change" fallback).

_VALLINAM = ("க்", "ச்", "ட்", "த்", "ப்", "ற்")
_DOUBLING_FINALS = ("ல்", "ள்", "ண்", "ன்")  # double before a vowel


def _insert_glide(after: LetterClass, final_char: str) -> str:
    """Choose the உடம்படுமெய் glide: front vowels take ய், everything else வ்.

    This is the single most productive sandhi rule in Tamil:
    பசி + இல்லை → பசியில்லை (ய்), திரு + அடி → திருவடி (வ்).
    """
    if after is not LetterClass.UYIR:
        return ""
    if final_char and final_char[-1] in _FRONT_MATRAS:
        return "ய்"
    if final_char in ("இ", "ஈ", "ஐ"):
        return "ய்"
    return "வ்"


# Independent vowel -> the matra it becomes when it rides on a consonant.
# அ keeps the consonant's inherent vowel, so its matra is the empty string. This
# table is why "திரு" + "அடி" surfaces as திருவடி and not திருவ்அடி: the glide
# consonant takes the following vowel onto itself, exactly as Tamil writes it.
VOWEL_TO_MATRA = {
    "அ": "",
    "ஆ": "ா",
    "இ": "ி",
    "ஈ": "ீ",
    "உ": "ு",
    "ஊ": "ூ",
    "எ": "ெ",
    "ஏ": "ே",
    "ஐ": "ை",
    "ஒ": "ொ",
    "ஓ": "ோ",
    "ஔ": "ௌ",
}

# Matra -> the independent vowel it stands for (the inverse of VOWEL_TO_MATRA;
# "" is the consonant's inherent அ). Punarchi reversal needs this direction: a
# written கல்லடி hides கல் + அடி, so the second ல must come back out as the free
# vowel அ.
MATRA_TO_VOWEL = {matra: vowel for vowel, matra in VOWEL_TO_MATRA.items()}
# Backwards-compatible private alias (older call sites used the underscore name).
_VOWEL_TO_MATRA = VOWEL_TO_MATRA


def attach_consonant(consonant: str, word: str) -> str:
    """Attach a bare consonant to ``word``, absorbing its initial vowel.

    ``attach_consonant("வ", "அடி")`` -> ``"வடி"``; ``attach_consonant("ய", "இல்லை")``
    -> ``"யில்லை"``. Inserting the glide or a doubled consonant without this
    step would produce the ungrammatical ``வ்அடி`` (a virama'd consonant followed
    by a free vowel).

    Public because the extended puṇarcci engine in
    :mod:`adhan_slm.tokenizer.punarchi` and the suffix analysis in this module
    (:func:`sandhi_splits`, :func:`split_suffix`) share it: one implementation
    keeps the forward and the reversed rule tables consistent.
    """
    if word and word[0] in VOWEL_TO_MATRA:
        return consonant + VOWEL_TO_MATRA[word[0]] + word[1:]
    return consonant + word


# Backwards-compatible private alias.
_attach = attach_consonant


def explain_join(left: str, right: str) -> SandhiResult:
    """Join two units and report the sandhi rule that applied.

    ``left`` is a word or morpheme, ``right`` the one that follows it. The
    returned surface form follows the classical rules; ``Vikaram.IYALBU`` means
    nothing changed.
    """
    left = unicodedata.normalize("NFC", left)
    right = unicodedata.normalize("NFC", right)
    if not left or not right:
        return SandhiResult(left + right, Vikaram.IYALBU, "empty input")

    last = final_akshara(left)
    final_class = letter_class(last)
    next_first = right[0]
    next_class = letter_class(next_first)

    # (a) Vowel-final + vowel-initial → insert the glide (தோன்றல்).
    #     பசி + இல்லை → பசியில்லை; பலா + அடி → பலாவடி
    if final_class is LetterClass.UYIR and next_class is LetterClass.UYIR:
        glide = _insert_glide(next_class, last)  # "ய்" or "வ்"
        bare = glide[:-1]  # drop the virama: the glide carries the next vowel
        return SandhiResult(
            left + _attach(bare, right),
            Vikaram.THONRAL,
            "uyir + uyir: உடம்படுமெய்",
            bare,
        )

    # (b) Uyirmei-final + vowel-initial → same glide, carrier vowel kept.
    #     திரு + அடி → திருவடி; கை + இல்லை → கையில்லை
    #     ("uyir-mei sandhi": the uyirmei's own vowel stays, the glide bridges.)
    if final_class is LetterClass.UYIRMEI and next_class is LetterClass.UYIR:
        glide = _insert_glide(next_class, last)
        bare = glide[:-1]
        return SandhiResult(
            left + _attach(bare, right),
            Vikaram.THONRAL,
            "uyirmei + uyir: உடம்படுமெய்",
            bare,
        )

    # From here on the final akshara is replaced, so work on the stem before it.
    stem = left[: -len(last)]

    # (c) ம் + vowel → the ம் becomes வ and takes the vowel (மரம் + அடி → மரவடி).
    if last == "ம்" and next_class is LetterClass.UYIR:
        return SandhiResult(stem + _attach("வ", right), Vikaram.VIKARAM, "ம் + uyir: ம் → வ")

    # (d) ம் + வல்லினம் → the hard consonant doubles (மரம் + தோப்பு → மரத்தோப்பு):
    #     the ம் goes away and a virama'd copy of the next consonant takes its place.
    if last == "ம்" and (next_first + _PULLI) in _VALLINAM:
        return SandhiResult(
            stem + next_first + _PULLI + right,
            Vikaram.MIGUTHAL,
            "ம் + vallinam: மிகுதல்",
        )

    # (e) ல்/ள்/ண்/ன் + vowel → the final consonant doubles (கல் + அடி → கல்லடி).
    if last in _DOUBLING_FINALS and next_class is LetterClass.UYIR:
        return SandhiResult(
            left + _attach(last[0], right), Vikaram.MIGUTHAL, f"{last} + uyir: மிகுதல்"
        )

    # (f) ய் + vowel → ய் doubles (நெய் + அப்பம் → நெய்யப்பம்).
    if last == "ய்" and next_class is LetterClass.UYIR:
        return SandhiResult(left + _attach("ய", right), Vikaram.MIGUTHAL, "ய் + uyir: மிகுதல்")

    # (g) Other mei finals + வல்லினம் → that consonant doubles too, which is the
    #     general வல்லினம் மிகுதல் rule (நீர் + துளி → நீர்த்துளி).
    if final_class is LetterClass.MEI and (next_first + _PULLI) in _VALLINAM:
        return SandhiResult(
            left + next_first + _PULLI + right,
            Vikaram.MIGUTHAL,
            f"{last} + vallinam: மிகுதல்",
        )

    # (h) ர் / ழ் / default mei final + vowel → இயல்பு, but orthographically the
    #     virama drops and the consonant takes the vowel (நீர் + அளவு → நீரளவு,
    #     தமிழ் + அறிஞர் → தமிழறிஞர்). No letter is added or changed.
    if final_class is LetterClass.MEI and next_class is LetterClass.UYIR:
        return SandhiResult(stem + _attach(last[0], right), Vikaram.IYALBU, "mei + uyir: இயல்பு")

    # (h) Everything else — consonant meets consonant, āytam, punctuation — is
    #     a plain concatenation. Classical Tamil does allow further rules
    #     (தொகுதல், குன்றல்); they are compounding-internal and out of scope here.
    return SandhiResult(left + right, Vikaram.IYALBU, "default: join as-is")


def join(left: str, right: str) -> str:
    """Convenience wrapper: the surface form of ``left`` + ``right``."""
    return explain_join(left, right).surface


def sandhi_splits(word: str, candidates: Sequence[str] | None = None) -> List[Tuple[str, str]]:
    """Candidate ``(left, right)`` splits of a written word that *explain* it.

    Used for alignment/diagnostics rather than production segmentation: for each
    possible cut, check whether re-joining the halves reproduces the word. That
    turns the rule table into a cheap "was a sandhi applied here?" test.
    """
    word = unicodedata.normalize("NFC", word)
    out: List[Tuple[str, str]] = []
    for cut in range(1, len(word)):
        left, right = word[:cut], word[cut:]
        if candidates is not None and (left, right) not in set(candidates):
            continue
        if join(left, right) == word:
            out.append((left, right))
    return out


# --- 3. Inflectional suffixes -------------------------------------------------
#
# Tamil is agglutinative: a stem accumulates case (ஐ, ஆல், உக்கு, இல், இன்),
# number (கள்), and clitics (உம், ஏ). The list below covers the productive
# endings; longer forms are matched first so "களுக்கு" is not read as "கள்"+"உக்கு"
# at the wrong offset.

SUFFIXES: Tuple[str, ...] = tuple(
    sorted(
        {
            # plural / number — கள் and its case-inflected forms
            "களிலிருந்து",
            "களுக்கு",
            "களில்",
            "களின்",
            "களை",
            "களால்",
            "கள்",
            # case endings (vibhakti) in their *surface* allomorphs: Tamil picks
            # the allomorph by the stem's final letter, so the table stores what
            # is actually written (வீட்டில், மரத்தில், தமிழுக்கு …), not the
            # citation form (இல், உக்கு).
            "ிலிருந்து",
            "இலிருந்து",
            "இருந்து",
            "க்காக",
            "க்கு",
            "உக்கு",
            "த்தில்",
            "இல்",
            "ில்",
            "இன்",
            "ின்",
            "ஆல்",
            "ால்",
            "ை",
            "ஓடு",
            "உடன்",
            "உடைய",
            "ஐ",
            # common clitics / converbs / verbal participles
            "கொண்டு",
            "ஆவது",
            "ஆக",
            "உம்",
            "ஏ",
        },
        key=len,
        reverse=True,  # longest first: greedy match gives the right boundary
    )
)


def split_suffix(word: str) -> Tuple[str, str]:
    """Greedy stem/suffix split; returns ``(word, "")`` when no suffix matches."""
    word = unicodedata.normalize("NFC", word)
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix):
            return word[: -len(suffix)], suffix
    return word, ""


def boundary_crossing_merges(
    merges: Sequence[Tuple[str, str]], suffixes: Sequence[str] = SUFFIXES
) -> Set[int]:
    """Indices of merges that cut *through* a known suffix.

    A merge ``(a, b)`` produces ``a + b``; if that string ends with a suffix and
    the join point falls strictly inside it, Layer B has learned to split
    morphology apart rather than span it (``மரங்`` + ``கள்`` is a good boundary;
    ``மரங்க`` + ``ள்`` is not). Those merges are what make a tokenizer efficient
    and morphologically blind at the same time.
    """
    bad: Set[int] = set()
    for i, (a, b) in enumerate(merges):
        joined = a + b
        for suffix in suffixes:
            idx = joined.rfind(suffix)
            # ``idx < len(a) < idx + len(suffix)`` means the join point is inside
            if idx != -1 and idx < len(a) < idx + len(suffix):
                bad.add(i)
                break
    return bad


def prune_boundary_crossing_merges(
    merges: Sequence[Tuple[str, str]], suffixes: Sequence[str] = SUFFIXES
) -> List[Tuple[str, str]]:
    """Drop the merges identified by :func:`boundary_crossing_merges`."""
    bad = boundary_crossing_merges(merges, suffixes)
    return [m for i, m in enumerate(merges) if i not in bad]


def morpheme_boundary_chars(word: str) -> Set[int]:
    """Character offsets in ``word`` where a morpheme starts (stem|suffix)."""
    stem, suffix = split_suffix(word)
    return {len(stem)} if suffix else set()


# --- 4. Alignment helpers used by the tokenizer and the morphology probe ------


@dataclass(frozen=True)
class TokenSpan:
    """A tokenizer piece plus the akshara range it covers in the source text.

    Carrying spans is what lets morphology line up with tokenization: the
    tokenizer says "these aksharas became one token", the morphology rules say
    "these aksharas are where the morpheme boundary is", and agreement between
    them is measurable instead of assumed.
    """

    piece: str
    akshara_start: int
    akshara_end: int


@dataclass
class BoundaryReport:
    """How a tokenizer's piece boundaries relate to stem/suffix boundaries.

    Three outcomes are distinguished, because they mean different things:

    * **exact** — a token boundary lands exactly on the stem|suffix break
      (``மரங்`` + ``கள்``). Ideal for morphology-aware modelling.
    * **whole-word** — the word is emitted as a single token. No boundary is
      cut, so the morphemes are preserved *inside* the token; high compression
      legitimately produces this, and penalising it (as the old single-number
      metric did) conflates "compressed" with "morphologically blind".
    * **cut** — some internal token boundary falls strictly inside the stem or
      inside the suffix (``மரங்க`` + ``ள்``). This is the actual defect: the
      tokenizer has learned to split a morpheme apart.
    """

    n_words: int = 0
    n_with_suffix: int = 0
    n_agree: int = 0
    n_whole_word: int = 0
    n_cut: int = 0
    examples: List[str] = None  # type: ignore[assignment]
    cut_examples: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.examples is None:
            self.examples = []
        if self.cut_examples is None:
            self.cut_examples = []

    @property
    def agreement_rate(self) -> float:
        """Exact-boundary rate (kept for backward compatibility)."""
        return self.n_agree / self.n_with_suffix if self.n_with_suffix else float("nan")

    @property
    def exact_rate(self) -> float:
        return self.agreement_rate

    @property
    def whole_word_rate(self) -> float:
        return self.n_whole_word / self.n_with_suffix if self.n_with_suffix else float("nan")

    @property
    def cut_rate(self) -> float:
        """Share of suffixed words where a token boundary cuts a morpheme."""
        return self.n_cut / self.n_with_suffix if self.n_with_suffix else float("nan")

    @property
    def integrity_rate(self) -> float:
        """Share of suffixed words with *no* morpheme cut (exact or whole-word)."""
        if not self.n_with_suffix:
            return float("nan")
        return (self.n_agree + self.n_whole_word) / self.n_with_suffix


def boundary_agreement(tokenizer, words: Sequence[str], max_examples: int = 20) -> BoundaryReport:
    """Classify suffixed words by how token boundaries meet morpheme boundaries.

    ``tokenizer`` only needs ``.tokenize(text) -> list[str]`` (both SwaramTokenizer
    and AksharaBPE satisfy this). See :class:`BoundaryReport` for the three
    outcomes. ``agreement_rate`` remains the exact-boundary rate so existing
    callers keep their meaning; use ``integrity_rate`` / ``cut_rate`` to judge
    whether the tokenizer actually damages morphology.
    """
    report = BoundaryReport()
    for word in words:
        stem, suffix = split_suffix(word)
        if not suffix:
            continue
        report.n_words += 1
        report.n_with_suffix += 1
        boundary = len(stem)

        pieces = tokenizer.tokenize(word)
        offset = 0
        boundaries = set()
        for piece in pieces:
            offset += len(piece)
            boundaries.add(offset)

        if boundary in boundaries:
            report.n_agree += 1
        elif len(pieces) == 1:
            # Whole word preserved as one token: morphemes are intact inside it.
            report.n_whole_word += 1
            if len(report.examples) < max_examples:
                report.examples.append(word)
        else:
            report.n_cut += 1
            if len(report.cut_examples) < max_examples:
                report.cut_examples.append(word)
    return report


def spans_from_pieces(pieces: Sequence[str], aksharas: Sequence[str]) -> List[TokenSpan]:
    """Attach akshara ranges to token pieces, left to right.

    Pieces are concatenations of whole aksharas (Layer A guarantees that), so the
    walk is exact: consume as many aksharas as each piece's text spans.
    """
    spans: List[TokenSpan] = []
    idx = 0
    for piece in pieces:
        start = idx
        consumed = ""
        while idx < len(aksharas) and len(consumed) < len(piece):
            consumed += aksharas[idx]
            idx += 1
        spans.append(TokenSpan(piece=piece, akshara_start=start, akshara_end=idx))
    return spans


def sandhi_points(text: str, segmenter) -> List[int]:
    """Akshara indices in ``text`` that carry a visible sandhi signature.

    A written word is the *result* of sandhi, so the pre-join forms cannot be
    recovered by splitting it — that needs a lexicon. What *can* be detected
    cheaply are the two footprints the rules leave on the surface:

    * a **doubled consonant** (கல் + அடி → கல்லடி, மரம் + தோப்பு → மரத்தோப்பு),
      where the morpheme boundary sits between the two identical consonants;
    * a **glide** ய/வ sitting between vowels (திரு + அடி → திருவடி,
      பசி + இல்லை → பசியில்லை), where the boundary sits just before the glide.

    ``segmenter`` is the akshara splitter (``segment_aksharas`` or
    ``indic_akshara.segment``). Returned indices are boundaries in the akshara
    list, not character offsets.
    """
    aksharas = segmenter(text)
    points: List[int] = []
    for i, ak in enumerate(aksharas):
        if i == 0:
            continue
        prev = aksharas[i - 1]
        # doubled consonant: same base letter twice in a row
        if ak[:1] == prev[:1] and letter_class(ak) in {
            LetterClass.MEI,
            LetterClass.UYIRMEI,
        }:
            points.append(i)
            continue
        # glide between vowels: ...V + glide + V...
        if letter_class(prev) in {LetterClass.UYIR, LetterClass.UYIRMEI}:
            if ak in ("ய", "வ", "யி", "யா", "வி", "வா") or ak[0] in ("ய", "வ"):
                nxt = aksharas[i + 1] if i + 1 < len(aksharas) else ""
                if nxt and letter_class(nxt) in {LetterClass.UYIR, LetterClass.UYIRMEI}:
                    points.append(i)
    return points
