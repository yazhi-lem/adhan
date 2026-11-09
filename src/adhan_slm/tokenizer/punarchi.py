"""புணர்ச்சி இயந்திரம் — the classical Tamil sandhi (puṇarcci) engine, both ways.

Why a *reversal* engine
-----------------------
Tamil is agglutinative and its orthography records the *result* of joining, not
the units that were joined. ``மரத்தோப்பு`` is ``மரம்`` + ``தோப்பு`` with the ம்
gone and த் doubled; ``கல்லடி`` is ``கல்`` + ``அடி`` with ல் doubled. A tokenizer
that only knows aksharas (Layer A) or corpus-frequency BPE merges (Layer B) cannot
tell where those units meet — it has to be *taught the grammar*, and the grammar
that governs the joins is Tholkappiyam's Puṇariyal (புணரியல்) and Naṉṉūl.

This module is the second, rule-based layer the blueprint calls for:

* **Forward** — :func:`join_units` takes two units and returns the surface form
  *plus the rule that produced it*. Useful for generating SFT data and for
  verifying a candidate analysis (``join(lemma, ending) == word``).
* **Reverse** — :func:`reverse_thondral`, :func:`reverse_miguthal`,
  :func:`reverse_thirithal`, :func:`reverse_keduthal` and :func:`reverse_all`
  take a written word and return the ``(left, right)`` cuts that *explain* it,
  each tagged with a rule id and a confidence. This is what
  :func:`adhan_slm.tokenizer.morphology.sandhi_splits` and the morphology-aware
  BPE in :mod:`adhan_slm.tokenizer.swaram_tokenizer` consume.

The classical categories
------------------------
=========================  ==========================================  ==========================
Classical (நன்னூல்)         Meaning                                     Pipeline name
=========================  ==========================================  ==========================
பகுபதம் / பகாபதம்            divisible word / indivisible root          word / lemma
பகுதி                        root or stem                                stem
விகுதி                       person-number-gender ending (PNG)           ``png``
இடைநிலை                     tense / aspect marker                       ``tense``
சாரியை                      euphonic link morpheme (empty morpheme)     ``sariyai``
வேற்றுமை உருபு               case marker                                 ``case``
இயல்பு புணர்ச்சி             words join unchanged                        ``IYALBU``
விகார புணர்ச்சி              words change on joining                     ``VIKARAM``
தோன்றல்                      a letter appears (உடம்படுமெய் ய்/வ்)          ``THONRAL``
திரிதல்                      a letter is substituted (ல் → ற் …)          ``THIRITHAL``
கெடுதல்                      a letter disappears (ம் + வ் → வ்)           ``KEDUTHAL``
மிகுதல்                      a consonant doubles                         ``MIGUTHAL``
=========================  ==========================================  ==========================

Design notes
------------
* No third-party dependency. Like ``morphology`` this stays pure python so it
  runs in the tokenizer's minimal environment and in shard prep.
* Forward and reverse share one table of :class:`PunarchiRule`, so a rule that
  can produce ``கல்லடி`` can always take it apart again. That symmetry is tested
  in :mod:`adhan_slm.tokenizer.punarchi_tests` (round-trip over every canonical
  example, plus the ambiguous cases).
* Reversal is deliberately *ambiguous*: without a lexicon a written ய/வ between
  two vowels may be an inserted glide or a real consonant (``பசியில்லை`` vs
  ``கயிறு``). We return every structural candidate with a confidence and let a
  lexicon or the model's context resolve it. Never silently pick one.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from adhan_slm.common.scripts import IDAIYINAM as _IDAIYINAM
from adhan_slm.common.scripts import MELLINAM as _MELLINAM
from adhan_slm.common.scripts import VALLINAM as _VALLINAM
from adhan_slm.tokenizer.morphology import (
    MATRA_TO_VOWEL,
    VOWEL_TO_MATRA,
    LetterClass,
    Vikaram,
    attach_consonant,
    final_akshara,
    letter_class,
)

# --- 1. Rule inventory --------------------------------------------------------

# The three consonantal classes (வல்லினம் / மெல்லினம் / இடையினம்) come from the
# shared registry, where they are a partition of the 18 native meis — not a
# second, hand-typed copy that can drift.
VALLINAM = _VALLINAM
MELLINAM = _MELLINAM
IDAIYINAM = _IDAIYINAM
# A word ending in one of these doubles before a vowel-initial unit.
_DOUBLING_FINALS = ("ல்", "ள்", "ண்", "ன்", "ய்")
# ல்/ள்/ண்/ன் mutate before வல்லினம் (நன்னூல்: வலித்தல்).
_THIRITHAL_BEFORE_VALLINAM: Dict[str, Tuple[str, ...]] = {
    "ல்": ("ற்",),
    "ள்": ("ட்",),
    "ண்": ("ட்",),
    "ன்": ("ற்",),
}
# Reverse map: written ற்/ட் may have been ல்/ள்/ண்/ன் before a hard consonant.
_REVERSE_THIRITHAL: Dict[str, Tuple[str, ...]] = {
    "ற்": ("ல்", "ன்"),
    "ட்": ("ள்", "ண்"),
}
# Front vowels take the ய் glide; every other vowel takes வ்.
_FRONT_VOWELS = ("இ", "ஈ", "ஐ", "ி", "ீ", "ை")


@dataclass(frozen=True)
class PunarchiRule:
    """One named sandhi rule, in classical and pipeline terms."""

    id: str
    vikaram: Vikaram
    kind: str  # thonral | thirithal | keduthal | miguthal | iyalbu
    name_ta: str
    name_en: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "vikaram": self.vikaram.value,
            "kind": self.kind,
            "name_ta": self.name_ta,
            "name_en": self.name_en,
        }


_THONRAL_GLIDE = PunarchiRule(
    "uyir_uyir_glide",
    Vikaram.THONRAL,
    "thonral",
    "உடம்படுமெய்",
    "vowel + vowel: insert the ய்/வ் glide",
)
_UYIRMEI_GLIDE = PunarchiRule(
    "uyirmei_uyir_glide",
    Vikaram.THONRAL,
    "thonral",
    "உடம்படுமெய் (உயிர்மெய்)",
    "uyirmei + vowel: glide bridges, carrier vowel kept",
)
_U_FINAL_DOUBLING = PunarchiRule(
    "u_final_miguthal",
    Vikaram.KUNRAL,
    "kunral",
    "உ ஈற்று மிகுதல்",
    "உ-final stem + vowel: the உ drops and the consonant doubles (வீடு+இல்)",
)
_M_TO_V = PunarchiRule(
    "m_uyir_to_v",
    Vikaram.VIKARAM,
    "thirithal",
    "ம் → வ்",
    "ம் + vowel: ம் becomes வ் and takes the vowel (மரம்+அடி)",
)
_M_DELETED_BEFORE_V = PunarchiRule(
    "m_keduthal_before_v",
    Vikaram.IYALBU,
    "keduthal",
    "ம் கெடுதல்",
    "ம் + வ்: the ம் disappears (மரம்+வேர் → மரவேர்)",
)
_AYTHAM_DELETED = PunarchiRule(
    "aytham_keduthal",
    Vikaram.IYALBU,
    "keduthal",
    "ஃ கெடுதல்",
    "āytam disappears before a consonant (அஃது → அது)",
)
_L_TO_R = PunarchiRule(
    "l_to_r_before_vallinam",
    Vikaram.VIKARAM,
    "thirithal",
    "ல் → ற்",
    "ல் mutates to ற் before a hard consonant (கல்+சீல → கற்சீல)",
)
_L_TO_D = PunarchiRule(
    "l_to_d_before_vallinam",
    Vikaram.VIKARAM,
    "thirithal",
    "ள் → ட்",
    "ள் mutates to ட் before a hard consonant (முள்+கொள → முட்கொள)",
)
_N_TO_D = PunarchiRule(
    "n_to_d_before_vallinam",
    Vikaram.VIKARAM,
    "thirithal",
    "ண் → ட்",
    "ண் mutates to ட் before a hard consonant (மண்+கலம் → மட்கலம்)",
)
_N_TO_R = PunarchiRule(
    "n_to_r_before_vallinam",
    Vikaram.VIKARAM,
    "thirithal",
    "ன் → ற்",
    "ன் mutates to ற் before a hard consonant (பொன்+காசு → பொற்காசு)",
)
_M_DOUBLING = PunarchiRule(
    "m_vallinam_miguthal",
    Vikaram.MIGUTHAL,
    "miguthal",
    "ம் + வல்லினம் மிகுதல்",
    "ம் + hard consonant: the hard consonant doubles (மரம்+தோப்பு)",
)
_DOUBLING = PunarchiRule(
    "mellinam_miguthal",
    Vikaram.MIGUTHAL,
    "miguthal",
    "மெல்லினம் மிகுதல்",
    "ல்/ள்/ண்/ன்/ய் + vowel: the final consonant doubles (கல்+அடி)",
)
_MEI_DOUBLING = PunarchiRule(
    "mei_vallinam_miguthal",
    Vikaram.MIGUTHAL,
    "miguthal",
    "வல்லினம் மிகுதல்",
    "any mei + hard consonant: the hard consonant doubles (நீர்+துளி)",
)
_MEI_UYIR = PunarchiRule(
    "mei_uyir_iyalbu",
    Vikaram.IYALBU,
    "iyalbu",
    "இயல்பு",
    "mei + vowel: the virama drops and the consonant takes the vowel",
)
_PLAIN = PunarchiRule(
    "plain_join",
    Vikaram.IYALBU,
    "iyalbu",
    "இயல்பு (மாற்றமில்லை)",
    "no rule applies: join as written",
)

RULES: Tuple[PunarchiRule, ...] = (
    _THONRAL_GLIDE,
    _UYIRMEI_GLIDE,
    _U_FINAL_DOUBLING,
    _M_TO_V,
    _M_DELETED_BEFORE_V,
    _AYTHAM_DELETED,
    _L_TO_R,
    _L_TO_D,
    _N_TO_D,
    _N_TO_R,
    _M_DOUBLING,
    _DOUBLING,
    _MEI_DOUBLING,
    _MEI_UYIR,
    _PLAIN,
)
RULES_BY_ID: Dict[str, PunarchiRule] = {r.id: r for r in RULES}


# --- 2. Small orthography helpers --------------------------------------------


def _base(akshara: str) -> str:
    """The consonant/vowel at the start of an akshara (``கி`` → ``க``)."""
    return akshara[:1]


def _vowel_of(akshara: str) -> str:
    """The independent vowel an akshara's matra stands for (``லா`` → ``ஆ``)."""
    base = _base(akshara)
    if base in VOWEL_TO_MATRA:  # a pure vowel is already its own vowel
        return base
    matra = akshara[1:]
    if matra in MATRA_TO_VOWEL:
        return MATRA_TO_VOWEL[matra]
    return ""  # pure mei (pulli): no vowel


def _aksharas(text: str) -> List[str]:
    from adhan_slm.tokenizer.swaram_tokenizer import segment_aksharas

    return segment_aksharas(text)


def _is_pure_mei(akshara: str) -> bool:
    return len(akshara) > 1 and akshara.endswith("\u0bcd")


def _is_front(akshara: str) -> bool:
    """True when the akshara's vowel selects the ய் glide (இ/ஈ/ஐ and their matras)."""
    if akshara in ("இ", "ஈ", "ஐ"):
        return True
    return len(akshara) > 1 and akshara[1:] in ("ி", "ீ", "ை")


def _virama(consonant: str) -> str:
    return consonant if consonant.endswith("\u0bcd") else consonant + "\u0bcd"


# --- 3. Forward direction -----------------------------------------------------


@dataclass(frozen=True)
class SandhiJoin:
    """Result of joining two units, with the rule that fired."""

    surface: str
    rule: PunarchiRule
    inserted: str = ""

    @property
    def vikaram(self) -> Vikaram:
        return self.rule.vikaram

    def as_dict(self) -> Dict[str, str]:
        return {"surface": self.surface, "rule": self.rule.id, "inserted": self.inserted}

    def __str__(self) -> str:  # pragma: no cover - log convenience
        return f"{self.surface} [{self.rule.id}]"


def join_units(
    left: str,
    right: str,
    *,
    case_suffix: bool = False,
) -> SandhiJoin:
    """Join ``left`` and ``right`` and say which puṇarcci rule produced the form.

    The order of the checks mirrors Puṇariyal: specific final-letter rules first,
    then the general doubling/iyalbu fallbacks.

    ``case_suffix`` selects the உ-ஈற்று விதி for *suffixes*: வீடு + இல் is
    வீட்டில், whereas the same stem in a compound keeps the glide (திரு + அடி →
    திருவடி). The classical grammar distinguishes the two joins; the engine
    cannot, so the caller says which one it is asking about.
    """
    if not left or not right:
        return SandhiJoin(left + right, _PLAIN, "empty input")
    left = unicodedata.normalize("NFC", left)
    right = unicodedata.normalize("NFC", right)
    if not left or not right:
        return SandhiJoin(left + right, _PLAIN, "empty input")

    last = final_akshara(left)
    final_class = letter_class(last)
    first = right[0]
    next_class = letter_class(first)
    stem = left[: -len(last)]

    # (a) Aytam disappears before anything it cannot stand in front of.
    if final_class is LetterClass.AYTHAM:
        joined = stem + right
        # ``அஃது`` (aytam + உ) keeps its vowel; a following hard consonant drops ஃ.
        if _virama(_base(first)) in VALLINAM:
            return SandhiJoin(joined, _AYTHAM_DELETED)
        return SandhiJoin(joined, _PLAIN)

    # (b) உ-final stem + vowel *suffix*: drop the உ, double the consonant
    #     (வீடு + இல் → வீட்டில்). Only for suffixes; a compound takes the glide.
    if (
        case_suffix
        and final_class is LetterClass.UYIRMEI
        and _vowel_of(last) == "உ"
        and next_class is LetterClass.UYIR
    ):
        # வீ + ட் + டில் = வீட்டில்: one copy keeps the virama, the other takes
        # the suffix's vowel (so it is never the ungrammatical வீட்ட்இல்).
        return SandhiJoin(
            stem + _virama(_base(last)) + attach_consonant(_base(last), right),
            _U_FINAL_DOUBLING,
        )

    # (c) Vowel-final + vowel-initial: the உடம்படுமெய் glide. Front vowels
    #     select ய், every other vowel வ்; the glide carries the next vowel.
    if final_class in (LetterClass.UYIR, LetterClass.UYIRMEI) and next_class is LetterClass.UYIR:
        bare = "ய" if _is_front(last) else "வ"
        rule = _THONRAL_GLIDE if final_class is LetterClass.UYIR else _UYIRMEI_GLIDE
        return SandhiJoin(left + attach_consonant(bare, right), rule, bare)

    # (d) ம் + vowel → வ்.
    if last == "ம்" and next_class is LetterClass.UYIR:
        return SandhiJoin(stem + attach_consonant("வ", right), _M_TO_V, "வ")

    # (e) ம் + வ் → the ம் is deleted (மரம் + வேர் → மரவேர்).
    if last == "ம்" and _base(first) == "வ":
        return SandhiJoin(stem + right, _M_DELETED_BEFORE_V)

    # (f) ல்/ள்/ண்/ன் + வல்லினம் → ற்/ட்/ட்/ற் (வலித்தல்).
    if last in _THIRITHAL_BEFORE_VALLINAM and _virama(_base(first)) in VALLINAM:
        mutated = _THIRITHAL_BEFORE_VALLINAM[last][0]
        rule = {
            "ல்": _L_TO_R,
            "ள்": _L_TO_D,
            "ண்": _N_TO_D,
            "ன்": _N_TO_R,
        }[last]
        return SandhiJoin(stem + mutated + right, rule, mutated)

    # (g) ம் + வல்லினம் → the hard consonant doubles.
    if last == "ம்" and _virama(_base(first)) in VALLINAM:
        return SandhiJoin(
            stem + _virama(_base(first)) + right,
            _M_DOUBLING,
            _virama(_base(first)),
        )

    # (h) ல்/ள்/ண்/ன்/ய் + vowel → the final consonant doubles.
    if last in _DOUBLING_FINALS and next_class is LetterClass.UYIR:
        return SandhiJoin(left + attach_consonant(_base(last), right), _DOUBLING, _base(last))

    # (i) Any other mei + வல்லினம் → the hard consonant doubles.
    if final_class is LetterClass.MEI and _virama(_base(first)) in VALLINAM:
        return SandhiJoin(
            left + _virama(_base(first)) + right,
            _MEI_DOUBLING,
            _virama(_base(first)),
        )

    # (j) mei + vowel → the virama drops (இயல்பு).
    if final_class is LetterClass.MEI and next_class is LetterClass.UYIR:
        return SandhiJoin(stem + attach_consonant(_base(last), right), _MEI_UYIR)

    return SandhiJoin(left + right, _PLAIN)


def join(left: str, right: str) -> str:
    """Surface form only — convenience wrapper over :func:`join_units`."""
    return join_units(left, right).surface


def apply_rules(word_units: Sequence[str]) -> SandhiJoin:
    """Join a chain of units left-to-right, returning the final surface form.

    The rule reported is the *last* one applied, which is what a left-to-right
    reader would call the outermost sandhi.
    """
    if not word_units:
        return SandhiJoin("", _PLAIN)
    surface = word_units[0]
    last = SandhiJoin(surface, _PLAIN)
    for unit in word_units[1:]:
        last = join_units(surface, unit)
        surface = last.surface
    return last


# --- 4. Reverse direction -----------------------------------------------------


@dataclass(frozen=True)
class PunarchiSplit:
    """A candidate ``left + right`` reading of a written word."""

    left: str
    right: str
    rule: PunarchiRule
    confidence: float
    restored: str = ""  # what the reversal inserted/replaced, for provenance

    def as_dict(self) -> Dict[str, object]:
        return {
            "left": self.left,
            "right": self.right,
            "rule": self.rule.id,
            "vikaram": self.rule.vikaram.value,
            "confidence": round(self.confidence, 3),
            "restored": self.restored,
        }

    def __str__(self) -> str:  # pragma: no cover - log convenience
        return f"{self.left} + {self.right} [{self.rule.id} {self.confidence:.2f}]"


def _boost(
    split: PunarchiSplit,
    lexicon: Optional[Iterable[str]],
    right_hints: Sequence[str] = (),
) -> PunarchiSplit:
    """Raise confidence when the lexicon or a known suffix endorses the cut.

    Structural reversal cannot know whether ``பசியில்லை`` is ``பசி + இல்லை`` or
    ``பசிய் + இலை``. Two cheap oracles — a lexicon and the suffix inventory the
    analyzer already carries — resolve most cases without a model.
    """
    conf = split.confidence
    if lexicon is not None and split.left in set(lexicon):
        conf = min(1.0, conf + 0.3)
    if right_hints and split.right in set(right_hints):
        conf = min(1.0, conf + 0.2)
    return PunarchiSplit(split.left, split.right, split.rule, conf, split.restored)


def reverse_thondral(word: str) -> List[PunarchiSplit]:
    """Undo தோன்றல்: a ய/வ between vowels may be an inserted glide.

    ``பசியில்லை`` → ``பசி + இல்லை``; ``திருவடி`` → ``திரு + அடி``; also the
    uyirmei-uyir variant (``கையில்லை`` → ``கை + இல்லை``). A real ய/வ consonant
    produces a structurally identical candidate, so confidence stays moderate
    until a lexicon votes.
    """
    word = unicodedata.normalize("NFC", word)
    aksharas = _aksharas(word)
    out: List[PunarchiSplit] = []
    offset = 0
    for i, ak in enumerate(aksharas):
        start = offset
        offset += len(ak)
        if i == 0 or _base(ak) not in ("ய", "வ"):
            continue
        prev = aksharas[i - 1]
        if letter_class(prev) not in (LetterClass.UYIR, LetterClass.UYIRMEI):
            continue
        vowel = _vowel_of(ak)
        if not vowel:
            continue
        left = word[:start]
        right = vowel + word[offset:]
        # A glide after an uyirmei keeps that carrier vowel; after a uyir the
        # glide is a pure insertion. Same reversal, different rule label.
        rule = _UYIRMEI_GLIDE if letter_class(prev) is LetterClass.UYIRMEI else _THONRAL_GLIDE
        out.append(PunarchiSplit(left, right, rule, 0.55, restored=ak))
    return out


def reverse_miguthal(word: str) -> List[PunarchiSplit]:
    """Undo மிகுதல்: a doubled consonant marks the join point.

    Three sub-cases, all reconstructed here:

    * ``கல்லடி`` (ல் + அடி) — a mellinam/idaiyinam doubles before a vowel; the
      second copy's matra is the next unit's vowel.
    * ``மரத்தோப்பு`` (மரம் + தோப்பு) — the ம் has already become the doubling
      consonant, so the left unit regains its ம்.
    * ``நீர்த்துளி`` (நீர் + துளி) — the left unit was a bare mei and is kept.
    """
    word = unicodedata.normalize("NFC", word)
    aksharas = _aksharas(word)
    out: List[PunarchiSplit] = []
    offset = 0
    for i in range(1, len(aksharas)):
        a, b = aksharas[i - 1], aksharas[i]
        start_a = offset
        offset += len(a)
        if not _is_pure_mei(a) or _base(a) != _base(b):
            continue
        prefix = word[:start_a]
        right_vowel = _vowel_of(b)
        mei = _virama(_base(a))
        suffix_rest = word[offset + len(b) :]

        if mei in ("ல்", "ள்", "ண்", "ன்", "ய்") and right_vowel:
            out.append(PunarchiSplit(prefix + mei, right_vowel + suffix_rest, _DOUBLING, 0.6, mei))
            continue

        if mei in VALLINAM:
            # (மரம் + தோப்பு): the ம் turned into the doubling letter. The
            # surface cannot say whether the left unit ended in ம் or in a bare
            # mei — மரத்தோப்பு is மரம் + தோப்பு, but நீர்த்துளி is நீர் + துளி.
            # Both are emitted, so the reader (or a lexicon) picks; the bare-mei
            # reading below carries the slightly lower prior confidence.
            if letter_class(final_akshara(prefix)) in (
                LetterClass.UYIR,
                LetterClass.MEI,
                LetterClass.UYIRMEI,
            ):
                out.append(PunarchiSplit(prefix + "ம்", b + suffix_rest, _M_DOUBLING, 0.5, "ம்"))
            # (வீடு + இல் → வீட்டில்): the உ dropped and the consonant doubled.
            out.append(
                PunarchiSplit(prefix + _base(a) + "ு", b + suffix_rest, _U_FINAL_DOUBLING, 0.5, "ு")
            )
            # (நீர் + துளி): the left unit ended in a bare mei and the doubling
            # consonant is a fresh letter — the prefix already holds the mei, so
            # the cut is right where the doubling starts.
            out.append(PunarchiSplit(prefix, b + suffix_rest, _MEI_DOUBLING, 0.45, ""))
            # (கல் + லட்சுமி): the left unit ended in this very mei, which is
            # what the doubling copy stands for.
            out.append(PunarchiSplit(prefix + mei, b + suffix_rest, _MEI_DOUBLING, 0.45, mei))
            # (கல் + தூண் → கற்றூண்): doubling hides an earlier mutation.
            for restored in _REVERSE_THIRITHAL.get(mei, ()):
                out.append(
                    PunarchiSplit(prefix + restored, b + suffix_rest, _DOUBLING, 0.35, restored)
                )
    return out


def reverse_thirithal(word: str) -> List[PunarchiSplit]:
    """Undo திரிதல்: a written ற்/ட் before வல்லினம் may have been ல்/ள்/ண்/ன்.

    ``முட்கொள`` → ``முள் + கொள``; ``கற்சீல`` → ``கல் + சீல``; ``பொற்காசு`` →
    ``பொன் + காசு``. Only substitutions that Puṇariyal licenses are offered, and
    confidence is low (0.4): the written ற்/ட் may genuinely be the root's.
    """
    word = unicodedata.normalize("NFC", word)
    aksharas = _aksharas(word)
    out: List[PunarchiSplit] = []
    offset = 0
    for i, ak in enumerate(aksharas):
        start = offset
        offset += len(ak)
        mei = _virama(_base(ak))
        if mei not in _REVERSE_THIRITHAL or not _is_pure_mei(ak):
            continue
        if i + 1 >= len(aksharas):
            continue
        nxt = aksharas[i + 1]
        if _virama(_base(nxt)) not in VALLINAM:
            continue
        rules_by_restored = {
            "ல்": _L_TO_R,
            "ள்": _L_TO_D,
            "ண்": _N_TO_D,
            "ன்": _N_TO_R,
        }
        for restored in _REVERSE_THIRITHAL[mei]:
            rule = rules_by_restored[restored]
            out.append(
                PunarchiSplit(
                    word[:start] + restored,
                    word[offset:],
                    rule,
                    0.4,
                    restored,
                )
            )
    return out


def reverse_keduthal(word: str) -> List[PunarchiSplit]:
    """Undo கெடுதல்: a deleted ம் (or ஃ) may have to be put back.

    ``மரவேர்`` → ``மரம் + வேர்``; ``அது`` ← ``அஃது`` is *not* recoverable from
    the surface (too many அது-shaped words exist), so only the ம்-deletion case is
    attempted, and only at வ-initial words — where the classical rule applies.
    """
    word = unicodedata.normalize("NFC", word)
    aksharas = _aksharas(word)
    out: List[PunarchiSplit] = []
    offset = 0
    for i, ak in enumerate(aksharas):
        start = offset
        offset += len(ak)
        if i == 0 or _base(ak) != "வ":
            continue
        left = word[:start]
        k = letter_class(final_akshara(left))
        # The deleted ம் stood after a vowel-bearing unit. On the surface that
        # unit's last akshara is a bare consonant carrying its inherent அ
        # (மர, திரு), which the letter taxonomy calls a mei — so MEI is a hit,
        # not a miss.
        if k not in (LetterClass.UYIR, LetterClass.MEI, LetterClass.UYIRMEI):
            continue
        out.append(
            PunarchiSplit(left + "ம்", word[start:], _M_DELETED_BEFORE_V, 0.3, "ம் inserted")
        )
    return out


def _explains(left: str, right: str, word: str) -> bool:
    """True when joining ``left`` and ``right`` rebuilds ``word``.

    The forward engine distinguishes a case join (வீட்டில்) from a compound
    (திருவடி) by flag, so both readings are tried; a reversal that neither
    reproduces is not an explanation of the word.
    """
    target = unicodedata.normalize("NFC", word)
    for case_suffix in (False, True):
        if join_units(left, right, case_suffix=case_suffix).surface == target:
            return True
    return False


def reverse_all(
    word: str,
    lexicon: Optional[Iterable[str]] = None,
    right_hints: Sequence[str] = (),
) -> List[PunarchiSplit]:
    """Every structural reversal of ``word``, deduplicated and ranked.

    ``lexicon`` (a set of known stems) and ``right_hints`` (the analyzer's suffix
    inventory) raise confidence for endorsed cuts. Ties keep forward order, so
    the result is deterministic.
    """
    word = unicodedata.normalize("NFC", word)
    candidates: List[PunarchiSplit] = []
    seen: set = set()
    for split in (
        *reverse_thondral(word),
        *reverse_miguthal(word),
        *reverse_thirithal(word),
        *reverse_keduthal(word),
    ):
        if not split.left or not split.right or split.left == split.right == "":
            continue
        key = (split.left, split.right)
        if key in seen:
            continue
        # A reversal only counts if the forward engine can rebuild the word from
        # it: same rule table, both directions. Structural candidates that do
        # not re-join are noise, and noise is what made the canonical
        # round-trips inconsistent.
        if not _explains(split.left, split.right, word):
            continue
        seen.add(key)
        candidates.append(_boost(split, lexicon, right_hints))
    candidates.sort(key=lambda s: -s.confidence)
    return candidates


def best_split(
    word: str,
    lexicon: Optional[Iterable[str]] = None,
    right_hints: Sequence[str] = (),
) -> Optional[PunarchiSplit]:
    """Highest-confidence reversal, or ``None`` when the word shows no sandhi."""
    splits = reverse_all(word, lexicon=lexicon, right_hints=right_hints)
    return splits[0] if splits else None


# --- 5. Classification helper -------------------------------------------------


def punarchi_class(left: str, right: str) -> str:
    """``"iyalbu"`` vs ``"vikaram"`` for a pair, in the blueprint's terms."""
    return "iyalbu" if join_units(left, right).vikaram is Vikaram.IYALBU else "vikaram"
