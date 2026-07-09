"""Thin bridge onto `open-tamil` (Ezhil Language Foundation), used as Adhan's base
Tamil-NLP layer wherever a mature, tested implementation beats a from-scratch one.

open-tamil is MIT-licensed and pip-installable (`pip install open-tamil`); see
requirements-jax.txt. One bundled submodule, `tamilsandhi`, is GPLv3 — we only
*call* it as a separate installed package, never vendor its source into this repo.

What Adhan takes from it, and why (see ROADMAP_JAX_SLM.md §Phase 1/4):
  - `tamil.utf8.get_letters`      reference akshara segmenter -> cross-check oracle
                                   for Layer A of the swaram/aksharam tokenizers
  - `tamilstemmer.TamilStemmer`   suffix-stripping stemmer -> morpheme-boundary
                                   signal for scoring Layer B (BPE) merges
  - `tamilsandhi.check_sandhi`    rule-based sandhi (புணர்ச்சி) grammar checker
                                   -> the "morphological analyzer" Phase 1 asks
                                   for, and a correctness probe for generations
  - `tamil.txt2unicode`           25-encoding auto-detector -> widens corpus
                                   sources beyond clean UTF-8 (Phase 2)
  - `transliterate.azhagi`        phonetic English->Tamil transliteration ->
                                   recovers Tanglish/romanized corpus text
  - `tamil.numeral`                number -> Tamil-word rendering -> synthetic
                                   kid-level data (counting, dates)
  - `solthiruthi` data/*.txt      categorized lexicons (animals, verbs, objects,
                                   relations, pronouns, stopwords) -> kid-level
                                   eval prompt seeds + vocab-coverage checks

The swaram/aksharam tokenizer *cores* stay pure-python and dependency-free (see
their module docstrings) — open-tamil is only used in eval/, corpus tooling, and
tests, never imported by the hot encode/decode path.
"""
from __future__ import annotations

import importlib.resources
import re
from pathlib import Path
from typing import List

try:
    import tamil.utf8 as _ot_utf8
    import tamil.numeral as _ot_numeral
    import tamil.txt2unicode as _ot_txt2unicode
    import tamilstemmer as _ot_stemmer_mod
    import tamilsandhi as _ot_sandhi
    import transliterate.azhagi as _ot_azhagi
    import solthiruthi

    HAS_OPEN_TAMIL = True
except ImportError:  # open-tamil is optional; callers must check HAS_OPEN_TAMIL
    HAS_OPEN_TAMIL = False
    _ot_utf8 = _ot_numeral = _ot_txt2unicode = None
    _ot_stemmer_mod = _ot_sandhi = _ot_azhagi = None
    solthiruthi = None


def _require_open_tamil() -> None:
    if not HAS_OPEN_TAMIL:
        raise ImportError(
            "open-tamil is not installed. Run: pip install -r requirements-jax.txt "
            "(or `pip install open-tamil` directly)."
        )


def reference_segment_aksharas(text: str) -> List[str]:
    """Akshara segmentation via open-tamil's `tamil.utf8.get_letters`.

    This is the cross-check oracle for `adhan_slm.tokenizer.segment_aksharas`
    (our from-scratch Layer A) — the two should agree on any well-formed Tamil
    string. Used in tests, not in the tokenizer's hot path.
    """
    _require_open_tamil()
    return _ot_utf8.get_letters(text)


_stemmer_singleton = None


def stem_word(word: str) -> str:
    """Strip tense/case/plural/question suffixes via open-tamil's TamilStemmer.

    Returns the stem; `word[len(stem):]` is the (approximate) suffix, used as a
    weak-supervision morpheme-boundary signal in `adhan_slm.eval.morphology`.
    """
    _require_open_tamil()
    global _stemmer_singleton
    if _stemmer_singleton is None:
        _stemmer_singleton = _ot_stemmer_mod.TamilStemmer()
    return _stemmer_singleton.stemWord(word)


def sandhi_check(phrase: str) -> List[str]:
    """Rule-based sandhi (புணர்ச்சி) grammar check via open-tamil's tamilsandhi.

    `phrase` must be whitespace-separated words (this is a per-word grammar
    checker/corrector, not a compound-word splitter). Returns a word list the
    same length as `phrase.split()`: words already following the ~40 sandhi
    rules pass through unchanged; words that don't are replaced with the
    checker's correction. Falls back to `phrase.split()` unchanged if the
    checker raises.
    """
    _require_open_tamil()
    try:
        words, _stemmer_engine = _ot_sandhi.check_sandhi(phrase)
        return list(words)
    except Exception:
        return phrase.split()


_TAMIL_BLOCK = re.compile(r"[஀-௿]")


def normalize_encoding(text: str) -> str:
    """Best-effort auto-detect + convert legacy Tamil encodings to Unicode.

    Wraps `tamil.txt2unicode.auto2unicode`, which recognizes ~25 legacy Tamil
    font encodings (TSCII, TAB, Bamini, dinamani, murasoli, ...). `auto2unicode`
    assumes its input is *not* already Unicode and fails on it (returns empty),
    so text already containing Tamil-block codepoints is passed through
    unchanged without calling it. Use before feeding scraped text of unknown
    provenance into the corpus pipeline (see docs/ARCHITECTURE_SWARAM_SLM.md).
    """
    _require_open_tamil()
    if not text or _TAMIL_BLOCK.search(text):
        return text
    try:
        converted = _ot_txt2unicode.auto2unicode(text)
    except Exception:
        return text
    return converted or text


def transliterate_tanglish(text: str) -> str:
    """Best-effort phonetic English (Tanglish) -> Tamil Unicode transliteration.

    Wraps open-tamil's Azhagi scheme (many->one phonetic map, longest-prefix
    greedy match). Useful for recovering Tamil meaning from romanized
    social-media text before it is discarded as non-Tamil during corpus
    filtering.
    """
    _require_open_tamil()
    import transliterate as _ot_transliterate_mod

    return _ot_transliterate_mod.iterative_transliterate(_ot_azhagi.Transliteration.table, text)


def number_to_tamil(n: int, casual: bool = False) -> str:
    """Render an integer as Tamil words, e.g. 12 -> 'பன்னிரண்டு'.

    Used for synthetic kid-level data (counting, ages, simple arithmetic).
    """
    _require_open_tamil()
    if casual:
        return _ot_numeral.num2tamilstr_casual(n)
    return _ot_numeral.num2tamilstr(n)


_LEXICON_FILES = {
    "animals": "animals.txt",
    "objects": "objects.txt",
    "relations": "relations.txt",
    "pronouns": "pronouns.txt",
    "verbs": "verbs.txt",
    "times": "times.txt",
    "categories": "categories.txt",
    "adjectives": "adjectives.txt",
    "adverbs": "adverbs.txt",
}
STOPWORDS_FILE = "TamilStopWords.txt"


def load_lexicon(category: str) -> List[str]:
    """Load a category word list bundled with open-tamil's `solthiruthi` data.

    Categories: animals, objects, relations, pronouns, verbs, times, categories,
    adjectives, adverbs. These are the seed vocabulary for the kid-level eval
    prompt set (`adhan_slm.eval.kid_level_prompts`) and for tokenizer
    vocab-coverage checks.
    """
    _require_open_tamil()
    if category not in _LEXICON_FILES:
        raise ValueError(f"unknown lexicon category {category!r}; have {sorted(_LEXICON_FILES)}")
    return _read_data_file(_LEXICON_FILES[category])


def load_stopwords() -> List[str]:
    """Load open-tamil's curated Tamil stopword list (solthiruthi data)."""
    _require_open_tamil()
    return _read_data_file(STOPWORDS_FILE)


_TAMIL_RUN = re.compile(r"[஀-௿]+(?:\s+[஀-௿]+)*")


def _read_data_file(filename: str) -> List[str]:
    """Parse a solthiruthi data file: `english<tab-or-space>tamil` rows with
    `#` comment and `>>section` header lines. Delimiters are inconsistent
    across files (tab in most, bare space in a few), so we extract the
    right-most run of Tamil-script text on each row instead of splitting.
    """
    data_dir = Path(str(importlib.resources.files(solthiruthi))) / "data"
    path = data_dir / filename
    words: List[str] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(">>"):
                continue
            matches = _TAMIL_RUN.findall(line)
            if not matches:
                continue
            word = matches[-1].strip()
            if word and word not in seen:
                seen.add(word)
                words.append(word)
    return words
