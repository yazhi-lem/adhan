"""Morphology probe (roadmap Phase 4: "morphology probe / sandhi split-join
accuracy", ROADMAP_JAX_SLM.md §Phase 1 & §Phase 4).

Uses open-tamil as the morphological oracle — its `TamilStemmer` (suffix
stripping) and `tamilsandhi` checker (rule-based சந்தி splitting) — to score
whether the swaram/aksharam tokenizer's learned Layer B (BPE) merges respect
real morpheme boundaries, instead of cutting through them.

Requires open-tamil (`pip install -r requirements-jax.txt`); raises ImportError
via the bridge if it isn't installed. This module is eval-only — it is never
imported by the tokenizer's encode/decode hot path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from adhan_slm.external.open_tamil_bridge import sandhi_check, stem_word


@dataclass
class BoundaryAgreementReport:
    n_words: int
    n_with_suffix: int
    n_agree: int
    disagreements: List[str] = field(default_factory=list)

    @property
    def agreement_rate(self) -> float:
        return self.n_agree / self.n_with_suffix if self.n_with_suffix else float("nan")


def stemmer_boundary_agreement(tokenizer, words: Sequence[str],
                                max_examples: int = 20) -> BoundaryAgreementReport:
    """Fraction of inflected words where a tokenizer merge boundary lands
    exactly at the stem/suffix split open-tamil's stemmer finds.

    `tokenizer` is a SwaramTokenizer/AksharamTokenizer (needs `.tokenize()`).
    Words the stemmer treats as unsuffixed (stem == word, or the stemmer
    diverges from a clean prefix of the word) are excluded from the rate —
    they carry no boundary signal either way.
    """
    n_agree = 0
    n_with_suffix = 0
    disagreements: List[str] = []
    for word in words:
        stem = stem_word(word)
        if stem == word or not word.startswith(stem):
            continue
        n_with_suffix += 1
        boundary = len(stem)

        pieces = tokenizer.tokenize(word)
        offset = 0
        token_boundaries = set()
        for piece in pieces:
            offset += len(piece)
            token_boundaries.add(offset)

        if boundary in token_boundaries:
            n_agree += 1
        elif len(disagreements) < max_examples:
            disagreements.append(word)

    return BoundaryAgreementReport(
        n_words=len(words), n_with_suffix=n_with_suffix, n_agree=n_agree,
        disagreements=disagreements,
    )


@dataclass
class SandhiCorrectnessReport:
    n_phrases: int
    n_words: int
    n_correct_words: int
    corrections: List[str] = field(default_factory=list)

    @property
    def word_correctness_rate(self) -> float:
        return self.n_correct_words / self.n_words if self.n_words else float("nan")


def sandhi_correctness_rate(phrases: Sequence[str], max_examples: int = 20) -> SandhiCorrectnessReport:
    """Fraction of words in `phrases` that already satisfy open-tamil's ~40
    sandhi (புணர்ச்சி) grammar rules, per `tamilsandhi.check_sandhi`.

    Feed this model generations (space-tokenized Tamil sentences) to get a
    cheap, rule-based grammaticality signal — exactly the "sandhi split/join
    accuracy" probe called for in ROADMAP_JAX_SLM.md §Phase 4, ahead of the
    human read-through rubric.
    """
    n_words = 0
    n_correct = 0
    corrections: List[str] = []
    for phrase in phrases:
        original_words = phrase.split()
        if not original_words:
            continue
        checked_words = sandhi_check(phrase)
        for orig, checked in zip(original_words, checked_words):
            n_words += 1
            if orig == checked:
                n_correct += 1
            elif len(corrections) < max_examples:
                corrections.append(f"{orig} -> {checked}")
    return SandhiCorrectnessReport(
        n_phrases=len(phrases), n_words=n_words, n_correct_words=n_correct,
        corrections=corrections,
    )
