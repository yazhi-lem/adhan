"""Classical n-gram akshara baseline (ROADMAP_JAX_SLM.md §Phase 4: "comparison
table vs baselines").

Wraps open-tamil's `ngram.LetterModels.Unigram` — a plain Tamil-akshara
frequency counter — into a per-akshara perplexity baseline, directly
comparable to `adhan-nano`'s own per-akshara ppl target (§Phase 3: "val
perplexity beats a distilgpt2 baseline"). This is a near-zero-cost floor: any
trained SLM should clear it easily; it exists to catch broken training runs
(loss not beating a unigram frequency model means something is badly wrong)
without waiting on a distilgpt2 comparison run.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Sequence

try:
    import tamil.utf8 as _ot_utf8
    from ngram.LetterModels import Unigram as _OtUnigram

    HAS_OPEN_TAMIL = True
except ImportError:
    HAS_OPEN_TAMIL = False
    _ot_utf8 = None
    _OtUnigram = None


class AksharaUnigramBaseline:
    """Add-one (Laplace) smoothed unigram model over Tamil aksharas."""

    def __init__(self, corpus: Sequence[str]):
        if not HAS_OPEN_TAMIL:
            raise ImportError(
                "open-tamil is not installed. Run: pip install -r requirements-jax.txt"
            )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as fh:
            fh.write("\n".join(corpus))
            corpus_path = fh.name
        try:
            model = _OtUnigram(corpus_path)
            model.frequency_model()
        finally:
            Path(corpus_path).unlink(missing_ok=True)

        self._counts = model.letter
        self._total = sum(self._counts.values())
        self._vocab_size = len(self._counts)

    def probability(self, akshara: str) -> float:
        count = self._counts.get(akshara, 0)
        return (count + 1) / (self._total + self._vocab_size)

    def perplexity(self, text: str) -> float:
        """Per-akshara perplexity of `text` under this unigram model.

        Aksharas outside the closed Tamil letter inventory (digits, Latin,
        punctuation) are skipped — this is a Tamil-script-only baseline.
        """
        aksharas = [a for a in _ot_utf8.get_letters(text) if a in self._counts]
        if not aksharas:
            return float("nan")
        log_prob_sum = sum(math.log(self.probability(a)) for a in aksharas)
        return math.exp(-log_prob_sum / len(aksharas))
