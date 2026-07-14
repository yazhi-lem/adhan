"""Swaram tokenizer — native Tamil akshara + morpheme tokenization for Adhan.

Two lossless layers (see docs/ARCHITECTURE_SWARAM_SLM.md):

  Layer A  akshara segmentation   deterministic, closed, join(aksharas) == input
  Layer B  morpheme merges        BPE trained *over aksharas*, bounded to vocab_size

The atomic unit is the Tamil grapheme cluster (akshara / உயிர்மெய்), not a byte or
an English-fit sub-word. A base consonant carries its combining vowel sign (matra)
or pulli (்); combining marks never start a cluster. This keeps Tamil morphology
boundaries intact and yields ~1 token per akshara before merges — far tighter than a
multilingual BPE tokenizer, which fragments each akshara into 3-6 pieces.

The core is pure-python (no external tokenizer dependency) so it runs anywhere,
including on-device.

CLI:
    python -m adhan_slm.tokenizer.swaram_tokenizer "படித்துக்கொண்டிருந்தேன்"
"""
from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# --- Tamil Unicode ranges (block U+0B80–U+0BFF) --------------------------------
_TAMIL_MATRAS = set(range(0x0BBE, 0x0BCD))      # vowel signs ா ி ீ … ௌ
_TAMIL_PULLI = 0x0BCD                           # virama ் (makes a pure consonant)
_COMBINING = _TAMIL_MATRAS | {_TAMIL_PULLI}     # marks that attach to a base

# Standalone building blocks used to seed the closed base inventory.
UYIR = list("அஆஇஈஉஊஎஏஐஒஓஔ")                    # 12 vowels (swaram)
_CONSONANTS = list("கஙசஜஞடணதநனபமயரறலளழவஶஷஸஹ")   # base consonants (+ Grantha)
AYTHAM = "ஃ"

# Tamil numerals + numeric/auspicious symbols (Tamil block, single codepoints).
# Adding these to the *closed* base inventory means classical and modern text that
# uses native Tamil numbers (e.g. ௰௲ = 1000) or the OM sign never falls back to
# <unk> — the "numeral handling" Phase-1 item in ROADMAP_JAX_SLM.md §Phase 1.
TAMIL_DIGITS = [chr(cp) for cp in range(0x0BE6, 0x0BF0)]      # ௦ ௧ … ௯ (0–9)
TAMIL_NUMERALS = [chr(cp) for cp in range(0x0BF0, 0x0BF3)]    # ௰ ௱ ௲ (10, 100, 1000)
# ௳ day, ௴ month, ௵ year, ௶ debit, ௷ credit, ௸ as-above, ௹ rupee, ௺ number sign
TAMIL_SYMBOLS = [chr(cp) for cp in range(0x0BF3, 0x0BFB)] + [chr(0x0BD0)]  # + ௐ OM

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"]
WORD_MARK = "▁"  # SentencePiece-style word-boundary marker (Layer B only)


def _is_combining(cp: int) -> bool:
    return cp in _COMBINING


def default_akshara_inventory() -> List[str]:
    """The closed base set: 12 uyir + 216 uyirmey + 18 mey + aytham + Grantha,
    plus Tamil digits/numerals/symbols (௦–௯, ௰௱௲, ௳–௺, ௐ).

    Generated combinatorially so the base vocabulary is complete regardless of
    which aksharas happen to appear in a given corpus.
    """
    inv: List[str] = list(UYIR) + [AYTHAM]
    inv += TAMIL_DIGITS + TAMIL_NUMERALS + TAMIL_SYMBOLS
    matras = [chr(cp) for cp in range(0x0BBE, 0x0BCD)]
    for c in _CONSONANTS:
        inv.append(c)                       # inherent-'a' uyirmey (க)
        inv.append(c + chr(_TAMIL_PULLI))   # pure consonant / mey (க்)
        for m in matras:                    # க + matra → காகிகீ …
            inv.append(c + m)
    # de-dup, keep order
    seen, out = set(), []
    for a in inv:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def segment_aksharas(text: str) -> List[str]:
    """Layer A: split text into grapheme clusters. Lossless: ''.join(out) == text.

    A new cluster starts on every base codepoint; matras and pulli attach to the
    preceding Tamil base. Non-Tamil characters (Latin, punctuation, whitespace,
    digits) are emitted verbatim as single-character clusters so code-switched text
    degrades gracefully.
    """
    text = unicodedata.normalize("NFC", text)
    out: List[str] = []
    for ch in text:
        cp = ord(ch)
        if out and _is_combining(cp):
            # attach to the current cluster (must follow a Tamil base)
            out[-1] += ch
        else:
            out.append(ch)
    return out


@dataclass
class SwaramTokenizer:
    """Akshara-native tokenizer with an optional learned morpheme-merge layer."""

    vocab: Dict[str, int] = field(default_factory=dict)
    merges: List[Tuple[str, str]] = field(default_factory=list)
    _ranks: Dict[Tuple[str, str], int] = field(default_factory=dict, repr=False)
    _inv_vocab: Dict[int, str] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._reindex()

    # -- introspection ---------------------------------------------------------
    def _reindex(self):
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._inv_vocab = {i: t for t, i in self.vocab.items()}

    @property
    def unk_id(self) -> int:
        return self.vocab.get("<unk>", 0)

    def __len__(self) -> int:
        return len(self.vocab)

    # -- segmentation hook (overridden by sibling tokenizers, e.g. Aksharam) ----
    def _segment(self, text: str) -> List[str]:
        """Layer A segmentation. Subclasses override for other scripts."""
        return segment_aksharas(text)

    @classmethod
    def _base_inventory(cls) -> List[str]:
        """Closed base akshara set. Subclasses override for other scripts."""
        return default_akshara_inventory()

    def aksharas(self, text: str) -> List[str]:
        """Layer A only — the grapheme clusters, for inspection/eval."""
        return self._segment(text)

    # -- Layer B: greedy BPE over aksharas ------------------------------------
    def _apply_merges(self, pieces: List[str]) -> List[str]:
        if not self._ranks:
            return pieces
        pieces = list(pieces)
        while len(pieces) > 1:
            best_rank, best_i = None, -1
            for i in range(len(pieces) - 1):
                r = self._ranks.get((pieces[i], pieces[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i < 0:
                break
            pieces[best_i : best_i + 2] = [pieces[best_i] + pieces[best_i + 1]]
        return pieces

    def _pretokenize(self, text: str) -> List[str]:
        """Segment, then insert word-boundary marks before non-space words."""
        text = unicodedata.normalize("NFC", text)
        pieces: List[str] = []
        for i, cluster in enumerate(self._segment(text)):
            if cluster == " ":
                pieces.append(WORD_MARK)
            else:
                if i == 0 or (pieces and pieces[-1] == WORD_MARK) is False:
                    pass
                pieces.append(cluster)
        return pieces

    # -- encode / decode -------------------------------------------------------
    def tokenize(self, text: str) -> List[str]:
        """Layer A + B merged subword-piece strings (no ids), for inspection/eval."""
        return self._apply_merges(self._pretokenize(text))

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        pieces = self.tokenize(text)
        ids = [self.vocab.get(p, self.unk_id) for p in pieces]
        if add_special:
            ids = [self.vocab.get("<bos>", self.unk_id), *ids,
                   self.vocab.get("<eos>", self.unk_id)]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        specials = set(SPECIAL_TOKENS)
        toks = []
        for i in ids:
            t = self._inv_vocab.get(i, "<unk>")
            if skip_special and t in specials:
                continue
            toks.append(t)
        return "".join(toks).replace(WORD_MARK, " ")

    def fertility(self, text: str) -> float:
        """Mean tokens per akshara (Layer B target: < 1.15). Lower = tighter."""
        n_aks = sum(1 for a in self._segment(text) if a.strip())
        if n_aks == 0:
            return 0.0
        n_tok = sum(1 for t in self._apply_merges(self._pretokenize(text)) if t != WORD_MARK)
        return n_tok / n_aks

    # -- training --------------------------------------------------------------
    @classmethod
    def train(cls, corpus: List[str], vocab_size: int = 8000,
              min_freq: int = 2) -> "SwaramTokenizer":
        """Train the merge layer over aksharas (bounded BPE).

        Base vocab = specials + full closed akshara inventory + any aksharas seen
        in the corpus. Merges are learned greedily on the most frequent adjacent
        akshara pair until vocab_size is reached (captures productive suffix chains
        like -கிற- -ந்த- -ஏன் -ஆக).
        """
        base = list(SPECIAL_TOKENS) + [WORD_MARK] + cls._base_inventory()
        seen = set(base)
        word_pieces: List[List[str]] = []
        for line in corpus:
            for tok in cls().__class__._pretokenize(cls(), line):
                if tok not in seen:
                    seen.add(tok)
                    base.append(tok)
            # group into "words" separated by WORD_MARK for merge counting
            pieces = cls()._pretokenize(line)
            word: List[str] = []
            for p in pieces:
                if p == WORD_MARK:
                    if word:
                        word_pieces.append(word)
                    word = []
                else:
                    word.append(p)
            if word:
                word_pieces.append(word)

        vocab = {t: i for i, t in enumerate(base)}
        merges: List[Tuple[str, str]] = []
        words = [list(w) for w in word_pieces]

        while len(vocab) < vocab_size:
            pairs: Counter = Counter()
            for w in words:
                for i in range(len(w) - 1):
                    pairs[(w[i], w[i + 1])] += 1
            if not pairs:
                break
            (a, b), freq = pairs.most_common(1)[0]
            if freq < min_freq:
                break
            merged = a + b
            merges.append((a, b))
            if merged not in vocab:
                vocab[merged] = len(vocab)
            for w in words:
                i = 0
                while i < len(w) - 1:
                    if w[i] == a and w[i + 1] == b:
                        w[i : i + 2] = [merged]
                    else:
                        i += 1

        return cls(vocab=vocab, merges=merges)

    # -- persistence -----------------------------------------------------------
    def save(self, vocab_path: str, merges_path: str) -> None:
        Path(vocab_path).write_text(
            json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8")
        Path(merges_path).write_text(
            "\n".join(f"{a}\t{b}" for a, b in self.merges), encoding="utf-8")

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str) -> "SwaramTokenizer":
        vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        merges = []
        mtext = Path(merges_path).read_text(encoding="utf-8").strip()
        if mtext:
            for line in mtext.splitlines():
                a, b = line.split("\t")
                merges.append((a, b))
        return cls(vocab=vocab, merges=merges)


def _demo(text: str) -> None:
    aks = segment_aksharas(text)
    print(f"input      : {text}")
    print(f"aksharas   : {aks}")
    print(f"n_aksharas : {sum(1 for a in aks if a.strip())}")
    # Train a tiny merge layer on the single input just to demonstrate ids/round-trip.
    tok = SwaramTokenizer.train([text], vocab_size=len(default_akshara_inventory()) + 64)
    ids = tok.encode(text, add_special=True)
    back = tok.decode(ids)
    print(f"n_tokens   : {len(ids)}  (with <bos>/<eos>)")
    print(f"ids        : {ids}")
    print(f"fertility  : {tok.fertility(text):.3f} tokens/akshara")
    print(f"round-trip : {'OK' if back == text else 'LOSSY'}  -> {back!r}")


if __name__ == "__main__":
    sample = sys.argv[1] if len(sys.argv) > 1 else "படித்துக்கொண்டிருந்தேன்"
    _demo(sample)
