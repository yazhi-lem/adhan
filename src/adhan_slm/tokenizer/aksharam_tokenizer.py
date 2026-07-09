"""Aksharam tokenizer — native Hindi/Devanagari akshara tokenization.

Sibling of the Swaram (Dravidian) tokenizer. Same two-layer design — lossless
akshara segmentation (Layer A) + bounded morpheme-merge BPE (Layer B) — retuned
for Devanagari script rules:

  - matras (vowel signs) U+093E–U+094C attach to the preceding consonant
  - virama / halant U+094D forms conjuncts: C + ् + C → one cluster (क्ष, त्र)
  - anusvara ं, visarga ः, chandrabindu ँ, nukta ़ attach to their base
  - Devanagari digits ०–९ handled as single clusters

Aksharam reuses SwaramTokenizer's merge/encode/decode machinery via the
`_segment` and `_base_inventory` hooks; only script logic differs. Encoding can be
JAX-accelerated in batch via `adhan_slm.tokenizer.jax_encode`.

CLI:
    python -m adhan_slm.tokenizer.aksharam_tokenizer "पढ़ रहा था"
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adhan_slm.tokenizer.swaram_tokenizer import SwaramTokenizer  # noqa: E402

# --- Devanagari Unicode (block U+0900–U+097F) ----------------------------------
_DEVA_MATRAS = set(range(0x093E, 0x094D))      # vowel signs ा ि ी … ौ
_DEVA_VIRAMA = 0x094D                           # halant ् (forms conjuncts)
_DEVA_NUKTA = 0x093C                            # ़
_DEVA_SIGNS = {0x0900, 0x0901, 0x0902, 0x0903} # chandrabindu/anusvara/visarga
_COMBINING = _DEVA_MATRAS | {_DEVA_VIRAMA, _DEVA_NUKTA} | _DEVA_SIGNS

VOWELS = list("अआइईउऊऋएऐओऔ")                    # independent vowels (svara)
_CONSONANTS = list("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")
_DEVA_DIGITS = list("०१२३४५६७८९")


def _is_combining(cp: int) -> bool:
    return cp in _COMBINING


def default_aksharam_inventory() -> List[str]:
    """Closed base set: independent vowels + consonants + C+virama + C+matra + digits."""
    inv: List[str] = list(VOWELS) + list(_DEVA_DIGITS)
    matras = [chr(cp) for cp in range(0x093E, 0x094D)]
    for c in _CONSONANTS:
        inv.append(c)                             # bare consonant (inherent 'a')
        inv.append(c + chr(_DEVA_VIRAMA))         # half-consonant क्
        for m in matras:                          # क + matra
            inv.append(c + m)
    seen, out = set(), []
    for a in inv:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def segment_devanagari(text: str) -> List[str]:
    """Layer A for Devanagari. Lossless: ''.join(out) == text.

    A new cluster starts on every base codepoint; matras, nukta, anusvara, visarga,
    chandrabindu attach to the base. A virama (halant) attaches, and the *following*
    consonant joins the same cluster to form a conjunct (क् + ष → क्ष).
    """
    text = unicodedata.normalize("NFC", text)
    out: List[str] = []
    prev_was_virama = False
    for ch in text:
        cp = ord(ch)
        if out and _is_combining(cp):
            out[-1] += ch
            prev_was_virama = (cp == _DEVA_VIRAMA)
        elif out and prev_was_virama and 0x0915 <= cp <= 0x0939:
            # consonant right after a halant → conjunct, stays in current cluster
            out[-1] += ch
            prev_was_virama = False
        else:
            out.append(ch)
            prev_was_virama = False
    return out


class AksharamTokenizer(SwaramTokenizer):
    """Hindi/Devanagari akshara tokenizer (Indic-script prototype)."""

    def _segment(self, text: str) -> List[str]:
        return segment_devanagari(text)

    @classmethod
    def _base_inventory(cls) -> List[str]:
        return default_aksharam_inventory()


def _demo(text: str) -> None:
    clusters = segment_devanagari(text)
    print(f"input      : {text}")
    print(f"aksharas   : {clusters}")
    print(f"n_aksharas : {sum(1 for a in clusters if a.strip())}")
    tok = AksharamTokenizer.train([text], vocab_size=len(default_aksharam_inventory()) + 64)
    ids = tok.encode(text, add_special=True)
    back = tok.decode(ids)
    print(f"n_tokens   : {len(ids)}  (with <bos>/<eos>)")
    print(f"fertility  : {tok.fertility(text):.3f} tokens/akshara")
    print(f"round-trip : {'OK' if back == text else 'LOSSY'}  -> {back!r}")


if __name__ == "__main__":
    sample = sys.argv[1] if len(sys.argv) > 1 else "पढ़ रहा था"
    _demo(sample)
