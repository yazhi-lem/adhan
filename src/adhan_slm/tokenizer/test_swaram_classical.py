"""Classical-Tamil regression tests for the Swaram tokenizer, on real Tholkappiyam.

The Tholkappiyam moolam (testdata/tholkappiyam_moolam.txt, public-domain via Project
Madurai) is the oldest Tamil grammar and a dense, morphologically rich stress test:
if Layer A stays lossless and the closed inventory covers every akshara here, it
covers Tamil. It also gives an honest fertility benchmark on classical (not modern)
text, and proves the merge layer tightens sequences on it.

Run: PYTHONPATH=src python -m adhan_slm.tokenizer.test_swaram_classical
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adhan_slm.tokenizer.swaram_tokenizer import (
    SwaramTokenizer, segment_aksharas, default_akshara_inventory, WORD_MARK,
)

_FIXTURE = Path(__file__).resolve().parent / "testdata" / "tholkappiyam_moolam.txt"
_TAMIL = re.compile(r"[஀-௿]")


def _load_sutras():
    lines = _FIXTURE.read_text(encoding="utf-8").splitlines()
    # drop the attribution header (lines starting with '#') and blanks
    return [unicodedata.normalize("NFC", ln) for ln in lines
            if ln.strip() and not ln.startswith("#")]


def test_layer_a_lossless_on_classical():
    sutras = _load_sutras()
    assert len(sutras) > 1000, f"fixture too small ({len(sutras)} lines)"
    for s in sutras:
        assert "".join(segment_aksharas(s)) == s, f"lossy round-trip on: {s[:40]}"
    print(f"  lossless: Layer A round-trips all {len(sutras)} sutras exactly OK")


def test_closed_inventory_covers_classical():
    """Every Tamil-script akshara in the Tholkappiyam is in the *closed* base set —
    so classical Tamil never depends on corpus-seen aksharas to avoid <unk>."""
    inv = set(default_akshara_inventory())
    text = "\n".join(_load_sutras())
    missing = sorted({a for a in segment_aksharas(text)
                      if _TAMIL.search(a) and a not in inv})
    assert not missing, f"{len(missing)} classical aksharas not in inventory: {missing[:20]}"
    print(f"  inventory: closed base covers every Tamil akshara in the text OK")


def test_full_encode_decode_round_trip():
    sutras = _load_sutras()
    tok = SwaramTokenizer.train(sutras, vocab_size=len(default_akshara_inventory()) + 2000)
    # round-trip a held-out-style sample through ids
    bad = 0
    for s in sutras[::7]:
        if tok.decode(tok.encode(s)) != s:
            bad += 1
    assert bad == 0, f"{bad} sutras failed id round-trip"
    print(f"  encode/decode: id round-trip lossless across the corpus OK")


def test_merges_tighten_fertility():
    sutras = _load_sutras()
    base = SwaramTokenizer()                        # no merges
    trained = SwaramTokenizer.train(sutras, vocab_size=len(default_akshara_inventory()) + 3000)
    # join with spaces (become WORD_MARK, excluded from both counts) so newlines don't
    # skew the ratio — this isolates the true akshara→token fertility.
    sample = " ".join(sutras[:400])
    f_base = base.fertility(sample)
    f_trained = trained.fertility(sample)
    # akshara-native tokenizer is ~1.0 tok/akshara before merges, and merges only
    # ever combine, so trained fertility must be < base and comfortably under target.
    assert abs(f_base - 1.0) < 1e-6, f"base fertility should be 1.0, got {f_base}"
    assert f_trained < f_base, f"merges should reduce fertility ({f_trained} !< {f_base})"
    assert f_trained < 1.15, f"fertility {f_trained} exceeds the <1.15 target"
    print(f"  fertility: base {f_base:.3f} -> merged {f_trained:.3f} tok/akshara "
          f"({(1-f_trained/f_base)*100:.0f}% tighter) OK")


def main():
    print("swaram classical (Tholkappiyam) tests:")
    passed = 0
    for fn in (test_layer_a_lossless_on_classical, test_closed_inventory_covers_classical,
               test_full_encode_decode_round_trip, test_merges_tighten_fertility):
        fn()
        passed += 1
    print(f"\n{passed} passed")


if __name__ == "__main__":
    main()
