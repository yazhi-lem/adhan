"""Tests for the Aksharam (Hindi/Devanagari) tokenizer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adhan_slm.tokenizer import (  # noqa: E402
    AksharamTokenizer,
    segment_devanagari,
    default_aksharam_inventory,
)

SAMPLES = [
    "नमस्ते",
    "पढ़ रहा था",
    "क्षत्रिय",              # conjuncts क्ष, त्र
    "मैं हिंदी बोलता हूँ",   # anusvara, chandrabindu
    "संख्या १२३ / 123",     # Devanagari + ASCII digits
    "Hindi AI 2026",         # code-switch
]


def test_segmentation_is_lossless():
    for s in SAMPLES:
        assert "".join(segment_devanagari(s)) == s, f"lossy: {s!r}"


def test_conjuncts_stay_together():
    # क् + ष -> क्ष (one cluster), त् + र + ि -> त्रि
    assert segment_devanagari("क्षत्रिय") == ["क्ष", "त्रि", "य"]
    assert segment_devanagari("नमस्ते") == ["न", "म", "स्ते"]


def test_matra_and_signs_attach():
    assert segment_devanagari("हूँ") == ["हूँ"]     # ह + ू + ँ
    assert segment_devanagari("हिंदी") == ["हिं", "दी"]


def test_inventory_closed_and_deduped():
    inv = default_aksharam_inventory()
    assert len(inv) == len(set(inv))
    assert "क" in inv and "क्" in inv and "कि" in inv
    assert len(inv) > 300


def test_encode_decode_round_trip():
    tok = AksharamTokenizer.train(
        SAMPLES, vocab_size=len(default_aksharam_inventory()) + 128)
    for s in SAMPLES:
        assert tok.decode(tok.encode(s, add_special=True)) == s, f"round-trip: {s!r}"


def _run():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run()
