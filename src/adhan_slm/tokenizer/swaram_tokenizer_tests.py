"""Tests for the Swaram (Tamil/Dravidian) akshara tokenizer.

Pure python — no pytest/numpy/JAX needed, matching the tokenizer core itself.

Run: PYTHONPATH=src python -m adhan_slm.tokenizer.swaram_tokenizer_tests
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on path

from adhan_slm.core.selftest import run_module_tests  # noqa: E402
from adhan_slm.tokenizer import (  # noqa: E402
    SwaramTokenizer,
    default_akshara_inventory,
    segment_aksharas,
)

SAMPLES = [
    "தமிழ்",
    "கற்போம்",
    "படித்துக்கொண்டிருந்தேன்",
    "வணக்கம் நண்பர்களே",
    "ஆதன் — தமிழ் முதல்",
    "Tamil AI 2026 model",  # code-switch
    "எண்: ௧௨௩ / 123",  # Tamil + ASCII digits
]


def test_segmentation_is_lossless():
    for s in SAMPLES:
        assert "".join(segment_aksharas(s)) == s, f"lossy segmentation: {s!r}"


def test_known_akshara_split():
    # கற்போம் -> க | ற் | போ | ம்
    assert segment_aksharas("கற்போம்") == ["க", "ற்", "போ", "ம்"]
    # pure vowel + uyirmey
    assert segment_aksharas("அப்பா") == ["அ", "ப்", "பா"]


def test_inventory_is_closed_and_deduped():
    inv = default_akshara_inventory()
    assert len(inv) == len(set(inv))
    assert "க" in inv and "க்" in inv and "கி" in inv and "ஃ" in inv
    # 12 uyir + aytham + 23 consonants*(1 inherent + 1 mey + 12 matras)
    assert len(inv) > 250


def test_encode_decode_round_trip():
    tok = SwaramTokenizer.train(SAMPLES, vocab_size=len(default_akshara_inventory()) + 128)
    for s in SAMPLES:
        ids = tok.encode(s, add_special=True)
        assert tok.decode(ids) == s, f"round-trip failed: {s!r}"


def test_fertility_at_most_one_before_merges_helps():
    # A tokenizer with no merges emits exactly one token per akshara (fertility ~1.0).
    tok = SwaramTokenizer(
        vocab={
            t: i
            for i, t in enumerate(
                ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>", "▁"] + default_akshara_inventory()
            )
        }
    )
    f = tok.fertility("கற்போம்")
    assert 0.99 <= f <= 1.01, f"expected ~1 token/akshara, got {f}"


def test_tokenizer_module_logger_initialized():
    import adhan_slm.tokenizer.swaram_tokenizer as st
    assert hasattr(st, "logger")
    assert st.logger is not None
    assert st.logger.name == "adhan_slm.tokenizer.swaram_tokenizer"


if __name__ == "__main__":
    run_module_tests(globals(), "swaram tokenizer")
