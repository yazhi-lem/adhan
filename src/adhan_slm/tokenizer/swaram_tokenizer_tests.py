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


# Trained on Tamil-only text, so English letters are out-of-vocabulary. (The
# round-trip test above trains ON SAMPLES, which include the English/digit
# strings, so it can never exercise the out-of-vocabulary path.)
_TAMIL_ONLY = ["இது ஒரு சோதனை வாக்கியம்", "மற்றொரு தமிழ் வாக்கியம் இது"]
_OOV_TEXT = "இது AI தான்"


def test_decode_shows_unk_by_default():
    # Out-of-vocabulary input must never vanish silently: <unk> stays visible
    # so lossy encoding can't masquerade as lossless.
    tok = SwaramTokenizer.train(_TAMIL_ONLY, vocab_size=200)
    ids = tok.encode(_OOV_TEXT)
    assert tok.unk_id in ids, "test setup: expected out-of-vocabulary tokens"
    assert "<unk>" in tok.decode(ids)


def test_decode_hide_unk_opt_out():
    tok = SwaramTokenizer.train(_TAMIL_ONLY, vocab_size=200)
    ids = tok.encode(_OOV_TEXT)
    assert "<unk>" not in tok.decode(ids, hide_unk=True)


def test_decode_still_hides_structural_specials():
    # <bos>/<eos>/<pad>/<mask> carry no content, so they stay hidden by default.
    tok = SwaramTokenizer.train(_TAMIL_ONLY, vocab_size=200)
    ids = tok.encode("இது", add_special=True)
    assert tok.decode(ids) == "இது"
    assert "<bos>" in tok.decode(ids, skip_special=False)


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


def _toy_corpus():
    words = ["படித்துக்கொண்டிருந்தேன்", "வணக்கம்", "நண்பர்களே", "தமிழ்", "கற்போம்"]
    return [" ".join(words[i % 5] for i in range(j, j + 6)) for j in range(300)]


def test_train_is_deterministic():
    a = SwaramTokenizer.train(_toy_corpus(), vocab_size=800)
    b = SwaramTokenizer.train(_toy_corpus(), vocab_size=800)
    assert a.vocab == b.vocab and a.merges == b.merges


def test_train_first_merge_is_the_most_frequent_pair():
    from collections import Counter

    corpus = _toy_corpus()
    tok = SwaramTokenizer.train(corpus, vocab_size=800)
    probe = SwaramTokenizer()
    pairs = Counter()
    for line in corpus:
        for w in line.split(" "):
            pieces = probe._pretokenize(w)
            pairs.update(zip(pieces, pieces[1:]))
    assert pairs[tok.merges[0]] == max(pairs.values())


def test_train_respects_vocab_size_and_min_freq():
    corpus = _toy_corpus()
    base = len(SwaramTokenizer.train(corpus, vocab_size=1))
    tok = SwaramTokenizer.train(corpus, vocab_size=base + 5)
    assert len(tok) == base + 5 and len(tok.merges) == 5
    assert len(SwaramTokenizer.train(corpus, vocab_size=800, min_freq=10**9).merges) == 0


def test_train_scales_to_many_merges_quickly():
    import time

    t0 = time.perf_counter()
    tok = SwaramTokenizer.train(_toy_corpus() * 20, vocab_size=2000)
    assert time.perf_counter() - t0 < 5.0
    text = "படித்துக்கொண்டிருந்தேன் வணக்கம்"
    assert tok.decode(tok.encode(text)) == text


if __name__ == "__main__":
    run_module_tests(globals(), "swaram tokenizer")

# -- byte fallback (issue #83) ------------------------------------------------

_BF_CORPUS = ["அவன் வீட்டுக்கு சென்றான்", "அவள் பள்ளிக்கு சென்றாள்"] * 5


def _train_byte_fallback(byte_fallback=True):
    from adhan_slm.tokenizer.swaram_tokenizer import SwaramTokenizer

    return SwaramTokenizer.train(
        _BF_CORPUS, vocab_size=700, min_freq=2, byte_fallback=byte_fallback
    )


def test_byte_fallback_roundtrips_numerals_english_and_emoji():
    import unicodedata

    tok = _train_byte_fallback()
    for text in ["௦௧௨௩௪௫௬௭௮௯", "இது ஒரு AI model தான்", "அவன் 2026 😀 சென்றான்"]:
        ids = tok.encode(text)
        assert tok.unk_id not in ids
        assert tok.decode(ids) == unicodedata.normalize("NFC", text)


def test_without_byte_fallback_out_of_vocab_is_still_unk():
    tok = _train_byte_fallback(byte_fallback=False)
    ids = tok.encode("௦")
    assert tok.unk_id in ids
    assert "<unk>" in tok.decode(ids)


def test_byte_fallback_survives_save_and_load(tmp_path):
    from adhan_slm.tokenizer.swaram_tokenizer import SwaramTokenizer

    tok = _train_byte_fallback()
    vocab_path, merges_path = tmp_path / "vocab.json", tmp_path / "merges.txt"
    tok.save(str(vocab_path), str(merges_path))
    loaded = SwaramTokenizer.from_files(str(vocab_path), str(merges_path))
    text = "AI ௧ சென்றான்"
    assert loaded.decode(loaded.encode(text)) == text
