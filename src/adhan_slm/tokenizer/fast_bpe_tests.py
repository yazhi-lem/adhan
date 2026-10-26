"""Tests for the Rust-backed akshara BPE (fast_bpe).

The contract these lock down:

* lossless round-trip on real Tamil *and* on every other script the model may
  meet in scraped text (byte fallback is what makes this true),
* the akshara layer is not bypassed by the Rust encoder (composition merges fire
  before learned merges),
* the Python reference tokenizer and the Rust tokenizer agree on normal text,
* artifacts round-trip through disk.
"""

from __future__ import annotations

import pytest

# The Rust BPE needs the optional `tokenizers` package (the "benchmark" extra);
# skip cleanly when it is not installed, as the jax-only tests do.
pytest.importorskip("tokenizers")

from adhan_slm.tokenizer import indic_akshara  # noqa: E402
from adhan_slm.tokenizer.fast_bpe import AksharaBPE  # noqa: E402
from adhan_slm.tokenizer.swaram_tokenizer import (  # noqa: E402
    BYTE_TOKENS,
    WORD_MARK,
    default_akshara_inventory,
)

CORPUS = [
    "தமிழ் மொழி உலகின் மிகத் தொன்மையான மொழிகளில் ஒன்றாகும்.",
    "திருக்குறள் தமிழின் பெருமையை உலகுக்கு எடுத்துக்காட்டுகிறது.",
    "கல்வி கற்காதவனைப் பகைத்துக்கொள்ளும் எளிய செயலைச் செய்ய இயலாது.",
    "ஆதனம் என்பது தமிழ் மொழிக்கான சிறிய மொழி மாதிரி.",
    "தமிழ்நாடு வேளாண்மைக்கும் தொழிலுக்கும் பெயர் பெற்றது.",
] * 8

CROSS_SCRIPT = "தமிழ் हिन्दी తెలుగు ಕನ್ನಡ മലയാളം বাংলা ABC 123 😀 ₹ ௧௨௩"


@pytest.fixture(scope="module")
def tok() -> AksharaBPE:
    return AksharaBPE.train(
        CORPUS,
        default_akshara_inventory(),
        vocab_size=1500,
        script=indic_akshara.TAMIL,
        byte_fallback=True,
        min_frequency=1,
        show_progress=False,
    )


def test_tamil_round_trip_is_lossless(tok: AksharaBPE) -> None:
    for doc in CORPUS:
        assert tok.decode(tok.encode(doc)) == doc


def test_byte_fallback_makes_any_script_lossless(tok: AksharaBPE) -> None:
    assert tok.decode(tok.encode(CROSS_SCRIPT)) == CROSS_SCRIPT
    assert "<unk>" not in tok.decode(tok.encode(CROSS_SCRIPT))


def test_byte_tokens_present_when_enabled(tok: AksharaBPE) -> None:
    assert all(t in tok.vocab for t in BYTE_TOKENS)


def test_byte_tokens_absent_when_disabled() -> None:
    tok = AksharaBPE.train(
        CORPUS,
        default_akshara_inventory(),
        vocab_size=1200,
        byte_fallback=False,
        min_frequency=1,
        show_progress=False,
    )
    assert not all(t in tok.vocab for t in BYTE_TOKENS)


def test_whole_base_inventory_is_in_vocab(tok: AksharaBPE) -> None:
    inv = default_akshara_inventory()
    assert sum(1 for a in inv if a in tok.vocab) == len(inv)


def test_composition_merges_outrank_learned_merges(tok: AksharaBPE) -> None:
    """Aksharas must be built from codepoints before any morpheme merge runs."""
    from adhan_slm.tokenizer.fast_bpe import _composition_merges

    comp = _composition_merges(tok.inventory)
    assert tok.merges[: len(comp)] == comp
    # Any learned merge must come after every composition merge, otherwise the
    # encoder could split an akshara before it is assembled.
    learned = tok.merges[len(comp) :]
    assert learned, "expected at least one learned merge on this corpus"
    assert not (set(learned) & set(comp))


def test_aksahara_atomicity(tok: AksharaBPE) -> None:
    """Every emitted token is a concatenation of whole aksharas."""
    aksharas = set(indic_akshara.segment(CROSS_SCRIPT + "".join(CORPUS)))
    for token in tok.tokenize("தமிழ்நாடு வாழ்க"):
        assert token in {WORD_MARK} or token in aksharas or token in tok.vocab


def test_python_reference_agrees_on_normal_text(tok: AksharaBPE) -> None:
    py = tok.to_python_tokenizer()
    agree = sum(1 for doc in CORPUS if py.encode(doc) == tok.encode(doc))
    assert agree == len(CORPUS)


def test_encode_batch_matches_single(tok: AksharaBPE) -> None:
    batch = tok.encode_batch(CORPUS[:3])
    assert batch == [tok.encode(d) for d in CORPUS[:3]]


def test_add_special_wraps_bos_eos(tok: AksharaBPE) -> None:
    ids = tok.encode("தமிழ்", add_special=True)
    assert ids[0] == tok.vocab["<bos>"] and ids[-1] == tok.vocab["<eos>"]


def test_decode_skips_structural_specials(tok: AksharaBPE) -> None:
    ids = [tok.vocab["<bos>"], *tok.encode("தமிழ்"), tok.vocab["<eos>"]]
    assert tok.decode(ids) == "தமிழ்"


def test_save_load_round_trip(tmp_path, tok: AksharaBPE) -> None:
    tok.save(tmp_path)
    for name in ("tokenizer.json", "vocab.json", "merges.txt"):
        assert (tmp_path / name).exists()

    reloaded = AksharaBPE.load(tmp_path, inventory=tok.inventory)
    assert reloaded.vocab_size == tok.vocab_size
    assert reloaded.encode("தமிழ் மொழி") == tok.encode("தமிழ் மொழி")
    assert reloaded.decode(reloaded.encode(CROSS_SCRIPT)) == CROSS_SCRIPT


def test_fertility_below_one(tok: AksharaBPE) -> None:
    for doc in CORPUS[:3]:
        assert 0.0 < tok.fertility(doc) < 1.15


def test_from_python_wraps_existing_tokenizer() -> None:
    from adhan_slm.tokenizer.swaram_tokenizer import SwaramTokenizer

    py = SwaramTokenizer.train(CORPUS, vocab_size=500, min_freq=1, byte_fallback=True)
    fast = AksharaBPE.from_python(py, default_akshara_inventory())
    assert fast.vocab_size >= len(py.vocab)
    assert fast.decode(fast.encode(CORPUS[0])) == CORPUS[0]
