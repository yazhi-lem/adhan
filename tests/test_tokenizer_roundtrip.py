"""Tokenizer round-trip property tests.

The swaram tokenizer is the foundation of everything Adhan does; if
``decode(encode(text)) != text`` then training targets are silently corrupted.
These tests fuzz the encoder/decoder over generated akshara strings and real
Tamil sentences to guard that invariant.
"""

from __future__ import annotations

import random

import pytest

from adhan_slm.tokenizer import SwaramTokenizer
from adhan_slm.tokenizer.swaram_tokenizer import default_akshara_inventory

SAMPLE_CORPUS = [
    "தமிழ் மொழி மிகவும் தொன்மையானது.",
    "சங்க இலக்கியம் நமது பெருமை.",
    "அம்மா வீட்டில் இருக்கிறாள்.",
    "கடற்கரையில் காற்று வீசியது.",
    "படித்துக்கொண்டிருந்தேன்.",
    "திருக்குறள் அறத்தைப் போதிக்கிறது.",
]


@pytest.fixture(scope="module")
def tokenizer() -> SwaramTokenizer:
    # byte_fallback=True mirrors scripts/freeze_fast_tokenizer.py, whose default is
    # byte fallback on. Without it, characters absent from the training corpus
    # (punctuation, digits, rare aksharas) collapse to <unk> and break round-trip.
    return SwaramTokenizer.train(
        SAMPLE_CORPUS, vocab_size=2048, min_freq=1, byte_fallback=True
    )


@pytest.mark.parametrize("text", SAMPLE_CORPUS)
def test_roundtrip_real_sentences(tokenizer, text):
    ids = tokenizer.encode(text, add_special=False)
    assert tokenizer.decode(ids) == text


def test_roundtrip_full_akshara_inventory(tokenizer):
    """Every base akshara must survive encode→decode."""
    for akshara in default_akshara_inventory():
        ids = tokenizer.encode(akshara, add_special=False)
        assert tokenizer.decode(ids) == akshara, f"round-trip failed for {akshara!r}"


def test_roundtrip_generated_strings(tokenizer):
    """Fuzz: random strings built from the closed akshara inventory."""
    rng = random.Random(1234)
    inventory = default_akshara_inventory()
    for _ in range(200):
        n = rng.randint(1, 12)
        text = "".join(rng.choice(inventory) for _ in range(n))
        ids = tokenizer.encode(text, add_special=False)
        assert tokenizer.decode(ids) == text, f"round-trip failed for {text!r}"


def test_special_tokens_roundtrip(tokenizer):
    text = "வணக்கம்"
    ids = tokenizer.encode(text, add_special=True)
    assert ids[0] == tokenizer.vocab["<bos>"]
    assert ids[-1] == tokenizer.vocab["<eos>"]
    # skip_special (default) strips <bos>/<eos>
    assert tokenizer.decode(ids) == text


def test_roundtrip_with_punctuation_and_spaces(tokenizer):
    text = "அவன் வந்தான், நான் சென்றேன்; மழை பெய்தது!"
    ids = tokenizer.encode(text, add_special=False)
    assert tokenizer.decode(ids) == text


def test_fertility_is_reasonable(tokenizer):
    """Sanity: fertility should be near 1 token/akshara for a trained vocab."""
    for text in SAMPLE_CORPUS:
        f = tokenizer.fertility(text)
        assert 0.0 < f < 3.0, f"implausible fertility {f} for {text!r}"
