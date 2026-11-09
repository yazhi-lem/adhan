"""Tests for Tamil morphology: letter classes, sandhi rules, suffix boundaries.

The sandhi table is the interesting part. Every case below is a canonical
Tholkappiyam/Naṉṉūl example, so this file doubles as the rule documentation
that CI enforces — if a refactor breaks திருவடி or மரத்தோப்பு, it fails here.
"""

from __future__ import annotations

import pytest

from adhan_slm.tokenizer import morphology as m
from adhan_slm.tokenizer.swaram_tokenizer import WORD_MARK, SwaramTokenizer, segment_aksharas

# (left, right, expected surface, expected vikaram)
SANDHI_CASES = [
    # உயிர் + உயிர் — the உடம்படுமெய் glide choices
    ("பசி", "இல்லை", "பசியில்லை", m.Vikaram.THONRAL),  # front vowel → ய்
    ("கை", "இல்லை", "கையில்லை", m.Vikaram.THONRAL),  # ஐ is front → ய்
    ("பலா", "அடி", "பலாவடி", m.Vikaram.THONRAL),  # non-front → வ்
    # உயிர்மெய் + உயிர் — the "uyir-mei sandhi" case, carrier vowel kept
    ("திரு", "அடி", "திருவடி", m.Vikaram.THONRAL),
    # மெய் + உயிர்
    ("மரம்", "அடி", "மரவடி", m.Vikaram.VIKARAM),  # ம் → வ
    ("நீர்", "அளவு", "நீரளவு", m.Vikaram.IYALBU),  # no change
    ("தமிழ்", "அறிஞர்", "தமிழறிஞர்", m.Vikaram.IYALBU),
    # மெய் + வல்லினம் — doubling
    ("மரம்", "தோப்பு", "மரத்தோப்பு", m.Vikaram.MIGUTHAL),
    ("நீர்", "துளி", "நீர்த்துளி", m.Vikaram.MIGUTHAL),
    # மெய் + உயிர் — final consonant doubles
    ("கல்", "அடி", "கல்லடி", m.Vikaram.MIGUTHAL),
    ("முள்", "அடி", "முள்ளடி", m.Vikaram.MIGUTHAL),
    ("மண்", "அடி", "மண்ணடி", m.Vikaram.MIGUTHAL),
    ("நெய்", "அப்பம்", "நெய்யப்பம்", m.Vikaram.MIGUTHAL),
]


@pytest.mark.parametrize("left,right,expected,vikaram", SANDHI_CASES)
def test_sandhi_join(left: str, right: str, expected: str, vikaram: m.Vikaram) -> None:
    result = m.explain_join(left, right)
    assert result.surface == expected
    assert result.vikaram is vikaram


def test_glide_is_not_a_virama_consonant() -> None:
    """The inserted glide absorbs the next vowel; it must not stay வ் + அ."""
    assert m.join("திரு", "அடி") == "திருவடி"
    assert "்அ" not in m.join("திரு", "அடி")


def test_plain_concatenation_when_no_rule_matches() -> None:
    # ல் + ம (a mellinam, not vallinam) triggers nothing: the units just meet.
    assert m.join("கல்", "மலை") == "கல்மலை"


def test_final_akshara() -> None:
    assert m.final_akshara("மரம்") == "ம்"  # not the lone virama
    assert m.final_akshara("திரு") == "ரு"
    assert m.final_akshara("பலா") == "லா"
    assert m.final_akshara("") == ""


@pytest.mark.parametrize(
    "char,expected",
    [
        ("அ", m.LetterClass.UYIR),
        ("க்", m.LetterClass.MEI),
        ("கு", m.LetterClass.UYIRMEI),
        ("ஃ", m.LetterClass.AYTHAM),
        ("a", m.LetterClass.OTHER),
    ],
)
def test_letter_class(char: str, expected: m.LetterClass) -> None:
    assert m.letter_class(char) is expected


# --- suffix boundaries --------------------------------------------------------

SUFFIX_CASES = [
    ("மரங்கள்", "மரங்", "கள்"),  # surface stem: ம் → ங் before க்
    ("தமிழுக்கு", "தமிழு", "க்கு"),
    ("மரத்தில்", "மர", "த்தில்"),
    ("வீட்டில்", "வீட்ட", "ில்"),  # வீடு + இல், written with the matra
    ("புத்தகத்தை", "புத்தகத்த", "ை"),
]


@pytest.mark.parametrize("word,stem,suffix", SUFFIX_CASES)
def test_split_suffix(word: str, stem: str, suffix: str) -> None:
    got_stem, got_suffix = m.split_suffix(word)
    assert (got_stem, got_suffix) == (stem, suffix)
    assert got_stem + got_suffix == word  # split must be lossless


def test_unsuffixed_word_returns_empty_suffix() -> None:
    assert m.split_suffix("வணக்கம்") == ("வணக்கம்", "")


def test_longest_suffix_wins() -> None:
    """களுக்கு must not be read as கள் + உக்கு at the wrong offset."""
    stem, suffix = m.split_suffix("மரங்களுக்கு")
    assert suffix == "களுக்கு"
    assert stem + suffix == "மரங்களுக்கு"


# --- sandhi signature detection ----------------------------------------------

SANDHI_SIGNATURES = {
    "கல்லடி": [2],  # doubled ல்
    "திருவடி": [2],  # glide வ between vowels
    "பசியில்லை": [4],  # glide ய
    "நீர்த்துளி": [3],  # doubled த்
    "மரங்கள்": [],  # no sandhi applied
}


@pytest.mark.parametrize("word,points", SANDHI_SIGNATURES.items())
def test_sandhi_points(word: str, points: list[int]) -> None:
    assert m.sandhi_points(word, segment_aksharas) == points


# --- tokenizer integration ---------------------------------------------------

CORPUS = ["மரங்கள் வளர்கின்றன", "திருவடி பசியில்லை", "தமிழ் மொழி வாழ்க"] * 4


@pytest.fixture(scope="module")
def tok() -> SwaramTokenizer:
    return SwaramTokenizer.train(CORPUS, vocab_size=800, min_freq=1)


def test_tokenize_spans_are_exact_and_contiguous(tok: SwaramTokenizer) -> None:
    text = "மரங்கள் வளர்கின்றன"
    spans = tok.tokenize_spans(text)
    assert "".join(x.piece for x in spans).replace(WORD_MARK, " ") == text
    assert spans[0].akshara_start == 0
    for prev, cur in zip(spans, spans[1:]):
        assert cur.akshara_start == prev.akshara_end
    assert spans[-1].akshara_end == len(segment_aksharas(text))


def test_span_akshara_ranges_match_piece_text(tok: SwaramTokenizer) -> None:
    text = "திருவடி பசியில்லை"
    aksharas = [WORD_MARK if a == " " else a for a in segment_aksharas(text)]
    for span in tok.tokenize_spans(text):
        assert "".join(aksharas[span.akshara_start : span.akshara_end]) == span.piece


def test_morphology_report_runs_on_tokenizer(tok: SwaramTokenizer) -> None:
    report = tok.morphology(["மரங்கள்"])
    assert report.n_with_suffix == 1
    assert 0.0 <= report.agreement_rate <= 1.0


def test_boundary_agreement_counts_perfect_split() -> None:
    class PerfectTokenizer:
        def tokenize(self, text: str) -> list[str]:
            stem, suffix = m.split_suffix(text)
            return [stem, suffix] if suffix else [text]

    report = m.boundary_agreement(PerfectTokenizer(), ["மரங்கள்", "தமிழுக்கு"])
    assert report.agreement_rate == 1.0


def test_boundary_agreement_detects_breakage() -> None:
    class WholeWordTokenizer:
        def tokenize(self, text: str) -> list[str]:
            return [text]

    report = m.boundary_agreement(WholeWordTokenizer(), ["மரங்கள்"])
    assert report.agreement_rate == 0.0
    assert report.examples == ["மரங்கள்"]
