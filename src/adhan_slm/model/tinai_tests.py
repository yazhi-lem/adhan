"""Tests for the pure-Python tiṇai vocabulary and weak labeler (no JAX needed)."""

import pytest

from adhan_slm.model.tinai import (
    N_TINAI,
    TINAI,
    TINAI_KEYWORDS,
    UNSPECIFIED,
    label_text,
    tinai_index,
)


def test_five_tinai_with_unique_names_in_fixed_order():
    assert N_TINAI == 5
    assert [t.key for t in TINAI] == ["kurinji", "mullai", "marutham", "neithal", "paalai"]
    assert len({t.tamil for t in TINAI}) == 5
    assert set(TINAI_KEYWORDS) == {t.key for t in TINAI}


def test_tinai_index_accepts_ascii_and_tamil_and_rejects_typos():
    assert tinai_index("mullai") == 1
    assert tinai_index(" Mullai ") == 1
    assert tinai_index("முல்லை") == 1
    assert tinai_index("பாலை") == 4
    with pytest.raises(KeyError):
        tinai_index("mulai")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("மலையில் அருவி பாய்கிறது", 0),  # hill + waterfall -> kurinji
        ("கடற்கரையில் பரதவர் வலை வீசினர்", 3),  # seashore + fisherfolk -> neithal
        ("வயலில் உழவர் நெல் விதைத்தார்", 2),  # field + farmers + paddy -> marutham
        ("வெயில் வறண்ட பாலை வழியே சென்றான்", 4),  # sun + dry + paalai -> paalai
    ],
)
def test_label_text_picks_the_dominant_landscape(text, expected):
    assert label_text(text) == expected


def test_label_text_never_guesses():
    assert label_text("") == UNSPECIFIED
    assert label_text("இது ஒரு சாதாரண வாக்கியம்") == UNSPECIFIED  # no landscape cue
    assert label_text("மலை கடல்") == UNSPECIFIED  # one hit each -> tie
