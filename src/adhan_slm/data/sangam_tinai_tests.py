"""Tests for the Tiṇai canon loader.

These check the *contract* (all five akam tiṇais present, every dimension populated,
unknown input yields nulls rather than a guess). They deliberately do not assert
scholarly claims — the canon is data in a JSON file so a Tamil scholar can correct
it without touching code.
"""

from __future__ import annotations

import pytest

from adhan_slm.data.sangam_tinai import (
    canonical_tinai,
    labels_for,
    load_canon,
    profile_for,
    quantise_kalam,
)

AKAM = ("kurinji", "mullai", "marutam", "neytal", "palai")


def test_canon_covers_the_five_akam_tinais() -> None:
    assert set(AKAM) <= set(load_canon())


@pytest.mark.parametrize("name", AKAM)
def test_every_tinai_has_all_four_dimensions(name: str) -> None:
    prof = load_canon()[name]
    assert prof.tamil, "tinai needs its Tamil name"
    assert prof.kalam, "kalam (season) must be populated"
    assert prof.uripporul and prof.uripporul_tamil, "uripporul must be populated"
    assert prof.karupporul, "karupporul imagery must be populated"


def test_uripporul_differs_across_akam_tinais() -> None:
    """The five akam tiṇais are distinguished by their essential action — if any two
    shared one, the labels would collapse and the conditioning signal would be lost."""
    actions = [load_canon()[t].uripporul for t in AKAM]
    assert len(set(actions)) == len(AKAM)


def test_unknown_tinai_yields_nulls_not_a_guess() -> None:
    for raw in ("unknown", "", None, "puram-ish", "veṭci"):
        lab = labels_for(raw)
        assert lab["tinai"] is None
        assert lab["uripporul"] is None
        assert lab["karupporul"] == []


def test_tamil_and_transliterated_names_normalise() -> None:
    for raw in ("குறிஞ்சி", "Kurinji", " kurinji ", "kurinci"):
        assert canonical_tinai(raw) == "kurinji"


def test_attested_filter_drops_disputed_imagery() -> None:
    prof = profile_for("mullai")
    assert prof is not None
    everything = prof.karupporul_tamil(attested_only=False)
    attested = prof.karupporul_tamil(attested_only=True)
    assert set(attested) <= set(everything)
    assert len(attested) <= len(everything)


def test_quantise_kalam_prefers_primary_but_honours_alternates() -> None:
    palai = profile_for("palai")
    assert palai is not None
    assert quantise_kalam(palai, None) == palai.kalam[0]
    assert quantise_kalam(palai, "முன்பனி") == "munpani"
    assert quantise_kalam(palai, "not-a-season") == palai.kalam[0]


def test_labels_are_json_serialisable() -> None:
    import json

    for name in AKAM:
        json.dumps(labels_for(name))
