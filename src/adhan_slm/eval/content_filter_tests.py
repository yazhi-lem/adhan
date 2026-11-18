"""Tests for the versioned content exclusion filter (adhan#112, K6).

Synthetic fixtures only; no real corpus text is read or stored here.

Run: PYTHONPATH=src python -m adhan_slm.eval.content_filter_tests
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on path

from adhan_slm.core.selftest import run_module_tests  # noqa: E402
from adhan_slm.eval import content_filter as cf  # noqa: E402

_BLOCKLIST = {
    "version": "test",
    "native_review": "pending",
    "categories": {
        "politics": {
            "latin": ["election", "vote", "dmk", "chief minister"],
            "tamil": ["தேர்தல்"],
        },
        "violence": {"latin": ["kill", "murder"], "tamil": ["கொலை"]},
    },
}


def _filter(raw=None):
    return cf.ContentFilter.from_dict(raw or _BLOCKLIST)


def _write_blocklist(tmp: Path, raw) -> Path:
    path = tmp / "blocklist.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return path


def test_latin_terms_match_as_whole_words_only():
    f = _filter()
    assert f.blocked("bro election la vote podu") == "politics"
    # A substring inside a longer word must not fire.
    assert f.blocked("this devotee is devoted") is None
    assert f.blocked("naan romba santhosham ah iruken") is None


def test_matching_is_case_insensitive():
    f = _filter()
    assert f.blocked("DMK and admk") == "politics"
    assert f.blocked("Election time") == "politics"


def test_multi_word_terms_allow_any_whitespace():
    f = _filter()
    assert f.blocked("the chief   minister said") == "politics"


def test_tamil_terms_match_as_substrings():
    f = _filter()
    assert f.blocked("இந்த தேர்தல் முடிந்தது") == "politics"


def test_first_matching_category_wins_in_file_order():
    f = _filter()
    assert f.blocked("election kill") == "politics"
    assert f.blocked("kill in election") == "politics"
    assert f.blocked("murder case") == "violence"


def test_clean_sentence_is_not_blocked():
    f = _filter()
    assert f.blocked("bro office ku late ah vanthen romba tired ah iruku") is None


def test_blocked_category_alias_matches_blocked():
    f = _filter()
    assert f.blocked_category("murder") == f.blocked("murder") == "violence"


def test_no_filter_is_inert():
    f = cf.ContentFilter.none()
    assert f.active is False
    assert f.blocked("election murder") is None
    assert f.describe()["version"] == cf.NO_FILTER


def test_describe_records_version_hash_and_categories():
    with tempfile.TemporaryDirectory() as d:
        path = _write_blocklist(Path(d), _BLOCKLIST)
        f = cf.ContentFilter.load(path)
        desc = f.describe()
        assert desc["version"] == "test"
        assert desc["categories"] == ["politics", "violence"]
        assert desc["native_review"] == "pending"
        assert len(desc["sha256"]) == 64
        assert f.source_path == path


def test_load_rejects_missing_file():
    try:
        cf.ContentFilter.load(Path("/nonexistent/blocklist.json"))
    except cf.ContentFilterError as exc:
        assert "cannot read" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ContentFilterError")


def test_load_rejects_invalid_json():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        try:
            cf.ContentFilter.load(path)
        except cf.ContentFilterError as exc:
            assert "not valid JSON" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected ContentFilterError")


def test_from_dict_requires_version_and_categories():
    for raw, fragment in (
        ({"categories": {"a": {}}}, "version"),
        ({"version": "x"}, "categories"),
        ({"version": "x", "categories": {}}, "categories"),
        ({"version": "x", "categories": {"a": {"latin": ["ok", " "]}}}, "empty"),
    ):
        try:
            cf.ContentFilter.from_dict(raw)
        except cf.ContentFilterError as exc:
            assert fragment in str(exc), (fragment, str(exc))
        else:  # pragma: no cover
            raise AssertionError(f"expected ContentFilterError for {raw}")


def test_shipped_blocklist_loads_and_is_well_formed():
    """The committed v1 gazetteer must parse and veto the obvious cases."""
    f = cf.load_named("v1")
    assert f.version == "v1"
    assert f.active
    assert f.native_review == "pending"
    assert "politics" in f.category_names()

    assert f.blocked("bro election la vote podu da") in ("politics", "abuse")
    assert f.blocked("naan office ku poren") is None


def test_latin_boundary_does_not_fire_mid_word():
    f = _filter()
    # "voter" is not listed; the bare "vote" term must not match inside it.
    assert f.blocked("he is a voter") is None


if __name__ == "__main__":
    raise SystemExit(run_module_tests(globals()))
