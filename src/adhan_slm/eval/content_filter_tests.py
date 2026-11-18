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
            "latin_stems": ["arasiyal"],
            "tamil": ["தேர்தல்", "அமைச்சர்", "சாதி"],
        },
        "abuse": {
            "latin": [],
            "latin_stems": ["pundai", "mayir"],
            "tamil": ["புண்ட"],
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
        assert desc["categories"] == ["politics", "abuse", "violence"]
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
    f = cf.load_named("v2")
    assert f.version == "v2"
    assert f.active
    assert f.native_review == "pending"
    assert "politics" in f.category_names()

    assert f.blocked("bro election la vote podu da") in ("politics", "abuse")
    assert f.blocked("naan office ku poren") is None


def test_latin_stems_accept_tanglish_case_suffixes():
    """Tanglish inflects with a small closed suffix set; the stems must cover it."""
    f = _filter()
    assert f.blocked("bro pundaiku pesuran") == "abuse"
    assert f.blocked("mayirum theriyala") == "abuse"
    assert f.blocked("arasiyaluku time illa") == "politics"


def test_latin_stem_suffixes_are_closed_not_open():
    """An open suffix would drag in unrelated words; only the listed markers may follow."""
    f = _filter()
    assert f.blocked("pundaikaran") is None  # not a listed suffix
    assert f.blocked("arasiyalism") is None


def test_tamil_terms_match_inflected_forms():
    """Tamil drops the pulli before a case suffix: தேர்தல் -> தேர்தலில்."""
    f = _filter()
    assert f.blocked("இந்த தேர்தலில் யாரு வருவாங்க") == "politics"
    assert f.blocked("அமைச்சரை பாத்திங்களா") == "politics"


def test_tamil_matching_does_not_fire_inside_longer_words():
    """The reviewer's false positives: சாதிக்க and புண்டரீகாட்சன் must stay clean."""
    f = _filter()
    assert f.blocked("நல்லா சாதிக்கணும்") is None
    assert f.blocked("புண்டரீகாட்சன் கோவிலுக்கு போனான்") is None
    # ...while the real forms still match.
    assert f.blocked("சாதி பாக்குறாங்க") == "politics"
    assert f.blocked("புண்டை") == "abuse"


def test_decomposed_input_is_blocked_after_nfc_normalisation():
    """A caller must not be able to slip past by passing NFD text."""
    import unicodedata

    f = _filter()
    nfd = unicodedata.normalize("NFD", "கொலை நடந்தது")
    assert nfd != unicodedata.normalize("NFC", nfd), "fixture must actually decompose"
    assert f.blocked(nfd) == "violence"


def test_terms_are_nfc_normalised_when_loaded():
    import unicodedata

    raw = {
        "version": "test",
        "categories": {"politics": {"tamil": [unicodedata.normalize("NFD", "தேர்தல்")]}},
    }
    f = cf.ContentFilter.from_dict(raw)
    assert f.blocked("தேர்தல் நாளைக்கு") == "politics"


def test_patterns_are_compiled_once_not_per_call():
    """The regex used to be rebuilt on every call (~4.7x slower)."""
    f = _filter()
    for cat in f.categories:
        assert cat._latin_re is not None or cat._tamil_re is not None
    before = [id(cat._latin_re) for cat in f.categories]
    for _ in range(3):
        f.blocked("election murder pundaiku")
    assert [id(cat._latin_re) for cat in f.categories] == before


def test_latin_boundary_does_not_fire_mid_word():
    f = _filter()
    # "voter" is not listed; the bare "vote" term must not match inside it.
    assert f.blocked("he is a voter") is None


if __name__ == "__main__":
    raise SystemExit(run_module_tests(globals()))
