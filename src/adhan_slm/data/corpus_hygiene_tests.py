"""Tests for the canonical corpus hygiene module (adhan#125).

Two jobs:

1. Pin the behaviour of the v1 and v2 configurations, so consolidating the duplicated helpers did
   not quietly change which documents pass the gate.
2. Pin the *absence* of duplication, so the copies cannot come back. That is the actual acceptance
   criterion of #125, and a test is the only thing that keeps it true.

Run: PYTHONPATH=src python -m adhan_slm.data.corpus_hygiene_tests
"""

import ast
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on path

from adhan_slm.core.selftest import run_module_tests  # noqa: E402
from adhan_slm.data import corpus_hygiene as ch  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]

_TAMIL = "தமிழ் மொழி உலகின் மிகத் தொன்மையான மொழிகளில் ஒன்றாகும். "


# -- hashing ------------------------------------------------------------------------


def test_h64_is_deterministic_and_64_bit():
    a = ch.h64(b"adhan")
    assert a == ch.h64(b"adhan")
    assert 0 <= a < 2**64
    assert ch.h64(b"adhan") != ch.h64(b"adhan ")


def test_sha256_hex_matches_hashlib():
    assert ch.sha256_hex("adhan") == hashlib.sha256(b"adhan").hexdigest()


def test_sha256_hex_ignore_errors_preserves_the_extractor_behaviour():
    """pdf_book_scraper hashed with errors='ignore'; that must not silently change."""
    text = "ok\ud800bad"
    assert (
        ch.sha256_hex(text, errors="ignore")
        == hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    )
    try:
        ch.sha256_hex(text)
    except UnicodeEncodeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("strict mode should reject a lone surrogate")


# -- text hygiene -------------------------------------------------------------------


def test_normalize_scrubs_pii_and_collapses_whitespace():
    out = ch.normalize("பார் https://x.com/a  mail@example.com  +919876543210   நன்றி")
    assert "https://" not in out and "mail@example.com" not in out and "9876543210" not in out
    assert "  " not in out
    assert out.startswith("பார்") and out.endswith("நன்றி")


def test_normalize_is_nfc():
    import unicodedata

    nfd = unicodedata.normalize("NFD", "கொலை")
    assert ch.normalize(nfd) == unicodedata.normalize("NFC", nfd)


def test_tamil_ratio_default_ignores_whitespace():
    assert ch.tamil_ratio("தமிழ் தமிழ்") == 1.0
    # 5 Tamil characters out of 8 non-whitespace characters.
    assert ch.tamil_ratio("தமிழ் abc") == 5 / 8


def test_tamil_ratio_raw_mode_preserves_the_old_sangam_measurement():
    """ingest_open_sangam divided by the unstripped length; that variant is still available."""
    text = "தமிழ்   "
    assert ch.tamil_ratio(text) == 1.0
    assert ch.tamil_ratio(text, strip_whitespace=False) < 1.0


def test_tamil_ratio_of_empty_text_is_zero():
    assert ch.tamil_ratio("") == 0.0
    assert ch.tamil_ratio("   ") == 0.0


# -- quality gate: v1 vs v2 ---------------------------------------------------------


def test_v1_and_v2_presets_keep_their_original_tunables():
    assert (ch.V1_QUALITY.max_chars, ch.V1_QUALITY.min_tamil_ratio) == (20000, 0.60)
    assert (ch.V2_QUALITY.max_chars, ch.V2_QUALITY.min_tamil_ratio) == (50000, 0.55)
    assert ch.V1_QUALITY.boilerplate == ()
    assert ch.V2_QUALITY.boilerplate == ch.BOILERPLATE


def test_v2_accepts_a_document_v1_rejects_on_length():
    text = _TAMIL * 400  # ~24k chars: over v1's 20k, under v2's 50k
    assert 20000 < len(text) < 50000
    assert ch.is_quality(text, ch.V1_QUALITY) is False
    assert ch.is_quality(text, ch.V2_QUALITY) is True


def test_v2_rejects_boilerplate_that_v1_accepts():
    # Enough Tamil to clear the ratio gate, so only the boilerplate rule can differ.
    text = _TAMIL * 3 + " javascript cookie privacy policy terms of use"
    assert ch.is_quality(text, ch.V1_QUALITY) is True
    assert ch.is_quality(text, ch.V2_QUALITY) is False


def test_both_presets_reject_short_low_tamil_and_repetitive_text():
    for config in (ch.V1_QUALITY, ch.V2_QUALITY):
        assert ch.is_quality("too short", config) is False
        assert ch.is_quality("english text only, no tamil here at all", config) is False
        assert ch.is_quality(_TAMIL + "aaaaaaaa", config) is False


def test_quality_config_overrides_are_immutable():
    looser = ch.V1_QUALITY.with_overrides(min_chars=1)
    assert looser.min_chars == 1
    assert ch.V1_QUALITY.min_chars == 40
    assert ch.is_quality("தமிழ்", looser) is True


# -- near-duplicate detection -------------------------------------------------------


def test_lsh_flags_exact_and_near_duplicates():
    dedup = ch.LshDedup()
    assert dedup.is_duplicate(_TAMIL) is False
    assert dedup.is_duplicate(_TAMIL) is True
    near = _TAMIL + "கூடுதல் வாக்கியம்."
    assert dedup.is_duplicate(near) is True


def test_lsh_keeps_unrelated_documents():
    dedup = ch.LshDedup()
    dedup.is_duplicate(_TAMIL)
    assert dedup.is_duplicate("முற்றிலும் வேறு ஒரு வாக்கியம் இது.") is False


def test_lsh_threshold_is_honoured():
    """v1 used 0.80 and v2 0.82; the threshold must actually change the verdict."""
    text = _TAMIL
    near = _TAMIL[:-20] + "வேறு முடிவு."
    strict = ch.LshDedup(threshold=0.99)
    loose = ch.LshDedup(threshold=0.10)
    strict.is_duplicate(text)
    loose.is_duplicate(text)
    assert strict.is_duplicate(near) is False
    assert loose.is_duplicate(near) is True


def test_lsh_rejects_inconsistent_banding():
    try:
        ch.LshDedup(num_perm=16, bands=5)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("accepted num_perm not divisible by bands")


def test_seed_exact_marks_a_document_without_a_signature():
    dedup = ch.LshDedup()
    dedup.seed_exact(_TAMIL)
    assert dedup.is_duplicate(_TAMIL) is True
    assert dedup.sigs == {}


# -- misc ---------------------------------------------------------------------------


def test_iter_strings_walks_nested_structures_in_order():
    obj = {"a": "one", "b": ["two", {"c": "three"}], "d": 4}
    assert list(ch.iter_strings(obj)) == ["one", "two", "three"]


# -- the acceptance criterion: no duplicate bodies ----------------------------------


def _defined_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def test_no_duplicate_helper_bodies_remain_in_scripts_or_extractors():
    """#125: one canonical implementation, thin wrappers only."""
    watched = {
        "_h64",
        "_shingles",
        "_signature",
        "_sig_similarity",
        "_iter_strings",
        "tamil_ratio",
        "sha256_hex",
    }
    offenders = []
    for path in list((ROOT / "scripts").glob("*.py")) + list(
        (ROOT / "src" / "data_scraper").rglob("*.py")
    ):
        for name in _defined_functions(path) & watched:
            body = ast.get_source_segment(path.read_text(encoding="utf-8"), _find(path, name)) or ""
            # A thin wrapper delegates; a duplicate body implements the logic itself.
            if "return _" not in body and "yield from" not in body:
                offenders.append(f"{path.relative_to(ROOT)}:{name}")
    assert offenders == [], f"duplicate implementations: {offenders}"


def _find(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def test_the_canonical_module_is_the_only_implementation():
    """The helpers must exist exactly once, in corpus_hygiene."""
    canonical = _defined_functions(ROOT / "src" / "adhan_slm" / "data" / "corpus_hygiene.py")
    for name in ("h64", "sha256_hex", "normalize", "tamil_ratio", "is_quality", "iter_strings"):
        assert name in canonical, name
    assert "LshDedup" in {
        n.name
        for n in ast.walk(ast.parse((ROOT / "src/adhan_slm/data/corpus_hygiene.py").read_text()))
        if isinstance(n, ast.ClassDef)
    }


if __name__ == "__main__":
    raise SystemExit(run_module_tests(globals()))
