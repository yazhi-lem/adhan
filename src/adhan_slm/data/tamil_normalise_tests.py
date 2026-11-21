"""Tests for Tamil normalisation and grapheme checks (`adhan_slm.data.tamil_normalise`).

Pure python, no numpy/JAX. The Tamil strings here are drafts that still need a
native reviewer's sign-off.

Run: PYTHONPATH=src python -m adhan_slm.data.tamil_normalise_tests
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adhan_slm.core.selftest import run_module_tests  # noqa: E402
from adhan_slm.data import tamil_normalise as tn  # noqa: E402
from adhan_slm.data._test_fixtures import TAMIL_LINES  # noqa: E402

# Decomposed input -> expected NFC output. These are the pairs a keyboard or an
# OCR/scrape step actually produces when it emits the two halves of a sign.
_NFC_PAIRS = [
    ("\u0b95\u0bc6\u0bbe", "\u0b95\u0bca"),  # க + ெ + ா -> கொ
    ("\u0b95\u0bc7\u0bbe", "\u0b95\u0bcb"),  # க + ே + ா -> கோ
    ("\u0b95\u0bc6\u0bd7", "\u0b95\u0bcc"),  # க + ெ + ௗ -> கௌ
    ("\u0b92\u0bd7", "\u0b94"),  # ஒ + ௗ -> ஔ
]


def test_normalise_composes_split_vowel_signs() -> None:
    for raw, want in _NFC_PAIRS:
        assert tn.normalise(raw) == want, (raw, want)


def test_normalise_is_idempotent_and_keeps_clean_text() -> None:
    for line in TAMIL_LINES:
        assert tn.normalise(line) == line
        assert tn.normalise(tn.normalise(line)) == tn.normalise(line)


def test_normalise_strips_invisibles_before_composing() -> None:
    # A zero-width space between the halves would otherwise block composition.
    assert tn.normalise("\ufeff\u0b95\u0bc6\u200b\u0bbe") == "\u0b95\u0bca"
    assert tn.normalise("தமி\u00adழ்\u2060") == "தமிழ்"


def test_normalise_keeps_joiners() -> None:
    text = "க்\u200dஷ"
    assert tn.normalise(text) == text


def test_graphemes_counts_tamil_letters() -> None:
    assert tn.graphemes("தமிழ்") == ["த", "மி", "ழ்"]
    assert tn.grapheme_count("கொ") == 1
    assert tn.graphemes("க்ஷ") == ["க்", "ஷ"]
    assert tn.grapheme_count("") == 0


def test_graphemes_tanglish_line() -> None:
    line = "naan school-ku போறேன்"
    clusters = tn.graphemes(line)
    assert "".join(clusters) == line
    assert clusters[-3:] == ["போ", "றே", "ன்"]
    assert tn.is_clean(line)


def test_clean_text_has_no_issues() -> None:
    for line in [*TAMIL_LINES, "hello world", "ஃ", "௧௨௩ 123", ""]:
        assert tn.find_issues(line) == [], line


def test_orphan_mark_at_start_and_after_space() -> None:
    issues = tn.find_issues("\u0bbeஅம்மா")
    assert [i.kind for i in issues] == [tn.IssueKind.ORPHAN_MARK]
    assert issues[0].offset == 0

    issues = tn.find_issues("அம்மா \u0bcdசாதம்")
    assert [i.kind for i in issues] == [tn.IssueKind.ORPHAN_MARK]
    assert issues[0].snippet == "\u0bcd"


def test_orphan_mark_after_latin() -> None:
    issues = tn.find_issues("a\u0bbf")
    assert [i.kind for i in issues] == [tn.IssueKind.ORPHAN_MARK]


def test_mark_on_independent_vowel_or_aytham() -> None:
    assert [i.kind for i in tn.find_issues("அ\u0bbe")] == [tn.IssueKind.MARK_ON_VOWEL]
    assert [i.kind for i in tn.find_issues("ஃ\u0bcd")] == [tn.IssueKind.MARK_ON_VOWEL]


def test_stacked_marks_on_consonant() -> None:
    issues = tn.find_issues("க\u0bbf\u0bc1")
    assert [i.kind for i in issues] == [tn.IssueKind.STACKED_MARKS]
    assert [i.kind for i in tn.find_issues("க\u0bcd\u0bbe")] == [tn.IssueKind.STACKED_MARKS]


def test_not_nfc_and_invisible_are_reported_on_original_text() -> None:
    issues = tn.find_issues("அ\u0b95\u0bc6\u0bbe")
    assert [i.kind for i in issues] == [tn.IssueKind.NOT_NFC]
    assert issues[0].offset == 2

    issues = tn.find_issues("தமிழ்\u200b")
    assert [(i.kind, i.offset, i.snippet) for i in issues] == [
        (tn.IssueKind.INVISIBLE_CHAR, 5, "U+200B")
    ]


def test_cluster_offsets_point_into_normalised_text() -> None:
    text = "\u200bஅம்மா அ\u0bbe"
    issues = [i for i in tn.find_issues(text) if i.kind is tn.IssueKind.MARK_ON_VOWEL]
    assert len(issues) == 1
    norm = tn.normalise(text)
    assert norm[issues[0].offset : issues[0].offset + 2] == "அ\u0bbe"


def test_safe_truncate_never_splits_a_letter() -> None:
    word = "தமிழ்"  # 5 code points, 3 letters
    assert tn.safe_truncate(word, 5) == word
    assert tn.safe_truncate(word, 4) == "தமி"
    assert tn.safe_truncate(word, 2) == "த"
    assert tn.safe_truncate(word, 0) == ""
    for line in TAMIL_LINES:
        for n in range(len(line) + 1):
            cut = tn.safe_truncate(line, n)
            assert len(cut) <= n
            assert line.startswith(cut)
            assert tn.is_clean(cut), (line, n, cut)


def test_safe_truncate_rejects_negative() -> None:
    try:
        tn.safe_truncate("அ", -1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_summarise_issues_counts_by_kind() -> None:
    counts = tn.summarise_issues([*TAMIL_LINES, "அ\u0bbe", "\u0bbe", "க\u0bc6\u0bbe"])
    assert counts["lines"] == len(TAMIL_LINES) + 3
    assert counts["dirty_lines"] == 3
    assert counts["mark_on_vowel"] == 1
    assert counts["orphan_mark"] == 1
    assert counts["not_nfc"] == 1
    assert counts["stacked_marks"] == 0


def test_main_passes_clean_corpus_and_fails_dirty_one() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        clean = root / "clean.txt"
        clean.write_text("\n".join(TAMIL_LINES) + "\n\n", encoding="utf-8")
        dirty = root / "dirty.jsonl"
        dirty.write_text(
            "\n".join(
                json.dumps(obj, ensure_ascii=False)
                for obj in ({"text": TAMIL_LINES[0]}, {"text": "அ\u0bbe"}, {"other": 1}, [1])
            )
            + "\n",
            encoding="utf-8",
        )
        assert tn.main([str(clean)]) == 0
        assert tn.main([str(clean), str(dirty)]) == 1
        # 1 dirty line out of 6 is within a 20% budget.
        assert tn.main([str(clean), str(dirty), "--max-dirty", "0.2"]) == 0
        assert list(tn.iter_text_lines(dirty)) == [TAMIL_LINES[0], "அ\u0bbe"]


if __name__ == "__main__":
    run_module_tests(globals(), "data tamil_normalise")
