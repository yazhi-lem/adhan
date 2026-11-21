"""Tests for the public-domain classical source builder (K6 / adhan#112).

Fixtures are synthetic records in the upstream shape; nothing is fetched.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_classical_corpus.py"
_spec = importlib.util.spec_from_file_location("build_classical_corpus", _SCRIPT)
classical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(classical)

_UPSTREAM = {
    "kural": [
        {
            "Number": 1,
            "Line1": "அகர முதல எழுத்தெல்லாம் ஆதி",
            "Line2": "பகவன் முதற்றே உலகு.",
            "Translation": "A leads letters",
            "mv": "மு. வரதராசனார் உரை",
            "sp": "சாலமன் பாப்பையா உரை",
            "mk": "கலைஞர் உரை",
        },
        {
            "Number": 2,
            "Line1": "கற்றதனால் ஆய பயனென்கொல்",
            "Line2": "வாலறிவன் நற்றாள் தொழாஅர் எனின்.",
            "Translation": "What is the use of learning",
            "mv": "மு. வரதராசனார் உரை",
        },
        {"Number": 3, "Line1": "", "Line2": "", "mv": "உரை மட்டும்"},
    ]
}


def _build(tmp_path: Path) -> tuple[list[dict], dict]:
    out = tmp_path / "verses.jsonl"
    stats = classical.build(_UPSTREAM, out)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    return rows, stats


def test_only_the_verse_fields_are_written(tmp_path: Path) -> None:
    rows, _ = _build(tmp_path)
    assert rows[0]["text"] == "அகர முதல எழுத்தெல்லாம் ஆதி\nபகவன் முதற்றே உலகு."
    assert rows[0]["id"] == "thirukkural-1"
    assert rows[0]["kural_num"] == 1
    assert set(rows[0]) == {"id", "text", "kural_num"}


def test_commentary_is_never_copied_into_the_output(tmp_path: Path) -> None:
    """The whole point of this source: no third-party commentary text."""
    out = tmp_path / "verses.jsonl"
    classical.build(_UPSTREAM, out)
    blob = out.read_text(encoding="utf-8")
    for marker in ("வரதராசனார்", "பாப்பையா", "கலைஞர்", "A leads letters"):
        assert marker not in blob


def test_records_without_verses_are_skipped(tmp_path: Path) -> None:
    rows, stats = _build(tmp_path)
    assert stats == {"kurals": 3, "written": 2, "skipped": 1}
    assert [r["kural_num"] for r in rows] == [1, 2]


def test_output_is_byte_stable(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    classical.build(_UPSTREAM, a)
    classical.build(_UPSTREAM, b)
    assert a.read_bytes() == b.read_bytes()
    assert classical.sha256_file(a) == classical.sha256_file(b)


def test_missing_kural_list_is_rejected(tmp_path: Path) -> None:
    for raw in ({}, {"kural": []}, {"kural": "nope"}):
        try:
            classical.build(raw, tmp_path / "x.jsonl")
        except ValueError as exc:
            assert "kural" in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"accepted {raw!r}")


def test_provenance_is_pinned_in_code() -> None:
    assert classical.DEFAULT_REPO == "tk120404/thirukkural"
    assert len(classical.DEFAULT_REVISION) == 40
    assert classical.DEFAULT_REVISION in classical.DEFAULT_URL
    assert "Public domain" in classical.LICENSE
    assert "not included" in classical.LICENSE
