"""Unit tests for the Tanglish content-filter measurement (K6 / adhan#112).

Synthetic fixtures only: no scraped corpus text is read or written here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from adhan_slm.eval import content_filter as cf

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tanglish_filter_report.py"
_spec = importlib.util.spec_from_file_location("tanglish_filter_report", _SCRIPT)
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

_BLOCKLIST = {
    "version": "test-v1",
    "native_review": "pending",
    "categories": {"politics": {"latin": ["election"], "tamil": []}},
}


def _write_corpus(path: Path, texts: list[str]) -> Path:
    path.write_text(
        "".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts),
        encoding="utf-8",
    )
    return path


def _flt() -> cf.ContentFilter:
    return cf.ContentFilter.from_dict(_BLOCKLIST)


def test_measure_counts_neutral_and_blocked_candidates(tmp_path: Path) -> None:
    corpus = _write_corpus(
        tmp_path / "tg.jsonl",
        [
            "bro office ku late ah vanthen romba tired ah iruku",
            "bro election la vote podu da",
            "short",
            "இந்த வாக்கியம் தமிழில் இருக்கிறது",
        ],
    )
    res = report.measure(corpus, _flt(), 15, 300)

    assert res["rows"] == 4
    assert res["neutral_candidates"] == 1
    assert res["blocked_per_category"] == {"politics": 1}
    assert res["candidates"] == 2
    assert res["dropped"]["length"] == 1
    assert res["dropped"]["script"] == 1
    assert res["dropped"]["blocked"] == 1


def test_measure_records_source_hash_and_filter_identity(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path / "tg.jsonl", ["bro office ku late ah vanthen romeda"])
    res = report.measure(corpus, _flt(), 15, 300)

    assert len(res["source"]["sha256"]) == 64
    assert res["source"]["bytes"] == corpus.stat().st_size
    assert res["filter"]["version"] == "test-v1"
    assert res["filter"]["categories"] == ["politics"]


def test_measure_is_independent_of_row_order(tmp_path: Path) -> None:
    texts = [
        "bro office ku late ah vanthen romba tired ah iruku",
        "machan salary cut pannanga romba tension ah iruku",
        "bro election la vote podu da",
    ]
    a = report.measure(_write_corpus(tmp_path / "a.jsonl", texts), _flt(), 15, 300)
    b = report.measure(_write_corpus(tmp_path / "b.jsonl", list(reversed(texts))), _flt(), 15, 300)
    assert a["neutral_candidates"] == b["neutral_candidates"]
    assert a["blocked_per_category"] == b["blocked_per_category"]


def test_measure_without_filter_blocks_nothing(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path / "tg.jsonl", ["bro election la vote podu da"])
    res = report.measure(corpus, cf.ContentFilter.none(), 15, 300)
    assert res["blocked_per_category"] == {}
    assert res["neutral_candidates"] == 1


def test_headroom_is_the_neutral_count_over_the_sample_size(tmp_path: Path) -> None:
    corpus = _write_corpus(tmp_path / "tg.jsonl", ["bro office ku late ah vanthen romba"])
    res = report.measure(corpus, _flt(), 15, 300)
    assert res["sample_size"] == 500
    assert abs(res["headroom"] - res["neutral_candidates"] / 500) < 1e-9


def test_markdown_states_counts_categories_and_the_review_caveat(tmp_path: Path) -> None:
    corpus = _write_corpus(
        tmp_path / "tg.jsonl",
        ["bro office ku late ah vanthen romba tired ah iruku", "bro election la vote podu da"],
    )
    md = report.to_markdown(report.measure(corpus, _flt(), 15, 300))

    assert "| **Neutral candidates** | **1** |" in md
    assert "| politics | 1 |" in md
    assert "native_review" in md
    assert "5.5%" in md  # the N3 cross-check
    assert "python scripts/tanglish_filter_report.py" in md


def test_reports_land_in_the_docs_reports_convention() -> None:
    assert report.DEFAULT_MD == "docs/reports/tanglish_filter_v2.md"
    assert report.DEFAULT_JSON == "docs/reports/tanglish_filter_v2.json"
