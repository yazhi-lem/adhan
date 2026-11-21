"""Unit tests for the modern Tamil source fetch (K6 / adhan#112).

No network and no HuggingFace cache is touched: the parquet round-trip test writes its own
tiny file and skips when the optional ``eval-corpus`` extra is not installed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_modern_tamil.py"
_spec = importlib.util.spec_from_file_location("fetch_modern_tamil", _SCRIPT)
fetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch)


def test_pinned_provenance_is_recorded_in_code() -> None:
    """The revision is pinned here, not passed ad hoc, so the fetch stays reproducible."""
    assert fetch.DEFAULT_REPO == "wikimedia/wikipedia"
    assert fetch.DEFAULT_CONFIG == "20231101.ta"
    assert len(fetch.DEFAULT_REVISION) == 40
    assert "CC BY-SA 4.0" in fetch.LICENSE and "GFDL" in fetch.LICENSE


def test_extract_writes_one_jsonl_record_per_article(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="needs the eval-corpus extra")
    import pyarrow as pa_mod
    import pyarrow.parquet as pq

    table = pa_mod.table(
        {
            "id": [1, 2, 3],
            "url": ["https://ta.wikipedia.org/1", "https://ta.wikipedia.org/2", ""],
            "title": ["கட்டிடக்கலை", "இயற்கை", "வெற்று"],
            "text": ["கட்டிடக்கலை என்பது கட்டிடங்கள்.", "  ", "இயற்கை பற்றிய கட்டுரை."],
        }
    )
    parquet_path = tmp_path / "shard.parquet"
    pq.write_table(table, parquet_path)

    out = tmp_path / "out" / "wikipedia_ta.jsonl"
    stats = fetch.extract(parquet_path, out)

    assert stats["articles"] == 3
    assert stats["written"] == 2  # the whitespace-only row is dropped
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [ln["title"] for ln in lines] == ["கட்டிடக்கலை", "வெற்று"]
    assert lines[0]["url"] == "https://ta.wikipedia.org/1"
    assert lines[1]["url"] == ""
    assert stats["bytes"] == out.stat().st_size


def test_extract_output_is_byte_stable(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="needs the eval-corpus extra")
    import pyarrow as pa_mod
    import pyarrow.parquet as pq

    table = pa_mod.table({"text": ["ஒன்று.", "இரண்டு."], "title": ["a", "b"], "url": ["", ""]})
    parquet_path = tmp_path / "shard.parquet"
    pq.write_table(table, parquet_path)

    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    fetch.extract(parquet_path, first)
    fetch.extract(parquet_path, second)

    assert fetch.sha256_file(first) == fetch.sha256_file(second)
    assert first.read_bytes() == second.read_bytes()


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "x.txt"
    path.write_bytes(b"adhan\n")
    assert fetch.sha256_file(path) == hashlib.sha256(b"adhan\n").hexdigest()


def test_parquet_shard_name_matches_the_pinned_layout() -> None:
    """The shard path is derived, not hand-typed, so it cannot drift from the config."""
    # huggingface_hub ships with the optional `eval-corpus` extra: skip, do not fail.
    hf_hub_url = pytest.importorskip(
        "huggingface_hub", reason="needs the eval-corpus extra"
    ).hf_hub_url

    name = f"{fetch.DEFAULT_CONFIG}/train-{0:05d}-of-00002.parquet"
    url = hf_hub_url(fetch.DEFAULT_REPO, name, revision=fetch.DEFAULT_REVISION, repo_type="dataset")
    assert name == "20231101.ta/train-00000-of-00002.parquet"
    assert fetch.DEFAULT_REVISION in url
