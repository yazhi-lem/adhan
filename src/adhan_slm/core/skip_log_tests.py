"""Tests for structured skip logging and the skipped-record metric (adhan#124).

The point of #124 is that a dropped record must be *visible*: logged with context and counted, so
a build can say "12,431 skipped: malformed_json 12,400" instead of silently producing a smaller
corpus. These tests pin both halves, plus the two failure paths the issue asks for.

Run: PYTHONPATH=src python -m adhan_slm.core.skip_log_tests
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on path

from adhan_slm.core import skip_log as sl  # noqa: E402


class _Capture(logging.Handler):
    """Collect records from a project logger (they set propagate=False, so caplog misses them)."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []
        self.levels: list[int] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())
        self.levels.append(record.levelno)


def _capture(logger_name: str):
    handler = _Capture()
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    return handler, logger


from adhan_slm.core.selftest import run_module_tests  # noqa: E402
from adhan_slm.data import corpus  # noqa: E402

# -- the counter --------------------------------------------------------------------


def test_counter_counts_by_reason_and_source():
    c = sl.SkipCounter()
    c.record(sl.MALFORMED_JSON, source="a.jsonl")
    c.record(sl.MALFORMED_JSON, source="a.jsonl")
    c.record(sl.UNREADABLE_FILE, source="b.jsonl")

    assert c.total == 3
    assert c.by_reason[sl.MALFORMED_JSON] == 2
    assert c.by_source["a.jsonl"][sl.MALFORMED_JSON] == 2
    assert c.by_source["b.jsonl"][sl.UNREADABLE_FILE] == 1


def test_counter_summary_is_human_readable_and_ordered():
    c = sl.SkipCounter()
    for _ in range(3):
        c.record(sl.MALFORMED_JSON)
    c.record(sl.EMPTY_TEXT)
    assert c.summary() == "4 skipped: malformed_json 3, empty_text 1"


def test_empty_counter_says_zero():
    assert sl.SkipCounter().summary() == "0 skipped"
    assert sl.SkipCounter().as_dict()["total"] == 0


def test_counter_as_dict_is_json_serialisable():
    c = sl.SkipCounter()
    c.record(sl.MALFORMED_JSON, source="a.jsonl", detail="line 7")
    payload = json.loads(json.dumps(c.as_dict()))
    assert payload["total"] == 1
    assert payload["by_reason"] == {sl.MALFORMED_JSON: 1}
    assert payload["by_source"] == {"a.jsonl": {sl.MALFORMED_JSON: 1}}
    assert payload["examples"] == {sl.MALFORMED_JSON: "line 7"}


def test_counter_keeps_a_bounded_number_of_examples():
    c = sl.SkipCounter(max_examples=2)
    for i in range(5):
        c.record(f"reason_{i}", detail=f"detail {i}")
    assert len(c.examples) == 2


def test_counter_merge_combines_two_runs():
    a = sl.SkipCounter()
    a.record(sl.MALFORMED_JSON, source="a.jsonl")
    b = sl.SkipCounter()
    b.record(sl.MALFORMED_JSON, source="a.jsonl")
    b.record(sl.EMPTY_TEXT, source="b.jsonl")
    a.merge(b)
    assert a.total == 3
    assert a.by_source["a.jsonl"][sl.MALFORMED_JSON] == 2


def test_skipped_reasons_are_ordered_by_count():
    c = sl.SkipCounter()
    c.record(sl.EMPTY_TEXT)
    for _ in range(3):
        c.record(sl.MALFORMED_JSON)
    assert list(sl.skipped_reasons(c)) == [sl.MALFORMED_JSON, sl.EMPTY_TEXT]


# -- the log line -------------------------------------------------------------------


def test_log_skipped_emits_structured_context():
    handler, logger = _capture("adhan.test.skip")
    try:
        sl.log_skipped(
            logger,
            "read shard",
            reason=sl.MALFORMED_JSON,
            path="data/x.jsonl",
            error=ValueError("bad"),
            line=12,
        )
    finally:
        logger.removeHandler(handler)
    message = handler.messages[-1]
    assert "skipped read shard" in message
    assert f"reason={sl.MALFORMED_JSON}" in message
    assert "path=data/x.jsonl" in message
    assert "error=ValueError: bad" in message
    assert "line=12" in message


def test_log_skipped_honours_the_level():
    handler, logger = _capture("adhan.test.skip.level")
    logger.setLevel(logging.DEBUG)
    try:
        sl.log_skipped(logger, "parse", reason=sl.SOURCE_FAILED, level=logging.DEBUG)
    finally:
        logger.removeHandler(handler)
    assert handler.levels[-1] == logging.DEBUG


# -- failure path 1: a malformed corpus line is logged and counted ------------------


def test_malformed_jsonl_line_is_logged_and_counted(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        '{"text": "நல்ல தமிழ் உரை ஒன்று"}\n' "{not json}\n" '{"text": "இரண்டாவது உரை"}\n',
        encoding="utf-8",
    )
    counter = sl.SkipCounter()
    handler, logger = _capture("adhan_slm.data.corpus")
    try:
        docs = corpus.read_corpus(path, skipped=counter)
    finally:
        logger.removeHandler(handler)

    assert len(docs) == 2, "the good lines must still be read"
    assert counter.total == 1
    assert counter.by_reason[sl.MALFORMED_JSON] == 1
    assert counter.by_source[str(path)][sl.MALFORMED_JSON] == 1
    assert any("malformed_json" in m for m in handler.messages)


def test_reading_without_a_counter_still_logs(tmp_path):
    """The counter is optional; the log line is not."""
    path = tmp_path / "corpus.jsonl"
    path.write_text("{broken}\n", encoding="utf-8")
    handler, logger = _capture("adhan_slm.data.corpus")
    try:
        assert corpus.read_corpus(path) == []
    finally:
        logger.removeHandler(handler)
    assert any("malformed_json" in m for m in handler.messages)


# -- failure path 2: an unreadable source file is reported, not swallowed -----------


def test_unreadable_sangam_source_is_logged_and_skipped(tmp_path):
    """A corrupt source file must not abort the build, and must not vanish either."""
    from adhan_slm.data import sangam_sft

    texts = tmp_path / "data" / "texts"
    (texts / "porul").mkdir(parents=True)
    (texts / "porul" / "broken.json").write_text("{not json", encoding="utf-8")
    (texts / "porul" / "good.json").write_text(
        json.dumps([{"text": "நல்ல தமிழ் உரை ஒன்று இரண்டு மூன்று"}]), encoding="utf-8"
    )

    handler, logger = _capture("adhan_slm.data.sangam_sft")
    try:
        result = sangam_sft.load_open_sangam(tmp_path)
    finally:
        logger.removeHandler(handler)

    assert isinstance(result, dict)
    assert any("unreadable_file" in m for m in handler.messages), handler.messages


# -- the lint rule that keeps it from regressing ------------------------------------


def test_ruff_config_enables_the_silent_except_rules():
    """S110/E722 are the guard: without them the next `except: pass` lands unnoticed."""
    import tomllib

    root = Path(__file__).resolve().parents[3]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    select = config["tool"]["ruff"]["lint"]["select"]
    assert "S110" in select, "try/except/pass must be a lint error"
    assert "E722" in select, "bare except must be a lint error"


if __name__ == "__main__":
    raise SystemExit(run_module_tests(globals()))
