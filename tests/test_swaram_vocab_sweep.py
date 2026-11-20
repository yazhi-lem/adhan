"""Unit tests for the Swaram vocabulary-size sweep (issue #99).

The sweep trains a tokenizer per row, so the tests drive ``measure`` with a stub
tokenizer and check the report shaping separately. Numbers themselves are checked
by re-running ``scripts/swaram_vocab_sweep.py``, which is what the committed report
claims to be reproducible from.
"""

from __future__ import annotations

import importlib.util
import json
import unicodedata
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "swaram_vocab_sweep.py"
_spec = importlib.util.spec_from_file_location("swaram_vocab_sweep", _SCRIPT)
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

WORD_MARK = sweep.WORD_MARK


class _StubTokenizer:
    """One token per character; spaces become the word mark."""

    unk_id = 0

    def __init__(self, unk_chars: str = "") -> None:
        self.unk_chars = set(unk_chars)

    def aksharas(self, text: str) -> list[str]:
        return [c for c in text if not c.isspace()]

    def tokenize(self, text: str) -> list[str]:
        return [WORD_MARK if c.isspace() else c for c in text]

    def encode(self, text: str) -> list[int]:
        return [self.unk_id if c in self.unk_chars else ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join("?" if i == self.unk_id else chr(i) for i in ids)


def test_measure_counts_tokens_per_akshara_and_word() -> None:
    m = sweep.measure(_StubTokenizer(), ["அ ப", "க த"])
    assert m["tok_per_akshara"] == 1.0
    assert m["tok_per_word"] == 1.0
    assert m["unk_rate"] == 0.0
    assert m["roundtrip"] == 1.0


def test_measure_ignores_the_word_mark_when_counting_tokens() -> None:
    """Word marks are separators, not content tokens; they must not inflate fertility."""
    with_space = sweep.measure(_StubTokenizer(), ["அ ப க"])
    # Four characters, three aksharas and three words: the word mark is excluded.
    assert with_space["tok_per_akshara"] == 1.0
    assert with_space["tok_per_word"] == 1.0


def test_measure_reports_unk_rate_from_encoded_ids() -> None:
    m = sweep.measure(_StubTokenizer(unk_chars="ஔ"), ["அஔ"])
    assert m["unk_rate"] == 0.5


def test_measure_roundtrip_requires_nfc_equality() -> None:
    """A decomposed string must not count as a lossless round-trip."""
    nfd = "e\u0301"
    nfc = "\u00e9"
    assert unicodedata.normalize("NFC", nfd) == nfc
    assert nfd != nfc

    m = sweep.measure(_StubTokenizer(), [nfd])
    assert m["roundtrip"] == 0.0

    ok = sweep.measure(_StubTokenizer(), [nfc])
    assert ok["roundtrip"] == 1.0


def _row(size: int, byte_fb: bool, fertility: float) -> dict:
    return {
        "requested": size,
        "actual": size,
        "byte_fallback": byte_fb,
        "tok_per_akshara": fertility,
        "tok_per_word": 2.0,
        "unk_rate": 0.0,
        "roundtrip": 1.0,
    }


def test_markdown_has_one_row_per_result_and_the_documented_columns() -> None:
    rows = [_row(1000, False, 0.5), _row(12000, True, 0.3)]
    md = sweep.to_markdown(rows, n_train=1197, n_eval=133)

    table = [ln for ln in md.splitlines() if ln.startswith("| 1000") or ln.startswith("| 12000")]
    assert len(table) == 2
    for line in table:
        assert len([c for c in line.split("|")[1:-1]]) == 7
    assert "1197 training docs" in md
    assert "133 held-out docs" in md


def test_markdown_states_the_small_corpus_caveat() -> None:
    md = sweep.to_markdown([_row(1000, False, 0.5)], 1197, 133)
    assert "small" in md
    assert "not a final 12k validation" in md


def test_markdown_carries_the_baselines_that_make_fertility_readable() -> None:
    md = sweep.to_markdown([_row(1000, False, 0.5)], 1197, 133)
    for name, _val, _role in sweep.BASELINES:
        assert name in md
    assert "merge rate" in md


def test_report_is_serialisable_and_keeps_the_target() -> None:
    rows = [_row(1000, False, 0.542)]
    report = sweep.to_report(rows, n_train=1197, n_eval=133)

    assert report["n_train_docs"] == 1197
    assert report["n_eval_docs"] == 133
    assert report["fertility_target_tok_per_akshara"] == 1.15
    assert report["rows"] == rows
    assert json.loads(json.dumps(report)) == report


def test_report_excludes_wall_clock_timings() -> None:
    """A per-machine timing column made every regeneration a diff; it must not come back."""
    rows = [_row(1000, False, 0.5)]
    assert "train_s" not in rows[0]
    assert "Train time" not in sweep.to_markdown(rows, 1197, 133)
    assert "train_s" not in json.dumps(sweep.to_report(rows, 1197, 133))


def test_reports_land_in_the_docs_reports_convention() -> None:
    assert sweep.DEFAULT_MD == "docs/reports/swaram_vocab_sweep.md"
    assert sweep.DEFAULT_JSON == "docs/reports/swaram_vocab_sweep.json"
