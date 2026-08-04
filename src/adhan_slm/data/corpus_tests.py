"""Tests for corpus reading (`adhan_slm.data.corpus`). Pure python, no numpy/JAX.

Run: PYTHONPATH=src python -m adhan_slm.data.corpus_tests
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adhan_slm.core.selftest import run_module_tests  # noqa: E402
from adhan_slm.data import corpus  # noqa: E402
from adhan_slm.data._test_fixtures import TAMIL_LINES  # noqa: E402


def test_reads_txt_and_jsonl_from_a_directory() -> None:
    """A corpus dir mixes plain text and JSONL; both must land in one doc stream."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "a.txt").write_text("\n".join(TAMIL_LINES) + "\n", encoding="utf-8")
        (root / "b.jsonl").write_text(
            "\n".join(json.dumps({"text": t}) for t in TAMIL_LINES) + "\n",
            encoding="utf-8",
        )
        docs = corpus.read_corpus(root)

    assert len(docs) == len(TAMIL_LINES) * 2, f"expected 8 docs, got {len(docs)}"


def test_drops_blank_and_whitespace_only_documents() -> None:
    """Blank lines would otherwise become `<bos><eos>` pairs padding the token stream."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "a.txt").write_text("\n".join(TAMIL_LINES) + "\n", encoding="utf-8")
        (root / "empty.txt").write_text("\n\n  \n", encoding="utf-8")
        docs = corpus.read_corpus(root)

    assert len(docs) == len(TAMIL_LINES)
    assert all(docs), "no empty docs should survive"


if __name__ == "__main__":
    run_module_tests(globals(), "data corpus")
