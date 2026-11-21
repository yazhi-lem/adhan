"""Exit-code tests for the review-signoff gate (K6 / adhan#112).

The gate is a release step, so its contract is its exit code: 0 only when a human has signed off
this exact corpus and gazetteer. These tests drive the real script in a subprocess against a
temporary fixture layout, which is also how it will be wired into a release job.

Note: this file is deliberately named after the module tests convention and lives in ``tests/``
because it exercises the CLI, not the library.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_review_signoff.py"
RECORDS = [
    {"id": "tanglish-00000", "register": "tanglish", "source": "tg", "text": "bro office ku late"},
    {"id": "modern-00000", "register": "modern", "source": "wiki", "text": "கட்டிடக்கலை என்பது"},
]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_layout(tmp: Path, *, approve_gazetteer: bool = True, empty_blocklist: bool = False):
    corpus = tmp / "corpus.jsonl"
    corpus.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in RECORDS),
        encoding="utf-8",
    )
    terms = [] if empty_blocklist else ["election"]
    blocklist = tmp / "blocklist.json"
    blocklist.write_text(
        json.dumps(
            {
                "version": "v9",
                "native_review": "reviewed 2026-11-18" if approve_gazetteer else "pending",
                "categories": {"politics": {"latin": terms}},
            }
        ),
        encoding="utf-8",
    )
    blocklist_bytes = blocklist.read_bytes()
    corpus_bytes = corpus.read_bytes()

    manifest = tmp / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "output_sha256": _sha(corpus_bytes),
                "spec": {"content_filter": "blocklist.json"},
                "content_filter": {
                    "version": "v9",
                    "sha256": _sha(blocklist_bytes),
                    "categories": ["politics"],
                    "native_review": "reviewed 2026-11-18" if approve_gazetteer else "pending",
                },
            }
        ),
        encoding="utf-8",
    )

    signoff = {
        "schema_version": 1,
        "corpus_output_sha256": _sha(corpus_bytes),
        "filter_version": "v9",
        "filter_sha256": _sha(blocklist_bytes),
        "reviewed_by": "Test Reviewer",
        "reviewed_at": "2026-11-18",
        "native_speaker": True,
        "gazetteer": {"version": "v9", "verdict": "keep", "missing_terms": [], "notes": ""},
        "sentences": {
            r["id"]: {
                "sha256": _sha(r["text"].encode("utf-8")),
                "verdict": "keep",
            }
            for r in RECORDS
        },
    }
    signoff_path = tmp / "signoff.json"
    signoff_path.write_text(json.dumps(signoff, ensure_ascii=False), encoding="utf-8")
    return corpus, manifest, signoff_path, signoff


def _run(tmp: Path, corpus: Path, manifest: Path, signoff: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base",
            str(tmp),
            "--corpus",
            str(corpus),
            "--manifest",
            str(manifest),
            "--signoff",
            str(signoff),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_complete_review_exits_zero(tmp_path: Path) -> None:
    corpus, manifest, signoff, _ = _write_layout(tmp_path)
    proc = _run(tmp_path, corpus, manifest, signoff)
    assert proc.returncode == 0, proc.stderr
    assert "OK:" in proc.stdout


def test_unreviewed_gazetteer_exits_one(tmp_path: Path) -> None:
    corpus, manifest, signoff, _ = _write_layout(tmp_path, approve_gazetteer=False)
    proc = _run(tmp_path, corpus, manifest, signoff)
    assert proc.returncode == 1
    assert "native_review" in proc.stderr


def test_emptied_blocklist_exits_one(tmp_path: Path) -> None:
    """The whole point of re-hashing: a manifest cannot vouch for a gazetteer that changed."""
    corpus, manifest, signoff, _ = _write_layout(tmp_path)
    (tmp_path / "blocklist.json").write_text(
        json.dumps({"version": "v9", "native_review": "reviewed", "categories": {}}),
        encoding="utf-8",
    )
    proc = _run(tmp_path, corpus, manifest, signoff)
    assert proc.returncode == 1
    assert "does not match the one the manifest records" in proc.stderr


def test_tampered_corpus_exits_one(tmp_path: Path) -> None:
    corpus, manifest, signoff, _ = _write_layout(tmp_path)
    corpus.write_text(
        corpus.read_text(encoding="utf-8")
        + json.dumps({"id": "x", "register": "modern", "source": "s", "text": "y"})
        + "\n",
        encoding="utf-8",
    )
    proc = _run(tmp_path, corpus, manifest, signoff)
    assert proc.returncode == 1
    assert "corpus on disk" in proc.stderr


def test_missing_signoff_exits_one_with_instructions(tmp_path: Path) -> None:
    corpus, manifest, _, _ = _write_layout(tmp_path)
    proc = _run(tmp_path, corpus, manifest, tmp_path / "nope.json")
    assert proc.returncode == 1
    assert "build_review_packet" in proc.stderr


def test_missing_corpus_exits_one_with_instructions(tmp_path: Path) -> None:
    _, manifest, signoff, _ = _write_layout(tmp_path)
    proc = _run(tmp_path, tmp_path / "nope.jsonl", manifest, signoff)
    assert proc.returncode == 1
    assert "rebuild the corpus" in proc.stderr


def test_pending_verdict_exits_one(tmp_path: Path) -> None:
    corpus, manifest, signoff, payload = _write_layout(tmp_path)
    payload["sentences"]["tanglish-00000"]["verdict"] = "pending"
    signoff.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    proc = _run(tmp_path, corpus, manifest, signoff)
    assert proc.returncode == 1
    assert "pending" in proc.stderr
