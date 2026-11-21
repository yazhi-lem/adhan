#!/usr/bin/env python3
"""Build the public-domain classical eval source (adhan#112, K6).

The Thirukkural itself is public domain, but the convenient corpus files mix the verses with
20th-century commentaries whose rights belong to their authors. The eval corpus must not need
a legal opinion, so this script keeps **the verses only**:

    Line1 + Line2 from github.com/tk120404/thirukkural @ cf3ff6c
    (Translation, mv, sp, mk and the other commentary fields are dropped, not parsed)

    python scripts/build_classical_corpus.py
    python scripts/build_classical_corpus.py --json-path /tmp/thirukkural.json   # offline

Writes ``data/raw/thirukkural/thirukkural_verses.jsonl`` (git-ignored) and prints its SHA-256,
which is what the corpus manifest pins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_REPO = "tk120404/thirukkural"
DEFAULT_REVISION = "cf3ff6c5a73c23b4092913156830457caab65fd4"
DEFAULT_URL = (
    f"https://raw.githubusercontent.com/{DEFAULT_REPO}/{DEFAULT_REVISION}/thirukkural.json"
)
DEFAULT_OUT = "data/raw/thirukkural/thirukkural_verses.jsonl"

#: Verses are public domain; the repository that transcribes them is Apache-2.0.
LICENSE = (
    "Public domain (Thirukkural, Tamil classical text; "
    f"transcription: github.com/{DEFAULT_REPO} @ {DEFAULT_REVISION[:7]}, Apache-2.0 repository). "
    "Third-party commentaries in the upstream file are deliberately not included."
)

#: Fields of the upstream record that hold the verse. Everything else is commentary.
VERSE_FIELDS = ("Line1", "Line2")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:  # noqa: S310 - fixed https URL above
        return resp.read()


def verse_text(record: dict) -> str:
    """The verse as the traditional two lines, with any commentary removed."""
    lines = [str(record.get(field) or "").strip() for field in VERSE_FIELDS]
    return "\n".join(line for line in lines if line)


def build(raw: dict, out: Path) -> dict:
    kurals = raw.get("kural")
    if not isinstance(kurals, list) or not kurals:
        raise ValueError("upstream JSON has no 'kural' list")

    out.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with out.open("w", encoding="utf-8") as fh:
        for record in kurals:
            text = verse_text(record)
            if not text:
                skipped += 1
                continue
            number = record.get("Number")
            fh.write(
                json.dumps(
                    {
                        "id": f"thirukkural-{number}",
                        "text": text,
                        "kural_num": number,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            written += 1
    return {"kurals": len(kurals), "written": written, "skipped": skipped}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--json-path", default="", help="use a local copy of the upstream JSON")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out

    if args.json_path:
        payload = Path(args.json_path).read_bytes()
        source = args.json_path
    else:
        payload = fetch(args.url)
        source = args.url

    stats = build(json.loads(payload.decode("utf-8")), out)
    print(f"source:  {source}")
    print(f"licence: {LICENSE}")
    print(
        f"kurals:  {stats['kurals']:,} (written {stats['written']:,}, skipped {stats['skipped']})"
    )
    print(f"output:  {args.out} ({out.stat().st_size:,} bytes)")
    print(f"sha256:  {sha256_file(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
