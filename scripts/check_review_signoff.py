#!/usr/bin/env python3
"""Check that a native reviewer has signed off the frozen eval corpus (adhan#112, K6).

Fails closed: exits 1 unless the sign-off covers *this exact* corpus bytes and gazetteer, the
reviewer is identified as a native speaker, the gazetteer was accepted, and no sentence was
flagged for removal.

    python scripts/check_review_signoff.py
    python scripts/check_review_signoff.py --signoff eval_manifests/k6_review_signoff.json

The check is the reason the sign-off can be committed: it carries ids, hashes and verdicts, and
any change to the corpus or the blocklist makes it stale instead of silently wrong. It is
written to be usable as a release gate in CI exactly as it is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.eval import review_packet as rp  # noqa: E402

DEFAULT_SIGNOFF = "eval_manifests/k6_review_signoff.json"
DEFAULT_CORPUS = "data/eval/tokenizer_bench.jsonl"
DEFAULT_MANIFEST = "eval_manifests/tokenizer_bench.manifest.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--signoff", default=DEFAULT_SIGNOFF)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = ap.parse_args()

    signoff_path = ROOT / args.signoff
    if not signoff_path.exists():
        print(
            f"FAIL: {args.signoff} not found. Generate the packet with\n"
            "      python scripts/build_review_packet.py\n"
            "      and have a native reviewer fill it in.",
            file=sys.stderr,
        )
        return 1

    corpus = ROOT / args.corpus
    if not corpus.exists():
        print(
            f"FAIL: {args.corpus} not found; fetch the sources and rebuild the corpus first.",
            file=sys.stderr,
        )
        return 1

    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    records = [
        json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    signoff = rp.load_signoff(signoff_path)

    problems = rp.validate_signoff(
        signoff,
        records,
        corpus_sha256=manifest["output_sha256"],
        filter_desc=manifest.get("content_filter"),
    )
    if problems:
        print(f"FAIL: the review does not cover this freeze ({len(problems)} problem(s)):")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    reviewed = len(signoff["sentences"])
    print(f"OK: {args.signoff} signs off {reviewed} sentences")
    print(f"    corpus {manifest['output_sha256']}")
    print(f"    reviewer {signoff['reviewed_by']} on {signoff['reviewed_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
