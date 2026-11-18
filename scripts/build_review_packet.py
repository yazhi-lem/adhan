#!/usr/bin/env python3
"""Build the native-review packet for the frozen eval corpus (adhan#112, K6).

Writes a reviewer-facing Markdown file containing the exclusion gazetteer and the sampled
sentences, plus a sign-off JSON skeleton with ids and per-sentence hashes. Both go to
`data/review/`, which is git-ignored: they contain sampled text and must not be committed.

    python scripts/build_review_packet.py
    python scripts/build_review_packet.py --reviewer "…"   # pre-fill the name

The reviewer fills in the JSON and commits it as `eval_manifests/k6_review_signoff.json`;
`scripts/check_review_signoff.py` then fails closed until the sign-off covers the exact frozen
corpus and gazetteer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.eval import eval_corpus as ec  # noqa: E402
from adhan_slm.eval import review_packet as rp  # noqa: E402

DEFAULT_CORPUS = "data/eval/tokenizer_bench.jsonl"
DEFAULT_MANIFEST = "eval_manifests/tokenizer_bench.manifest.json"
DEFAULT_PACKET = "data/review/k6_review_packet.md"
DEFAULT_SIGNOFF = "data/review/k6_review_signoff.json"
DEFAULT_SPEC = "eval_manifests/tokenizer_bench.sources.json"


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_blocklist_terms(spec_path: Path, base: Path) -> dict:
    """Return the gazetteer terms so the reviewer reads them without leaving the packet."""
    spec = ec.load_spec(spec_path)
    if (spec.content_filter or "none") == "none":
        return {}
    path = Path(spec.content_filter)
    if not path.is_absolute():
        path = base / path
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("categories", {})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    ap.add_argument("--packet", default=DEFAULT_PACKET)
    ap.add_argument("--signoff", default=DEFAULT_SIGNOFF)
    ap.add_argument("--reviewer", default="", help="pre-fill reviewed_by")
    ap.add_argument(
        "--registers",
        nargs="+",
        default=list(rp.REVIEW_FOCUS),
        help="registers to list in full first (default: the scraped ones)",
    )
    args = ap.parse_args()

    corpus = ROOT / args.corpus
    if not corpus.exists():
        print(f"FAIL: {corpus} not found; build the corpus first.", file=sys.stderr)
        return 1

    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    records = load_records(corpus)
    if len(records) != manifest["counts"]["total"]:
        print(
            f"FAIL: {corpus} has {len(records)} rows but the manifest says "
            f"{manifest['counts']['total']}; the corpus is stale.",
            file=sys.stderr,
        )
        return 1

    terms = load_blocklist_terms(ROOT / args.spec, ROOT)
    md = rp.packet_markdown(
        records,
        corpus_sha256=manifest["output_sha256"],
        filter_desc=manifest.get("content_filter"),
        blocklist_terms=terms,
        focus=tuple(args.registers),
    )
    signoff = rp.signoff_template(
        records, manifest["output_sha256"], manifest.get("content_filter")
    )
    if args.reviewer:
        signoff["reviewed_by"] = args.reviewer

    packet_path = ROOT / args.packet
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(md, encoding="utf-8")

    signoff_path = ROOT / args.signoff
    signoff_path.parent.mkdir(parents=True, exist_ok=True)
    signoff_path.write_text(
        json.dumps(signoff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"packet:  {args.packet} ({len(md):,} chars, {len(records):,} sentences)")
    print(f"signoff: {args.signoff} (ids + hashes only; safe to commit once filled in)")
    print(f"freeze:  {manifest['output_sha256']}")
    print("next:    reviewer fills the sign-off, then scripts/check_review_signoff.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
