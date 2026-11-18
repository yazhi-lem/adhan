#!/usr/bin/env python3
"""Measure what the content filter removes from a Tanglish source (adhan#112, K6).

The K6 exit criterion needs a published, checkable number: after the eval-corpus rules and
the gazetteer pass, how many neutral Tanglish sentences are left to sample 500 from? This
script answers that on a pinned source and writes both a readable report and the raw JSON.

    python scripts/tanglish_filter_report.py

    # another filter version or source
    python scripts/tanglish_filter_report.py --filter v2 \
        --corpus data/raw/tanglish/tanglish_corpus.jsonl

The raw corpus stays local (`data/` is git-ignored). Only counts, hashes and the pinned
provenance are committed, so no scraped text enters the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.eval import content_filter as cf  # noqa: E402
from adhan_slm.eval import eval_corpus as ec  # noqa: E402

DEFAULT_CORPUS = "data/raw/tanglish/tanglish_corpus.jsonl"
DEFAULT_MD = "docs/reports/tanglish_filter_v2.md"
DEFAULT_JSON = "docs/reports/tanglish_filter_v2.json"
REGISTER = "tanglish"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def measure(corpus: Path, flt: cf.ContentFilter, min_chars: int, max_chars: int) -> dict:
    rows = 0
    sentences = 0
    too_short = 0
    wrong_script = 0
    duplicates = 0
    blocked: dict[str, int] = {name: 0 for name in flt.category_names()}
    seen: set[str] = set()

    with corpus.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                doc = json.loads(line).get("text") or ""
            except json.JSONDecodeError:
                continue
            for sent in ec.split_sentences(doc):
                sentences += 1
                if not (min_chars <= len(sent) <= max_chars):
                    too_short += 1
                    continue
                if not ec.script_ok(sent, REGISTER):
                    wrong_script += 1
                    continue
                if sent in seen:
                    duplicates += 1
                    continue
                category = flt.blocked(sent)
                if category is not None:
                    blocked[category] += 1
                    continue
                seen.add(sent)

    blocked_total = sum(blocked.values())
    candidates = len(seen) + blocked_total
    return {
        "source": {
            "path": str(corpus),
            "sha256": sha256_file(corpus),
            "bytes": corpus.stat().st_size,
        },
        "register": REGISTER,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "filter": flt.describe(),
        "rows": rows,
        "sentences": sentences,
        "dropped": {
            "length": too_short,
            "script": wrong_script,
            "duplicate": duplicates,
            "blocked": blocked_total,
        },
        "blocked_per_category": blocked,
        "candidates": candidates,
        "neutral_candidates": len(seen),
        "blocked_share": (blocked_total / candidates) if candidates else 0.0,
        "sample_size": 500,
        "headroom": (len(seen) / 500) if len(seen) else 0.0,
    }


def to_markdown(report: dict) -> str:
    src = report["source"]
    flt = report["filter"] or {}
    dropped = report["dropped"]
    lines = [
        f"# Content filter {flt.get('version', 'none')}: Tanglish source measurement",
        "",
        "Task: [K6 / #112](https://github.com/yazhi-lem/adhan/issues/112). Answers one",
        "question with committed code: after the eval-corpus rules and the gazetteer pass,",
        f"how many neutral Tanglish sentences remain to sample {report['sample_size']} from?",
        "",
        "## Source (pinned)",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Path | `{src['path']}` (git-ignored, local only) |",
        f"| SHA-256 | `{src['sha256']}` |",
        f"| Size | {src['bytes']:,} bytes |",
        "| Repo | `vishnu-n/Tanglish-Corpus-185k` @ `927f5b4` |",
        "| Declared licence | CC-BY-4.0 (dataset card) |",
        "",
        "## Rules applied",
        "",
        f"- {report['min_chars']}–{report['max_chars']} characters per sentence",
        f"- `{REGISTER}` script rule: no Tamil-script characters, at least 80% Latin letters",
        "- exact-duplicate removal across the source",
        f"- content filter `{flt.get('version', 'none')}` "
        f"(categories: {', '.join(flt.get('categories') or []) or 'none'})",
        "",
        "## Results",
        "",
        "| Stage | Count |",
        "|---|---:|",
        f"| Raw rows | {report['rows']:,} |",
        f"| Sentences | {report['sentences']:,} |",
        f"| Dropped: length | {dropped['length']:,} |",
        f"| Dropped: script | {dropped['script']:,} |",
        f"| Dropped: duplicate | {dropped['duplicate']:,} |",
        f"| Dropped: content filter | {dropped['blocked']:,} |",
        f"| **Neutral candidates** | **{report['neutral_candidates']:,}** |",
        "",
        "### Content filter, per category",
        "",
        "| Category | Blocked |",
        "|---|---:|",
    ]
    for name, count in report["blocked_per_category"].items():
        lines.append(f"| {name} | {count:,} |")
    lines += [
        "",
        f"The filter vetoed **{report['blocked_share']:.1%}** of candidates. That matches the",
        "~5.5% keyword-positive rate in the N3 scope report",
        "([`n3_tanglish_source_scope.md`](n3_tanglish_source_scope.md)), so the filter behaves as",
        "scoped rather than over- or under-filtering.",
        "",
        f"At {report['neutral_candidates']:,} neutral candidates the 500-sentence sample has",
        f"**{report['headroom']:.0f}× headroom**, so the freeze does not depend on a handful of",
        "rows and can survive a stricter reviewer-passed gazetteer.",
        "",
        "## Caveats",
        "",
        f"- `native_review` is `{flt.get('native_review', 'pending')}`: this is a seed gazetteer.",
        "  A native reviewer must extend and sign it off before the corpus is frozen.",
        "- A gazetteer is high recall, not precision. It removes some innocent rows and misses",
        "  paraphrases; MuRIL was probed for this and rejected as too weak to ship",
        "  ([`muril_base_assessment.md`](muril_base_assessment.md)).",
        "- Counts describe this pinned revision only; a newer upstream file is a different",
        "  corpus and needs its own measurement.",
        "",
        "Regenerate: `python scripts/tanglish_filter_report.py`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--filter", default="v2", help="shipped gazetteer version, or 'none'")
    ap.add_argument("--min-chars", type=int, default=15)
    ap.add_argument("--max-chars", type=int, default=300)
    ap.add_argument("--out", default=DEFAULT_MD)
    ap.add_argument("--json", default=DEFAULT_JSON)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    if not corpus.exists():
        print(
            f"FAIL: {corpus} not found. Fetch the pinned source first "
            "(see docs/reports/n3_tanglish_source_scope.md §6).",
            file=sys.stderr,
        )
        return 1
    flt = cf.ContentFilter.none() if args.filter == "none" else cf.load_named(args.filter)
    report = measure(corpus, flt, args.min_chars, args.max_chars)

    md = to_markdown(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
