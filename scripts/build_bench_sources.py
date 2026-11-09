#!/usr/bin/env python3
"""Derive the per-source pools the tokenizer benchmark samples from.

The raw web crawl (``data/raw/expanded``) is one big pile of labelled documents:
``source`` names the upstream corpus, ``tier``/``quality_score`` say how much the
crawler trusts it. The eval-corpus spec (``eval_manifests/tokenizer_bench.sources.json``)
wants one file per source, with a stated licence — so this script does the split,
deterministically, and records exactly what it kept.

    python scripts/build_bench_sources.py            # write data/intermediate/bench_sources/
    python scripts/build_bench_sources.py --check    # rebuild and compare hashes

Rules (identical for every source, so the pools are comparable):

* read the shards in sorted filename order;
* keep a document only if ``source`` is in the source's label set, the text is
  40–6,000 characters after NFC + whitespace collapsing, and it carries no control
  characters other than tab/newline;
* drop exact duplicates by SHA-256 of the normalised text (first occurrence wins);
* take the first ``--docs-per-source`` survivors.

The output is JSONL with a single ``text`` field plus a ``<id>.manifest.json``
sidecar (counts, input/output SHA-256, the rule and the label set). Raw text stays
under ``data/``, which is gitignored; only the manifest and the eval-corpus
manifest are committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_CORPUS = "data/raw/expanded"
DEFAULT_OUT = "data/intermediate/bench_sources"
MIN_CHARS = 40
MAX_CHARS = 6000

#: source id -> (eval register, upstream ``source`` labels in the crawl)
SOURCES: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    # Thirukkural is public domain; the crawl stored it as a classical local source.
    "thirukkural": ("classical", ("local:thirukkural-classical",)),
    # Kalithokai (Sangam anthology) is public domain, ingested from open-sangam.
    "kalithokai": ("classical", ("local:kalithokai",)),
    # Tamil Wikipedia: CC BY-SA 4.0. Tier 1 is the curated crawl, tier 2 the bulk one.
    "wikipedia_ta": ("modern", ("local:wikipedia-ta", "wikipedia-ta")),
}

_ALLOWED_CONTROL = {"\t", "\n"}


def normalise(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def acceptable(text: str, min_chars: int, max_chars: int) -> bool:
    if not (min_chars <= len(text) <= max_chars):
        return False
    return not any(unicodedata.category(ch) == "Cc" and ch not in _ALLOWED_CONTROL for ch in text)


def iter_documents(corpus_dir: Path) -> Iterator[Tuple[str, str]]:
    """Yield ``(source label, text)`` for every document, shard order."""
    shards = sorted(corpus_dir.glob("*.jsonl"))
    if not shards:
        raise SystemExit(f"no *.jsonl shards under {corpus_dir}")
    for shard in shards:
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                yield str(record.get("source") or ""), str(record.get("text") or "")


def build(corpus_dir: Path, out_dir: Path, per_source: int) -> Dict[str, Dict[str, object]]:
    wanted = {label: sid for sid, (_reg, labels) in SOURCES.items() for label in labels}
    seen: Dict[str, Set[str]] = {sid: set() for sid in SOURCES}
    texts: Dict[str, List[str]] = {sid: [] for sid in SOURCES}
    dropped = {"label": 0, "length": 0, "control": 0, "duplicate": 0}
    scanned = 0

    for label, text in iter_documents(corpus_dir):
        sid = wanted.get(label)
        if sid is None:
            dropped["label"] += 1
            continue
        if len(texts[sid]) >= per_source:
            continue
        scanned += 1
        clean = normalise(text)
        if not acceptable(clean, MIN_CHARS, MAX_CHARS):
            dropped["control" if clean and not acceptable(clean, 0, MAX_CHARS) else "length"] += 1
            continue
        digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        if digest in seen[sid]:
            dropped["duplicate"] += 1
            continue
        seen[sid].add(digest)
        texts[sid].append(clean)

    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Dict[str, object]] = {}
    for sid, (register, labels) in SOURCES.items():
        path = out_dir / f"{sid}.jsonl"
        payload = "".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts[sid])
        path.write_text(payload, encoding="utf-8")
        manifest = {
            "id": sid,
            "register": register,
            "labels": list(labels),
            "docs": len(texts[sid]),
            "chars": sum(len(t) for t in texts[sid]),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "rule": (
                "shard order; source label in {labels}; 40 <= NFC+ws-collapsed chars <= 6000; "
                "no control chars; exact-duplicate SHA-256 removed; first {per_source} kept"
            ).format(labels=list(labels), per_source=per_source),
            "corpus": (
                str(corpus_dir.relative_to(ROOT))
                if corpus_dir.is_relative_to(ROOT)
                else str(corpus_dir)
            ),
        }
        (out_dir / f"{sid}.manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report[sid] = manifest
    report["_dropped"] = dropped  # type: ignore[assignment]
    report["_scanned"] = scanned  # type: ignore[assignment]
    return report


def check(out_dir: Path, per_source: int) -> List[str]:
    problems: List[str] = []
    for sid in SOURCES:
        manifest_path = out_dir / f"{sid}.manifest.json"
        corpus_path = out_dir / f"{sid}.jsonl"
        if not manifest_path.exists() or not corpus_path.exists():
            problems.append(f"{sid}: missing output (run without --check first)")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
        if digest != manifest["sha256"]:
            problems.append(f"{sid}: sha256 {digest} != manifest {manifest['sha256']}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default=DEFAULT_CORPUS, help="dir of raw *.jsonl shards")
    ap.add_argument("--out", default=DEFAULT_OUT, help="where the per-source pools go")
    ap.add_argument("--docs-per-source", type=int, default=8000)
    ap.add_argument("--check", action="store_true", help="verify the existing pools instead")
    args = ap.parse_args()

    corpus_dir = ROOT / args.corpus if not Path(args.corpus).is_absolute() else Path(args.corpus)
    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)

    if args.check:
        problems = check(out_dir, args.docs_per_source)
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        if problems:
            return 1
        print(f"OK: {len(SOURCES)} pools match their manifests")
        return 0

    report = build(corpus_dir, out_dir, args.docs_per_source)
    for sid in SOURCES:
        m = report[sid]
        print(f"  {sid:<14} {m['register']:<10} {m['docs']:>6,} docs {m['chars']:>12,} chars")
    print(f"  dropped: {report['_dropped']}")
    width = max(len(str(m["sha256"])) for sid, m in report.items() if not sid.startswith("_"))
    print(f"  manifests written to {out_dir} (sha256 width {width})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
