#!/usr/bin/env python3
"""Expand the Adhan Tamil pretraining corpus from local + HuggingFace sources.

Why this script exists
----------------------
The previous ingestors (``scripts/ingest_all_prioritized_data.py`` and
``scripts/phase2_corpus_build.py``, both removed) did a linear scan of every
previously seen MinHash signature (O(n^2)), which made them unusable past a few
thousand documents. This script keeps the same source tiering and quality gates
but uses

  * exact SHA-1 dedup, and
  * banded LSH near-dedup over cached shingle hashes (xxhash when available),

so it streams a few hundred thousand documents in minutes, and it is resumable
(shards + a state file), because IndicCorp/Wikipedia streams are slow to restart.

Sources
-------
tier 1  local classical corpora   open-sangam/data/texts, Yazhi/repos/data/texts
tier 2  Tamil Wikipedia           wikimedia/wikipedia (20231101.ta)
tier 3  AI4Bharat IndicCorp v2    ai4bharat/IndicCorpV2 (tam_Taml)

Output
------
data/raw/expanded/corpus-NNNNN.jsonl     one JSON record per line
data/raw/expanded/state.json             per-source progress (resume)
data/raw/expanded/stats.json             source/tier counts, dedup + quality stats

Usage
-----
    python scripts/expand_tamil_corpus.py --wiki-limit 150000 --indic-limit 300000
    python scripts/expand_tamil_corpus.py --local-only
    python scripts/expand_tamil_corpus.py --reset          # start over
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from _hf_pins import describe_pins, load_tamil_dataset  # noqa: E402

from adhan_slm.core.logging import get_logger  # noqa: E402
from adhan_slm.data import corpus_hygiene as hygiene  # noqa: E402

logger = get_logger(__name__)


def is_quality(
    text: str,
    min_chars: int = 40,
    max_chars: int = 20000,
    min_tamil_ratio: float = 0.60,
) -> bool:
    """v1 quality gate: thin wrapper over the canonical gate with v1 tunables."""
    return hygiene.is_quality(
        text,
        hygiene.V1_QUALITY.with_overrides(
            min_chars=min_chars, max_chars=max_chars, min_tamil_ratio=min_tamil_ratio
        ),
    )


# --- Source readers ---------------------------------------------------------
# `_iter_strings` is imported from adhan_slm.data.corpus_hygiene (adhan#125).


def iter_local_json(path: Path) -> Iterator[Tuple[str, str]]:
    """Yield (text, sub_source) from a JSON/JSONL file."""
    try:
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict) and isinstance(rec.get("text"), str):
                        yield rec["text"], rec.get("source", path.stem)
                    else:
                        for s in hygiene.iter_strings(rec):
                            yield s, path.stem
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("text"), str):
                yield data["text"], path.stem
            else:
                for s in hygiene.iter_strings(data):
                    yield s, path.stem
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return


def iter_local_corpus(roots: Iterable[Path]) -> Iterator[Dict]:
    for root in roots:
        if not root.exists():
            logger.warning(f"local source missing: {root}")
            continue
        files = sorted(list(root.rglob("*.json")) + list(root.rglob("*.jsonl")))
        logger.info(f"local source {root}: {len(files)} files")
        for path in files:
            for text, sub in iter_local_json(path):
                text = hygiene.normalize(text)
                if is_quality(text):
                    yield {
                        "text": text,
                        "source": f"local:{sub}",
                        "tier": 1,
                        "quality_score": 0.95,
                    }


def iter_wikipedia(limit: int) -> Iterator[Dict]:
    if limit <= 0:
        return

    logger.info(f"streaming Tamil Wikipedia (limit={limit:,})")
    ds = load_tamil_dataset("wikimedia/wikipedia", "20231101.ta", split="train", streaming=True)
    n = 0
    for row in ds:
        text = hygiene.normalize(row.get("text", ""))
        if is_quality(text, min_chars=100):
            yield {
                "text": text,
                "title": (row.get("title") or "")[:200],
                "source": "wikipedia-ta",
                "tier": 2,
                "quality_score": 0.90,
            }
            n += 1
            if n >= limit:
                break
    logger.info(f"wikipedia stream finished: {n:,} accepted")


def iter_indiccorp(limit: int) -> Iterator[Dict]:
    if limit <= 0:
        return

    logger.info(f"streaming IndicCorpV2 tam_Taml (limit={limit:,})")
    ds = load_tamil_dataset(
        "ai4bharat/IndicCorpV2", "indiccorp_v2", split="tam_Taml", streaming=True
    )
    n = 0
    for row in ds:
        text = hygiene.normalize(row.get("text", ""))
        if is_quality(text, min_chars=60, min_tamil_ratio=0.70):
            yield {
                "text": text,
                "source": "ai4bharat-indiccorp-v2",
                "tier": 3,
                "quality_score": 0.80,
            }
            n += 1
            if n >= limit:
                break
    logger.info(f"indiccorp stream finished: {n:,} accepted")


def iter_fineweb2(limit: int, min_language_score: float = 0.90) -> Iterator[Dict]:
    """Stream Tamil-script pages from HuggingFace FineWeb-2 (web crawl).

    Complements IndicCorp (news/books) and Wikipedia (encyclopaedic) with
    forum/blog/consumer web register. FineWeb-2 ships a per-document
    ``language_score``, which is a far better gate than a character ratio, so
    documents below it are dropped before the (more expensive) quality pass.
    """
    if limit <= 0:
        return

    logger.info(f"streaming FineWeb-2 tam_Taml (limit={limit:,}, lang_score>={min_language_score})")
    ds = load_tamil_dataset("HuggingFaceFW/fineweb-2", "tam_Taml", split="train", streaming=True)
    n = 0
    for row in ds:
        score = row.get("language_score")
        if score is not None and float(score) < min_language_score:
            continue
        text = hygiene.normalize(row.get("text", ""))
        if is_quality(text, min_chars=80, min_tamil_ratio=0.70):
            yield {
                "text": text,
                "url": (row.get("url") or "")[:300],
                "source": "fineweb-2-ta",
                "tier": 3,
                "quality_score": round(min(0.95, float(score or 0.85)), 4),
            }
            n += 1
            if n >= limit:
                break
    logger.info(f"fineweb-2 stream finished: {n:,} accepted")


# --- Orchestration ----------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Expand the Adhan Tamil corpus")
    ap.add_argument("--out-dir", default="data/raw/expanded")
    ap.add_argument("--shard-size", type=int, default=20000, help="docs per jsonl shard")
    ap.add_argument("--wiki-limit", type=int, default=100000)
    ap.add_argument("--indic-limit", type=int, default=200000)
    ap.add_argument(
        "--fineweb-limit",
        type=int,
        default=0,
        help="docs to stream from FineWeb-2 tam_Taml (0 = skip)",
    )
    ap.add_argument(
        "--fineweb-min-lang-score", type=float, default=0.90, help="FineWeb-2 language_score floor"
    )
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--no-local", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument(
        "--no-seed-existing",
        action="store_true",
        help="do not preload exact hashes from existing shards (only for a fresh out-dir)",
    )
    ap.add_argument(
        "--local-roots",
        default=",".join(
            [
                str(ROOT.parent / "open-sangam" / "data" / "texts"),
                str(ROOT.parent / "data" / "texts"),
            ]
        ),
        help="comma-separated local corpus roots",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.reset and out_dir.exists():
        for p in sorted(out_dir.glob("corpus-*.jsonl")):
            p.unlink()
        logger.info("reset: removed existing shards")
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = out_dir / "state.json"
    stats_path = out_dir / "stats.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    done: Set[str] = set(state.get("done_sources", []))
    if args.reset:
        done.clear()

    # do not re-read our own output
    local_roots = [
        Path(p)
        for p in args.local_roots.split(",")
        if p.strip() and Path(p).resolve() != out_dir.resolve()
    ]

    sources: List[Tuple[str, Iterable[Dict]]] = []
    if not args.no_local:
        sources.append(("local", iter_local_corpus(local_roots)))
    if not args.local_only:
        if "wikipedia" not in done:
            sources.append(("wikipedia", iter_wikipedia(args.wiki_limit)))
        else:
            logger.info("wikipedia: already complete (state.json), skipping")
        if "indiccorp" not in done:
            sources.append(("indiccorp", iter_indiccorp(args.indic_limit)))
        else:
            logger.info("indiccorp: already complete (state.json), skipping")
        if "fineweb2" not in done and args.fineweb_limit > 0:
            sources.append(
                (
                    "fineweb2",
                    iter_fineweb2(args.fineweb_limit, args.fineweb_min_lang_score),
                )
            )
        elif "fineweb2" in done:
            logger.info("fineweb2: already complete (state.json), skipping")

    shard_idx = len(sorted(out_dir.glob("corpus-*.jsonl")))
    fh = (out_dir / f"corpus-{shard_idx:05d}.jsonl").open("a", encoding="utf-8")
    in_shard = 0

    dedup = hygiene.LshDedup()

    # Extension/resume guard: preload exact hashes of everything already written
    # so a second run (e.g. raising --indic-limit) does not duplicate the archive.
    existing_shards = sorted(out_dir.glob("corpus-*.jsonl"))
    if existing_shards and not args.no_seed_existing and not args.reset:
        seeded = 0
        for p in existing_shards:
            with p.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        dedup.seed_exact(json.loads(line)["text"])
                        seeded += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
        logger.info(f"seeded {seeded:,} existing documents for exact dedup")

    counts: Counter = Counter()
    tier_counts: Counter = Counter()
    dropped = Counter()
    kept = 0

    def rotate() -> None:
        nonlocal fh, shard_idx, in_shard
        fh.close()
        shard_idx += 1
        in_shard = 0
        new_path = out_dir / f"corpus-{shard_idx:05d}.jsonl"
        logger.info(f"rotating shard -> {new_path.name} ({kept:,} docs so far)")
        fh = new_path.open("a", encoding="utf-8")

    try:
        for name, iterator in sources:
            logger.info(f"=== source: {name} ===")
            for rec in iterator:
                text = rec["text"]
                if dedup.is_duplicate(text):
                    dropped["duplicate"] += 1
                    continue
                rec["id"] = f"{rec['source'].replace(':', '-')}-{kept}"
                rec["chars"] = len(text)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[rec["source"]] += 1
                tier_counts[rec["tier"]] += 1
                kept += 1
                in_shard += 1
                if in_shard >= args.shard_size:
                    rotate()
                if kept % 5000 == 0:
                    logger.info(
                        f"  kept={kept:,}  dropped={sum(dropped.values()):,}  "
                        f"current={counts.most_common(3)}"
                    )
                    fh.flush()
                    state.update(
                        {"done_sources": sorted(done), "kept": kept, "shards": shard_idx + 1}
                    )
                    state_path.write_text(json.dumps(state, indent=2))
            if name in ("wikipedia", "indiccorp", "fineweb2"):
                done.add(name)
    finally:
        fh.close()

    # Recompute totals AND per-source/per-tier counts from every shard, so the
    # datasheet describes the whole archive rather than only this run's additions
    # (each extension run appends shards).
    total_chars = 0
    total_docs = 0
    counts = Counter()
    tier_counts = Counter()
    for p in sorted(out_dir.glob("corpus-*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total_docs += 1
                total_chars += len(rec.get("text", ""))
                counts[rec.get("source", "unknown")] += 1
                tier_counts[rec.get("tier", 0)] += 1

    stats = {
        "documents": total_docs,
        "characters": total_chars,
        # Measured, not guessed: 800-doc sample with the frozen swaram
        # tokenizer gives 0.311 tokens/char (0.573 tokens/akshara). The old
        # 0.95/char figure assumed ~1 token per character and over-reported
        # the corpus 3x.
        "approx_tokens_swaram": int(total_chars * 0.311),
        "source_counts": dict(counts),
        "tier_counts": {str(k): v for k, v in tier_counts.items()},
        "dropped": dict(dropped),
        "hf_revisions": describe_pins(),
        "limit_wiki": args.wiki_limit,
        "limit_indic": args.indic_limit,
        "shards": len(sorted(out_dir.glob("corpus-*.jsonl"))),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    state.update({"done_sources": sorted(done), "kept": kept, "shards": stats["shards"]})
    state_path.write_text(json.dumps(state, indent=2))

    logger.info(
        f"✅ corpus expanded: {total_docs:,} docs, {total_chars/1e6:.1f}M chars "
        f"(~{stats['approx_tokens_swaram']/1e6:.1f}M swaram tokens), "
        f"{stats['shards']} shards -> {out_dir}"
    )
    logger.info(f"stats: {stats_path}")


if __name__ == "__main__":
    main()
