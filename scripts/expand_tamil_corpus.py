#!/usr/bin/env python3
"""Expand the Adhan Tamil pretraining corpus from local + HuggingFace sources.

Why this script exists
----------------------
``scripts/ingest_all_prioritized_data.py`` does a linear scan of every previously
seen MinHash signature (O(n^2)), which makes it unusable past a few thousand
documents. This script keeps the same source tiering and quality gates but uses

  * exact SHA-1 dedup, and
  * banded LSH near-dedup over cached shingle hashes (xxhash when available),

so it streams a few hundred thousand documents in minutes, and it is resumable
(shards + a state file), because IndicCorp/Wikipedia streams are slow to restart.

Sources
-------
tier 1  local classical corpora   open-sangam/data/texts, Yazhi/repos/data/texts,
                                  existing adhan raw JSONL
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
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.core.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

try:  # fast path; pure-python fallback below
    import xxhash  # type: ignore

    def _h64(data: bytes) -> int:
        return xxhash.xxh64_intdigest(data)

except Exception:  # pragma: no cover
    def _h64(data: bytes) -> int:
        return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


# --- Text hygiene -----------------------------------------------------------
TAMIL_RANGE = re.compile(r"[\u0B80-\u0BFF]")
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_RE = re.compile(r"(\+91[\-\s]?)?[6-9]\d{9}")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
REPEAT_RE = re.compile(r"(.)\1{6,}")
WS_RE = re.compile(r"[ \t]+")
NL_RE = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    """NFC-normalise, scrub PII, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = WS_RE.sub(" ", text)
    text = NL_RE.sub("\n\n", text)
    return text.strip()


def tamil_ratio(text: str) -> float:
    clean = text.replace(" ", "").replace("\n", "").replace("\t", "")
    if not clean:
        return 0.0
    return len(TAMIL_RANGE.findall(clean)) / len(clean)


def is_quality(
    text: str,
    min_chars: int = 40,
    max_chars: int = 20000,
    min_tamil_ratio: float = 0.60,
) -> bool:
    if not text:
        return False
    n = len(text)
    if n < min_chars or n > max_chars:
        return False
    if tamil_ratio(text) < min_tamil_ratio:
        return False
    if REPEAT_RE.search(text):
        return False
    # reject markup / boilerplate-heavy documents
    if text.count("<") > 20 or text.count("|") > n * 0.02:
        return False
    return True


# --- Fast banded LSH dedup --------------------------------------------------
class LshDedup:
    """Exact + near-duplicate filtering with banded LSH.

    Shingle hashes are cached so each document costs one pass over its text;
    candidates are only pulled from matching LSH bands, then verified against
    the true shingle sets (no false positives from band luck).
    """

    def __init__(
        self,
        num_perm: int = 32,
        bands: int = 8,
        shingle_k: int = 5,
        threshold: float = 0.80,
        max_shingle_chars: int = 4000,
    ) -> None:
        assert num_perm % bands == 0
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.k = shingle_k
        self.threshold = threshold
        self.max_chars = max_shingle_chars
        self.exact: Set[str] = set()
        self.buckets: List[Dict[Tuple[int, ...], List[int]]] = [
            {} for _ in range(bands)
        ]
        # Only signatures are retained (not raw shingle sets): at 500k docs the
        # raw sets cost >10 GiB, while 32 permille signatures cost ~0.6 GiB.
        self.sigs: Dict[int, Tuple[int, ...]] = {}
        self._next_id = 0
        # deterministic permutation coefficients (odd multipliers)
        self._a = [(i * 2 + 1) * 0x9E3779B97F4A7C15 % (2**64) for i in range(num_perm)]
        self._b = [(i * 0xBF58476D1CE4E5B9 + 1) % (2**64) for i in range(num_perm)]

    def _shingles(self, text: str) -> Set[int]:
        clean = text[: self.max_chars]
        if len(clean) < self.k:
            return {_h64(clean.encode("utf-8"))} if clean else set()
        return {_h64(clean[i : i + self.k].encode("utf-8")) for i in range(len(clean) - self.k + 1)}

    def _signature(self, shingles: Set[int]) -> Tuple[int, ...]:
        if not shingles:
            return tuple([0] * self.num_perm)
        return tuple(
            min((a * s + b) & 0xFFFFFFFFFFFFFFFF for s in shingles)
            for a, b in zip(self._a, self._b)
        )

    @staticmethod
    def _sig_similarity(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
        matches = sum(1 for x, y in zip(a, b) if x == y)
        return matches / len(a)

    def is_duplicate(self, text: str) -> bool:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if digest in self.exact:
            return True

        shingles = self._shingles(text)
        sig = self._signature(shingles)

        candidates: Set[int] = set()
        keys: List[Tuple[int, Tuple[int, ...]]] = []
        for band in range(self.bands):
            start = band * self.rows
            key = sig[start : start + self.rows]
            keys.append((band, key))
            bucket = self.buckets[band].get(key)
            if bucket:
                candidates.update(bucket)

        for cand in candidates:
            if self._sig_similarity(sig, self.sigs[cand]) >= self.threshold:
                return True

        doc_id = self._next_id
        self._next_id += 1
        self.exact.add(digest)
        self.sigs[doc_id] = sig
        for band, key in keys:
            self.buckets[band].setdefault(key, []).append(doc_id)
        return False


# --- Source readers ---------------------------------------------------------
def _iter_strings(obj) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_strings(v)


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
                        for s in _iter_strings(rec):
                            yield s, path.stem
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("text"), str):
                yield data["text"], path.stem
            else:
                for s in _iter_strings(data):
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
                text = normalize(text)
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
    from datasets import load_dataset

    logger.info(f"streaming Tamil Wikipedia (limit={limit:,})")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ta", split="train", streaming=True)
    n = 0
    for row in ds:
        text = normalize(row.get("text", ""))
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
    from datasets import load_dataset

    logger.info(f"streaming IndicCorpV2 tam_Taml (limit={limit:,})")
    ds = load_dataset(
        "ai4bharat/IndicCorpV2", "indiccorp_v2", split="tam_Taml", streaming=True
    )
    n = 0
    for row in ds:
        text = normalize(row.get("text", ""))
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


# --- Orchestration ----------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Expand the Adhan Tamil corpus")
    ap.add_argument("--out-dir", default="data/raw/expanded")
    ap.add_argument("--shard-size", type=int, default=20000, help="docs per jsonl shard")
    ap.add_argument("--wiki-limit", type=int, default=100000)
    ap.add_argument("--indic-limit", type=int, default=200000)
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--no-local", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument(
        "--local-roots",
        default=",".join(
            [
                str(ROOT / "data" / "raw" / "unified_prioritized"),
                str(ROOT / "data" / "raw" / "thirukkural"),
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

    shard_idx = len(sorted(out_dir.glob("corpus-*.jsonl")))
    fh = (out_dir / f"corpus-{shard_idx:05d}.jsonl").open("a", encoding="utf-8")
    in_shard = 0

    dedup = LshDedup()
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
            if name in ("wikipedia", "indiccorp"):
                done.add(name)
    finally:
        fh.close()

    total_chars = 0
    total_docs = 0
    for p in out_dir.glob("corpus-*.jsonl"):
        with p.open(encoding="utf-8") as f:
            for line in f:
                total_docs += 1
                total_chars += len(json.loads(line).get("text", ""))

    stats = {
        "documents": total_docs,
        "characters": total_chars,
        # aksharas are expensive to recount; ~0.95 tokens/char is the observed
        # swaram fertility ratio on Tamil text.
        "approx_tokens_swaram": int(total_chars * 0.95),
        "source_counts": dict(counts),
        "tier_counts": {str(k): v for k, v in tier_counts.items()},
        "dropped": dict(dropped),
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
