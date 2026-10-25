#!/usr/bin/env python3
"""Expand Adhan Tamil corpus — Phase 2: Multi-source mega-ingestion.

Targets 1.5–2 BILLION clean swaram tokens by adding:
  • mC4/ta (Colossal Clean Crawled Corpus) — ~500M tokens
  • CC-100/ta (Facebook CommonCrawl) — ~200M tokens
  • OSCAR/ta (Open Super-large Crawled Aggregated coRpus) — ~150M tokens
  • Tamil News Corpus (HF: SEBIS/tamil_news) — ~50M tokens
  • Tamil Books / Project Madurai expansion — ~30M tokens
  • Tamil Wikipedia expanded (full dump, not limited) — ~50M tokens
  • Tamil WebText (cleaned CommonCrawl snapshots) — ~100M tokens
  • IndicCorp v2 unlimited (raise limit from 1M → 3M docs)
  • FineWeb-2 unlimited (raise limit from 157K → 500K docs)

Quality strategy: less aggressive filtering for Tier 2+ sources
(they've already been cleaned by upstream), aggressive for raw crawl.

Usage:
    # Full expansion (hours, needs 50GB disk, 16GB RAM)
    python scripts/expand_tamil_corpus_v2.py --full

    # Incremental — only new sources
    python scripts/expand_tamil_corpus_v2.py \
        --mc4-limit 1000000 \
        --cc100-limit 500000 \
        --oscar-limit 300000 \
        --indic-limit 3000000 \
        --fineweb-limit 1000000

    # Resume after interruption
    python scripts/expand_tamil_corpus_v2.py --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.core.logging import get_logger

logger = get_logger(__name__)

try:
    import xxhash
    def _h64(data: bytes) -> int:
        return xxhash.xxh64_intdigest(data)
except Exception:
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
    *,
    min_chars: int = 40,
    max_chars: int = 50000,
    min_tamil_ratio: float = 0.55,
    max_markup: int = 30,
    max_pipe_ratio: float = 0.03,
) -> bool:
    """Quality gate — tunable per source."""
    if not text:
        return False
    n = len(text)
    if n < min_chars or n > max_chars:
        return False
    if tamil_ratio(text) < min_tamil_ratio:
        return False
    if REPEAT_RE.search(text):
        return False
    if text.count("<") > max_markup or text.count("|") > n * max_pipe_ratio:
        return False
    # Block common boilerplate patterns
    boiler = ["javascript", "cookie", "privacy policy", "terms of use",
              "all rights reserved", "click here to", "subscribe now",
              "பதிவு செய்ய", "விளம்பரம்", "advertisement"]
    lower = text.lower()
    if sum(1 for b in boiler if b in lower) >= 3:
        return False
    return True


# --- Fast banded LSH dedup (same as v1, but with larger cache) --------------
class LshDedup:
    def __init__(
        self,
        num_perm: int = 16,
        bands: int = 4,
        shingle_k: int = 5,
        threshold: float = 0.82,
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
        self.buckets: List[Dict[Tuple[int, ...], List[int]]] = [{} for _ in range(bands)]
        self.sigs: Dict[int, Tuple[int, ...]] = {}
        self._next_id = 0
        self._a = [(i * 2 + 1) * 0x9E3779B97F4A7C15 % (2**64) for i in range(num_perm)]
        self._b = [(i * 0xBF58476D1CE4E5B9 + 1) % (2**64) for i in range(num_perm)]

    def _shingles(self, text: str) -> Set[int]:
        clean = text[:self.max_chars]
        if len(clean) < self.k:
            return {_h64(clean.encode("utf-8"))} if clean else set()
        return {_h64(clean[i:i+self.k].encode("utf-8"))
                for i in range(len(clean) - self.k + 1)}

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

    def seed_exact(self, text: str) -> None:
        self.exact.add(hashlib.sha1(text.encode("utf-8")).hexdigest())

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
            key = sig[start:start + self.rows]
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
                if is_quality(text, min_chars=30, min_tamil_ratio=0.50):
                    yield {
                        "text": text,
                        "source": f"local:{sub}",
                        "tier": 1,
                        "quality_score": 0.95,
                    }


# --- HuggingFace dataset streams -------------------------------------------

def iter_wikipedia(limit: int) -> Iterator[Dict]:
    if limit <= 0:
        return
    from datasets import load_dataset
    logger.info(f"streaming Tamil Wikipedia (limit={limit:,})")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ta", split="train", streaming=True)
    n = 0
    for row in ds:
        text = normalize(row.get("text", ""))
        if is_quality(text, min_chars=80, min_tamil_ratio=0.55):
            yield {"text": text, "source": "wikipedia-ta", "tier": 2, "quality_score": 0.92}
            n += 1
            if n >= limit:
                break
    logger.info(f"wikipedia: {n:,} accepted")


def iter_indiccorp(limit: int) -> Iterator[Dict]:
    if limit <= 0:
        return
    from datasets import load_dataset
    logger.info(f"streaming IndicCorpV2 tam_Taml (limit={limit:,})")
    ds = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split="tam_Taml", streaming=True)
    n = 0
    for row in ds:
        text = normalize(row.get("text", ""))
        if is_quality(text, min_chars=50, min_tamil_ratio=0.60):
            yield {"text": text, "source": "indiccorp-v2", "tier": 3, "quality_score": 0.82}
            n += 1
            if n >= limit:
                break
    logger.info(f"indiccorp: {n:,} accepted")


def iter_fineweb2(limit: int, min_lang_score: float = 0.88) -> Iterator[Dict]:
    if limit <= 0:
        return
    from datasets import load_dataset
    logger.info(f"streaming FineWeb-2 tam_Taml (limit={limit:,})")
    ds = load_dataset("HuggingFaceFW/fineweb-2", "tam_Taml", split="train", streaming=True)
    n = 0
    for row in ds:
        score = row.get("language_score")
        if score is not None and float(score) < min_lang_score:
            continue
        text = normalize(row.get("text", ""))
        if is_quality(text, min_chars=60, min_tamil_ratio=0.60):
            yield {"text": text, "source": "fineweb-2", "tier": 3,
                   "quality_score": round(min(0.95, float(score or 0.85)), 4)}
            n += 1
            if n >= limit:
                break
    logger.info(f"fineweb-2: {n:,} accepted")


def iter_mc4(limit: int) -> Iterator[Dict]:
    """mC4 Tamil — Colossal Clean Crawled Corpus. Massive but noisier."""
    if limit <= 0:
        return
    from datasets import load_dataset
    logger.info(f"streaming mC4/ta (limit={limit:,})")
    # mC4 uses 'ta' language code
    ds = load_dataset("allenai/c4", "ta", split="train", streaming=True)
    n = 0
    for row in ds:
        text = normalize(row.get("text", ""))
        # mC4 is already cleaned by upstream; use relaxed gate
        if is_quality(text, min_chars=80, min_tamil_ratio=0.50, max_markup=50):
            yield {"text": text, "source": "mc4", "tier": 3, "quality_score": 0.78}
            n += 1
            if n >= limit:
                break
    logger.info(f"mc4: {n:,} accepted")


def iter_cc100(limit: int) -> Iterator[Dict]:
    """CC-100 Tamil — Facebook CommonCrawl extraction."""
    if limit <= 0:
        return
    from datasets import load_dataset
    logger.info(f"streaming CC-100 tam (limit={limit:,})")
    ds = load_dataset("facebook/cc100", "ta", split="train", streaming=True)
    n = 0
    for row in ds:
        text = normalize(row.get("text", ""))
        if is_quality(text, min_chars=60, min_tamil_ratio=0.50):
            yield {"text": text, "source": "cc100", "tier": 3, "quality_score": 0.75}
            n += 1
            if n >= limit:
                break
    logger.info(f"cc100: {n:,} accepted")


def iter_oscar(limit: int) -> Iterator[Dict]:
    """OSCAR Tamil — deduplicated CommonCrawl."""
    if limit <= 0:
        return
    from datasets import load_dataset
    logger.info(f"streaming OSCAR unshuffled/deduplicated ta (limit={limit:,})")
    ds = load_dataset("oscar-corpus/OSCAR-2301", "ta", split="train", streaming=True)
    n = 0
    for row in ds:
        text = normalize(row.get("text", ""))
        if is_quality(text, min_chars=60, min_tamil_ratio=0.50):
            yield {"text": text, "source": "oscar", "tier": 3, "quality_score": 0.76}
            n += 1
            if n >= limit:
                break
    logger.info(f"oscar: {n:,} accepted")


def iter_tamil_news(limit: int) -> Iterator[Dict]:
    """Tamil News Dataset — high-quality journalism text."""
    if limit <= 0:
        return
    from datasets import load_dataset
    logger.info(f"streaming Tamil News (limit={limit:,})")
    # Try multiple news datasets
    datasets_to_try = [
        ("SEBIS/tamil_news", None),
        ("kdebnath/tamil_news_dataset", None),
    ]
    n = 0
    for ds_name, config in datasets_to_try:
        try:
            ds = load_dataset(ds_name, config, split="train", streaming=True)
            for row in ds:
                text = normalize(row.get("text", row.get("content", row.get("article", ""))))
                if is_quality(text, min_chars=100, min_tamil_ratio=0.60):
                    yield {"text": text, "source": "tamil-news", "tier": 2, "quality_score": 0.88}
                    n += 1
                    if n >= limit:
                        break
            if n >= limit:
                break
        except Exception as e:
            logger.warning(f"news dataset {ds_name} failed: {e}")
            continue
    logger.info(f"tamil-news: {n:,} accepted")


def iter_tamil_books(limit: int) -> Iterator[Dict]:
    """Tamil books from HF datasets — literary prose."""
    if limit <= 0:
        return
    from datasets import load_dataset
    logger.info(f"streaming Tamil Books (limit={limit:,})")
    datasets_to_try = [
        "Tamil-NLP/Tamil_Books",
    ]
    n = 0
    for ds_name in datasets_to_try:
        try:
            ds = load_dataset(ds_name, split="train", streaming=True)
            for row in ds:
                text = normalize(row.get("text", row.get("content", "")))
                if is_quality(text, min_chars=200, min_tamil_ratio=0.55):
                    yield {"text": text, "source": "tamil-books", "tier": 1, "quality_score": 0.92}
                    n += 1
                    if n >= limit:
                        break
            if n >= limit:
                break
        except Exception as e:
            logger.warning(f"books dataset {ds_name} failed: {e}")
            continue
    logger.info(f"tamil-books: {n:,} accepted")


# --- Orchestration ----------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Expand Adhan Tamil corpus — Phase 2 mega-ingestion")
    ap.add_argument("--out-dir", default="data/raw/expanded_v2")
    ap.add_argument("--shard-size", type=int, default=25000)
    ap.add_argument("--resume", action="store_true", help="resume from state.json")
    ap.add_argument("--full", action="store_true", help="run with default full limits")

    # Source limits
    ap.add_argument("--wiki-limit", type=int, default=300000)
    ap.add_argument("--indic-limit", type=int, default=3000000)
    ap.add_argument("--fineweb-limit", type=int, default=1000000)
    ap.add_argument("--mc4-limit", type=int, default=0)
    ap.add_argument("--cc100-limit", type=int, default=0)
    ap.add_argument("--oscar-limit", type=int, default=0)
    ap.add_argument("--news-limit", type=int, default=0)
    ap.add_argument("--books-limit", type=int, default=0)
    ap.add_argument("--local-roots", default=",".join([
        str(ROOT.parent / "open-sangam" / "data" / "texts"),
        str(ROOT.parent / "data" / "texts"),
    ]))
    ap.add_argument("--no-local", action="store_true")
    ap.add_argument("--seed-v1", default="data/raw/expanded",
                    help="path to v1 corpus for cross-version dedup")
    args = ap.parse_args()

    # --full sets all major sources
    if args.full:
        args.wiki_limit = 500000
        args.indic_limit = 3000000
        args.fineweb_limit = 1500000
        args.mc4_limit = 2000000
        args.cc100_limit = 1000000
        args.oscar_limit = 500000
        args.news_limit = 500000
        args.books_limit = 200000
        logger.info("FULL MODE: ingesting all major sources at maximum limits")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    stats_path = out_dir / "stats.json"
    state = json.loads(state_path.read_text()) if (args.resume and state_path.exists()) else {}
    done: Set[str] = set(state.get("done_sources", []))

    local_roots = [Path(p) for p in args.local_roots.split(",")
                   if p.strip() and Path(p).resolve() != out_dir.resolve()]

    sources: List[Tuple[str, Iterable[Dict]]] = []
    if not args.no_local:
        sources.append(("local", iter_local_corpus(local_roots)))
    for src_name, limit, fn in [
        ("wikipedia", args.wiki_limit, iter_wikipedia),
        ("indiccorp", args.indic_limit, iter_indiccorp),
        ("fineweb2", args.fineweb_limit, iter_fineweb2),
        ("mc4", args.mc4_limit, iter_mc4),
        ("cc100", args.cc100_limit, iter_cc100),
        ("oscar", args.oscar_limit, iter_oscar),
        ("news", args.news_limit, iter_tamil_news),
        ("books", args.books_limit, iter_tamil_books),
    ]:
        if limit > 0 and src_name not in done:
            sources.append((src_name, fn(limit)))
        elif src_name in done:
            logger.info(f"{src_name}: already complete (state.json), skipping")

    shard_idx = len(sorted(out_dir.glob("corpus-*.jsonl")))
    fh = (out_dir / f"corpus-{shard_idx:05d}.jsonl").open("a", encoding="utf-8")
    in_shard = 0

    dedup = LshDedup(threshold=0.82)

    # Seed from v1 corpus for cross-version dedup
    v1_path = Path(args.seed_v1)
    if v1_path.exists() and not args.resume:
        seeded = 0
        for p in sorted(v1_path.glob("corpus-*.jsonl")):
            with p.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        dedup.seed_exact(json.loads(line)["text"])
                        seeded += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
        logger.info(f"seeded {seeded:,} v1 documents for cross-version dedup")

    # Seed from our own existing shards
    existing = sorted(out_dir.glob("corpus-*.jsonl"))
    if existing and args.resume:
        seeded = 0
        for p in existing:
            with p.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        dedup.seed_exact(json.loads(line)["text"])
                        seeded += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
        logger.info(f"seeded {seeded:,} existing v2 documents for resume dedup")

    counts: Counter = Counter()
    tier_counts: Counter = Counter()
    dropped = Counter()
    kept = 0
    chars_kept = 0

    def rotate() -> None:
        nonlocal fh, shard_idx, in_shard
        fh.close()
        shard_idx += 1
        in_shard = 0
        new_path = out_dir / f"corpus-{shard_idx:05d}.jsonl"
        logger.info(f"rotating shard -> {new_path.name} ({kept:,} docs, {chars_kept:,} chars)")
        fh = new_path.open("a", encoding="utf-8")

    try:
        for name, iterator in sources:
            logger.info(f"=== source: {name} ===")
            for rec in iterator:
                text = rec["text"]
                if dedup.is_duplicate(text):
                    dropped["duplicate"] += 1
                    continue
                rec["id"] = f"{rec['source']}-{kept}"
                rec["chars"] = len(text)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[rec["source"]] += 1
                tier_counts[rec["tier"]] += 1
                kept += 1
                chars_kept += len(text)
                in_shard += 1
                if in_shard >= args.shard_size:
                    rotate()
                if kept % 10000 == 0:
                    logger.info(
                        f"  kept={kept:,}  chars={chars_kept:,}  "
                        f"dropped={sum(dropped.values()):,}  "
                        f"top={counts.most_common(3)}"
                    )
                    fh.flush()
                    state.update({"done_sources": sorted(done), "kept": kept,
                                  "shards": shard_idx + 1, "chars": chars_kept})
                    state_path.write_text(json.dumps(state, indent=2))
            done.add(name)
    finally:
        fh.close()

    # Write final stats
    approx_tokens = int(chars_kept / 3.2)  # swaram ~3.2 chars/token
    stats = {
        "documents": kept,
        "characters": chars_kept,
        "approx_tokens_swaram": approx_tokens,
        "source_counts": dict(counts),
        "tier_counts": dict(tier_counts),
        "dropped": dict(dropped),
        "shards": shard_idx + 1,
        "out_dir": str(out_dir),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    logger.info(
        f"FINAL: {kept:,} docs | {chars_kept:,} chars | ~{approx_tokens:,} tokens | "
        f"{shard_idx + 1} shards"
    )
    for src, cnt in counts.most_common():
        logger.info(f"  {src:25s}: {cnt:,}")


if __name__ == "__main__":
    main()
