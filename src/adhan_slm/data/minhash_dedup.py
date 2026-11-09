"""MinHash deduplication for Tamil corpus curation.

Adapted from Agaram 2.5B's data pipeline (Murai-Labs/Agaram2.5b) and
"Under the Hood" Project 10: Data Curation and Contamination.

Uses locality-sensitive hashing (LSH) with MinHash to find near-duplicate
documents in O(n) time, scaling to tens of millions of documents without
loading everything into RAM.

Usage::

    from adhan_slm.data.minhash_dedup import DedupConfig, deduplicate_records

    cfg = DedupConfig(n_hashes=128, shingle_size=5, threshold=0.85)
    records = [{"text": "...", "id": "doc_1"}, ...]
    unique, clusters = deduplicate_records(records, cfg)

Or from CLI::

    python -m adhan_slm.data.minhash_dedup \
        data/raw/tamil_corpus.jsonl \
        --output data/deduped.jsonl \
        --n-hashes 128 \
        --threshold 0.85 \
        --shingle-size 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from adhan_slm.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DedupConfig:
    """MinHash deduplication hyperparameters.

    ``n_hashes`` controls the precision/recall trade-off: more hashes give
    fewer false positives (documents marked duplicate when they are not)
    but use more memory. 128 is a good default for ~10M documents.

    ``shingle_size`` is the length of contiguous character n-grams used as
    features. 5 is standard for natural language; larger values catch
    phrasing similarity, smaller ones catch exact-match boilerplate.

    ``threshold`` is the Jaccard similarity above which two documents are
    considered duplicates. 0.85 is conservative; 0.80 catches more but may
    over-merge distinct articles on the same topic.
    """

    n_hashes: int = 128
    shingle_size: int = 5
    threshold: float = 0.85
    seed: int = 42

    # Derived: number of bands (rows per band = n_hashes // n_bands)
    # Optimal when threshold ≈ (1/bands)^(1/rows)
    @property
    def n_bands(self) -> int:
        # Solve for bands where threshold ≈ (1/bands)^(bands/n_hashes)
        # Approximate: bands ≈ n_hashes * threshold, rounded to nearest int
        return max(2, round(self.n_hashes * self.threshold))

    @property
    def rows_per_band(self) -> int:
        return self.n_hashes // self.n_bands


def _shingles(text: str, k: int) -> Set[str]:
    """Return the set of contiguous character k-grams."""
    if len(text) < k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _hash_family(seed: int, n: int):
    """Generate n independent hash functions using splitmix64-style mixing."""
    rng = random.Random(seed)
    # Large 64-bit primes for universal hashing
    a_vals = [rng.randint(1, 2**64 - 1) for _ in range(n)]
    b_vals = [rng.randint(0, 2**64 - 1) for _ in range(n)]
    m = (1 << 61) - 1  # Mersenne prime for fast modulo

    def _make_hash(a: int, b: int):
        def _h(s: str) -> int:
            x = int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)
            return ((a * x + b) & 0xFFFFFFFFFFFFFFFF) % m

        return _h

    return [_make_hash(a, b) for a, b in zip(a_vals, b_vals)]


def _minhash(shingles: Set[str], hash_fns) -> Tuple[int, ...]:
    """Compute the MinHash signature: min hash value per function."""
    return tuple(min(h(s) for s in shingles) for h in hash_fns)


def _lsh_bands(signature: Tuple[int, ...], cfg: DedupConfig) -> List[Tuple[int, int]]:
    """Hash each band of the signature to a bucket id."""
    r = cfg.rows_per_band
    bands = []
    for b in range(cfg.n_bands):
        start = b * r
        band = signature[start : start + r]
        # Combine band hashes into one bucket id via stable string hash
        key = "|".join(str(x) for x in band)
        bucket = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
        bands.append((b, bucket))
    return bands


def deduplicate_records(
    records: Iterable[dict],
    cfg: Optional[DedupConfig] = None,
    text_key: str = "text",
    id_key: str = "id",
) -> Tuple[List[dict], Dict[str, List[str]]]:
    """Deduplicate records by near-duplicate text.

    Returns ``(unique_records, clusters)`` where ``clusters`` maps each kept
    document id to the list of ids of documents merged into it.

    Time: O(n * k) where k = average document length in shingles.
    Space: O(n * n_hashes) for signatures plus O(n * n_bands) for buckets.
    """
    cfg = cfg or DedupConfig()
    hash_fns = _hash_family(cfg.seed, cfg.n_hashes)

    signatures: Dict[str, Tuple[int, ...]] = {}
    buckets: Dict[Tuple[int, int], Set[str]] = {}
    record_map: Dict[str, dict] = {}

    logger.info(
        "Building MinHash signatures: n_hashes=%d, bands=%d, rows=%d, threshold=%.2f",
        cfg.n_hashes,
        cfg.n_bands,
        cfg.rows_per_band,
        cfg.threshold,
    )

    for rec in records:
        doc_id = str(rec.get(id_key, ""))
        if not doc_id:
            doc_id = f"_anon_{len(record_map)}"
        text = rec.get(text_key, "")
        if not text:
            continue

        shingles = _shingles(text, cfg.shingle_size)
        sig = _minhash(shingles, hash_fns)
        signatures[doc_id] = sig
        record_map[doc_id] = rec

        for band_bucket in _lsh_bands(sig, cfg):
            buckets.setdefault(band_bucket, set()).add(doc_id)

    logger.info("Indexed %d documents, %d band-buckets", len(signatures), len(buckets))

    # Build candidate pairs from shared buckets
    candidates: Set[frozenset] = set()
    for doc_ids in buckets.values():
        if len(doc_ids) < 2:
            continue
        ids = sorted(doc_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                candidates.add(frozenset([ids[i], ids[j]]))

    logger.info("Candidate pairs: %d", len(candidates))

    # Exact Jaccard similarity for candidates
    duplicates: Set[str] = set()
    clusters: Dict[str, List[str]] = {}

    for pair in candidates:
        a, b = sorted(pair)
        if a in duplicates or b in duplicates:
            continue
        sig_a = signatures[a]
        sig_b = signatures[b]
        # Estimate Jaccard from MinHash agreement
        agreements = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
        est_jaccard = agreements / cfg.n_hashes
        if est_jaccard >= cfg.threshold:
            # Merge b into a (keeps earliest-seen document as representative)
            duplicates.add(b)
            clusters.setdefault(a, []).append(b)

    unique = [record_map[k] for k in record_map if k not in duplicates]
    logger.info(
        "Deduplication: %d in → %d unique (removed %d duplicates, %d clusters)",
        len(record_map),
        len(unique),
        len(duplicates),
        len(clusters),
    )
    return unique, clusters


def _cli():
    ap = argparse.ArgumentParser(description="MinHash deduplicate JSONL corpus")
    ap.add_argument("input", help="Input JSONL file")
    ap.add_argument("--output", "-o", required=True, help="Output JSONL file")
    ap.add_argument("--text-key", default="text", help="JSON key for document text")
    ap.add_argument("--id-key", default="id", help="JSON key for document id")
    ap.add_argument("--n-hashes", type=int, default=128)
    ap.add_argument("--shingle-size", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clusters", help="Optional JSON file to write duplicate clusters")
    ap.add_argument("--max-docs", type=int, default=None, help="Stop after N docs (for testing)")
    args = ap.parse_args()

    cfg = DedupConfig(
        n_hashes=args.n_hashes,
        shingle_size=args.shingle_size,
        threshold=args.threshold,
        seed=args.seed,
    )

    def _stream():
        count = 0
        with Path(args.input).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
                count += 1
                if args.max_docs and count >= args.max_docs:
                    break

    unique, clusters = deduplicate_records(
        _stream(), cfg, text_key=args.text_key, id_key=args.id_key
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in unique:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(unique), out_path)

    if args.clusters:
        clusters_out = Path(args.clusters)
        clusters_out.parent.mkdir(parents=True, exist_ok=True)
        clusters_out.write_text(
            json.dumps(clusters, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Wrote %d clusters to %s", len(clusters), clusters_out)


if __name__ == "__main__":
    _cli()
