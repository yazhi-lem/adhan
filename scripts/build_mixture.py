#!/usr/bin/env python3
"""Build a *weighted* training mixture from the archive: data-mixture as code.

The archive's implicit mixture is whatever the ingest order produced — currently
**72% IndicCorp**, which is not a decision anyone made. This tool makes the mixture
explicit and reproducible:

    python scripts/build_mixture.py \\
        --corpus data/raw/expanded --out data/final/mix_v1 \\
        --max-tokens 30000000 \\
        --weights "ai4bharat-indicorp-v2:0.45,fineweb-2-ta:0.25,wikipedia-ta:0.15,local:0.15"

Design decisions worth knowing:

* **Deterministic subsampling.** Each source is thinned by hashing the document
  index, not by taking a prefix — otherwise a mixture would be "the first N docs of
  each source", i.e. whatever the scraper happened to fetch first.
* **Two passes, no giant buffers.** Pass 1 reads `stats.json` plus a small sample to
  estimate tokens per source; the keep-rate per source is then target/available.
  Pass 2 streams, encodes with the Rust tokenizer and packs straight to disk.
* **Realized proportions are reported, not assumed.** Weighting is an intent;
  `mixture.json` records what actually landed in the shards so a training run can be
  reproduced and audited.
* **Weights are relative, not percentages.** They are normalised, so `{a: 3, b: 1}`
  and `{a: 0.75, b: 0.25}` mean the same thing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.core.logging import get_logger  # noqa: E402
from adhan_slm.tokenizer.fast_bpe import AksharaBPE  # noqa: E402
from adhan_slm.tokenizer.swaram_tokenizer import default_akshara_inventory  # noqa: E402

logger = get_logger(__name__)


def parse_weights(spec: str) -> Dict[str, float]:
    """`"src:0.5,prefix*:0.5"` -> {pattern: weight} (relative, normalised later)."""
    out: Dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, val = part.partition(":")
        out[name.strip()] = float(val)
    total = sum(out.values())
    if total <= 0:
        raise SystemExit("weights must sum to > 0")
    return {k: v / total for k, v in out.items()}


def match_weight(source: str, weights: Dict[str, float]) -> float:
    """Exact match wins; otherwise the longest matching prefix (``local:`` etc.)."""
    if source in weights:
        return weights[source]
    best, best_len = 0.0, -1
    for pattern, w in weights.items():
        prefix = pattern.rstrip("*:")
        if source.startswith(prefix) and len(prefix) > best_len:
            best, best_len = w, len(prefix)
    return best


def _stable_hash(n: int, seed: int = 0) -> int:
    """SplitMix64 — deterministic, well-distributed, no dependency on PYTHONHASHSEED."""
    x = (n + seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def iter_shards(corpus_dir: Path) -> Iterator[Tuple[int, Dict]]:
    idx = 0
    for shard in sorted(corpus_dir.glob("*.jsonl")):
        with shard.open(encoding="utf-8") as fh:
            for line in fh:
                idx += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("text"):
                    yield idx, rec


class ShardWriter:
    """Append fixed-length sequences to a .bin shard without buffering everything."""

    def __init__(self, path: Path, seq_len: int, vocab_size: int) -> None:
        import array

        self._array = array
        self.path, self.seq_len, self.vocab_size = path, seq_len, vocab_size
        self.typecode = "H" if vocab_size <= 0xFFFF else "I"
        self.dtype = "<u2" if self.typecode == "H" else "<u4"
        self._fh = path.open("wb")
        self._buf: List[int] = []
        self.n_sequences = 0

    def add(self, seq: Sequence[int]) -> None:
        self._buf.extend(seq)
        self.n_sequences += 1
        if len(self._buf) >= self.seq_len * 2048:
            self._flush()

    def _flush(self) -> None:
        if self._buf:
            self._array.array(self.typecode, self._buf).tofile(self._fh)
            self._buf.clear()

    def close(self) -> Dict[str, int]:
        self._flush()
        self._fh.close()
        meta = {
            "path": str(self.path),
            "seq_len": self.seq_len,
            "n_sequences": self.n_sequences,
            "n_tokens": self.n_sequences * self.seq_len,
            "vocab_size": self.vocab_size,
            "dtype": self.dtype,
        }
        Path(str(self.path) + ".manifest.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return meta


def estimate_tokens_per_source(
    corpus_dir: Path, weights: Dict[str, float], sample_docs: int = 4000
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Mean tokens/doc and doc count per source group, from a small sample + stats.json."""
    stats_path = corpus_dir / "stats.json"
    counts: Dict[str, int] = {}
    if stats_path.exists():
        counts = json.loads(stats_path.read_text(encoding="utf-8")).get("source_counts", {})
    sums: Dict[str, int] = defaultdict(int)
    n: Dict[str, int] = defaultdict(int)
    seen = 0
    stride = max(1, sum(counts.values()) // max(1, sample_docs)) if counts else 1
    for idx, rec in iter_shards(corpus_dir):
        if idx % stride:
            continue
        src = rec.get("source", "?")
        if not match_weight(src, weights):
            continue
        sums[src] += len(rec["text"])
        n[src] += 1
        seen += 1
        if seen >= sample_docs:
            break
    mean_tokens = {src: (sums[src] / n[src]) * 0.311 for src in sums if n[src]}  # measured tok/char
    return mean_tokens, counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a weighted training mixture")
    ap.add_argument("--corpus", default="data/raw/expanded")
    ap.add_argument("--tokenizer", default="data/final/tamil_slm_v2")
    ap.add_argument("--out", default="data/final/mix_v1")
    ap.add_argument("--max-tokens", type=int, default=30_000_000)
    ap.add_argument(
        "--weights",
        default="ai4bharat-indiccorp-v2:0.45,fineweb-2-ta:0.25,wikipedia-ta:0.15,local:0.15",
        help="relative weights, `source:weight` or `prefix*:weight`, comma separated",
    )
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    weights = parse_weights(args.weights)
    logger.info("requested weights: %s", weights)

    tok = AksharaBPE.load(args.tokenizer, inventory=default_akshara_inventory())
    mean_tokens, counts = estimate_tokens_per_source(corpus, weights)
    avail = {
        src: mean_tokens.get(src, 0) * cnt
        for src, cnt in counts.items()
        if match_weight(src, weights)
    }

    # Allocate the budget per group (a pattern and its matching sources), then turn
    # each group's token target into a keep-rate for its members proportional to how
    # much they actually hold. A group that cannot fill its target is capped at 1.0
    # and the shortfall is reported instead of being silently redistributed.
    keep: Dict[str, float] = {}
    group_report: Dict[str, Dict[str, float]] = {}
    for pattern, weight in weights.items():
        members = [src for src in avail if src == pattern or src.startswith(pattern.rstrip("*:"))]
        if not members:
            logger.warning("no source matches weight pattern %r", pattern)
            continue
        target = weight * args.max_tokens  # weights were normalised in parse_weights
        available = sum(avail[s] for s in members)
        scale = min(1.0, target / available) if available else 0.0
        for src in members:
            keep[src] = scale
        group_report[pattern] = {
            "target_tokens": int(target),
            "available_tokens": int(available),
            "take_tokens": int(min(target, available)),
            "keep_rate": round(scale, 4),
            "sources": len(members),
        }
    shortfall = sum(g["target_tokens"] - g["take_tokens"] for g in group_report.values())
    if shortfall > 0:
        logger.warning(
            "mixture cannot reach --max-tokens: ~%s tokens unavailable; "
            "grow the corpus or lower the budget",
            f"{shortfall:,}",
        )

    train_w = ShardWriter(out / "train.bin", args.seq_len, len(tok))
    val_w = ShardWriter(out / "val.bin", args.seq_len, len(tok))
    realized: Counter = Counter()
    docs_by_source: Counter = Counter()
    batch: List[Tuple[Dict, bool]] = []

    # Tokens are packed across document boundaries, not per document. Packing each
    # document on its own silently discards every document shorter than seq_len —
    # which is most of IndicCorp (median 237 chars ≈ 80 tokens) — so a "30M-token"
    # mixture delivered 10.5M and was 45% short on exactly the short-document source.
    streams: Dict[bool, List[int]] = {False: [], True: []}
    leftover = {False: 0, True: 0}

    def push(is_val: bool, ids: List[int]) -> None:
        buf = streams[is_val]
        buf.extend(ids)
        writer = val_w if is_val else train_w
        while len(buf) >= args.seq_len:
            writer.add(buf[: args.seq_len])
            del buf[: args.seq_len]

    def flush() -> None:
        if not batch:
            return
        texts = [rec["text"] for rec, _ in batch]
        for (rec, is_val), ids in zip(batch, tok.encode_batch(texts)):
            src = rec.get("source", "?")
            seq_ids = [tok.vocab.get("<bos>", 0), *ids, tok.vocab.get("<eos>", 0)]
            realized[src] += len(seq_ids)
            docs_by_source[src] += 1
            push(is_val, seq_ids)
        batch.clear()

    for idx, rec in iter_shards(corpus):
        src = rec.get("source", "?")
        rate = keep.get(src, 0.0)
        if rate <= 0:
            continue
        # deterministic thinning: hash(index) is uniform, so the kept subset is a
        # random spread across the source rather than its first N documents
        if (_stable_hash(idx, args.seed) & 0xFFFFFF) / float(0xFFFFFF) >= rate:
            continue
        is_val = ((_stable_hash(idx, args.seed ^ 0x5EED) & 0xFFFF) / 65536.0) < args.val_frac
        batch.append((rec, is_val))
        if len(batch) >= args.batch:
            flush()

    flush()
    for is_val, buf in streams.items():
        leftover[is_val] = len(buf)
    train_meta = train_w.close()
    val_meta = val_w.close()
    # tokens consumed but not representable at this seq_len (a few hundred at most)
    packed_tokens = train_meta["n_tokens"] + val_meta["n_tokens"]

    total_realized = sum(realized.values())
    achieved = {src: round(v / max(1, total_realized), 4) for src, v in realized.most_common()}
    desired: Dict[str, float] = {}
    for pattern, weight in weights.items():
        members = [s for s in realized if s == pattern or s.startswith(pattern.rstrip("*:"))]
        if not members:
            continue
        avail_m = sum(avail.get(s, 0) for s in members) or 1.0
        for s in members:
            share = weight * (avail.get(s, 0) / avail_m)
            desired[s] = round(share, 4)

    datasheet = {
        "corpus": str(corpus),
        "tokenizer": args.tokenizer,
        "vocab_size": len(tok),
        "seq_len": args.seq_len,
        "max_tokens": args.max_tokens,
        "requested_weights": weights,
        "group_plan": group_report,
        "realized_tokens": dict(realized.most_common()),
        "realized_share": achieved,
        "requested_share_within_realized": desired,
        "documents": dict(docs_by_source.most_common()),
        "tokens_fed_to_packer": total_realized,
        "tokens_packed": packed_tokens,
        "tokens_left_in_partial_window": leftover[False] + leftover[True],
        "train": train_meta,
        "val": val_meta,
        "total_tokens": total_realized,
    }
    (out / "mixture.json").write_text(json.dumps(datasheet, indent=2), encoding="utf-8")
    tok.save(out)

    logger.info("mixture written to %s", out)
    logger.info(
        "tokens %s (target %s) | packed %s | train %s seqs, val %s seqs",
        f"{total_realized:,}",
        f"{args.max_tokens:,}",
        f"{packed_tokens:,}",
        f"{train_meta['n_sequences']:,}",
        f"{val_meta['n_sequences']:,}",
    )
    print("\nrealized mixture vs requested:")
    for src in sorted(achieved, key=lambda s: -realized[s])[:10]:
        print(
            f"  {src:34} {achieved[src] * 100:5.1f}% realized"
            f"   (requested {desired.get(src, 0) * 100:5.1f}%)"
        )


if __name__ == "__main__":
    main()
