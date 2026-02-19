#!/usr/bin/env python3
"""
merge_corpora.py - Merge ADHAN and VAZHI datasets into a unified Tamil training corpus.

Usage:
    python scripts/merge_corpora.py \
        --adhan_dir data/final/tamil_texts/hf \
        --vazhi_repo ../vazhi \
        --output data/unified/tamil_6k.jsonl \
        --target_count 6000
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

# Domain-specific weights for VAZHI knowledge packs
DOMAIN_WEIGHTS = {
    "security": 4.0,
    "government": 3.5,
    "healthcare": 3.0,
    "culture": 2.5,
    "education": 2.0,
    "legal": 2.0,
}

# Tamil language gets full weight; English gets 30%
TAMIL_LANG_WEIGHT = 1.0
ENGLISH_LANG_WEIGHT = 0.3


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_adhan_corpus(adhan_dir: Path) -> list[dict]:
    """Load records from ADHAN train/validation/test JSONL files."""
    records = []
    splits = ["train", "validation", "test"]
    for split in splits:
        path = adhan_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text", "")
                if not text:
                    continue
                records.append(
                    {
                        "text": text,
                        "source": f"adhan_{obj.get('source', split)}",
                        "quality_score": float(obj.get("quality_score", 1.0)),
                        "weight": 1.0,
                    }
                )
    return records


def load_vazhi_packs(vazhi_repo: Path) -> list[dict]:
    """Load records from VAZHI knowledge-pack JSON files."""
    packs_dir = vazhi_repo / "vazhi-packs"
    if not packs_dir.exists():
        return []

    records = []
    for pack_file in sorted(packs_dir.glob("*_pack.json")):
        # Infer domain from filename, e.g. government_pack.json -> government
        domain = pack_file.stem.replace("_pack", "").lower()
        domain_weight = DOMAIN_WEIGHTS.get(domain, 1.0)

        try:
            with pack_file.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue

        # Support both a list of conversations and a dict with a "conversations" key
        if isinstance(data, list):
            conversations = data
        elif isinstance(data, dict):
            conversations = data.get("conversations", [])
        else:
            continue

        for conv in conversations:
            if not isinstance(conv, dict):
                continue
            for lang, lang_weight in [("ta", TAMIL_LANG_WEIGHT), ("en", ENGLISH_LANG_WEIGHT)]:
                text = conv.get(lang, "")
                if not text:
                    continue
                quality_score = float(conv.get("quality_score", 1.0))
                effective_weight = domain_weight * lang_weight
                records.append(
                    {
                        "text": text,
                        "source": f"vazhi_{domain}",
                        "quality_score": quality_score,
                        "weight": effective_weight,
                    }
                )
    return records


def deduplicate(records: list[dict]) -> list[dict]:
    """Remove exact-text duplicates, keeping the occurrence with the highest quality_score."""
    seen: dict[str, dict] = {}
    for rec in records:
        h = sha256(rec["text"])
        if h not in seen:
            seen[h] = rec
        else:
            # Prefer the record with the higher quality_score
            if rec["quality_score"] > seen[h]["quality_score"]:
                seen[h] = rec
    return list(seen.values())


def weighted_sample(records: list[dict], target: int, seed: int = 42) -> list[dict]:
    """Sample *target* records proportionally to (quality_score * weight).

    If fewer records than *target* are available, all records are returned.
    """
    if len(records) <= target:
        return records

    weights = [max(r["quality_score"] * r["weight"], 1e-9) for r in records]
    rng = random.Random(seed)
    sampled = rng.choices(records, weights=weights, k=target)
    return sampled


def print_distribution(records: list[dict]) -> None:
    total = len(records)
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        counts[rec["source"]] += 1
    print("\n📊 Final distribution:")
    for source, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total else 0
        print(f"   {source}: {count:,} ({pct:.1f}%)")


def write_jsonl(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(
                json.dumps(
                    {
                        "text": rec["text"],
                        "source": rec["source"],
                        "quality_score": rec["quality_score"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_splits(records: list[dict], output: Path, seed: int = 42) -> None:
    """Write train/val/test splits (80/10/10) alongside the main output file."""
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)
    splits = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }
    stem = output.stem
    for split_name, split_records in splits.items():
        split_path = output.parent / f"{stem}_{split_name}.jsonl"
        write_jsonl(split_records, split_path)
        print(f"   ✅ {split_name}: {len(split_records):,} records → {split_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge ADHAN + VAZHI datasets into a unified Tamil training corpus."
    )
    parser.add_argument(
        "--adhan_dir",
        type=Path,
        default=Path("data/final/tamil_texts/hf"),
        help="Directory containing ADHAN JSONL files (train/validation/test.jsonl).",
    )
    parser.add_argument(
        "--vazhi_repo",
        type=Path,
        default=Path("../vazhi"),
        help="Path to the VAZHI repository root (expects vazhi-packs/ subdirectory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/unified/tamil_6k.jsonl"),
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--target_count",
        type=int,
        default=6000,
        help="Target number of records in the merged output.",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Also write train/val/test splits (80/10/10) alongside the main output file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load ADHAN ---
    print("📚 Loading ADHAN corpus...")
    if not args.adhan_dir.exists():
        print(f"   ❌ ADHAN directory not found: {args.adhan_dir}")
        raise SystemExit(1)
    adhan_records = load_adhan_corpus(args.adhan_dir)
    print(f"   ✅ {len(adhan_records):,} records")

    # --- Load VAZHI ---
    print("\n📦 Loading VAZHI knowledge packs...")
    vazhi_records: list[dict] = []
    if not args.vazhi_repo.exists():
        print(f"   ⚠️  VAZHI repo not found at {args.vazhi_repo} — continuing with ADHAN only.")
    else:
        vazhi_records = load_vazhi_packs(args.vazhi_repo)
        if vazhi_records:
            print(f"   ✅ {len(vazhi_records):,} records")
        else:
            print("   ⚠️  No VAZHI pack files found — continuing with ADHAN only.")

    # --- Merge & deduplicate ---
    combined = adhan_records + vazhi_records
    print(f"\n🔗 Total before dedup: {len(combined):,}")
    deduped = deduplicate(combined)
    print(f"   ✅ After dedup: {len(deduped):,}")

    # --- Weighted sampling ---
    sampled = weighted_sample(deduped, args.target_count, seed=args.seed)
    print(f"\n   ✅ Final sampled: {len(sampled):,} / {args.target_count:,} (target)")

    # --- Statistics ---
    print_distribution(sampled)

    # --- Write output ---
    write_jsonl(sampled, args.output)
    print(f"\n✅ Saved to {args.output}")

    if args.split:
        print("\n📂 Writing train/val/test splits...")
        write_splits(sampled, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
