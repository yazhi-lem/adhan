#!/usr/bin/env python3
"""Train (only train) a Swaram/Aksharam tokenizer from a stated corpus mix.

`prepare_slm_corpus.py` trains *and* packs; freezing a vocab is a decision of its
own, and an experiment on the merge budget should not pay for 300M tokens of
shard packing. This script does step 3 alone:

    python scripts/train_tokenizer.py --vocab-size 64000 \
        --corpus data/raw/expanded --chars 20000000 \
        --corpus data/raw/tanglish/tanglish_corpus.jsonl --chars 6000000 \
        --out data/final/swaram_v4_64k

What it writes:
  vocab.json, merges.txt, tokenizer.json   the frozen tokenizer
  datasheet.json                           the exact recipe (paths, char counts,
                                           inventory size, merge count, seconds)

The sampling rule is deterministic: shards in sorted filename order, documents in
file order, first ``--chars`` characters per ``--corpus`` entry. No shuffling, so
two runs on the same inputs produce the same vocab.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.common.scripts import DEVANAGARI, TAMIL  # noqa: E402
from adhan_slm.tokenizer import indic_akshara  # noqa: E402
from adhan_slm.tokenizer.fast_bpe import AksharaBPE  # noqa: E402
from adhan_slm.tokenizer.swaram_tokenizer import default_akshara_inventory  # noqa: E402

SCRIPTS = {"tamil": TAMIL, "devanagari": DEVANAGARI}


def documents(path: Path) -> Iterator[str]:
    """Yield document text from a *.jsonl file or a directory of them."""
    paths: Sequence[Path] = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for file in paths:
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if file.suffix == ".jsonl":
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = record.get("text") if isinstance(record, dict) else record
                else:
                    text = line
                if isinstance(text, str) and text:
                    yield text


def sample(entries: List[tuple[Path, int]]) -> List[str]:
    """Take the first N characters from each corpus entry, in file order."""
    out: List[str] = []
    for path, budget in entries:
        taken = 0
        for text in documents(path):
            out.append(text)
            taken += len(text)
            if taken >= budget:
                break
        print(f"  {path}: {len(out)} docs so far, {taken:,} chars from this source")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", action="append", required=True, help="*.jsonl file or dir")
    ap.add_argument(
        "--chars",
        action="append",
        type=int,
        help="character budget for the matching --corpus (default: whole file)",
    )
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--script", choices=sorted(SCRIPTS), default="tamil")
    ap.add_argument("--no-byte-fallback", action="store_true")
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--progress", action="store_true", help="show tokenizers progress bar")
    ap.add_argument(
        "--extend-inventory-min-count",
        type=int,
        default=0,
        help="also make aksharas the table lacks atomic, if seen at least N times",
    )
    args = ap.parse_args()

    budgets = list(args.chars or [])
    entries: List[tuple[Path, int]] = []
    for i, spec in enumerate(args.corpus):
        path = Path(spec) if Path(spec).is_absolute() else ROOT / spec
        if not path.exists():
            print(f"ERROR: corpus not found: {path}", file=sys.stderr)
            return 2
        entries.append((path, budgets[i] if i < len(budgets) else 10**12))

    script = SCRIPTS[args.script]
    inventory = (
        default_akshara_inventory() if script is TAMIL else indic_akshara.akshara_inventory(script)
    )
    print(f"  script={script.name} inventory={len(inventory)} vocab_size={args.vocab_size:,}")

    print("[1/2] sampling corpus")
    docs = sample(entries)
    chars = sum(len(d) for d in docs)
    print(f"  {len(docs):,} docs, {chars:,} chars")

    if args.extend_inventory_min_count:
        # Data-driven inventory extension. An akshara the table does not know
        # cannot be protected by a composition merge, so a learned merge can cut
        # it. Adding the *systematic* ones (a ZWJ, a doubled vowel sign) as their
        # own vocabulary entries makes them atomic too; one-off typos are left
        # alone so the vocabulary does not fill with noise.
        observed: Counter = Counter()
        for doc in docs:
            for cluster in indic_akshara.segment(doc, [script]):
                if cluster in inventory or not cluster.strip():
                    continue
                # ZWJ/ZWNJ are format characters, not part of an akshara: giving
                # them their own composed token makes the Rust and Python paths
                # rank-compete over it. Leave them to byte fallback.
                if any(ch in cluster for ch in "\u200c\u200d"):
                    continue
                observed[cluster] += 1
        extra = [c for c, n in observed.most_common() if n >= args.extend_inventory_min_count]
        if extra:
            inventory = list(inventory) + extra
            print(
                f"  inventory extended by {len(extra)} cluster(s) seen >= "
                f"{args.extend_inventory_min_count}x (of {len(observed)} unknown)"
            )

    print("[2/2] training tokenizer")
    started = time.perf_counter()
    tok = AksharaBPE.train(
        docs,
        inventory,
        vocab_size=args.vocab_size,
        script=script,
        byte_fallback=not args.no_byte_fallback,
        min_frequency=args.min_frequency,
        show_progress=args.progress,
    )
    seconds = time.perf_counter() - started

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    datasheet = {
        "tokenizer": "swaram" if script is TAMIL else "aksharam",
        "script": script.name,
        "backend": "rust",
        "vocab_size": len(tok),
        "n_merges": len(tok.merges),
        "inventory_size": len(inventory),
        "inventory_extension_min_count": args.extend_inventory_min_count,
        "byte_fallback": not args.no_byte_fallback,
        "min_frequency": args.min_frequency,
        "train_docs": len(docs),
        "train_chars": chars,
        "splits_in_inventory": None,
        "train_seconds": round(seconds, 1),
        "corpora": [
            {
                "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "chars_budget": None if budget >= 10**12 else budget,
            }
            for path, budget in entries
        ],
        "sampling": "shards sorted by filename, documents in file order, first N chars per source",
    }
    (out / "datasheet.json").write_text(
        json.dumps(datasheet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"  vocab={len(tok):,} merges={len(tok.merges):,} in {seconds:.0f}s -> "
        f"{datasheet['vocab_size']:,} tokens in {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
