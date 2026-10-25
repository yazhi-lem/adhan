#!/usr/bin/env python3
"""Build Sangam SFT data and a held-out tiṇai benchmark from the labelled corpus.

Inputs (already produced by ``scripts/ingest_open_sangam.py``):
  data/final/tinai_labels.jsonl      tiṇai/kalam/uripporul/karupporul per song
  data/raw/sangam/*/verse.jsonl      verse text
  data/raw/sangam/*/urai.jsonl       traditional commentary text

Outputs:
  data/final/sft_sangam/{train,val,test}.jsonl
  data/final/sft_sangam/stats.json
  bench/tamil-bench/data/sangam_ta_test.jsonl   (MCQ, test poems only)

Splits are by poem (stable hash), so a poem's verse, urai and labels are never
split across train and evaluation.

    python scripts/build_sangam_sft.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.data import sangam_sft  # noqa: E402


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default="data/final/tinai_labels.jsonl")
    ap.add_argument("--sangam-dir", default="data/raw/sangam")
    ap.add_argument(
        "--open-sangam-root",
        default="../open-sangam",
        help="Open-Sangam checkout; its per-poem `english` field feeds eurai_en",
    )
    ap.add_argument("--out-dir", default="data/final/sft_sangam")
    ap.add_argument(
        "--bench-out",
        default="bench/tamil-bench/data/sangam_ta_test.jsonl",
        help="MCQ test rows for bench/tamil-bench (text poems only)",
    )
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    labels = ROOT / args.labels
    sangam_dir = ROOT / args.sangam_dir
    if not labels.exists():
        sys.exit(f"labels not found: {labels}")
    if not sangam_dir.exists():
        sys.exit(f"sangam dir not found: {sangam_dir}")

    open_sangam_root = ROOT / args.open_sangam_root
    if not open_sangam_root.exists():
        print(f"warning: Open-Sangam not found at {open_sangam_root}; eurai_en will be empty")
        open_sangam_root = None
    records = sangam_sft.load_records(labels, sangam_dir, open_sangam_root)
    if not records:
        sys.exit("no labelled poems joined with verse text")

    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for rec in records:
        by_split[sangam_sft.split_of(rec["poem_id"], args.val_frac, args.test_frac)].append(rec)
    examples = {split: sangam_sft.build_examples(recs) for split, recs in by_split.items()}

    out = ROOT / args.out_dir
    for split, rows in examples.items():
        _write_jsonl(out / f"{split}.jsonl", rows)

    report = sangam_sft.stats(records, examples)
    overlap = report["poem_overlap"]
    if any(overlap.values()):
        sys.exit(f"poem-level split leaked across splits: {overlap}")

    bench_rows = sangam_sft.build_benchmark(by_split["test"], seed=args.seed)
    bench_out = ROOT / args.bench_out
    _write_jsonl(bench_out, bench_rows)
    report["bench_rows"] = len(bench_rows)
    report["bench_out"] = str(bench_out.relative_to(ROOT))
    (out / "stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"poems: {report['poems']}  splits: {report['poems_by_split']}")
    print(f"examples: {report['examples_by_split']}  by task: {report['examples_by_task']}")
    print(f"tinai distribution by split: {report['tinai_by_split']}")
    print(f"bench MCQ rows: {len(bench_rows)} -> {report['bench_out']}")
    print(f"written: {out.relative_to(ROOT)}/{{train,val,test}}.jsonl + stats.json")


if __name__ == "__main__":
    main()
