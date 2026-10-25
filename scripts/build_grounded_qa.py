#!/usr/bin/env python3
"""Build grounded QA SFT data (answer + span + citation, or refusal).

Sources
  indicqa  the Tamil IndicQA file vendored by bench/tamil-bench (ready today)
  corpus   a directory of SCERT/textbook .txt/.jsonl files (cloze items)

Outputs
  data/final/sft_qa/{train,val,test}.jsonl
  data/final/sft_qa/stats.json

15% of rows are made unanswerable by construction:
  * indicqa: pair the question with a different article that lacks the answer
  * corpus:  remove the answer-bearing sentence from the context

    python scripts/build_grounded_qa.py                       # IndicQA v0
    python scripts/build_grounded_qa.py --source corpus \
        --corpus data/raw/scert                               # SCERT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.data import grounded_qa  # noqa: E402


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_rows(rows, key_fn, val_frac: float, test_frac: float) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for row in rows:
        out[grounded_qa.split_of(key_fn(row), val_frac, test_frac)].append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["indicqa", "corpus"], default="indicqa")
    ap.add_argument("--indicqa", default="bench/tamil-bench/data/indicqa.ta.json")
    ap.add_argument("--corpus", default="data/raw/scert")
    ap.add_argument("--out-dir", default="data/final/sft_qa")
    ap.add_argument("--unanswerable", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-examples", type=int, default=None)
    args = ap.parse_args()

    if args.source == "indicqa":
        path = ROOT / args.indicqa
        if not path.exists():
            sys.exit(f"IndicQA file not found: {path}")
        records = grounded_qa.load_indicqa(path)
        by_context = _split_rows(
            records, lambda r: r["context_id"], args.val_frac, args.test_frac
        )
        examples = {
            split: grounded_qa.build_indicqa_split(
                rows, split, seed=args.seed, unanswerable_frac=args.unanswerable
            )
            for split, rows in by_context.items()
        }
    else:
        corpus = ROOT / args.corpus
        if not corpus.exists():
            sys.exit(
                f"corpus dir not found: {corpus}\n"
                "Put SCERT .txt/.jsonl files there (one document per file or a "
                "`text` field per JSONL line), then rerun."
            )
        rows = grounded_qa.build_corpus_examples(
            corpus,
            unanswerable_frac=args.unanswerable,
            seed=args.seed,
            max_examples=args.max_examples,
        )
        examples = _split_rows(
            rows, lambda e: e["meta"]["doc_id"], args.val_frac, args.test_frac
        )

    out = ROOT / args.out_dir
    for split, rows in examples.items():
        _write_jsonl(out / f"{split}.jsonl", rows)

    report = grounded_qa.stats(examples)
    report.update(
        {
            "source": args.source,
            "unanswerable_target": args.unanswerable,
            "split_rule": "context_id" if args.source == "indicqa" else "doc_id",
        }
    )
    (out / "stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"source: {args.source}")
    print(f"examples: {report['by_split']}  total: {report['total']}")
    print(f"answerable: {report['answerable']}")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"written: {shown}/{{train,val,test}}.jsonl + stats.json")


if __name__ == "__main__":
    main()
