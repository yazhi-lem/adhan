#!/usr/bin/env python3
"""Run exact deduplication, near-duplicate removal, and train/eval overlap filtering.

Pipeline:
  1. Exact deduplication (SHA-256 on NFC-normalized text)
  2. Near-duplicate deduplication (MinHash/LSH with configurable similarity threshold)
  3. Train-vs-eval overlap filter (removes any train line appearing in eval corpus)

Outputs before/after counts for each stage and writes a reproducible JSON report.

Usage:
    python scripts/run_dedup.py --corpus data/train.txt --eval-corpus data/eval.txt \
        --output data/train_clean.txt --report dedup_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.core.logging import get_logger  # noqa: E402
from adhan_slm.data import corpus as corpus_mod  # noqa: E402
from adhan_slm.data.deduplicator import (  # noqa: E402
    TextDeduplicator,
    compute_train_eval_overlap,
    filter_train_eval_overlap,
)

logger = get_logger(__name__)


def run_dedup_pipeline(
    corpus_path: str | Path,
    eval_corpus_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    report_path: Optional[str | Path] = None,
    threshold: float = 0.85,
    shingle_size: int = 5,
    num_perm: int = 128,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute the full 3-stage deduplication and train/eval filtering pipeline.

    Args:
        corpus_path: Path to input training corpus (file or directory).
        eval_corpus_path: Optional path to eval corpus for leak prevention.
        output_path: Optional path to save filtered corpus.
        report_path: Optional path to write JSON report.
        threshold: Jaccard similarity threshold for near-duplicates (0-1).
        shingle_size: Character shingle size for MinHash.
        num_perm: Number of hash permutations for MinHash.
        seed: Random seed for MinHash reproducibility.

    Returns:
        Dict[str, Any]: Report dictionary containing before/after counts for all stages.
    """
    c_path = Path(corpus_path)
    if not c_path.exists():
        raise FileNotFoundError(f"Corpus path not found: {c_path}")

    # Read training corpus lines
    logger.info("Reading training corpus from %s...", c_path)
    lines: List[str] = corpus_mod.read_corpus(c_path, line_documents=True)
    initial_count = len(lines)

    # Stage 1: Exact deduplication
    logger.info("[Stage 1/3] Running exact deduplication...")
    exact_before = len(lines)
    exact_deduper = TextDeduplicator()
    exact_docs = [{"id": f"doc-{i}", "text": line} for i, line in enumerate(lines)]
    exact_gen, _ = exact_deduper.deduplicate(
        exact_docs, track_sources=False, check_near=False
    )
    lines_after_exact: List[str] = [str(d["text"]) for d in exact_gen]
    exact_after = len(lines_after_exact)
    exact_removed = exact_before - exact_after

    # Stage 2: Near-duplicate deduplication
    logger.info(
        "[Stage 2/3] Running near-duplicate deduplication (threshold=%.2f)...",
        threshold,
    )
    near_before = exact_after
    near_deduper = TextDeduplicator(
        num_perm=num_perm,
        threshold=threshold,
        shingle_size=shingle_size,
        seed=seed,
    )
    near_docs = [{"id": f"doc-{i}", "text": line} for i, line in enumerate(lines_after_exact)]
    near_gen, _ = near_deduper.deduplicate(
        near_docs, track_sources=False, check_near=True
    )
    lines_after_near: List[str] = [str(d["text"]) for d in near_gen]
    near_after = len(lines_after_near)
    near_removed = near_before - near_after

    # Stage 3: Train-vs-eval filter
    logger.info("[Stage 3/3] Checking train-vs-eval overlap...")
    filter_before = near_after
    filtered_lines: List[str] = lines_after_near
    eval_removed = 0
    final_eval_overlap = 0

    if eval_corpus_path is not None:
        e_path = Path(eval_corpus_path)
        if not e_path.exists():
            raise FileNotFoundError(f"Eval corpus path not found: {e_path}")
        logger.info("Reading evaluation corpus from %s...", e_path)
        eval_lines: List[str] = corpus_mod.read_corpus(e_path, line_documents=True)
        filtered_items, eval_removed = filter_train_eval_overlap(lines_after_near, eval_lines)
        filtered_lines = [str(item) for item in filtered_items]
        final_eval_overlap = compute_train_eval_overlap(filtered_lines, eval_lines)
        if final_eval_overlap != 0:
            raise ValueError(
                f"Train/eval overlap check failed: expected 0 overlap, found {final_eval_overlap}"
            )

    filter_after = len(filtered_lines)
    total_removed = initial_count - filter_after

    # Save output if requested
    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as fh:
            if out_p.suffix in (".jsonl", ".json"):
                for line in filtered_lines:
                    fh.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")
            else:
                for line in filtered_lines:
                    fh.write(line + "\n")
        logger.info("Filtered corpus written to %s (%d lines)", out_p, len(filtered_lines))

    report: Dict[str, Any] = {
        "corpus_path": str(c_path),
        "eval_corpus_path": str(eval_corpus_path) if eval_corpus_path else None,
        "stages": {
            "initial_count": initial_count,
            "exact_dedup": {
                "before": exact_before,
                "after": exact_after,
                "removed": exact_removed,
            },
            "near_dedup": {
                "before": near_before,
                "after": near_after,
                "removed": near_removed,
                "threshold": threshold,
                "shingle_size": shingle_size,
                "num_perm": num_perm,
            },
            "train_eval_filter": {
                "before": filter_before,
                "after": filter_after,
                "removed": eval_removed,
                "final_eval_overlap": final_eval_overlap,
            },
        },
        "final_count": filter_after,
        "total_removed": total_removed,
    }

    if report_path is not None:
        rep_p = Path(report_path)
        rep_p.parent.mkdir(parents=True, exist_ok=True)
        with rep_p.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.info("JSON report written to %s", rep_p)

    return report


def print_report(report: Dict[str, Any]) -> None:
    """Print readable deduplication report to stdout."""
    stages = report["stages"]
    print("=" * 60)
    print("DEDUPLICATION & TRAIN/EVAL OVERLAP REPORT")
    print("=" * 60)
    print(f"Corpus path:         {report['corpus_path']}")
    print(f"Eval corpus path:    {report['eval_corpus_path'] or 'None (skipped)'}")
    print(f"Initial lines:       {stages['initial_count']}")
    print("-" * 60)
    print("[Stage 1] Exact Deduplication:")
    print(f"  Before:            {stages['exact_dedup']['before']}")
    print(f"  After:             {stages['exact_dedup']['after']}")
    print(f"  Removed:           {stages['exact_dedup']['removed']}")
    print("-" * 60)
    print(f"[Stage 2] Near-duplicate Deduplication (threshold={stages['near_dedup']['threshold']}):")
    print(f"  Before:            {stages['near_dedup']['before']}")
    print(f"  After:             {stages['near_dedup']['after']}")
    print(f"  Removed:           {stages['near_dedup']['removed']}")
    print("-" * 60)
    print("[Stage 3] Train vs Eval Overlap Filter:")
    print(f"  Before:            {stages['train_eval_filter']['before']}")
    print(f"  After:             {stages['train_eval_filter']['after']}")
    print(f"  Removed (overlap): {stages['train_eval_filter']['removed']}")
    print(f"  Final overlap:     {stages['train_eval_filter']['final_eval_overlap']}")
    print("=" * 60)
    print(f"Final corpus lines:  {report['final_count']}")
    print(f"Total lines removed: {report['total_removed']}")
    print("=" * 60)


def main() -> None:
    """CLI entry point for run_dedup.py."""
    parser = argparse.ArgumentParser(
        description="Run exact dedup, near-duplicate removal, and train/eval overlap filter."
    )
    parser.add_argument(
        "corpus_pos",
        nargs="?",
        default=None,
        help="Path to input training corpus (txt, jsonl, or dir)",
    )
    parser.add_argument(
        "--corpus",
        "-c",
        dest="corpus_flag",
        default=None,
        help="Path to input training corpus",
    )
    parser.add_argument(
        "--eval-corpus",
        "-e",
        default=None,
        help="Path to evaluation corpus (txt, jsonl, or dir)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to save deduplicated corpus",
    )
    parser.add_argument(
        "--report",
        "-r",
        default="dedup_report.json",
        help="Path to output JSON report (default: dedup_report.json)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.85,
        help="Near-duplicate similarity threshold (default: 0.85)",
    )
    parser.add_argument(
        "--shingle-size",
        type=int,
        default=5,
        help="Character shingle size (default: 5)",
    )
    parser.add_argument(
        "--num-perm",
        type=int,
        default=128,
        help="MinHash permutations (default: 128)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()
    corpus_path = args.corpus_flag or args.corpus_pos

    if not corpus_path:
        parser.print_help()
        sys.exit(1)

    try:
        report = run_dedup_pipeline(
            corpus_path=corpus_path,
            eval_corpus_path=args.eval_corpus,
            output_path=args.output,
            report_path=args.report,
            threshold=args.threshold,
            shingle_size=args.shingle_size,
            num_perm=args.num_perm,
            seed=args.seed,
        )
        print_report(report)
    except Exception as exc:
        logger.error("Deduplication pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
