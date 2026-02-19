#!/usr/bin/env python3
"""Top-level model orchestrator: deps + scraper + training (+ optional merge)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str], dry_run: bool = False) -> None:
    print(f"\n$ {' '.join(command)}")
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full model orchestration")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--requirements", default="requirements.txt")

    # scraper/data build args
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument(
        "--existing-corpus",
        default="data/intermediate/rebalancing/v3_modern_enhanced.jsonl",
    )
    parser.add_argument(
        "--corpus-output",
        default="data/intermediate/rebalancing/unified_modern.jsonl",
    )
    parser.add_argument("--hf-output", default="data/final/tamil_texts/hf")
    parser.add_argument(
        "--strategy",
        choices=["balanced", "modern", "rebalanced"],
        default="modern",
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--modern-only", action="store_true")
    parser.add_argument("--hf-strategy", choices=["standard", "modern"], default="modern")
    parser.add_argument("--modern-ratio", type=float, default=0.60)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-dedupe", action="store_true")

    # training args
    parser.add_argument("--config")
    parser.add_argument("--model-name")
    parser.add_argument("--train-data-dir")
    parser.add_argument("--output-dir", default="models/adhan")
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--use-wandb", action="store_true")

    # merge args
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--adhan-dir", default="data/final/tamil_texts/hf")
    parser.add_argument("--vazhi-repo", default="../vazhi")
    parser.add_argument("--merge-output", default="data/unified/tamil_6k.jsonl")
    parser.add_argument("--target-count", type=int, default=6000)
    parser.add_argument("--split", action="store_true")

    args = parser.parse_args()

    if args.install_deps:
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(Path(args.requirements)),
            ],
            args.dry_run,
        )

    scraper_cmd = [
        sys.executable,
        str(ROOT / "scripts/run_scraper.py"),
        "--data-dir",
        args.data_dir,
        "--existing-corpus",
        args.existing_corpus,
        "--corpus-output",
        args.corpus_output,
        "--hf-output",
        args.hf_output,
        "--strategy",
        args.strategy,
        "--hf-strategy",
        args.hf_strategy,
        "--modern-ratio",
        str(args.modern_ratio),
        "--train-ratio",
        str(args.train_ratio),
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
    ]
    if args.max_records is not None:
        scraper_cmd.extend(["--max-records", str(args.max_records)])
    if args.modern_only:
        scraper_cmd.append("--modern-only")
    if args.no_dedupe:
        scraper_cmd.append("--no-dedupe")
    if args.dry_run:
        scraper_cmd.append("--dry-run")

    run_command(scraper_cmd, args.dry_run)

    training_data_dir = args.train_data_dir or args.hf_output
    train_cmd = [
        sys.executable,
        str(ROOT / "scripts/run_training.py"),
        "--data-dir",
        training_data_dir,
        "--output-dir",
        args.output_dir,
    ]
    if args.config:
        train_cmd.extend(["--config", args.config])
    if args.model_name:
        train_cmd.extend(["--model-name", args.model_name])
    if args.num_epochs is not None:
        train_cmd.extend(["--num-epochs", str(args.num_epochs)])
    if args.batch_size is not None:
        train_cmd.extend(["--batch-size", str(args.batch_size)])
    if args.learning_rate is not None:
        train_cmd.extend(["--learning-rate", str(args.learning_rate)])
    if args.use_wandb:
        train_cmd.append("--use-wandb")
    if args.dry_run:
        train_cmd.append("--dry-run")

    run_command(train_cmd, args.dry_run)

    if args.merge:
        merge_cmd = [
            sys.executable,
            str(ROOT / "src/data_scraper/merge_corpora.py"),
            "--adhan_dir",
            args.adhan_dir,
            "--vazhi_repo",
            args.vazhi_repo,
            "--output",
            args.merge_output,
            "--target_count",
            str(args.target_count),
            "--seed",
            str(args.seed),
        ]
        if args.split:
            merge_cmd.append("--split")
        run_command(merge_cmd, args.dry_run)


if __name__ == "__main__":
    main()
