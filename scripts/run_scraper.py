#!/usr/bin/env python3
"""Run corpus build + HF export pipeline.

Optional social-media pre-collection:
  --social reddit    Run the Reddit Tamil scraper before building the corpus.
  --social twitter   Run the Twitter/X Tamil collector before building the corpus.
  --social all       Run both social scrapers.
"""

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
    parser = argparse.ArgumentParser(description="Run scraper/data preparation pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")

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

    # Social media collection
    parser.add_argument(
        "--social",
        choices=["reddit", "twitter", "all"],
        default=None,
        help="Run social-media scraper(s) before corpus build.",
    )
    parser.add_argument("--social-output-dir", default="data/raw/social",
                        help="Base output dir for social scrapers.")
    parser.add_argument("--social-max-posts", type=int, default=500,
                        help="Max posts per subreddit (Reddit scraper).")
    parser.add_argument("--social-max-requests", type=int, default=100,
                        help="Max HTTP requests (Twitter scraper).")

    args = parser.parse_args()

    # ── optional social collection ─────────────────────────────────────────────
    if args.social in ("reddit", "all"):
        reddit_cmd = [
            sys.executable,
            str(ROOT / "src/data_scraper/raw_extractors/reddit_scraper.py"),
            "--output-dir", str(Path(args.social_output_dir) / "reddit"),
            "--max-posts", str(args.social_max_posts),
        ]
        run_command(reddit_cmd, args.dry_run)

    if args.social in ("twitter", "all"):
        twitter_cmd = [
            sys.executable,
            str(ROOT / "src/data_scraper/raw_extractors/twitter_scraper.py"),
            "--output-dir", str(Path(args.social_output_dir) / "twitter"),
            "--max-requests", str(args.social_max_requests),
        ]
        run_command(twitter_cmd, args.dry_run)

    # ── corpus build ───────────────────────────────────────────────────────────
    build_cmd = [
        sys.executable,
        str(ROOT / "src/data_scraper/processing/build_unified_corpus.py"),
        "--data-dir",
        args.data_dir,
        "--existing-corpus",
        args.existing_corpus,
        "--output",
        args.corpus_output,
        "--strategy",
        args.strategy,
    ]
    if args.max_records is not None:
        build_cmd.extend(["--max-records", str(args.max_records)])
    if args.modern_only:
        build_cmd.append("--modern-only")

    export_cmd = [
        sys.executable,
        str(ROOT / "src/data_scraper/export/export_unified_hf.py"),
        "--input",
        args.corpus_output,
        "--output",
        args.hf_output,
        "--strategy",
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
    if args.no_dedupe:
        export_cmd.append("--no-dedupe")

    run_command(build_cmd, args.dry_run)
    run_command(export_cmd, args.dry_run)


if __name__ == "__main__":
    main()
