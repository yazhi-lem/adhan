#!/usr/bin/env python3
"""Run model training."""

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
    parser = argparse.ArgumentParser(description="Run training pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--config")
    parser.add_argument("--model-name")
    parser.add_argument("--data-dir", default="data/final/tamil_texts/hf")
    parser.add_argument("--output-dir", default="models/adhan")
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--use-wandb", action="store_true")

    args = parser.parse_args()

    train_cmd = [
        sys.executable,
        str(ROOT / "src/models/sangam_gpt/train_enhanced.py"),
        "--data-dir",
        args.data_dir,
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

    run_command(train_cmd, args.dry_run)


if __name__ == "__main__":
    main()
