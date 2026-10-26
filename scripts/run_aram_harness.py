#!/usr/bin/env python3
"""Run the Aram Harness safety evaluation on a trained Adhan model.

Usage:
    # Full evaluation
    python scripts/run_aram_harness.py \
        --checkpoint checkpoints/adhan-nano-sft \
        --tokenizer data/final/tamil_slm \
        --output aram_report.json

    # Quick check (critical prompts only)
    python scripts/run_aram_harness.py \
        --checkpoint checkpoints/adhan-nano-sft \
        --tokenizer data/final/tamil_slm \
        --critical-only \
        --output aram_critical.json

    # Specific category
    python scripts/run_aram_harness.py \
        --checkpoint checkpoints/adhan-nano-sft \
        --tokenizer data/final/tamil_slm \
        --category harmful_instructions \
        --output aram_harmful.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adhan_slm.core.logging import get_logger
from adhan_slm.inference import load_model, load_tokenizer
from adhan_slm.safety import AramHarness
from adhan_slm.safety.red_team_prompts import (
    RED_TEAM_PROMPTS,
    RedTeamCategory,
    get_critical_prompts,
)

logger = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aram Harness — Tamil AI Safety Evaluation")
    ap.add_argument("--checkpoint", required=True, help="Path to Orbax checkpoint dir")
    ap.add_argument("--tokenizer", required=True, help="Path to tokenizer dir")
    ap.add_argument("--config", default=None, help="Optional model config YAML")
    ap.add_argument("--output", default="aram_report.json", help="Output JSON path")
    ap.add_argument(
        "--critical-only", action="store_true", help="Only run CRITICAL severity prompts"
    )
    ap.add_argument(
        "--category", choices=[c.value for c in RedTeamCategory], help="Filter by category"
    )
    ap.add_argument("--device", choices=["auto", "cpu", "gpu", "tpu"], default="auto")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    from adhan_slm.training.device import configure_backend

    configure_backend(args.device)

    logger.info(f"Loading model from {args.checkpoint}")
    tok = load_tokenizer(args.tokenizer)
    model, params, _ = load_model(
        config_path=args.config,
        checkpoint_dir=args.checkpoint,
        vocab_size=len(tok),
    )

    harness = AramHarness(
        model=model,
        params=params,
        tokenizer=tok,
        model_name=Path(args.checkpoint).name,
        checkpoint_path=args.checkpoint,
        generation_max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Filter prompts
    prompts = RED_TEAM_PROMPTS
    if args.critical_only:
        prompts = get_critical_prompts()
        logger.info(f"Running {len(prompts)} CRITICAL prompts")
    elif args.category:
        cat = RedTeamCategory(args.category)
        prompts = [p for p in RED_TEAM_PROMPTS if p.category == cat]
        logger.info(f"Running {len(prompts)} prompts in category '{args.category}'")
    else:
        logger.info(f"Running full evaluation: {len(prompts)} prompts")

    # Run eval
    report = harness.run_full_eval()

    # Override results with filtered set if needed
    if args.critical_only or args.category:
        # Re-run just red-team with filtered prompts
        report.results = harness.run_red_team_eval(prompts)
        report.total_prompts = len(report.results)
        report.critical_failures = sum(1 for r in report.results if r.verdict.value == "critical")
        report.high_failures = sum(1 for r in report.results if r.verdict.value == "high")
        report.medium_failures = sum(1 for r in report.results if r.verdict.value == "medium")
        report.overall_score = sum(r.score for r in report.results) / max(len(report.results), 1)

    # Print and save
    AramHarness.print_report(report)
    AramHarness.save_report(report, args.output)
    logger.info(f"Report saved to {args.output}")

    # Exit code based on critical failures
    if report.critical_failures > 0:
        logger.error(
            f"❌ {report.critical_failures} CRITICAL failures detected. Model is NOT safe for deployment."
        )
        sys.exit(1)
    elif report.high_failures > 0:
        logger.warning(f"⚠️ {report.high_failures} HIGH-risk failures. Review before deployment.")
        sys.exit(2)
    else:
        logger.info("✅ No critical or high failures. Model passes Aram safety gate.")
        sys.exit(0)


if __name__ == "__main__":
    main()
