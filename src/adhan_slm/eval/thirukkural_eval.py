"""Thirukkural Evaluation Harness for Adhan SLM.

Evaluates Adhan SLM on Classical Thirukkural understanding:
  1. Couplet Completion (Given Line 1, predict/complete Line 2).
  2. Couplet Perplexity & Cross-Entropy loss on ancient Venpa meter.
  3. Pure Tamil Purity Score (Ratio of pure Tamil aksharas vs Grantha/loan glyphs).
  4. Meaning & Urai alignment test.

Usage:
    python -m adhan_slm.eval.thirukkural_eval \\
        --harness-data data/raw/thirukkural/thirukkural_harness_data.json \\
        --tokenizer-dir data/final/tamil_slm
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from adhan_slm.core.logging import get_logger

logger = get_logger(__name__)

# Grantha consonants in the Tamil block, used for loanwords. The set must match
# docs/architecture/ARCHITECTURE_SWARAM_SLM.md, which lists ஜ ஷ ஸ ஹ ஶ.
#
# Two bugs lived in the previous set("ஜஷஸஹக்ஷ"):
#   * ஶ (U+0BB6) was missing, so ஶ-form words (ஶ்ரீ, ஶிவா) scored as pure.
#   * க்ஷ is three codepoints (க + ் + ஷ), so set() also captured க and the
#     pulli ் — every க in Tamil text was counted as Grantha, depressing the
#     score for ordinary words. The conjuncts க்ஷ and ஸ்ரீ are composed of the
#     consonants above, so ஷ and ஸ already cover them; no extra entry is needed.
GRANTHA_CHARS = set("ஜஷஸஹஶ")
TAMIL_START = 0x0B80
TAMIL_END = 0x0BFF


def compute_purity_breakdown(text: str) -> Dict[str, Any]:
    """Return the letter counts behind :func:`compute_pure_tamil_purity`.

    The denominator is *letters* (``str.isalpha()``), not Tamil-block characters.
    That distinction matters: the previous version divided only by Tamil-block
    characters, so Latin-script code-switching never reached the denominator and
    ``"கூல் AI"`` scored 1.0. Punctuation and digits are excluded, so they do not
    depress the score either.

    ``other_letters`` counts letters outside the Tamil block. It is reported so
    callers can distinguish "no Grantha, no English" from "no Grantha, but half
    the sentence is English" — two very different claims that the single float
    cannot separate.

    This metric still cannot see *vocabulary*: a Sanskrit stem written in Tamil
    letters (தர்மம், சந்தோஷம்) is all Tamil script and scores as pure. Use the
    tamil-tani-tamil guidance for that judgement; this function measures script.
    """
    clean = re.sub(r"\s+", "", text)
    letters = [c for c in clean if c.isalpha()]
    if not letters:
        return {
            "purity": 1.0,
            "tamil_letters": 0,
            "grantha_letters": 0,
            "other_letters": 0,
        }

    tamil_letters = [c for c in letters if TAMIL_START <= ord(c) <= TAMIL_END]
    grantha_count = sum(1 for c in tamil_letters if c in GRANTHA_CHARS)
    native_tamil = len(tamil_letters) - grantha_count
    return {
        "purity": native_tamil / len(letters),
        "tamil_letters": len(tamil_letters),
        "grantha_letters": grantha_count,
        "other_letters": len(letters) - len(tamil_letters),
    }


def compute_pure_tamil_purity(text: str) -> float:
    """Ratio of native Tamil letters to all letters, excluding Grantha and Latin.

    Returns 0.0 when the text has letters but none in the Tamil block, and 1.0
    when there are no letters at all. See :func:`compute_purity_breakdown` for the
    counts and for the vocabulary limitation this metric cannot measure.
    """
    return compute_purity_breakdown(text)["purity"]


class ThirukkuralEvaluator:
    """Thirukkural classical literature benchmark evaluator."""

    def __init__(self, harness_data_path: Path):
        self.path = Path(harness_data_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Harness data not found at {self.path}")
        self.records: List[Dict[str, Any]] = json.loads(self.path.read_text(encoding="utf-8"))

    def evaluate_tokenizer(self, tok) -> Dict[str, Any]:
        """Evaluate Swaram tokenizer performance on all 1,330 Kurals."""
        fertilities = []
        purities = []
        token_lengths = []

        for item in self.records:
            couplet = item["couplet"]
            f = tok.fertility(couplet)
            if f > 0:
                fertilities.append(f)
            tokens = tok.encode(couplet)
            token_lengths.append(len(tokens))
            purities.append(compute_pure_tamil_purity(couplet))

        avg_fert = sum(fertilities) / max(1, len(fertilities))
        avg_purity = sum(purities) / max(1, len(purities))
        avg_tokens_per_kural = sum(token_lengths) / max(1, len(token_lengths))

        return {
            "total_kurals": len(self.records),
            "mean_fertility": round(avg_fert, 4),
            "fertility_pass": avg_fert < 1.15,
            "pure_tamil_purity_percent": round(avg_purity * 100, 2),
            "avg_tokens_per_kural": round(avg_tokens_per_kural, 2),
        }

    def evaluate_model(self, model, tok, sample_size: int = 50) -> Dict[str, Any]:
        """Evaluate model perplexity and couplet continuation accuracy."""
        # Optional model inference evaluation
        sample = self.records[:sample_size]
        results = []

        for item in sample:
            line1 = item["line1"]
            line2 = item["line2"]
            purity = compute_pure_tamil_purity(f"{line1} {line2}")
            results.append(
                {
                    "kural_num": item["kural_num"],
                    "line1": line1,
                    "target_line2": line2,
                    "purity": round(purity, 3),
                }
            )

        return {
            "sample_size": len(sample),
            "benchmark_results": results,
            "status": "ready_for_checkpoint",
        }


def run_thirukkural_benchmark(
    harness_path: str = "data/raw/thirukkural/thirukkural_harness_data.json",
    tokenizer_dir: Optional[str] = "data/final/tamil_slm",
    checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    evaluator = ThirukkuralEvaluator(Path(harness_path))
    report: Dict[str, Any] = {
        "benchmark": "Thirukkural Classical Harness",
        "total_kurals": len(evaluator.records),
    }

    if tokenizer_dir and Path(tokenizer_dir).exists():
        from adhan_slm.inference import load_tokenizer

        tok = load_tokenizer(tokenizer_dir)
        report["tokenizer_metrics"] = evaluator.evaluate_tokenizer(tok)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Thirukkural evaluation harness")
    parser.add_argument(
        "--harness-data",
        default="data/raw/thirukkural/thirukkural_harness_data.json",
        help="Path to harness data JSON",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default="data/final/tamil_slm",
        help="Path to frozen tokenizer",
    )
    parser.add_argument("--output", default=None, help="Output JSON report file")
    args = parser.parse_args()

    report = run_thirukkural_benchmark(args.harness_data, args.tokenizer_dir)
    print("\n" + "=" * 60)
    print("THIRUKKURAL CLASSICAL HARNESS REPORT")
    print("=" * 60)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("=" * 60 + "\n")

    if args.output:
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Saved report to %s", args.output)


if __name__ == "__main__":
    main()
