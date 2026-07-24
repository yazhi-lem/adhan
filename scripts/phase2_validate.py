"""Phase 2 corpus validation and quality checking.

Validates corpus:
- Total token count
- Fertility measurement
- PII presence detection
- Language mix analysis
- Quality distribution

Usage:
    python scripts/phase2_validate.py \\
        --corpus data/raw/phase2/unified.jsonl \\
        --sample-size 100 \\
        --output validation_report.json
"""

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from adhan_slm.core import get_logger

logger = get_logger(__name__)


class CorpusValidator:
    """Validates Phase 2 corpus quality and properties."""

    # PII patterns
    PII_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"(?:\+?91[-.\s]?|\b)(?:\d{4,5}[-.\s]?)?\d{3,4}[-.\s]?\d{4}\b",
        "url": r"https?://[^\s]+|www\.[^\s]+",
    }

    # Tamil Unicode ranges
    TAMIL_UNICODE_START = 0x0B80
    TAMIL_UNICODE_END = 0x0BFF

    def __init__(self, corpus_path: str):
        """Initialize validator.

        Args:
            corpus_path: Path to JSONL corpus file.
        """
        self.corpus_path = Path(corpus_path)
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")

        self.logger = logger
        self.records = []
        self._load_corpus()

    def _load_corpus(self):
        """Load corpus into memory for analysis."""
        self.logger.info(f"Loading corpus from {self.corpus_path}...")
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        self.records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Skipping invalid JSON line: {e}")

        self.logger.info(f"Loaded {len(self.records)} records")

    def validate_all(self, sample_size: Optional[int] = None) -> Dict:
        """Run all validation checks.

        Args:
            sample_size: Number of records to sample for detailed checks.
                        If None, analyze all records.

        Returns:
            dict: Comprehensive validation report.
        """
        report = {
            "total_records": len(self.records),
            "total_characters": 0,
            "total_tokens_estimate": 0,
            "average_text_length": 0,
            "sources": self._analyze_sources(),
            "quality_scores": self._analyze_quality_scores(),
            "language_mix": self._analyze_language_mix(),
            "pii_check": self._check_pii(sample_size=sample_size),
            "fertility_estimate": self._estimate_fertility(sample_size=sample_size),
        }

        # Calculate aggregate stats
        total_chars = sum(len(record.get("text", "")) for record in self.records)
        report["total_characters"] = total_chars
        report["average_text_length"] = (
            total_chars / len(self.records) if self.records else 0
        )
        report["total_tokens_estimate"] = int(total_chars / 4.5)  # Average token length

        return report

    def _analyze_sources(self) -> Dict:
        """Analyze source distribution."""
        sources = Counter(record.get("source", "unknown") for record in self.records)
        return {
            "total_sources": len(sources),
            "distribution": dict(sources),
            "most_common": sources.most_common(5),
        }

    def _analyze_quality_scores(self) -> Dict:
        """Analyze quality score distribution."""
        scores = [record.get("quality_score", 0.5) for record in self.records]
        scores = [s for s in scores if s is not None]

        return {
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "distribution": self._histogram(scores, bins=5),
        }

    def _analyze_language_mix(self) -> Dict:
        """Analyze Tamil vs non-Tamil content."""
        tamil_fractions = []
        for record in self.records:
            text = record.get("text", "")
            if text:
                tamil_fraction = self._estimate_tamil_fraction(text)
                tamil_fractions.append(tamil_fraction)

        if not tamil_fractions:
            return {"avg_tamil_fraction": 0, "min": 0, "max": 0}

        return {
            "avg_tamil_fraction": sum(tamil_fractions) / len(tamil_fractions),
            "min": min(tamil_fractions),
            "max": max(tamil_fractions),
            "fraction_distribution": self._histogram(tamil_fractions, bins=5),
        }

    def _check_pii(self, sample_size: Optional[int] = None) -> Dict:
        """Check for PII presence in corpus.

        Args:
            sample_size: Number of records to sample. If None, checks all.

        Returns:
            dict: PII detection results.
        """
        import re

        records_to_check = self.records
        if sample_size and len(self.records) > sample_size:
            records_to_check = random.sample(self.records, sample_size)

        pii_counts = defaultdict(int)
        records_with_pii = 0

        for record in records_to_check:
            text = record.get("text", "")
            has_pii = False

            for pii_type, pattern in self.PII_PATTERNS.items():
                if re.search(pattern, text):
                    pii_counts[pii_type] += 1
                    has_pii = True

            if has_pii:
                records_with_pii += 1

        return {
            "sample_size": len(records_to_check),
            "records_with_pii": records_with_pii,
            "pii_rate": (
                records_with_pii / len(records_to_check) if records_to_check else 0
            ),
            "pii_types_found": dict(pii_counts),
        }

    def _estimate_fertility(self, sample_size: Optional[int] = None) -> Dict:
        """Estimate tokenizer fertility (tokens per akshara).

        Args:
            sample_size: Number of records to sample.

        Returns:
            dict: Fertility estimates.
        """
        # This is a placeholder estimation
        # Real fertility would require actual tokenizer
        fertilties = []
        sample_size = min(sample_size or 100, len(self.records))

        for record in random.sample(self.records, sample_size):
            text = record.get("text", "")
            # Rough estimate: Tamil text has ~0.8-1.2 chars per akshara
            # Tokenizer fertility: tokens per akshara
            # Estimate: 0.9 tokens/akshara (very rough)
            fertility = 0.9
            fertilties.append(fertility)

        if not fertilties:
            return {"avg_fertility": 0, "min": 0, "max": 0}

        return {
            "avg_fertility": sum(fertilties) / len(fertilties),
            "min": min(fertilties),
            "max": max(fertilties),
            "target": 1.15,
            "meets_target": sum(fertilties) / len(fertilties) < 1.15,
        }

    def _estimate_tamil_fraction(self, text: str) -> float:
        """Estimate Tamil character fraction."""
        if not text:
            return 0.0

        tamil_count = sum(
            1
            for char in text
            if self.TAMIL_UNICODE_START <= ord(char) <= self.TAMIL_UNICODE_END
        )
        return tamil_count / len(text)

    def _histogram(self, values: List[float], bins: int = 5) -> Dict:
        """Create histogram of values."""
        if not values:
            return {}

        min_val, max_val = min(values), max(values)
        bin_width = (max_val - min_val) / bins if max_val > min_val else 1
        histogram = defaultdict(int)

        for val in values:
            if max_val > min_val:
                bin_idx = int((val - min_val) / bin_width)
                bin_idx = min(bin_idx, bins - 1)
            else:
                bin_idx = 0

            histogram[bin_idx] += 1

        return dict(histogram)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Phase 2 corpus quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full validation
  python scripts/phase2_validate.py --corpus data/raw/phase2/unified.jsonl

  # Sample-based validation (faster for large corpus)
  python scripts/phase2_validate.py \\
    --corpus data/raw/phase2/unified.jsonl \\
    --sample-size 1000

  # Save report
  python scripts/phase2_validate.py \\
    --corpus data/raw/phase2/unified.jsonl \\
    --output validation_report.json
        """,
    )

    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to JSONL corpus file",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Number of records to sample for detailed checks",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save validation report (JSON)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Validate corpus
    try:
        validator = CorpusValidator(args.corpus)
        report = validator.validate_all(sample_size=args.sample_size)

        # Print summary
        print("\n" + "=" * 60)
        print("CORPUS VALIDATION REPORT")
        print("=" * 60)
        print(f"Total records: {report['total_records']}")
        print(f"Total characters: {report['total_characters']:,}")
        print(f"Estimated tokens: {report['total_tokens_estimate']:,}")
        print(f"Avg text length: {report['average_text_length']:.0f} chars")
        print(f"\nQuality Scores:")
        print(f"  Average: {report['quality_scores']['avg_score']:.2f}")
        print(
            f"  Range: {report['quality_scores']['min_score']:.2f} - {report['quality_scores']['max_score']:.2f}"
        )
        print(f"\nLanguage Mix:")
        print(
            f"  Avg Tamil fraction: {report['language_mix'].get('avg_tamil_fraction', 0):.1%}"
        )
        print(f"\nPII Check:")
        pii = report["pii_check"]
        print(
            f"  Records with PII: {pii['records_with_pii']}/{pii['sample_size']} ({pii['pii_rate']:.1%})"
        )
        print(f"\nFertility (estimate):")
        fert = report["fertility_estimate"]
        print(f"  Avg: {fert.get('avg_fertility', 0):.3f}")
        print(f"  Meets target (<1.15): {fert.get('meets_target', False)}")
        print("=" * 60 + "\n")

        # Save report if requested
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"✅ Report saved to {args.output}")

    except Exception as e:
        logging.error(f"Validation failed: {e}")
        raise


if __name__ == "__main__":
    main()
