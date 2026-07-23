"""Phase 2 corpus curation orchestrator.

Builds 300M-1B token corpus by:
1. Ingesting data from yazhi projects + scrapers
2. Deduplicating (exact + near-duplicates)
3. Filtering (quality, language, PII)
4. Validating and generating datasheet

Usage:
    python scripts/phase2_corpus_build.py \\
        --yazhi-projects /path/to/yazhi \\
        --scrapers wikipedia,reddit,twitter,news \\
        --output data/raw/phase2 \\
        --dedup-threshold 0.85 \\
        --min-tamil-fraction 0.7
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Generator, Optional

from adhan_slm.core import get_logger
from adhan_slm.data.deduplicator import TextDeduplicator
from adhan_slm.data.filters import CorpusFilter
from src.data_scraper.yazhi_integrations import (
    CorpusTamilImporter,
    SangamImporter,
    VazhiImporter,
)

logger = get_logger(__name__)


class Phase2CorpusBuilder:
    """Orchestrates Phase 2 corpus curation pipeline."""

    def __init__(
        self,
        output_dir: str,
        dedup_threshold: float = 0.85,
        min_tamil_fraction: float = 0.7,
        min_quality_score: float = 0.5,
    ):
        """Initialize corpus builder.

        Args:
            output_dir: Directory to save corpus and reports.
            dedup_threshold: Jaccard similarity threshold for dedup.
            min_tamil_fraction: Minimum Tamil content fraction.
            min_quality_score: Minimum quality score threshold.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.deduplicator = TextDeduplicator(threshold=dedup_threshold)
        self.filter = CorpusFilter(
            min_tamil_fraction=min_tamil_fraction, min_quality_score=min_quality_score
        )

        self.logger = logger
        self.stats = {
            "sources": {},
            "total_ingested": 0,
            "total_after_dedup": 0,
            "total_after_filter": 0,
        }

    def ingest_yazhi_projects(self, yazhi_root: Optional[str] = None) -> Generator[dict, None, None]:
        """Import data from yazhi ecosystem projects.

        Args:
            yazhi_root: Root directory containing yazhi projects.

        Yields:
            dict: Corpus records from yazhi projects.
        """
        if not yazhi_root:
            self.logger.warning("No yazhi projects directory specified")
            return

        yazhi_root = Path(yazhi_root)

        # Import from yazhi-lem/vazhi
        vazhi_path = yazhi_root / "vazhi"
        if vazhi_path.exists():
            self.logger.info("Importing from yazhi-lem/vazhi...")
            try:
                importer = VazhiImporter(str(vazhi_path))
                count = 0
                for record in importer.import_from_repo():
                    self.stats["sources"]["vazhi"] = self.stats["sources"].get("vazhi", 0) + 1
                    self.stats["total_ingested"] += 1
                    count += 1
                    yield record

                self.logger.info(f"Imported {count} records from vazhi")
            except Exception as e:
                self.logger.warning(f"Failed to import from vazhi: {e}")

        # Import from yazhi-lem/corpus-tamil
        corpus_tamil_path = yazhi_root / "corpus-tamil"
        if corpus_tamil_path.exists():
            self.logger.info("Importing from yazhi-lem/corpus-tamil...")
            try:
                importer = CorpusTamilImporter(str(corpus_tamil_path))
                count = 0
                for record in importer.import_from_repo():
                    self.stats["sources"]["corpus-tamil"] = (
                        self.stats["sources"].get("corpus-tamil", 0) + 1
                    )
                    self.stats["total_ingested"] += 1
                    count += 1
                    yield record

                self.logger.info(f"Imported {count} records from corpus-tamil")
            except Exception as e:
                self.logger.warning(f"Failed to import from corpus-tamil: {e}")

        # Import from open Sangam sources
        sangam_path = yazhi_root / "sangam"
        if sangam_path.exists():
            self.logger.info("Importing from open Sangam sources...")
            try:
                importer = SangamImporter()
                count = 0
                for record in importer.import_from_open_sangam(str(sangam_path)):
                    self.stats["sources"]["sangam"] = self.stats["sources"].get("sangam", 0) + 1
                    self.stats["total_ingested"] += 1
                    count += 1
                    yield record

                self.logger.info(f"Imported {count} records from Sangam")
            except Exception as e:
                self.logger.warning(f"Failed to import from Sangam: {e}")

    def process_pipeline(
        self,
        documents: Generator[dict, None, None],
        skip_pii: bool = False,
    ) -> Generator[dict, None, None]:
        """Process documents through full pipeline: dedup → filter.

        Args:
            documents: Input document stream.
            skip_pii: If True, skip PII scrubbing.

        Yields:
            dict: Processed documents ready for packing.
        """
        # Step 1: Deduplication
        self.logger.info("Starting deduplication phase...")
        dedup_gen, _ = self.deduplicator.deduplicate(documents)

        count_after_dedup = 0
        for doc in dedup_gen:
            self.stats["total_after_dedup"] += 1
            count_after_dedup += 1
            yield doc

        self.logger.info(
            f"After deduplication: {count_after_dedup} documents "
            f"({count_after_dedup / self.stats['total_ingested']:.1%} of ingested)"
        )

        # Note: Filtering happens in post-processing step via apply_all_filters

    def build_corpus(
        self,
        yazhi_root: Optional[str] = None,
        output_file: str = "unified.jsonl",
        skip_pii: bool = False,
    ) -> str:
        """Build complete corpus with all processing steps.

        Args:
            yazhi_root: Root directory of yazhi projects.
            output_file: Output JSONL filename.
            skip_pii: If True, skip PII scrubbing.

        Returns:
            str: Path to output corpus file.
        """
        output_path = self.output_dir / output_file

        self.logger.info("Starting Phase 2 corpus build...")
        start_time = time.time()

        # Step 1: Ingest from all sources
        self.logger.info("Phase 1: Ingesting from all sources...")
        documents = self.ingest_yazhi_projects(yazhi_root)

        # Step 2: Process pipeline (dedup + filter)
        self.logger.info("Phase 2: Processing through pipeline...")
        processed = self.process_pipeline(documents, skip_pii=skip_pii)

        # Step 3: Apply filters and write
        self.logger.info("Phase 3: Applying filters and writing...")
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in self.filter.apply_all_filters(processed, skip_pii=skip_pii):
                self.stats["total_after_filter"] += 1
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        elapsed = time.time() - start_time

        # Log summary
        self.logger.info(f"Corpus building complete in {elapsed:.1f}s")
        self.logger.info(f"Total ingested: {self.stats['total_ingested']}")
        self.logger.info(f"After dedup: {self.stats['total_after_dedup']}")
        self.logger.info(f"After filter: {self.stats['total_after_filter']}")
        self.logger.info(f"Output file: {output_path}")

        return str(output_path)

    def write_report(self, report_file: str = "phase2_report.json") -> str:
        """Write corpus building report.

        Args:
            report_file: Report filename.

        Returns:
            str: Path to report file.
        """
        report_path = self.output_dir / report_file

        report = {
            "timestamp": time.time(),
            "corpus_size": self.stats["total_after_filter"],
            "ingestion_stats": self.stats,
            "dedup_settings": {
                "threshold": self.deduplicator.threshold,
                "num_perm": self.deduplicator.num_perm,
            },
            "filter_settings": {
                "min_length": self.filter.min_length,
                "max_length": self.filter.max_length,
                "min_quality_score": self.filter.min_quality_score,
                "min_tamil_fraction": self.filter.min_tamil_fraction,
            },
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Report written to {report_path}")
        return str(report_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 2 corpus curation orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with yazhi projects
  python scripts/phase2_corpus_build.py --yazhi-projects ~/yazhi

  # With custom settings
  python scripts/phase2_corpus_build.py \\
    --yazhi-projects ~/yazhi \\
    --output data/phase2 \\
    --dedup-threshold 0.85 \\
    --min-tamil-fraction 0.8
        """,
    )

    parser.add_argument(
        "--yazhi-projects",
        type=str,
        help="Root directory containing yazhi projects (vazhi, corpus-tamil, sangam)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/phase2",
        help="Output directory for corpus and reports (default: data/raw/phase2)",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.85,
        help="Jaccard similarity threshold for near-duplicates (default: 0.85)",
    )
    parser.add_argument(
        "--min-tamil-fraction",
        type=float,
        default=0.7,
        help="Minimum Tamil content fraction (default: 0.7)",
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.5,
        help="Minimum quality score to keep (default: 0.5)",
    )
    parser.add_argument(
        "--skip-pii",
        action="store_true",
        help="Skip PII scrubbing (faster for trusted sources)",
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

    # Build corpus
    builder = Phase2CorpusBuilder(
        output_dir=args.output,
        dedup_threshold=args.dedup_threshold,
        min_tamil_fraction=args.min_tamil_fraction,
        min_quality_score=args.min_quality_score,
    )

    # Build corpus
    corpus_path = builder.build_corpus(
        yazhi_root=args.yazhi_projects,
        skip_pii=args.skip_pii,
    )

    # Write report
    report_path = builder.write_report()

    print(f"\n✅ Corpus building complete!")
    print(f"   Output: {corpus_path}")
    print(f"   Report: {report_path}")
    print(f"   Records: {builder.stats['total_after_filter']}")


if __name__ == "__main__":
    main()
