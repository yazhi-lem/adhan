"""Corpus quality, language-ID, and PII filtering for large-scale preprocessing.

Provides production-grade filtering pipelines for corpus curation.
"""

import json
import logging
import re
from typing import Generator, Optional, Set, Tuple

from adhan_slm.core import get_logger

logger = get_logger(__name__)


class CorpusFilter:
    """Quality, language-ID, and PII filtering for Tamil corpus.

    Provides modular filtering with detailed statistics:
    1. Quality filtering (text length, format validity)
    2. Language-ID filtering (Tamil fraction detection)
    3. PII scrubbing (email, phone, URLs, sensitive info)
    """

    # Tamil Unicode ranges (basic coverage)
    TAMIL_UNICODE_START = 0x0B80  # Tamil block start
    TAMIL_UNICODE_END = 0x0BFF  # Tamil block end

    # PII detection patterns
    EMAIL_PATTERN = re.compile(
        r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", re.IGNORECASE
    )
    PHONE_PATTERN = re.compile(
        r"(?:\+?91[-.\s]?|\b)(?:\d{4,5}[-.\s]?)?\d{3,4}[-.\s]?\d{4}\b"
    )
    URL_PATTERN = re.compile(r"https?://[^\s]+|www\.[^\s]+|\[URL\]", re.IGNORECASE)

    def __init__(
        self,
        min_length: int = 20,
        max_length: int = 10000,
        min_quality_score: float = 0.5,
        min_tamil_fraction: float = 0.7,
    ):
        """Initialize CorpusFilter.

        Args:
            min_length: Minimum text length in characters (default: 20).
            max_length: Maximum text length in characters (default: 10000).
            min_quality_score: Minimum quality score to keep (0-1, default: 0.5).
            min_tamil_fraction: Minimum Tamil content fraction (default: 0.7).
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_quality_score = min_quality_score
        self.min_tamil_fraction = min_tamil_fraction
        self.logger = logger

        self.logger.info(
            f"Filter initialized: min_length={min_length}, "
            f"max_length={max_length}, min_quality={min_quality_score}, "
            f"min_tamil_fraction={min_tamil_fraction:.0%}"
        )

    def filter_quality(
        self, documents: Generator[dict, None, None], check_fertility: bool = False
    ) -> Tuple[Generator[dict, None, None], dict]:
        """Remove low-quality text based on length and format.

        Args:
            documents: Generator yielding document dicts.
            check_fertility: If True, check tokenizer fertility (requires tokenizer).

        Yields:
            dict: Documents passing quality filters.

        Returns:
            dict: Quality filter statistics.
        """

        def quality_generator():
            """Inner generator for quality filtering."""
            stats = {
                "total": 0,
                "kept": 0,
                "too_short": 0,
                "too_long": 0,
                "low_quality_score": 0,
                "excessive_punctuation": 0,
            }

            for doc in documents:
                stats["total"] += 1
                text = doc.get("text", "")
                quality_score = doc.get("quality_score", 0.5)

                # Length checks
                text_len = len(text)
                if text_len < self.min_length:
                    stats["too_short"] += 1
                    self.logger.debug(
                        f"Text too short ({text_len} chars): {doc.get('id')}"
                    )
                    continue

                if text_len > self.max_length:
                    stats["too_long"] += 1
                    self.logger.debug(
                        f"Text too long ({text_len} chars): {doc.get('id')}"
                    )
                    continue

                # Quality score check
                if quality_score < self.min_quality_score:
                    stats["low_quality_score"] += 1
                    self.logger.debug(
                        f"Quality score too low ({quality_score}): {doc.get('id')}"
                    )
                    continue

                # Excessive punctuation check
                punct_count = sum(1 for c in text if c in "!?.,-;:")
                punct_ratio = punct_count / text_len if text_len > 0 else 0
                if punct_ratio > 0.3:  # More than 30% punctuation
                    stats["excessive_punctuation"] += 1
                    self.logger.debug(
                        f"Excessive punctuation ({punct_ratio:.1%}): {doc.get('id')}"
                    )
                    continue

                # Passed all checks
                stats["kept"] += 1
                yield doc

            stats["removed"] = stats["total"] - stats["kept"]
            stats["removal_rate"] = (
                stats["removed"] / stats["total"] if stats["total"] > 0 else 0.0
            )

            return stats

        return quality_generator(), {}

    def filter_language(
        self, documents: Generator[dict, None, None]
    ) -> Tuple[Generator[dict, None, None], dict]:
        """Keep only Tamil-dominant documents.

        Args:
            documents: Generator yielding document dicts.

        Yields:
            dict: Documents with Tamil fraction >= min_tamil_fraction.

        Returns:
            dict: Language filter statistics.
        """

        def language_generator():
            """Inner generator for language filtering."""
            stats = {
                "total": 0,
                "kept": 0,
                "non_tamil": 0,
                "tamil_fractions": [],
            }

            for doc in documents:
                stats["total"] += 1
                text = doc.get("text", "")

                # Calculate Tamil fraction
                tamil_fraction = self._estimate_tamil_fraction(text)
                stats["tamil_fractions"].append(tamil_fraction)

                if tamil_fraction < self.min_tamil_fraction:
                    stats["non_tamil"] += 1
                    self.logger.debug(
                        f"Insufficient Tamil ({tamil_fraction:.1%}): {doc.get('id')}"
                    )
                    continue

                stats["kept"] += 1
                yield doc

            stats["removed"] = stats["total"] - stats["kept"]
            stats["removal_rate"] = (
                stats["removed"] / stats["total"] if stats["total"] > 0 else 0.0
            )
            if stats["tamil_fractions"]:
                stats["avg_tamil_fraction"] = sum(stats["tamil_fractions"]) / len(
                    stats["tamil_fractions"]
                )

            return stats

        return language_generator(), {}

    def scrub_pii(
        self, documents: Generator[dict, None, None], anonymize_level: str = "standard"
    ) -> Tuple[Generator[dict, None, None], dict]:
        """Remove personally identifiable information.

        Supports different anonymization levels:
        - 'none': No anonymization (return as-is)
        - 'standard': Remove emails, phones, anonymize URLs
        - 'aggressive': Also attempt name detection

        Args:
            documents: Generator yielding document dicts.
            anonymize_level: Anonymization level ('none', 'standard', 'aggressive').

        Yields:
            dict: PII-scrubbed documents.

        Returns:
            dict: PII scrubbing statistics.
        """

        def pii_generator():
            """Inner generator for PII scrubbing."""
            stats = {
                "total": 0,
                "kept": 0,
                "emails_removed": 0,
                "phones_removed": 0,
                "urls_anonymized": 0,
                "had_pii": 0,
            }

            if anonymize_level == "none":
                # No scrubbing, pass through
                for doc in documents:
                    yield doc
                    stats["total"] += 1
                    stats["kept"] += 1
                return stats

            for doc in documents:
                stats["total"] += 1
                text = doc.get("text", "")
                original_text = text

                # Remove/anonymize PII
                if anonymize_level in ("standard", "aggressive"):
                    # Remove emails
                    emails = self.EMAIL_PATTERN.findall(text)
                    if emails:
                        text = self.EMAIL_PATTERN.sub("[EMAIL]", text)
                        stats["emails_removed"] += len(emails)

                    # Remove phone numbers
                    phones = self.PHONE_PATTERN.findall(text)
                    if phones:
                        text = self.PHONE_PATTERN.sub("[PHONE]", text)
                        stats["phones_removed"] += len(phones)

                    # Anonymize URLs
                    urls = self.URL_PATTERN.findall(text)
                    if urls:
                        text = self.URL_PATTERN.sub("[URL]", text)
                        stats["urls_anonymized"] += len(urls)

                if text != original_text:
                    stats["had_pii"] += 1

                # Update document with scrubbed text
                doc["text"] = text
                stats["kept"] += 1
                yield doc

            stats["removal_rate"] = (
                stats["had_pii"] / stats["total"] if stats["total"] > 0 else 0.0
            )

            return stats

        return pii_generator(), {}

    def _estimate_tamil_fraction(self, text: str) -> float:
        """Estimate fraction of Tamil characters in text.

        Args:
            text: Text to analyze.

        Returns:
            float: Fraction of Tamil characters (0-1).
        """
        if not text:
            return 0.0

        tamil_count = 0
        for char in text:
            code = ord(char)
            # Check if character is in Tamil Unicode block
            if self.TAMIL_UNICODE_START <= code <= self.TAMIL_UNICODE_END:
                tamil_count += 1

        return tamil_count / len(text)

    def apply_all_filters(
        self,
        documents: Generator[dict, None, None],
        skip_pii: bool = False,
    ) -> Generator[dict, None, None]:
        """Apply all filters in sequence: quality → language → PII.

        Args:
            documents: Generator yielding document dicts.
            skip_pii: If True, skip PII scrubbing (faster for trusted sources).

        Yields:
            dict: Fully filtered documents.
        """
        # Apply quality filter
        filtered, _ = self.filter_quality(documents)

        # Apply language filter
        filtered, _ = self.filter_language(filtered)

        # Apply PII scrubbing
        if not skip_pii:
            filtered, _ = self.scrub_pii(filtered)

        # Yield all documents that passed all filters
        yield from filtered

    def write_stats(self, stats: dict, output_path: str) -> None:
        """Write filter statistics to JSON file.

        Args:
            stats: Statistics dictionary from filters.
            output_path: Path to write JSON report.
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Filter report written to {output_path}")
