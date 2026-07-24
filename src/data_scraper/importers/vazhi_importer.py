"""Importer for yazhi-lem/vazhi knowledge/QA dataset."""

import json
import logging
from pathlib import Path
from typing import Generator, Optional

from adhan_slm.core import get_logger

logger = get_logger(__name__)


class VazhiImporter:
    """Import knowledge/QA pairs from yazhi-lem/vazhi project.

    Vazhi is a Tamil knowledge base with QA pairs. This importer extracts
    Tamil text from questions and answers and converts to unified corpus format.
    """

    def __init__(self, repo_path: Optional[str] = None):
        """Initialize VazhiImporter.

        Args:
            repo_path: Path to yazhi-lem/vazhi repository. If None, will search
                      common locations or require explicit path in import calls.
        """
        self.repo_path = Path(repo_path) if repo_path else None
        self.logger = logger

    def import_from_repo(
        self, repo_path: Optional[str] = None
    ) -> Generator[dict, None, None]:
        """Load QA pairs from vazhi repository and convert to corpus format.

        Yields JSONL records with format:
        {
            "id": "vazhi-qa-{index}",
            "text": "Combined question and answer text",
            "source": "vazhi-qa",
            "url": null,
            "quality_score": 0.75
        }

        Args:
            repo_path: Path to yazhi-lem/vazhi repository.
                      If not provided, uses self.repo_path.

        Yields:
            dict: JSONL-compatible corpus records.

        Raises:
            FileNotFoundError: If repo path not found or no data files exist.
            json.JSONDecodeError: If data files are malformed JSON.
        """
        path = Path(repo_path) if repo_path else self.repo_path
        if not path:
            raise ValueError(
                "repo_path must be provided either in __init__ or import_from_repo()"
            )

        path = Path(path)
        if not path.exists():
            self.logger.error(f"Vazhi repository not found at {path}")
            raise FileNotFoundError(f"Vazhi repository not found at {path}")

        # Look for common data locations in vazhi project
        data_patterns = [
            path / "data" / "*.jsonl",
            path / "data" / "*.json",
            path / "qa" / "*.jsonl",
            path / "qa" / "*.json",
            path / "*.jsonl",
            path / "*.json",
        ]

        processed = 0
        for pattern in data_patterns:
            for file_path in path.glob(pattern.name.replace("*.", "*")):
                if not file_path.is_file():
                    continue

                try:
                    yield from self._process_file(file_path)
                    processed += 1
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    self.logger.warning(f"Error processing {file_path}: {e}")
                    continue

        if processed == 0:
            self.logger.warning(f"No data files found in {path}")

    def _process_file(self, file_path: Path) -> Generator[dict, None, None]:
        """Process a single JSONL or JSON file from vazhi.

        Args:
            file_path: Path to JSONL or JSON file.

        Yields:
            dict: Corpus records.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        records = []

        # Try JSONL format first (one record per line)
        if file_path.suffix == ".jsonl":
            for line_idx, line in enumerate(content.strip().split("\n")):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    self.logger.warning(
                        f"Invalid JSON at line {line_idx} in {file_path.name}: {e}"
                    )
        else:
            # Try JSON format (single array or object)
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict):
                    # Handle various JSON object structures
                    if "data" in data:
                        records = (
                            data["data"]
                            if isinstance(data["data"], list)
                            else [data["data"]]
                        )
                    elif "qa" in data:
                        records = (
                            data["qa"] if isinstance(data["qa"], list) else [data["qa"]]
                        )
                    else:
                        records = [data]
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in {file_path.name}: {e}")
                raise

        # Convert each record to corpus format
        for idx, record in enumerate(records):
            try:
                yield self._convert_record(record, idx)
            except (KeyError, ValueError, TypeError) as e:
                self.logger.debug(f"Skipping record {idx} in {file_path.name}: {e}")
                continue

    def _convert_record(self, record: dict, idx: int) -> dict:
        """Convert vazhi QA record to unified corpus format.

        Args:
            record: Raw QA record from vazhi.
            idx: Record index for ID generation.

        Returns:
            dict: Corpus-formatted record.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If text is empty or invalid.
        """
        # Extract question and answer
        question = (
            record.get("question") or record.get("q") or record.get("query") or ""
        )
        answer = record.get("answer") or record.get("a") or record.get("response") or ""

        if not question and not answer:
            raise ValueError("No question or answer found in record")

        # Combine question and answer
        text = f"{question} {answer}".strip()

        if not text:
            raise ValueError("Empty text after combining question and answer")

        # Determine quality score (high for structured QA data)
        quality_score = 0.75  # QA pairs are generally high-quality

        return {
            "id": f"vazhi-qa-{idx}",
            "text": text,
            "source": "vazhi-qa",
            "url": record.get("url") or record.get("source_url"),
            "quality_score": quality_score,
        }

    def import_from_directory(self, directory: str) -> Generator[dict, None, None]:
        """Import from a directory of vazhi JSONL/JSON files.

        Convenience method for batch importing all files in a directory.

        Args:
            directory: Path to directory containing vazhi data files.

        Yields:
            dict: Corpus records.
        """
        yield from self.import_from_repo(directory)
