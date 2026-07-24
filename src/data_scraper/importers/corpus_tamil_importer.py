"""Importer for yazhi-lem/corpus-tamil pre-curated dataset."""

import json
import logging
from pathlib import Path
from typing import Generator, Optional

from adhan_slm.core import get_logger

logger = get_logger(__name__)


class CorpusTamilImporter:
    """Import pre-curated Tamil corpus from yazhi-lem/corpus-tamil project.

    Corpus-tamil is a collection of high-quality Tamil text from various sources.
    This importer handles multiple format support (JSONL, TXT, JSON).
    """

    def __init__(self, repo_path: Optional[str] = None):
        """Initialize CorpusTamilImporter.

        Args:
            repo_path: Path to yazhi-lem/corpus-tamil repository. If None, will
                      require explicit path in import calls.
        """
        self.repo_path = Path(repo_path) if repo_path else None
        self.logger = logger

    def import_from_repo(
        self, repo_path: Optional[str] = None
    ) -> Generator[dict, None, None]:
        """Load pre-curated corpus from corpus-tamil repository.

        Yields JSONL records with format:
        {
            "id": "corpus-tamil-{source}-{index}",
            "text": "Tamil text content",
            "source": "corpus-tamil-{type}",
            "url": source_url or null,
            "quality_score": 0.85
        }

        Args:
            repo_path: Path to yazhi-lem/corpus-tamil repository.
                      If not provided, uses self.repo_path.

        Yields:
            dict: JSONL-compatible corpus records.

        Raises:
            FileNotFoundError: If repo path not found or no data files exist.
        """
        path = Path(repo_path) if repo_path else self.repo_path
        if not path:
            raise ValueError(
                "repo_path must be provided either in __init__ or import_from_repo()"
            )

        path = Path(path)
        if not path.exists():
            self.logger.error(f"Corpus-tamil repository not found at {path}")
            raise FileNotFoundError(f"Corpus-tamil repository not found at {path}")

        # Look for common data locations in corpus-tamil project
        data_dirs = [
            path / "data",
            path / "corpus",
            path / "texts",
            path,
        ]

        processed = 0
        for data_dir in data_dirs:
            if not data_dir.exists():
                continue

            # Process JSONL files
            for jsonl_file in data_dir.glob("*.jsonl"):
                try:
                    yield from self._process_jsonl(jsonl_file)
                    processed += 1
                except Exception as e:
                    self.logger.warning(f"Error processing {jsonl_file}: {e}")

            # Process JSON files
            for json_file in data_dir.glob("*.json"):
                try:
                    yield from self._process_json(json_file)
                    processed += 1
                except Exception as e:
                    self.logger.warning(f"Error processing {json_file}: {e}")

            # Process TXT files
            for txt_file in data_dir.glob("*.txt"):
                try:
                    yield from self._process_txt(txt_file)
                    processed += 1
                except Exception as e:
                    self.logger.warning(f"Error processing {txt_file}: {e}")

        if processed == 0:
            self.logger.warning(f"No data files found in {path}")

    def _process_jsonl(self, file_path: Path) -> Generator[dict, None, None]:
        """Process JSONL file from corpus-tamil.

        Args:
            file_path: Path to JSONL file.

        Yields:
            dict: Corpus records.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                    yield self._convert_record(record, line_idx, file_path.stem)
                except json.JSONDecodeError as e:
                    self.logger.debug(
                        f"Invalid JSON at line {line_idx} in {file_path.name}: {e}"
                    )

    def _process_json(self, file_path: Path) -> Generator[dict, None, None]:
        """Process JSON file from corpus-tamil.

        Args:
            file_path: Path to JSON file.

        Yields:
            dict: Corpus records.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in {file_path.name}: {e}")
                return

        records = []
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            if "data" in data:
                records = (
                    data["data"] if isinstance(data["data"], list) else [data["data"]]
                )
            elif "texts" in data:
                records = (
                    data["texts"]
                    if isinstance(data["texts"], list)
                    else [data["texts"]]
                )
            else:
                records = [data]

        for idx, record in enumerate(records):
            try:
                yield self._convert_record(record, idx, file_path.stem)
            except (KeyError, ValueError) as e:
                self.logger.debug(f"Skipping record {idx} in {file_path.name}: {e}")

    def _process_txt(self, file_path: Path) -> Generator[dict, None, None]:
        """Process TXT file from corpus-tamil (one text per file or line-separated).

        Args:
            file_path: Path to TXT file.

        Yields:
            dict: Corpus records.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # If content is very long, split by paragraph
        # Otherwise treat entire file as single record
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        if len(paragraphs) > 1:
            # Multiple paragraphs - yield each as separate record
            for idx, text in enumerate(paragraphs):
                if text:
                    yield {
                        "id": f"corpus-tamil-txt-{file_path.stem}-{idx}",
                        "text": text,
                        "source": "corpus-tamil-txt",
                        "url": None,
                        "quality_score": 0.85,
                    }
        else:
            # Single text - yield as single record
            if content.strip():
                yield {
                    "id": f"corpus-tamil-txt-{file_path.stem}",
                    "text": content.strip(),
                    "source": "corpus-tamil-txt",
                    "url": None,
                    "quality_score": 0.85,
                }

    def _convert_record(self, record: dict, idx: int, source_name: str) -> dict:
        """Convert corpus-tamil record to unified corpus format.

        Args:
            record: Raw record from corpus-tamil.
            idx: Record index for ID generation.
            source_name: Name of source file for categorization.

        Returns:
            dict: Corpus-formatted record.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If text is empty or invalid.
        """
        # Extract text from various possible field names
        text = (
            record.get("text")
            or record.get("content")
            or record.get("body")
            or record.get("data")
            or ""
        )

        if not text or not str(text).strip():
            raise ValueError("No text content found in record")

        text = str(text).strip()

        # Determine source type and quality
        source_type = "corpus-tamil"
        quality_score = record.get("quality_score", 0.85)

        # If record has category or type, use it for source
        if "category" in record:
            source_type = f"corpus-tamil-{record['category']}"
        elif "type" in record:
            source_type = f"corpus-tamil-{record['type']}"

        return {
            "id": f"{source_type}-{source_name}-{idx}",
            "text": text,
            "source": source_type,
            "url": record.get("url") or record.get("source_url"),
            "quality_score": float(quality_score) if quality_score else 0.85,
        }

    def import_from_directory(self, directory: str) -> Generator[dict, None, None]:
        """Import from a directory of corpus-tamil files.

        Convenience method for batch importing all files in a directory.

        Args:
            directory: Path to directory containing corpus-tamil data files.

        Yields:
            dict: Corpus records.
        """
        yield from self.import_from_repo(directory)
