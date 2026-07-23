"""Importer for classical Tamil literature from Yazhi API and open Sangam sources."""

import json
import logging
from pathlib import Path
from typing import Generator, Optional
from urllib.parse import urljoin

from adhan_slm.core import get_logger

logger = get_logger(__name__)


class SangamImporter:
    """Import classical Tamil literature from Yazhi API and open Sangam sources.

    Sangam literature is the oldest extant Tamil literature, including works like
    Thirukkural, Silappatikaram, and other classical texts. This importer handles:
    1. Yazhi API queries for Sangam works
    2. Open Sangam repository scraping (sangam.org, etc.)
    3. Local Sangam text files
    """

    def __init__(self, yazhi_api_endpoint: Optional[str] = None):
        """Initialize SangamImporter.

        Args:
            yazhi_api_endpoint: Base URL for Yazhi API (e.g., https://api.yazhi.ai).
                               If None, local/open sources only.
        """
        self.yazhi_api_endpoint = yazhi_api_endpoint
        self.logger = logger

    def import_from_yazhi_api(self, query: Optional[str] = None) -> Generator[dict, None, None]:
        """Query Yazhi API for Sangam works.

        Note: This is a placeholder for when Yazhi API is available.
        Currently returns empty as API endpoint would need authentication.

        Args:
            query: Optional search query (e.g., "thirukkural", "akananuru").

        Yields:
            dict: Corpus records from Yazhi API.
        """
        if not self.yazhi_api_endpoint:
            self.logger.warning("Yazhi API endpoint not configured, skipping API import")
            return

        try:
            import requests
        except ImportError:
            self.logger.warning("requests library not available for Yazhi API queries")
            return

        # Placeholder for actual API implementation
        self.logger.info("Yazhi API import not yet implemented (awaiting API credentials)")

        # When implemented, would look like:
        # endpoint = urljoin(self.yazhi_api_endpoint, "/sangam/texts")
        # response = requests.get(endpoint, params={"query": query or "all"})
        # for record in response.json()["results"]:
        #     yield self._convert_api_record(record)

    def import_from_open_sangam(self, source_path: Optional[str] = None) -> Generator[dict, None, None]:
        """Import from open Sangam resources (local files or well-known sources).

        Supports:
        1. Local Sangam text files
        2. sangam.org repository data
        3. Project Madurai texts (pmworks)
        4. Other open Tamil literature archives

        Args:
            source_path: Path to local Sangam text files. If None, searches common locations.

        Yields:
            dict: Corpus records from open Sangam sources.
        """
        processed = 0

        # If local path provided, process it
        if source_path:
            source_path = Path(source_path)
            if source_path.exists():
                yield from self._process_sangam_files(source_path, "open-sangam-local")
                processed += 1
            else:
                self.logger.warning(f"Sangam source path not found: {source_path}")

        # Try common locations for Sangam texts
        common_paths = [
            Path("/opt/sangam"),
            Path("/data/sangam"),
            Path("./data/sangam"),
            Path("./sangam"),
        ]

        for path in common_paths:
            if path.exists() and path.is_dir():
                yield from self._process_sangam_files(path, "open-sangam-local")
                processed += 1

        if processed == 0:
            self.logger.info(
                "No local Sangam files found. Provide source_path or check common locations."
            )

    def import_from_pmworks(
        self, pmworks_path: Optional[str] = None
    ) -> Generator[dict, None, None]:
        """Import classical Tamil texts from Project Madurai pmworks collection.

        Project Madurai is a repository of Tamil literature. This importer handles
        the pmworks format used there.

        Args:
            pmworks_path: Path to Project Madurai pmworks files. If None, searches common locations.

        Yields:
            dict: Corpus records from Project Madurai.
        """
        if not pmworks_path:
            # Try common locations
            common_paths = [
                Path("/opt/pmworks"),
                Path("/data/pmworks"),
                Path("./data/pmworks"),
                Path("./pmworks"),
            ]
            for path in common_paths:
                if path.exists():
                    pmworks_path = path
                    break

        if not pmworks_path:
            self.logger.info("No Project Madurai pmworks found. Provide pmworks_path if available.")
            return

        pmworks_path = Path(pmworks_path)
        yield from self._process_pmworks_files(pmworks_path)

    def _process_sangam_files(
        self, directory: Path, source_type: str
    ) -> Generator[dict, None, None]:
        """Process Sangam text files from a directory.

        Supports JSONL, JSON, and plain text formats.

        Args:
            directory: Path to directory containing Sangam texts.
            source_type: Classification for source (e.g., "open-sangam-local").

        Yields:
            dict: Corpus records.
        """
        if not directory.exists():
            self.logger.warning(f"Directory not found: {directory}")
            return

        # Process JSON/JSONL files
        for json_file in directory.glob("**/*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                records = data if isinstance(data, list) else [data]
                for idx, record in enumerate(records):
                    try:
                        yield self._convert_sangam_record(record, idx, json_file.stem)
                    except (KeyError, ValueError):
                        continue
            except json.JSONDecodeError as e:
                self.logger.debug(f"Invalid JSON in {json_file}: {e}")

        for jsonl_file in directory.glob("**/*.jsonl"):
            try:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                            yield self._convert_sangam_record(record, line_idx, jsonl_file.stem)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                self.logger.debug(f"Error processing {jsonl_file}: {e}")

        # Process plain text files
        for txt_file in directory.glob("**/*.txt"):
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    content = f.read()

                if content.strip():
                    yield {
                        "id": f"sangam-classical-txt-{txt_file.stem}",
                        "text": content.strip(),
                        "source": "sangam-classical",
                        "url": None,
                        "quality_score": 0.9,  # Classical texts are high quality
                    }
            except Exception as e:
                self.logger.debug(f"Error reading {txt_file}: {e}")

    def _process_pmworks_files(self, directory: Path) -> Generator[dict, None, None]:
        """Process Project Madurai pmworks format files.

        Args:
            directory: Path to pmworks directory.

        Yields:
            dict: Corpus records.
        """
        # Project Madurai uses specific format, typically XML or plain text
        for xml_file in directory.glob("**/*.xml"):
            try:
                yield from self._parse_pmworks_xml(xml_file)
            except Exception as e:
                self.logger.debug(f"Error processing PMWorks XML {xml_file}: {e}")

        for txt_file in directory.glob("**/*.txt"):
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    content = f.read()

                if content.strip():
                    yield {
                        "id": f"sangam-pmworks-{txt_file.stem}",
                        "text": content.strip(),
                        "source": "sangam-pmworks",
                        "url": None,
                        "quality_score": 0.9,
                    }
            except Exception as e:
                self.logger.debug(f"Error reading PMWorks text {txt_file}: {e}")

    def _parse_pmworks_xml(self, file_path: Path) -> Generator[dict, None, None]:
        """Parse Project Madurai XML format.

        Attempts to extract text content from XML structure.

        Args:
            file_path: Path to PMWorks XML file.

        Yields:
            dict: Corpus records.
        """
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            self.logger.warning("xml.etree not available for PMWorks XML parsing")
            return

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Extract all text content from XML
            texts = []
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    texts.append(elem.text.strip())

            if texts:
                combined_text = " ".join(texts)
                yield {
                    "id": f"sangam-pmworks-{file_path.stem}",
                    "text": combined_text,
                    "source": "sangam-pmworks",
                    "url": None,
                    "quality_score": 0.9,
                }
        except Exception as e:
            self.logger.debug(f"Error parsing PMWorks XML {file_path}: {e}")

    def _convert_sangam_record(self, record: dict, idx: int, source_name: str) -> dict:
        """Convert Sangam/PMWorks record to unified corpus format.

        Args:
            record: Raw record from Sangam/PMWorks source.
            idx: Record index for ID generation.
            source_name: Name of source for categorization.

        Returns:
            dict: Corpus-formatted record.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If text is empty.
        """
        text = (
            record.get("text")
            or record.get("content")
            or record.get("body")
            or record.get("data")
            or ""
        )

        if not text or not str(text).strip():
            raise ValueError("No text content in record")

        text = str(text).strip()

        # Determine source type
        source_type = record.get("source", "sangam-classical")
        if "category" in record:
            source_type = f"sangam-{record['category']}"

        return {
            "id": f"{source_type}-{source_name}-{idx}",
            "text": text,
            "source": source_type,
            "url": record.get("url") or record.get("source_url"),
            "quality_score": float(record.get("quality_score", 0.9)),
        }

    def _convert_api_record(self, record: dict) -> dict:
        """Convert Yazhi API record to unified corpus format.

        Args:
            record: Record from Yazhi API response.

        Returns:
            dict: Corpus-formatted record.
        """
        return {
            "id": record.get("id", "sangam-api-unknown"),
            "text": record.get("text", ""),
            "source": "sangam-yazhi-api",
            "url": record.get("url"),
            "quality_score": float(record.get("quality_score", 0.9)),
        }
