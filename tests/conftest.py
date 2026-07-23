"""Pytest configuration and shared fixtures for Adhan SLM tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables."""
    old_env = os.environ.copy()
    # Disable JSON logging for tests (use colored output)
    os.environ["ADHAN_JSON_LOGS"] = "false"
    os.environ["ADHAN_LOG_LEVEL"] = "WARNING"
    yield
    # Restore environment
    os.environ.clear()
    os.environ.update(old_env)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text() -> str:
    """Sample Tamil text for testing."""
    return "தமிழ் என்பது ஒரு அழகான மொழி. நாம் தமிழை பேசுகிறோம், வாழ்க்கை நல்லது."


@pytest.fixture
def sample_corpus(temp_dir: Path) -> Path:
    """Create a sample corpus file for testing."""
    corpus_file = temp_dir / "corpus.txt"
    corpus_file.write_text("""தமிழ் என்பது ஒரு அழகான மொழி.
நாம் தமிழை பேசுகிறோம், வாழ்க்கை நல்லது.
கணிணி ஒரு பயனுள்ள கருவி.
உலகம் உழைப்பாலும், அறிவாலும் மாறுகிறது.""")
    return corpus_file


@pytest.fixture
def sample_jsonl_corpus(temp_dir: Path) -> Path:
    """Create a sample JSONL corpus file for testing."""
    corpus_file = temp_dir / "corpus.jsonl"
    corpus_file.write_text(
        '{"text": "தமிழ் என்பது ஒரு அழகான மொழி."}\n'
        '{"text": "நாம் தமிழை பேசுகிறோம்."}\n'
        '{"text": "வாழ்க்கை நல்லது."}\n'
    )
    return corpus_file
