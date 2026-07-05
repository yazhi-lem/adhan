"""
Base scraper class for Tamil corpus sources.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from ..corpus import CorpusSample


class BaseScraper(ABC):
    """Abstract base scraper."""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
    
    @abstractmethod
    def scrape(self) -> List[CorpusSample]:
        """Scrape Tamil corpus from source."""
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Return human-readable source name."""
        pass
