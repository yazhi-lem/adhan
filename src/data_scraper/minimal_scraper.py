#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Tamil Corpus Scraper

Focused scraper targeting high-value Tamil sources:
  - TN Government Schemes (quality_score=0.9)
  - Tamil News Headlines (quality_score=0.7)
  - Wikipedia Trending Topics (quality_score=0.8)

Usage:
    python src/data_scraper/minimal_scraper.py \
        --domains govt,news,wiki \
        --limit 100 \
        --output data/scraped/minimal.jsonl
"""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from random import uniform
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; MinimalTamilScraper/1.0; "
        "+https://github.com/yazhi-lem/adhan)"
    ),
    "Accept-Language": "ta,en;q=0.9",
}


def is_tamil_content(text: str, min_ratio: float = 0.5) -> bool:
    """Return True if *text* has at least *min_ratio* Tamil Unicode characters."""
    if not text:
        return False
    tamil_chars = sum(1 for c in text if "\u0B80" <= c <= "\u0BFF")
    return (tamil_chars / len(text)) >= min_ratio


def sha256_id(text: str) -> str:
    """Return a truncated SHA-256 hex digest of *text* (16 characters)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _make_record(
    text: str,
    source: str,
    url: str,
    quality_score: float,
) -> Dict:
    """Build a standard output record."""
    return {
        "text": text,
        "source": source,
        "url": url,
        "quality_score": quality_score,
        "id": sha256_id(text),
    }


# ---------------------------------------------------------------------------
# Scraper class
# ---------------------------------------------------------------------------


class MinimalTamilScraper:
    """Focused Tamil corpus scraper targeting three high-value source domains."""

    # Source-level quality scores
    QUALITY = {
        "tn_govt_schemes": 0.9,
        "tamil_news_headlines": 0.7,
        "wikipedia_trending": 0.8,
    }

    # News sources: (url, CSS selector for headline elements)
    NEWS_SOURCES = [
        ("https://www.dinamalar.com", ".headlinetext"),
        ("https://www.dailythanthi.com", ".story-title"),
    ]

    def __init__(self, output_dir: Path = Path("data/scraped")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = self._build_session()

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(HEADERS)
        return session

    def _get(self, url: str, timeout: int = 15) -> Optional[requests.Response]:
        """Perform a GET request with error handling."""
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            logger.warning("Request failed for %s: %s", url, exc)
            return None

    def _sleep(self) -> None:
        """Rate-limit: sleep 1–2 seconds between requests."""
        time.sleep(uniform(1.0, 2.0))

    # ------------------------------------------------------------------
    # Domain scrapers
    # ------------------------------------------------------------------

    def scrape_tn_govt(self, limit: int = 100) -> List[Dict]:
        """
        Scrape TN Government scheme pages.

        Fetches the scheme listing at https://www.tn.gov.in/scheme and
        follows each scheme link to extract Tamil body text.
        """
        print("🏛️  Scraping TN Govt schemes...")
        base_url = "https://www.tn.gov.in/scheme"
        quality = self.QUALITY["tn_govt_schemes"]
        records: List[Dict] = []

        response = self._get(base_url)
        if response is None:
            print("   ⚠️  Could not reach tn.gov.in — skipping.")
            return records

        soup = BeautifulSoup(response.text, "html.parser")

        # Collect scheme links from the listing page
        scheme_links: List[str] = []
        for anchor in soup.find_all("a", href=True):
            href: str = anchor["href"]
            if "scheme" in href.lower():
                if href.startswith("http"):
                    scheme_links.append(href)
                elif href.startswith("/"):
                    scheme_links.append("https://www.tn.gov.in" + href)
            if len(scheme_links) >= limit * 2:
                break

        # Also try to extract text directly from the listing page
        for element in soup.find_all(["p", "li", "td", "div"]):
            text = element.get_text(separator=" ", strip=True)
            if len(text) > 80 and is_tamil_content(text):
                records.append(_make_record(text, "tn_govt_schemes", base_url, quality))
            if len(records) >= limit:
                break

        # Follow individual scheme links for more content
        for link in scheme_links:
            if len(records) >= limit:
                break
            self._sleep()
            page = self._get(link)
            if page is None:
                continue
            page_soup = BeautifulSoup(page.text, "html.parser")
            for element in page_soup.find_all(["p", "li", "div"]):
                text = element.get_text(separator=" ", strip=True)
                if len(text) > 80 and is_tamil_content(text):
                    records.append(_make_record(text, "tn_govt_schemes", link, quality))
                if len(records) >= limit:
                    break

        print(f"   ✅ {len(records)} records extracted")
        return records

    def scrape_tamil_news_headlines(self, limit: int = 200) -> List[Dict]:
        """
        Scrape Tamil news headlines from Dinamalar and Daily Thanthi.

        Extracts only headline text (fast, no full-article fetching).
        *limit* is split equally across configured news sources.
        """
        print("📰 Scraping Tamil news headlines...")
        quality = self.QUALITY["tamil_news_headlines"]
        records: List[Dict] = []
        per_site = max(1, limit // len(self.NEWS_SOURCES))

        for site_url, selector in self.NEWS_SOURCES:
            site_records: List[Dict] = []
            self._sleep()
            response = self._get(site_url)
            if response is None:
                logger.warning("Skipping %s", site_url)
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            elements = soup.select(selector)

            # Fallback: try common headline tags when CSS selector yields nothing
            if not elements:
                elements = soup.find_all(["h1", "h2", "h3"])

            for element in elements:
                text = element.get_text(separator=" ", strip=True)
                if len(text) > 10 and is_tamil_content(text):
                    site_records.append(
                        _make_record(text, "tamil_news_headlines", site_url, quality)
                    )
                if len(site_records) >= per_site:
                    break

            logger.info("  %s → %d headlines", site_url, len(site_records))
            records.extend(site_records)

        print(f"   ✅ {len(records)} records extracted")
        return records

    def scrape_wikipedia_trending(self, limit: int = 50) -> List[Dict]:
        """
        Scrape Tamil Wikipedia current-events portal paragraphs.

        Targets the Tamil Portal:நடப்பு_நிகழ்வுகள் (current events) page.
        """
        print("📚 Scraping Wikipedia trending...")
        # URL uses Tamil Unicode characters (valid UTF-8, requests handles encoding)
        url = "https://ta.wikipedia.org/wiki/போர்ட்டல்:நடப்பு_நிகழ்வுகள்"
        quality = self.QUALITY["wikipedia_trending"]
        records: List[Dict] = []

        response = self._get(url)
        if response is None:
            print("   ⚠️  Could not reach Tamil Wikipedia — skipping.")
            return records

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove navigation / boilerplate elements
        for tag in soup.find_all(["script", "style", "nav", "footer"]):
            tag.decompose()

        for para in soup.find_all("p"):
            text = para.get_text(separator=" ", strip=True)
            if len(text) > 50 and is_tamil_content(text):
                records.append(_make_record(text, "wikipedia_trending", url, quality))
            if len(records) >= limit:
                break

        print(f"   ✅ {len(records)} records extracted")
        return records

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def scrape_all(
        self,
        domains: Optional[List[str]] = None,
        limit_per_domain: int = 100,
    ) -> List[Dict]:
        """
        Orchestrate scraping across requested *domains*.

        Deduplicates results by SHA-256 hash of content before returning.

        Parameters
        ----------
        domains:
            List of domain names to scrape.  Recognised values:
            ``"govt"``, ``"news"``, ``"wiki"``.
            Defaults to all three when *None*.
        limit_per_domain:
            Maximum records to collect per domain.

        Returns
        -------
        List of unique record dicts.
        """
        if domains is None:
            domains = ["govt", "news", "wiki"]

        all_records: List[Dict] = []

        if "govt" in domains:
            all_records.extend(self.scrape_tn_govt(limit=limit_per_domain))

        if "news" in domains:
            all_records.extend(
                self.scrape_tamil_news_headlines(limit=limit_per_domain * 2)
            )

        if "wiki" in domains:
            all_records.extend(
                self.scrape_wikipedia_trending(limit=limit_per_domain // 2)
            )

        # Deduplicate by content hash
        seen: set[str] = set()
        unique: List[Dict] = []
        for record in all_records:
            if record["id"] not in seen:
                seen.add(record["id"])
                unique.append(record)

        print(f"\n✅ Scraped {len(unique)} unique records")
        return unique

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, records: List[Dict], output_path: Path) -> None:
        """Write *records* to *output_path* as JSONL (UTF-8)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"💾 Saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal Tamil corpus scraper (govt / news / wiki)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--domains",
        default="govt,news,wiki",
        help="Comma-separated list of domains to scrape: govt, news, wiki",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum records per domain",
    )
    parser.add_argument(
        "--output",
        default="data/scraped/minimal.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--output-dir",
        default="data/scraped",
        help="Directory for saving output (overridden by --output if absolute)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    domains = [d.strip().lower() for d in args.domains.split(",") if d.strip()]
    output_path = Path(args.output)

    scraper = MinimalTamilScraper(output_dir=output_path.parent)
    records = scraper.scrape_all(domains=domains, limit_per_domain=args.limit)
    scraper.save(records, output_path)


if __name__ == "__main__":
    main()
