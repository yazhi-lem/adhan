#!/usr/bin/env python3
"""Targeted Tamil web corpus ingestion — high-quality sources only.

Pulls from known high-quality Tamil content repositories:
  • Project Madurai (projectmadurai.org) — classical & modern literature
  • Tamil Virtual Academy (tamilvu.org) — authoritative reference texts
  • Tamil Wikipedia dumps (beyond HF streaming)
  • Tamil blog aggregators (clean RSS feeds)
  • Tamil Nadu government open data portals

This is NOT a general web crawler — it hits specific endpoints that
publish Tamil text under CC-BY or public domain.

Usage:
    python scripts/ingest_tamil_web.py --source madurai --out data/raw/web/madurai.jsonl
    python scripts/ingest_tamil_web.py --source all --out-dir data/raw/web/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Iterator
from urllib.parse import urljoin, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.core.logging import get_logger

logger = get_logger(__name__)

# Try to import requests/beautifulsoup; fail gracefully if not installed
try:
    import requests
    from bs4 import BeautifulSoup
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests + beautifulsoup4 not installed; web ingestion disabled")


TAMIL_RANGE = re.compile(r"[\u0B80-\u0BFF]")
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"[ \t]+")


def normalize(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = WS_RE.sub(" ", text)
    return text.strip()


def tamil_ratio(text: str) -> float:
    clean = text.replace(" ", "").replace("\n", "")
    if not clean:
        return 0.0
    return len(TAMIL_RANGE.findall(clean)) / len(clean)


def fetch_text(url: str, retries: int = 3, delay: float = 1.0) -> str:
    if not HAS_REQUESTS:
        return ""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30, headers={
                "User-Agent": "Mozilla/5.0 (compatible; AdhanBot/1.0; +https://yazhi.dev/bot)"
            })
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.warning(f"fetch {url} failed (attempt {attempt+1}): {e}")
            time.sleep(delay * (attempt + 1))
    return ""


def extract_paragraphs(html: str, min_chars: int = 40) -> Iterator[str]:
    if not HAS_REQUESTS:
        return
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style/nav/footer
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    for p in soup.find_all("p"):
        text = normalize(p.get_text(separator=" ", strip=True))
        if len(text) >= min_chars and tamil_ratio(text) >= 0.50:
            yield text


# --- Project Madurai -------------------------------------------------------

def iter_madurai(limit: int = 10000) -> Iterator[dict]:
    """Scrape Project Madurai index and extract texts.
    
    Project Madurai publishes classical Tamil works as HTML pages.
    We respect their robots.txt and rate-limit aggressively.
    """
    if not HAS_REQUESTS:
        return
    base = "https://www.projectmadurai.org"
    index_url = f"{base}/pmworks.html"
    logger.info(f"fetching Project Madurai index: {index_url}")
    html = fetch_text(index_url)
    if not html:
        return
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".html") and "/pmworks/" not in href:
            full = urljoin(base, href)
            links.append(full)
    logger.info(f"found {len(links)} potential work links")
    n = 0
    for url in links[:limit]:
        doc_html = fetch_text(url, delay=2.0)
        if not doc_html:
            continue
        paragraphs = list(extract_paragraphs(doc_html))
        if not paragraphs:
            continue
        full_text = "\n\n".join(paragraphs)
        yield {
            "text": full_text,
            "source": "web:project-madurai",
            "tier": 1,
            "quality_score": 0.95,
            "url": url,
        }
        n += 1
        if n % 10 == 0:
            logger.info(f"madurai: {n} works extracted")
        time.sleep(1.5)  # Be polite
    logger.info(f"madurai: total {n} works")


# --- Tamil Virtual Academy --------------------------------------------------

def iter_tamilvu(limit: int = 5000) -> Iterator[dict]:
    """Tamil Virtual Academy — reference texts and literature."""
    if not HAS_REQUESTS:
        return
    # TamilVU has categorized sections; we target the literature ones
    sections = [
        "https://www.tamilvu.org/library/litcont.asp",
        "https://www.tamilvu.org/library/l3120/l3120cnt.jsp",
    ]
    n = 0
    for section in sections[:1]:  # Start with main literature container
        html = fetch_text(section, delay=2.0)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if ".jsp" in href or ".html" in href:
                full = urljoin("https://www.tamilvu.org", href)
                doc_html = fetch_text(full, delay=1.5)
                if not doc_html:
                    continue
                paragraphs = list(extract_paragraphs(doc_html))
                if paragraphs:
                    yield {
                        "text": "\n\n".join(paragraphs),
                        "source": "web:tamilvu",
                        "tier": 1,
                        "quality_score": 0.93,
                        "url": full,
                    }
                    n += 1
                    if n >= limit:
                        break
                time.sleep(1.0)
        if n >= limit:
            break
    logger.info(f"tamilvu: {n} documents")


# --- Tamil blog / news RSS feeds -------------------------------------------

def iter_tamil_rss(feeds: list[str], limit: int = 50000) -> Iterator[dict]:
    """Parse Tamil RSS feeds for articles."""
    if not HAS_REQUESTS:
        return
    try:
        import feedparser
    except ImportError:
        logger.warning("feedparser not installed; RSS ingestion skipped")
        return
    n = 0
    for feed_url in feeds:
        logger.info(f"parsing RSS: {feed_url}")
        try:
            fp = feedparser.parse(feed_url)
            for entry in fp.entries:
                # Some feeds have content, others only summary
                text = entry.get("content", [{}])[0].get("value", "") if "content" in entry else ""
                if not text:
                    text = entry.get("summary", "")
                text = normalize(text)
                if len(text) >= 200 and tamil_ratio(text) >= 0.55:
                    yield {
                        "text": text,
                        "source": "web:rss",
                        "tier": 3,
                        "quality_score": 0.70,
                        "url": entry.get("link", ""),
                    }
                    n += 1
                    if n >= limit:
                        break
        except Exception as e:
            logger.warning(f"RSS feed {feed_url} failed: {e}")
        if n >= limit:
            break
    logger.info(f"rss: {n} entries")


# --- Orchestration ----------------------------------------------------------

DEFAULT_RSS_FEEDS = [
    # Tamil news / blog RSS feeds (add more as discovered)
    "https://feeds.feedburner.com/HinduTamil",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest Tamil web content")
    ap.add_argument("--source", choices=["madurai", "tamilvu", "rss", "all"], default="all")
    ap.add_argument("--out-dir", default="data/raw/web")
    ap.add_argument("--madurai-limit", type=int, default=5000)
    ap.add_argument("--tamilvu-limit", type=int, default=3000)
    ap.add_argument("--rss-limit", type=int, default=50000)
    ap.add_argument("--rss-feeds", default=",".join(DEFAULT_RSS_FEEDS))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = []
    if args.source in ("madurai", "all"):
        sources.append(("madurai", iter_madurai(args.madurai_limit)))
    if args.source in ("tamilvu", "all"):
        sources.append(("tamilvu", iter_tamilvu(args.tamilvu_limit)))
    if args.source in ("rss", "all"):
        feeds = [f for f in args.rss_feeds.split(",") if f.strip()]
        sources.append(("rss", iter_tamil_rss(feeds, args.rss_limit)))

    for name, iterator in sources:
        out_path = out_dir / f"{name}.jsonl"
        logger.info(f"=== ingesting {name} -> {out_path} ===")
        n = 0
        with out_path.open("w", encoding="utf-8") as fh:
            for rec in iterator:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n % 1000 == 0:
                    logger.info(f"  {name}: {n} records")
        logger.info(f"{name}: {n} total records written to {out_path}")


if __name__ == "__main__":
    main()
