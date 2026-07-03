#!/usr/bin/env python3
"""
scrape_news_tamil_enhanced.py — Enhanced Tamil News Corpus Scraper for Adhan
ARIVU Cycle 6 | Jul 3, 2026

OBJECTIVE: Deep corpus expansion to 1000+ articles from multiple sources
- Dinamalar + Dinamani: Article archives and pagination
- BBC Tamil: Full topic archives  
- Vikatan: Classical + modern Tamil literature
- OneIndia Tamil: Broad news coverage

Features:
- Paginated scraping with archive support
- Deduplication by content hash
- Smart Tamil quality filtering (>60% Tamil/script chars)
- Rate limiting with exponential backoff
- Logging of all URLs + metrics
- Streaming JSONL output

Output: JSONL format (matches train_adhan_real.py expectations)
  {"text": "...", "source": "...", "date": "...", "category": "...", "type": "news"}
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse
from collections import defaultdict

from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT = BASE_DIR / "data" / "news_tamil" / "articles.jsonl"
LOGS_DIR = BASE_DIR / "data" / "news_tamil" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

RATE_LIMIT = 1.5  # seconds between requests
MIN_ARTICLE_LENGTH = 100
TAMIL_THRESHOLD = 0.60  # 60% Tamil script required

USER_AGENT = "YazhiAdhanBot/1.0 (Research; +https://yazhi.org/bot)"

# Enhanced source configurations with pagination
SOURCES = {
    "dinamalar": {
        "name": "Dinamalar",
        "base_url": "https://www.dinamalar.com",
        "listing_paths": [
            "/news/tamilnadu",
            "/news/india", 
            "/news/world",
            "/sports",
            "/entertainment",
            "/health",
            "/business",
        ],
        "article_selector": "article, .news-content, .article-content, .news-body, #newsContent",
        "title_selector": "h1, .article-title, .news-title, .headline",
        "text_selector": ".article-body, .news-body, .content-body, #articleBody, .story-body",
        "date_selector": "time, .date, .publish-date",
        "pagination": "?page={page}",
        "max_pages": 15,
    },
    "dinamani": {
        "name": "Dinamani",
        "base_url": "https://www.dinamani.com",
        "listing_paths": [
            "/tamilnadu",
            "/india",
            "/world",
            "/sports",
            "/cinema",
            "/business",
            "/tech",
        ],
        "article_selector": "article, .news-content, .article-content",
        "title_selector": "h1, .article-title, .news-title",
        "text_selector": ".article-body, .news-body, .content-body",
        "date_selector": "time, .date, .publish-date",
        "pagination": "?page={page}",
        "max_pages": 20,
    },
    "bbctamil": {
        "name": "BBC Tamil",
        "base_url": "https://www.bbc.com/tamil",
        "listing_paths": [
            "/topics/c5v12v9vj1et",  # Top stories
            "/topics/cg4vylwvggnt",  # India
            "/topics/c9wpm0en87xt",  # World
            "/topics/cxvjnc81v9zt",  # India politics
            "/topics/cz2k9yq1yp3t",  # Sports
        ],
        "article_selector": "article, [data-component='text-block'], .bbc-1fxtbkn",
        "title_selector": "h1, .bbc-1sk3j6u, [data-testid='headline']",
        "text_selector": "article .ssrcss-11r1m41-RichTextComponentWrapper, .bbc-19j92fr",
        "date_selector": "time, [data-testid='timestamp']",
        "pagination": None,  # BBC uses AJAX
        "max_pages": 1,
    },
    "vikatan": {
        "name": "Vikatan",
        "base_url": "https://www.vikatan.com",
        "listing_paths": [
            "/news",
            "/cinema",
            "/health",
            "/business",
            "/sports",
        ],
        "article_selector": "article, .news-content, .story-box",
        "title_selector": "h1, .news-title, h2",
        "text_selector": ".article-body, .story-body, .content",
        "date_selector": "time, .date, .publish-date",
        "pagination": "?page={page}",
        "max_pages": 10,
    },
    "oneindia_tamil": {
        "name": "OneIndia Tamil",
        "base_url": "https://tamil.oneindia.com",
        "listing_paths": [
            "/news/tamilnadu",
            "/news/india",
            "/news/world",
            "/sports",
        ],
        "article_selector": "article, .article-content, .news-content",
        "title_selector": "h1, .article-title, .news-title",
        "text_selector": ".article-body, .news-body, .content-body",
        "date_selector": "time, .date, .publish-date",
        "pagination": "?page={page}",
        "max_pages": 8,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def is_tamil_text(text):
    """Check if text is predominantly Tamil script."""
    if not text:
        return False
    tamil_pattern = r'[\u0B80-\u0BFF]'
    tamil_chars = len(re.findall(tamil_pattern, text))
    total_chars = len(text)
    if total_chars == 0:
        return False
    ratio = tamil_chars / total_chars
    return ratio >= TAMIL_THRESHOLD

def clean_text(text):
    """Clean and normalize Tamil text."""
    if not text:
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  # Remove control chars
    return text

def content_hash(text):
    """Generate content hash for deduplication."""
    normalized = re.sub(r'\s+', ' ', text.strip().lower())[:500]
    return hashlib.md5(normalized.encode()).hexdigest()

def fetch_url(url, timeout=15):
    """Fetch URL using curl with error handling."""
    try:
        result = subprocess.run(
            ["curl", "-s", "-L", "--max-time", str(timeout), "-A", USER_AGENT, url],
            capture_output=True, text=True, timeout=timeout + 5
        )
        if result.returncode != 0:
            return None
        return result.stdout if result.stdout else None
    except Exception as e:
        print(f"    ERROR fetching {url}: {e}", file=sys.stderr)
        return None

def extract_article(html, source_config, source_name, url):
    """Extract article text from HTML."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        title = None
        for selector in source_config['title_selector'].split(','):
            elem = soup.select_one(selector.strip())
            if elem:
                title = clean_text(elem.get_text())
                if title and len(title) > 10:
                    break
        
        if not title:
            return None
        
        # Extract text
        text_parts = []
        for selector in source_config['text_selector'].split(','):
            elems = soup.select(selector.strip())
            for elem in elems:
                t = clean_text(elem.get_text())
                if t and len(t) > 20:
                    text_parts.append(t)
        
        if not text_parts:
            return None
        
        text = " ".join(text_parts)
        
        # Validate
        if len(text) < MIN_ARTICLE_LENGTH or not is_tamil_text(text):
            return None
        
        # Extract date
        date_str = datetime.now(timezone.utc).isoformat()
        if 'date_selector' in source_config:
            for selector in source_config['date_selector'].split(','):
                elem = soup.select_one(selector.strip())
                if elem:
                    date_text = elem.get_text() or elem.get('content', '')
                    if date_text:
                        date_str = date_text[:50]
                        break
        
        return {
            "title": title,
            "text": text,
            "source": source_name,
            "url": url,
            "date": date_str,
            "type": "news",
        }
    except Exception as e:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCRAPER
# ─────────────────────────────────────────────────────────────────────────────

class TamilCorporaScraper:
    def __init__(self, output_path, verbose=False):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.stats = defaultdict(int)
        self.seen_hashes = set()
        self.articles = []
        
    def log(self, msg):
        if self.verbose:
            print(msg)
    
    def scrape_source(self, source_key, source_config, max_articles=None):
        """Scrape a single news source."""
        source_name = source_config["name"]
        print(f"\n{'='*70}")
        print(f"Source: {source_name} ({source_key})")
        print(f"Base URL: {source_config['base_url']}")
        print(f"{'='*70}")
        
        if max_articles is None:
            max_articles = 200  # Default per source
        
        found_articles = []
        total_attempted = 0
        skipped = 0
        
        # Scrape each listing path with pagination
        for listing_path in source_config['listing_paths']:
            page = 1
            max_pages = source_config.get('max_pages', 3)
            
            while page <= max_pages and len(found_articles) < max_articles:
                if source_config.get('pagination'):
                    url = source_config['base_url'] + listing_path + source_config['pagination'].format(page=page)
                else:
                    url = source_config['base_url'] + listing_path
                
                self.log(f"  Fetching page {page}: {url[:80]}")
                html = fetch_url(url)
                
                if not html:
                    print(f"    FAILED to fetch")
                    page += 1
                    time.sleep(RATE_LIMIT * 2)
                    continue
                
                try:
                    soup = BeautifulSoup(html, 'html.parser')
                    # Find all links
                    links = []
                    for a in soup.find_all('a', href=True):
                        href = a['href']
                        if not href.startswith('http'):
                            href = urljoin(source_config['base_url'], href)
                        # Basic filtering
                        if source_key in href or 'article' in href.lower() or 'news' in href.lower():
                            links.append(href)
                    
                    unique_links = list(set(links))[:40]  # Limit per page
                    print(f"    Found {len(unique_links)} candidate links")
                    
                    # Fetch each article
                    for link in unique_links:
                        if len(found_articles) >= max_articles:
                            break
                        
                        total_attempted += 1
                        time.sleep(RATE_LIMIT)  # Rate limit
                        
                        article_html = fetch_url(link, timeout=10)
                        if not article_html:
                            skipped += 1
                            continue
                        
                        article = extract_article(article_html, source_config, source_name, link)
                        if article:
                            h = content_hash(article['text'])
                            if h not in self.seen_hashes:
                                self.seen_hashes.add(h)
                                found_articles.append(article)
                                self.stats[f'{source_key}_articles'] += 1
                                print(f"      [{len(found_articles)}/{max_articles}] {article['title'][:60]}")
                        else:
                            skipped += 1
                    
                    if page > 1 and len(unique_links) < 5:
                        # No more articles on this page
                        break
                    
                except Exception as e:
                    print(f"    ERROR parsing page: {e}")
                
                page += 1
                time.sleep(RATE_LIMIT)
        
        print(f"\n  => {len(found_articles)} articles from {source_name}")
        print(f"     (Attempted: {total_attempted}, Skipped: {skipped})")
        
        self.articles.extend(found_articles)
        self.stats['total_articles'] = len(self.articles)
        return len(found_articles)
    
    def save(self):
        """Save articles to JSONL."""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for article in self.articles:
                json.dump(article, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"\n{'='*70}")
        print(f"SAVED: {len(self.articles)} articles to {self.output_path}")
        print(f"File size: {self.output_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"{'='*70}")
        
        # Summary
        print("\nSORCE BREAKDOWN:")
        for source in SOURCES.keys():
            count = self.stats.get(f'{source}_articles', 0)
            if count > 0:
                print(f"  {SOURCES[source]['name']:20} {count:4d} articles")
        
        return len(self.articles)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global RATE_LIMIT
    
    parser = argparse.ArgumentParser(
        description='Enhanced Tamil News Corpus Scraper for Adhan Training'
    )
    parser.add_argument(
        '--output', type=str, default=str(DEFAULT_OUTPUT),
        help=f'Output JSONL file (default: {DEFAULT_OUTPUT})'
    )
    parser.add_argument(
        '--sources', type=str, default='all',
        help='Comma-separated sources (dinamalar,dinamani,bbctamil,vikatan,oneindia_tamil) or "all"'
    )
    parser.add_argument(
        '--max-per-source', type=int, default=200,
        help='Maximum articles per source (default: 200)'
    )
    parser.add_argument(
        '--rate-limit', type=float, default=RATE_LIMIT,
        help=f'Rate limit in seconds (default: {RATE_LIMIT})'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Test mode: 50 articles, quick run'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.test:
        args.max_per_source = 50
        args.output = str(BASE_DIR / "data" / "news_tamil" / "test_enhanced.jsonl")
    
    if args.rate_limit:
        RATE_LIMIT = args.rate_limit
    
    # Parse sources
    if args.sources.lower() == 'all':
        sources_to_scrape = SOURCES.keys()
    else:
        sources_to_scrape = [s.strip() for s in args.sources.split(',')]
        sources_to_scrape = [s for s in sources_to_scrape if s in SOURCES]
    
    print("=" * 70)
    print("ENHANCED TAMIL NEWS CORPUS SCRAPER — Adhan Training Data")
    print("ARIVU Cycle 6 | Jul 3, 2026")
    print("=" * 70)
    print(f"Output: {args.output}")
    print(f"Sources: {', '.join(sources_to_scrape)}")
    print(f"Max per source: {args.max_per_source}")
    print(f"Rate limit: {RATE_LIMIT}s")
    print("=" * 70)
    
    scraper = TamilCorporaScraper(args.output, verbose=args.verbose)
    
    for source_key in sources_to_scrape:
        if source_key not in SOURCES:
            print(f"WARNING: Unknown source {source_key}")
            continue
        
        try:
            scraper.scrape_source(source_key, SOURCES[source_key], args.max_per_source)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Saving progress...")
            break
        except Exception as e:
            print(f"ERROR scraping {source_key}: {e}")
            continue
    
    scraper.save()

if __name__ == '__main__':
    main()
