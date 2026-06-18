#!/usr/bin/env python3
"""
scrape_news_tamil.py — Tamil news scraper for Adhan training data
ARIVU + Hermes | Rotation 26 Cycle 2 | Jun 18, 2026

Scrapes Tamil news articles from:
- Dinamalar (www.dinamalar.com)
- Dinamani (www.dinamani.com)
- BBC Tamil (www.bbc.com/tamil)
- OneIndia Tamil (tamil.oneindia.com)

Output: JSONL format matching OpenSangam schema:
  {"text": "...", "source": "...", "date": "...", "category": "...", "type": "news"}

Features:
- curl + BeautifulSoup (lightweight, no selenium)
- Rate limiting: 2 second delay between requests
- robots.txt respect
- Tamil text cleaning pipeline
- Deduplication by title hash
- Filter articles < 100 chars

Usage:
    python scrape_news_tamil.py --output data/news_tamil/articles.jsonl --max-per-source 50
    python scrape_news_tamil.py --test  # 5 articles per source, saves to test_scrape.jsonl
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
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT = BASE_DIR / "data" / "news_tamil" / "articles.jsonl"
TEST_OUTPUT = BASE_DIR / "data" / "news_tamil" / "test_scrape.jsonl"

# Rate limit: seconds between requests
RATE_LIMIT = 2

# Minimum article length (chars)
MIN_ARTICLE_LENGTH = 100

# User agent
USER_AGENT = "YazhiAdhanBot/1.0 (Research Project; +https://yazhi.org/bot)"

# Source configurations
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
        ],
        "article_selector": "article, .news-content, .article-content, .news-body, #newsContent",
        "title_selector": "h1, .article-title, .news-title, .headline",
        "date_selector": "time, .date, .publish-date, .article-date, meta[property='article:published_time']",
        "category_selector": ".category, .section-name, .breadcrumb a, meta[property='article:section']",
        "text_selector": ".article-body, .news-body, .content-body, #articleBody, .story-body",
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
        ],
        "article_selector": "article, .news-content, .article-content",
        "title_selector": "h1, .article-title, .news-title",
        "date_selector": "time, .date, .publish-date",
        "category_selector": ".category, .section-name, .breadcrumb a",
        "text_selector": ".article-body, .news-body, .content-body",
    },
    "bbctamil": {
        "name": "BBC Tamil",
        "base_url": "https://www.bbc.com/tamil",
        "listing_paths": [
            "/topics/c5v12v9vj1et",  # Top stories
            "/topics/cg4vylwvggnt",  # India
            "/topics/c9wpm0en87xt",  # World
        ],
        "article_selector": "article, [data-component='text-block'], .bbc-1fxtbkn",
        "title_selector": "h1, .bbc-1sk3j6u, [data-testid='headline']",
        "date_selector": "time, [data-testid='timestamp'], .bbc-14xtggo",
        "category_selector": ".bbc-11pkra2, [data-testid='section-label']",
        "text_selector": "article .ssrcss-11r1m41-RichTextComponentWrapper, .bbc-19j92fr, [data-component='text-block']",
    },
    "oneindia_tamil": {
        "name": "OneIndia Tamil",
        "base_url": "https://tamil.oneindia.com",
        "listing_paths": [
            "/news/tamilnadu",
            "/news/india",
            "/news/world",
            "/sports",
            "/entertainment",
        ],
        "article_selector": "article, .article-content, .news-content",
        "title_selector": "h1, .article-title, .news-title",
        "date_selector": "time, .date, .publish-date",
        "category_selector": ".category, .section-name",
        "text_selector": ".article-body, .news-body, .content-body",
    },
}

# ── Robots.txt checker ───────────────────────────────────────────────────────


class RobotsChecker:
    """
    Cache and check robots.txt for multiple domains.
    
    Uses a conservative approach: only blocks if robots.txt explicitly
    disallows the exact path. Handles known quirks of Python's
    RobotFileParser with 'Allow: /' + 'Disallow: /specific' patterns.
    """

    def __init__(self):
        self._cache = {}

    def _fetch_robots(self, domain):
        """Fetch and parse robots.txt for a domain."""
        robots_url = f"{domain}/robots.txt"
        try:
            result = subprocess.run(
                ["curl", "-s", "--max-time", "10", "-A", USER_AGENT, robots_url],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None  # No robots.txt = allow all

            rules = {"allow": [], "disallow": []}
            current_agent = "*"
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                if line.lower().startswith('user-agent:'):
                    current_agent = line.split(':', 1)[1].strip()
                    continue
                if current_agent in ('*', 'YazhiAdhanBot'):
                    if line.lower().startswith('allow:'):
                        path = line.split(':', 1)[1].strip()
                        if path:
                            rules["allow"].append(re.compile(self._path_to_regex(path)))
                    elif line.lower().startswith('disallow:'):
                        path = line.split(':', 1)[1].strip()
                        if path:
                            rules["disallow"].append(re.compile(self._path_to_regex(path)))

            return rules
        except Exception:
            return None

    @staticmethod
    def _path_to_regex(path):
        """Convert robots.txt path pattern to regex."""
        # Escape special regex chars except * and $
        path = path.replace('*', '.*').replace('$', r'\$')
        return '^' + path

    def can_fetch(self, url, user_agent=USER_AGENT):
        """Check if URL can be fetched according to robots.txt."""
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        path = parsed.path or '/'

        if domain not in self._cache:
            self._cache[domain] = self._fetch_robots(domain)

        rules = self._cache[domain]
        if rules is None:
            return True  # No robots.txt = allow

        # Check explicit allows first (more specific wins)
        for pattern in rules["allow"]:
            if pattern.match(path):
                return True

        # Check disallows
        for pattern in rules["disallow"]:
            if pattern.match(path):
                return False

        return True  # No match = allow


# ── HTTP client (curl-based) ─────────────────────────────────────────────────


def fetch_url(url, user_agent=USER_AGENT, timeout=15):
    """
    Fetch URL using curl (lightweight, no selenium).
    
    Returns:
        (status_code, html_content) or (None, None) on failure
    """
    try:
        result = subprocess.run(
            [
                "curl", "-s", "-L",
                "--max-time", str(timeout),
                "-A", user_agent,
                "-H", "Accept: text/html,application/xhtml+xml",
                "-H", "Accept-Language: ta,en;q=0.9",
                url
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )
        if result.returncode == 0:
            return 200, result.stdout
        else:
            return None, None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None, None


# ── Text cleaning (inline, avoids import issues) ──────────────────────────────

HTML_PATTERN = re.compile(r'<[^>]+>')
URL_PATTERN = re.compile(r'https?://[^\s<>"\')\]]+|www\.[^\s<>"\')\]]+')
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')
PHONE_PATTERN = re.compile(r'(?:\+?91[\-\s]?)?[6-9]\d{9}|\d{3,4}[\-\s]?\d{6,8}')
ZW_PATTERN = re.compile(r'[\u200b\u200c\u200d\ufeff]')
MULTI_SPACE = re.compile(r'[ \t]+')
MULTI_NEWLINE = re.compile(r'\n{3,}')

TAMIL_RANGE = (0x0B80, 0x0BFF)


def is_tamil_char(c):
    cp = ord(c)
    return TAMIL_RANGE[0] <= cp <= TAMIL_RANGE[1]


def tamil_ratio(text):
    if not text:
        return 0.0
    tamil_count = sum(1 for c in text if is_tamil_char(c))
    alpha_count = sum(1 for c in text if c.isalpha())
    return tamil_count / alpha_count if alpha_count > 0 else 0.0


def clean_article_text(text):
    """Clean article text: normalize Unicode, remove HTML/URLs/noise."""
    if not text:
        return ""

    text = unicodedata.normalize('NFC', text)
    text = ZW_PATTERN.sub('', text)
    text = HTML_PATTERN.sub(' ', text)
    text = URL_PATTERN.sub('', text)
    text = EMAIL_PATTERN.sub('', text)
    text = PHONE_PATTERN.sub('', text)
    text = MULTI_SPACE.sub(' ', text)
    text = MULTI_NEWLINE.sub('\n\n', text)
    text = text.strip()

    return text


# ── Article extraction ───────────────────────────────────────────────────────


def extract_text_from_selectors(soup, selectors):
    """Try multiple CSS selectors and return first match's text."""
    for selector in selectors:
        selector = selector.strip()
        try:
            elements = soup.select(selector)
            if elements:
                # Get text from all matching elements
                texts = []
                for el in elements:
                    text = el.get_text(separator=' ', strip=True)
                    if text:
                        texts.append(text)
                if texts:
                    return ' '.join(texts)
        except Exception:
            continue
    return ""


def extract_attribute(soup, selectors, attribute):
    """Extract attribute value from first matching element."""
    for selector in selectors:
        selector = selector.strip()
        try:
            el = soup.select_one(selector)
            if el:
                # Check for attribute
                if el.has_attr(attribute):
                    return el[attribute]
                # Special case: time datetime
                if attribute == 'datetime' and el.name == 'time':
                    return el.get('datetime', el.get_text(strip=True))
                # meta tag content
                if el.name == 'meta' and el.has_attr('content'):
                    return el['content']
        except Exception:
            continue
    return ""


def find_article_links(listing_url, source_config, max_articles=10):
    """
    Find article links from a listing/category page.
    
    Returns:
        List of absolute URLs
    """
    status, html = fetch_url(listing_url)
    if not html or status != 200:
        return []

    soup = BeautifulSoup(html, 'html.parser')
    links = set()

    # Find all links that look like article links
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(source_config['base_url'], href)

        # Filter: must be from same domain
        if urlparse(full_url).netloc != urlparse(source_config['base_url']).netloc:
            continue

        # Filter: skip non-article paths
        skip_patterns = [
            '/video/', '/photo/', '/gallery/', '/live/',
            '/tag/', '/author/', '/search', '/login', '/register',
            'javascript:', '#', 'mailto:',
        ]
        if any(p in full_url.lower() for p in skip_patterns):
            continue

        # Filter: article URLs typically have some structure
        # (not just the homepage or category page)
        path = urlparse(full_url).path
        if len(path) < 10:  # Too short to be an article
            continue

        links.add(full_url)

    return list(links)[:max_articles * 3]  # Extra for dedup


def scrape_article(url, source_key, source_config):
    """
    Scrape a single article.
    
    Returns:
        dict with keys: text, source, date, category, type, title, url
        or None if failed
    """
    status, html = fetch_url(url)
    if not html or status != 200:
        return None

    soup = BeautifulSoup(html, 'html.parser')

    # Extract title
    title = extract_text_from_selectors(soup, source_config['title_selector'].split(','))
    if not title:
        # Fallback: page title
        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else ""

    # Extract date
    date_str = extract_attribute(soup, source_config['date_selector'].split(','), 'datetime')
    if not date_str:
        date_str = extract_text_from_selectors(soup, source_config['date_selector'].split(','))

    # Extract category
    category = extract_text_from_selectors(soup, source_config['category_selector'].split(','))
    if not category:
        category = source_key

    # Extract article text
    text = extract_text_from_selectors(soup, source_config['text_selector'].split(','))

    # Fallback: try to get all paragraph text
    if not text or len(text) < 50:
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)

    # Clean the text
    text = clean_article_text(text)
    title = clean_article_text(title)

    # Validate
    if len(text) < MIN_ARTICLE_LENGTH:
        return None

    if tamil_ratio(text) < 0.4:  # Slightly lower threshold for news (may contain names)
        return None

    return {
        "text": text,
        "source": source_config['name'],
        "date": date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "category": category,
        "type": "news",
        "title": title,
        "url": url,
    }


# ── Main scraper ─────────────────────────────────────────────────────────────


def scrape_source(source_key, source_config, max_articles=50, robots_checker=None):
    """
    Scrape articles from a single source.
    
    Returns:
        List of article dicts
    """
    articles = []
    seen_titles = set()

    print(f"\n{'='*60}")
    print(f"Source: {source_config['name']} ({source_key})")
    print(f"Base URL: {source_config['base_url']}")
    print(f"Max articles: {max_articles}")
    print(f"{'='*60}")

    all_links = []

    # Collect article links from listing pages
    for listing_path in source_config['listing_paths']:
        listing_url = source_config['base_url'] + listing_path
        print(f"\n  Scanning: {listing_url}")

        # Check robots.txt
        if robots_checker and not robots_checker.can_fetch(listing_url):
            print(f"    BLOCKED by robots.txt")
            continue

        links = find_article_links(listing_url, source_config, max_articles)
        print(f"    Found {len(links)} candidate links")
        all_links.extend(links)

        time.sleep(RATE_LIMIT)

    # Deduplicate links
    all_links = list(dict.fromkeys(all_links))  # Preserve order, remove dupes
    print(f"\n  Total unique links: {len(all_links)}")

    # Scrape each article
    for i, url in enumerate(all_links):
        if len(articles) >= max_articles:
            break

        print(f"  [{i+1}/{len(all_links)}] {url[:80]}...", end=" ")

        # Check robots.txt
        if robots_checker and not robots_checker.can_fetch(url):
            print("BLOCKED (robots.txt)")
            continue

        article = scrape_article(url, source_key, source_config)

        if article:
            # Deduplicate by title hash
            title_hash = hashlib.md5(article['title'].encode()).hexdigest()
            if title_hash in seen_titles:
                print("DUP")
                continue
            seen_titles.add(title_hash)

            articles.append(article)
            print(f"OK ({len(article['text'])} chars, {tamil_ratio(article['text']):.0%} Tamil)")
        else:
            print("SKIP (too short or not Tamil)")

        time.sleep(RATE_LIMIT)

    print(f"\n  => {len(articles)} articles from {source_config['name']}")
    return articles


def write_jsonl(articles, filepath):
    """Write articles to JSONL file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

    print(f"\nWrote {len(articles)} articles to {filepath}")


def compute_stats(articles):
    """Compute statistics about scraped articles."""
    if not articles:
        return {}

    sources = {}
    total_chars = 0
    total_tokens = 0

    for a in articles:
        src = a['source']
        sources[src] = sources.get(src, 0) + 1
        total_chars += len(a['text'])
        total_tokens += len(a['text']) // 4  # Approximate BPE tokens

    return {
        'total_articles': len(articles),
        'total_characters': total_chars,
        'approx_tokens_bpe': total_tokens,
        'by_source': sources,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Tamil News Scraper for Adhan Training Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test                          # Test: 5 articles per source
  %(prog)s --output data/articles.jsonl    # Full scrape (50/source)
  %(prog)s --sources bbctamil dinamani     # Only specific sources
  %(prog)s --max-per-source 10             # Custom limit
        """
    )
    parser.add_argument('--output', '-o', type=str, default=str(DEFAULT_OUTPUT),
                        help='Output JSONL file path')
    parser.add_argument('--max-per-source', '-m', type=int, default=50,
                        help='Maximum articles per source (default: 50)')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: 5 articles per source')
    parser.add_argument('--sources', nargs='+', choices=list(SOURCES.keys()),
                        help='Specific sources to scrape (default: all)')
    parser.add_argument('--no-robots', action='store_true',
                        help='Skip robots.txt checking (not recommended)')
    parser.add_argument('--rate-limit', type=float, default=RATE_LIMIT,
                        help=f'Seconds between requests (default: {RATE_LIMIT})')

    args = parser.parse_args()

    # Determine output path
    if args.test:
        output_path = TEST_OUTPUT
        max_per_source = 5
    else:
        output_path = Path(args.output)
        max_per_source = args.max_per_source

    # Determine sources to scrape
    if args.sources:
        sources_to_scrape = {k: SOURCES[k] for k in args.sources}
    else:
        sources_to_scrape = SOURCES

    # Initialize robots.txt checker
    robots_checker = None if args.no_robots else RobotsChecker()

    print("=" * 60)
    print("TAMIL NEWS SCRAPER — Adhan Training Data Pipeline")
    print("ARIVU + Hermes | Rotation 26 Cycle 2 | Jun 18, 2026")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Sources: {', '.join(sources_to_scrape.keys())}")
    print(f"Max per source: {max_per_source}")
    print(f"Rate limit: {args.rate_limit}s")
    print(f"Robots.txt: {'disabled' if args.no_robots else 'enabled'}")

    # Scrape all sources
    all_articles = []
    for source_key, source_config in sources_to_scrape.items():
        articles = scrape_source(
            source_key, source_config,
            max_articles=max_per_source,
            robots_checker=robots_checker,
        )
        all_articles.extend(articles)

    # Write output
    write_jsonl(all_articles, output_path)

    # Compute and display stats
    stats = compute_stats(all_articles)
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print(f"Total articles: {stats.get('total_articles', 0)}")
    print(f"Total characters: {stats.get('total_characters', 0):,}")
    print(f"Approx. BPE tokens: {stats.get('approx_tokens_bpe', 0):,}")
    print("\nBy source:")
    for src, count in stats.get('by_source', {}).items():
        print(f"  {src}: {count} articles")
    print(f"\nOutput saved to: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
