#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Tamil Corpus Scraper - Simple, Secure, Focused

Simplified architecture with focus on:
- Code minimalism (< 200 lines)
- Security best practices
- Core functionality only
"""
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TamilScraper:
    """Minimal, secure Tamil corpus scraper"""
    
    # Security: Whitelist of allowed domains
    ALLOWED_DOMAINS = {'ta.wikipedia.org', 'api.wikimedia.org'}
    
    def __init__(self, output_dir: str = "data/raw", timeout: int = 10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create secure session with retry logic"""
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retry))
        session.headers.update({'User-Agent': 'TamilScraper/1.0'})
        return session
    
    def _validate_url(self, url: str) -> bool:
        """Security: Validate URL is from allowed domain"""
        try:
            domain = urlparse(url).netloc
            return domain in self.ALLOWED_DOMAINS
        except Exception:
            return False
    
    def _is_tamil(self, text: str) -> bool:
        """Check if text contains Tamil characters"""
        tamil_chars = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
        return tamil_chars > len(text) * 0.3  # 30% threshold
    
    def fetch_wikipedia_articles(self, category: str = "Tamil_language", limit: int = 50) -> List[Dict]:
        """Fetch Tamil Wikipedia articles from category"""
        url = "https://ta.wikipedia.org/w/api.php"
        
        # Security: Validate URL
        if not self._validate_url(url):
            logger.error(f"URL not in whitelist: {url}")
            return []
        
        records = []
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'categorymembers',
            'cmtitle': f'Category:{category}',
            'cmlimit': min(limit, 50),  # Security: Cap limit
            'cmtype': 'page'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            for page in data.get('query', {}).get('categorymembers', []):
                text = self._fetch_page_content(page['pageid'])
                if text and self._is_tamil(text):
                    records.append({
                        'text': text[:1000],  # Security: Limit text size
                        'source': 'wikipedia',
                        'title': page['title']
                    })
                    
        except Exception as e:
            logger.error(f"Error fetching articles: {e}")
        
        return records
    
    def _fetch_page_content(self, page_id: int) -> Optional[str]:
        """Fetch content of a single Wikipedia page"""
        url = "https://ta.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'format': 'json',
            'pageids': page_id,
            'prop': 'extracts',
            'explaintext': True
        }
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            return pages.get(str(page_id), {}).get('extract', '')
        except Exception as e:
            logger.error(f"Error fetching page {page_id}: {e}")
            return None
    
    def save(self, records: List[Dict], filename: str = "tamil_corpus.jsonl"):
        """Save records securely to JSONL"""
        output_path = self.output_dir / filename
        
        # Security: Validate filename
        if not re.match(r'^[\w\-. ]+$', filename):
            raise ValueError("Invalid filename")
        
        try:
            with output_path.open('w', encoding='utf-8') as f:
                for record in records:
                    # Security: Validate record structure
                    if isinstance(record, dict) and 'text' in record:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
            logger.info(f"Saved {len(records)} records to {output_path}")
        except Exception as e:
            logger.error(f"Error saving records: {e}")
            raise


def main():
    """Simple CLI"""
    import argparse
    parser = argparse.ArgumentParser(description='Minimal Tamil Scraper')
    parser.add_argument('--category', default='Tamil_language', help='Wikipedia category')
    parser.add_argument('--limit', type=int, default=50, help='Max articles (max 50)')
    parser.add_argument('--output', default='tamil_corpus.jsonl', help='Output file')
    args = parser.parse_args()
    
    # Security: Cap limit
    limit = min(args.limit, 50)
    
    scraper = TamilScraper()
    records = scraper.fetch_wikipedia_articles(args.category, limit)
    scraper.save(records, args.output)
    logger.info(f"Complete. Scraped {len(records)} articles.")


if __name__ == '__main__':
    main()
