#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Tamil Corpus Scraper with modular design and better error handling.

Key improvements:
- Modular base scraper class
- Configurable retry logic
- Better error handling and logging
- Caching mechanism
- Rate limiting configuration
- Progress tracking
"""
import argparse
import json
import hashlib
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScraperConfig:
    """Configuration for the scraper"""
    
    def __init__(
        self,
        base_dir: str = "data/raw",
        cache_dir: str = "data/raw/.cache",
        rate_limit: float = 1.0,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        timeout: int = 30,
        user_agent: str = None,
        enable_cache: bool = True
    ):
        self.base_dir = Path(base_dir)
        self.cache_dir = Path(cache_dir)
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.timeout = timeout
        self.user_agent = user_agent or (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/91.0.4472.124 Safari/537.36'
        )
        self.enable_cache = enable_cache
        
        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)


def rate_limit(func: Callable) -> Callable:
    """Decorator to enforce rate limiting"""
    last_call = {'time': 0}
    
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Get rate limit from config
        rate_limit_seconds = getattr(self.config, 'rate_limit', 1.0)
        
        # Calculate time since last call
        elapsed = time.time() - last_call['time']
        if elapsed < rate_limit_seconds:
            time.sleep(rate_limit_seconds - elapsed)
        
        result = func(self, *args, **kwargs)
        last_call['time'] = time.time()
        return result
    
    return wrapper


def retry_on_error(max_retries: int = 3, backoff: float = 2.0) -> Callable:
    """Decorator to retry on errors with exponential backoff"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                    wait_time = backoff ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        return wrapper
    
    return decorator


class BaseScraper:
    """Base scraper with common functionality"""
    
    def __init__(self, config: ScraperConfig = None):
        self.config = config or ScraperConfig()
        self.session = self._create_session()
        self.progress = {'total': 0, 'success': 0, 'failed': 0}
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic"""
        session = requests.Session()
        
        # Configure retries
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers
        session.headers.update({
            'User-Agent': self.config.user_agent
        })
        
        return session
    
    def _get_cache_path(self, url: str) -> Path:
        """Get cache file path for a URL"""
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        return self.config.cache_dir / f"{url_hash}.json"
    
    def _load_from_cache(self, url: str) -> Optional[str]:
        """Load content from cache if available"""
        if not self.config.enable_cache:
            return None
        
        cache_path = self._get_cache_path(url)
        if not cache_path.exists():
            return None
        
        try:
            with cache_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
                # Check if cache is expired (7 days)
                cached_time = datetime.fromisoformat(data['timestamp'])
                if datetime.now() - cached_time > timedelta(days=7):
                    logger.debug(f"Cache expired for {url}")
                    return None
                return data['content']
        except Exception as e:
            logger.warning(f"Error loading cache for {url}: {e}")
            return None
    
    def _save_to_cache(self, url: str, content: str):
        """Save content to cache"""
        if not self.config.enable_cache:
            return
        
        cache_path = self._get_cache_path(url)
        try:
            with cache_path.open('w', encoding='utf-8') as f:
                json.dump({
                    'url': url,
                    'content': content,
                    'timestamp': datetime.now().isoformat()
                }, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Error saving cache for {url}: {e}")
    
    @rate_limit
    @retry_on_error()
    def fetch_url(self, url: str, use_cache: bool = True) -> Optional[str]:
        """Fetch content from URL with caching and error handling"""
        # Try cache first
        if use_cache:
            cached_content = self._load_from_cache(url)
            if cached_content:
                logger.debug(f"Using cached content for {url}")
                return cached_content
        
        # Fetch from URL
        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            content = response.text
            
            # Save to cache
            if use_cache:
                self._save_to_cache(url, content)
            
            self.progress['success'] += 1
            return content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            self.progress['failed'] += 1
            raise
    
    def is_tamil_text(self, text: str, min_fraction: float = 0.5) -> bool:
        """Check if text contains sufficient Tamil characters"""
        if not text:
            return False
        
        # Tamil Unicode range: 0x0B80-0x0BFF
        tamil_chars = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
        total_chars = len([c for c in text if c.strip()])
        
        if total_chars == 0:
            return False
        
        tamil_fraction = tamil_chars / total_chars
        return tamil_fraction >= min_fraction
    
    def clean_tamil_text(self, text: str) -> str:
        """Clean and normalize Tamil text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # Normalize Unicode (NFC normalization)
        import unicodedata
        text = unicodedata.normalize('NFC', text)
        
        return text.strip()
    
    def extract_text_from_html(self, html: str, selector: str = None) -> str:
        """Extract text from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract text
            if selector:
                elements = soup.select(selector)
                text = '\n'.join([el.get_text().strip() for el in elements])
            else:
                text = soup.get_text()
            
            return self.clean_tamil_text(text)
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return ""
    
    def make_record(
        self,
        text: str,
        source: str,
        url: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Create a standardized record"""
        record = {
            'id': hashlib.sha256(text.encode()).hexdigest()[:16],
            'text': text,
            'source': source,
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'char_count': len(text),
            'tamil_fraction': self._calculate_tamil_fraction(text),
        }
        
        if metadata:
            record.update(metadata)
        
        return record
    
    def _calculate_tamil_fraction(self, text: str) -> float:
        """Calculate fraction of Tamil characters in text"""
        if not text:
            return 0.0
        
        tamil_chars = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
        total_chars = len([c for c in text if c.strip()])
        
        return tamil_chars / total_chars if total_chars > 0 else 0.0
    
    def save_records(self, records: List[Dict], filename: str):
        """Save records to JSONL file"""
        output_path = self.config.base_dir / filename
        
        try:
            with output_path.open('w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved {len(records)} records to {output_path}")
        except Exception as e:
            logger.error(f"Error saving records to {output_path}: {e}")
            raise
    
    def print_progress(self):
        """Print scraping progress"""
        total = self.progress['total']
        success = self.progress['success']
        failed = self.progress['failed']
        
        logger.info(
            f"Progress: {success + failed}/{total} "
            f"(Success: {success}, Failed: {failed})"
        )


class TamilCorpusScraper(BaseScraper):
    """Enhanced Tamil corpus scraper"""
    
    def __init__(self, config: ScraperConfig = None):
        super().__init__(config)
    
    def scrape_wikipedia_category(
        self,
        category: str = "Tamil_language",
        max_articles: int = 50
    ) -> List[Dict]:
        """Scrape articles from Tamil Wikipedia category"""
        logger.info(f"Scraping Tamil Wikipedia category: {category}")
        records = []
        
        try:
            # Get category members using API
            url = "https://ta.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'categorymembers',
                'cmtitle': f"Category:{category}",
                'cmlimit': max_articles
            }
            
            content = self.fetch_url(
                f"{url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
            )
            
            if not content:
                return records
            
            data = json.loads(content)
            members = data.get('query', {}).get('categorymembers', [])
            
            self.progress['total'] = len(members)
            
            for member in members:
                if member.get('ns') == 0:  # Only articles
                    article_url = f"https://ta.wikipedia.org/wiki/{member['title'].replace(' ', '_')}"
                    article_text = self._scrape_wikipedia_article(article_url)
                    
                    if article_text:
                        record = self.make_record(
                            text=article_text,
                            source='wikipedia',
                            url=article_url,
                            metadata={'category': category, 'title': member['title']}
                        )
                        records.append(record)
            
            self.print_progress()
            
        except Exception as e:
            logger.error(f"Error scraping Wikipedia category {category}: {e}")
        
        return records
    
    def _scrape_wikipedia_article(self, url: str) -> Optional[str]:
        """Scrape content from a single Wikipedia article"""
        try:
            html = self.fetch_url(url)
            if not html:
                return None
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract main content
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                return None
            
            # Remove unwanted elements
            for element in content_div.find_all(['table', 'img', 'script', 'style', 'sup']):
                element.decompose()
            
            # Extract paragraphs
            paragraphs = content_div.find_all('p')
            text = '\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            # Clean and validate
            text = self.clean_tamil_text(text)
            
            if text and self.is_tamil_text(text):
                return text
            
            return None
            
        except Exception as e:
            logger.error(f"Error scraping Wikipedia article {url}: {e}")
            return None
    
    def scrape_literature_site(
        self,
        url: str,
        selector: str = None
    ) -> List[Dict]:
        """Scrape Tamil literature from a website"""
        logger.info(f"Scraping literature site: {url}")
        records = []
        
        try:
            html = self.fetch_url(url)
            if not html:
                return records
            
            text = self.extract_text_from_html(html, selector)
            
            if text and self.is_tamil_text(text):
                record = self.make_record(
                    text=text,
                    source='literature',
                    url=url
                )
                records.append(record)
            
        except Exception as e:
            logger.error(f"Error scraping literature site {url}: {e}")
        
        return records


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Enhanced Tamil Corpus Scraper')
    parser.add_argument('--base-dir', default='data/raw', help='Base directory for output')
    parser.add_argument('--cache-dir', default='data/raw/.cache', help='Cache directory')
    parser.add_argument('--rate-limit', type=float, default=1.0, help='Rate limit in seconds')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum retries')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--source', choices=['wikipedia', 'literature'], 
                       default='wikipedia', help='Source to scrape')
    parser.add_argument('--category', default='Tamil_language', 
                       help='Wikipedia category to scrape')
    parser.add_argument('--max-articles', type=int, default=50, 
                       help='Maximum articles to scrape')
    parser.add_argument('--output', default='tamil_corpus_enhanced.jsonl', 
                       help='Output filename')
    
    args = parser.parse_args()
    
    # Create config
    config = ScraperConfig(
        base_dir=args.base_dir,
        cache_dir=args.cache_dir,
        rate_limit=args.rate_limit,
        max_retries=args.max_retries,
        timeout=args.timeout,
        enable_cache=not args.no_cache
    )
    
    # Create scraper
    scraper = TamilCorpusScraper(config)
    
    # Scrape based on source
    records = []
    if args.source == 'wikipedia':
        records = scraper.scrape_wikipedia_category(
            category=args.category,
            max_articles=args.max_articles
        )
    
    # Save results
    if records:
        scraper.save_records(records, args.output)
        logger.info(f"Successfully scraped {len(records)} records")
    else:
        logger.warning("No records scraped")


if __name__ == '__main__':
    main()
