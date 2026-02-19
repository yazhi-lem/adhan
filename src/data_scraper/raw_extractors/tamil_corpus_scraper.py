#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tamil Corpus Scraper for AADHAN Model Training
"""
import argparse
import os
import re
import json
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urlparse, urljoin
from urllib import robotparser
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
import logging
from typing import List, Dict, Optional

# Import core constants
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core import TAMIL_HASHTAGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TamilCorpusScraper:
    """Scraper for collecting Tamil text data from various sources"""
    
    def __init__(self, base_dir="data/raw"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_wikipedia(self, category: str = "Tamil_language") -> List[str]:
        """Scrape Tamil Wikipedia articles"""
        logger.info(f"Scraping Tamil Wikipedia category: {category}")
        articles = []
        
        try:
            # Get category members
            url = f"https://ta.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'categorymembers',
                'cmtitle': f"Category:{category}",
                'cmlimit': 50
            }
            
            response = self.session.get(url, params=params)
            data = response.json()
            
            for member in data.get('query', {}).get('categorymembers', []):
                if member['ns'] == 0:  # Only articles
                    article_url = f"https://ta.wikipedia.org/wiki/{member['title'].replace(' ', '_')}"
                    article_text = self._scrape_wikipedia_article(article_url)
                    if article_text:
                        articles.append(article_text)
                        time.sleep(1)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Error scraping Wikipedia: {e}")
        
        return articles
    
    def _scrape_wikipedia_article(self, url: str) -> Optional[str]:
        """Scrape content from a single Wikipedia article"""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract main content
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if not content_div:
                return None
            
            # Remove unwanted elements
            for element in content_div.find_all(['table', 'img', 'script', 'style']):
                element.decompose()
            
            # Extract text
            paragraphs = content_div.find_all('p')
            text = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            # Clean text
            text = self._clean_tamil_text(text)
            
            return text if text else None
            
        except Exception as e:
            logger.error(f"Error scraping article {url}: {e}")
            return None
    
    def scrape_tamil_literature_sites(self) -> List[str]:
        """Scrape Tamil literature websites"""
        logger.info("Scraping Tamil literature sites...")
        texts = []
        
        # Sangam literature sites
        sites = [
            "https://sangam.org",
            "https://tamilvu.org",
            "https://projectmadurai.org"
        ]
        
        for site in sites:
            texts.extend(self._scrape_literature_site(site))
            time.sleep(2)  # Rate limiting
        
        return texts
    
    def _scrape_literature_site(self, url: str) -> List[str]:
        """Scrape content from a literature website"""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find text content
            texts = []
            for element in soup.find_all(['p', 'div', 'span']):
                text = element.get_text().strip()
                if self._is_tamil_text(text):
                    texts.append(self._clean_tamil_text(text))
            
            return texts
            
        except Exception as e:
            logger.error(f"Error scraping literature site {url}: {e}")
            return []
    
    def scrape_tamil_news_sites(self) -> List[str]:
        """Scrape Tamil news websites"""
        logger.info("Scraping Tamil news sites...")
        texts = []
        
        news_sites = [
            "https://tamil.oneindia.com",
            "https://tamil.samayam.com",
            "https://tamil.indianexpress.com"
        ]
        
        for site in news_sites:
            texts.extend(self._scrape_news_site(site))
            time.sleep(2)  # Rate limiting
        
        return texts
    
    def _scrape_news_site(self, url: str) -> List[str]:
        """Scrape news articles from a site"""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news articles
            articles = []
            for article in soup.find_all('article'):
                text = article.get_text().strip()
                if self._is_tamil_text(text):
                    articles.append(self._clean_tamil_text(text))
            
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping news site {url}: {e}")
            return []
    
    def scrape_tamil_books(self, book_paths: List[str]) -> List[str]:
        """Scrape Tamil books from local files or URLs"""
        logger.info("Scraping Tamil books...")
        texts = []
        
        for path in book_paths:
            if path.startswith("http"):
                # Download and scrape
                response = self.session.get(path)
                response.raise_for_status()
                text = response.text
            else:
                # Read local file
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            if text:
                cleaned_text = self._clean_tamil_text(text)
                texts.append(cleaned_text)
                time.sleep(1)  # Rate limiting
        
        return texts
    
    def scrape_social_media(self, hashtags: List[str]) -> List[str]:
        """Scrape Tamil content from social media"""
        logger.info("Scraping social media for Tamil content...")
        texts = []
        
        # Twitter/Telegram scraping (simplified)
        for hashtag in hashtags:
            # Simulate social media scraping
            sample_texts = [
                f"#{hashtag} தமிழ் வெற்றி பெற்றது",
                f"#{hashtag} தமிழ் ",
                f"#{hashtag} தமிழ் ",
                f"#{hashtag} நமது தமிழ் மொழி மிகவும் சிறப்பு",
                f"#{hashtag} தமிழ் பண்பாடு பேணுங்கள்"
            ]
            texts.extend(sample_texts)
            time.sleep(1)  # Rate limiting
        
        return texts
    
    def _is_tamil_text(self, text: str) -> bool:
        """Check if text contains Tamil characters"""
        tamil_pattern = re.compile(r'[\u0B80-\u0BFF]+')
        return bool(tamil_pattern.search(text))
    
    def _clean_tamil_text(self, text: str) -> str:
        """Clean and preprocess Tamil text"""
        # Remove unwanted characters
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s\u0B80-\u0BFF.,!?]', '', text)  # Remove special chars
        text = text.strip()
        
        return text

    def make_record(self, text: str, source: str = "unknown", url: Optional[str] = None) -> Dict:
        """Create a standardized record with metadata for each scraped text."""
        cleaned = self._clean_tamil_text(text)
        rec_id = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        record = {
            "id": rec_id,
            "text": cleaned,
            "source": source,
            "url": url,
            "crawl_date": datetime.utcnow().isoformat() + "Z",
            "lang": "ta" if self._is_tamil_text(cleaned) else "unknown",
            "char_count": len(cleaned),
            "word_count": len(cleaned.split()),
        }
        return record

    def dedupe_records(self, records: List[Dict]) -> List[Dict]:
        """Remove exact and near-duplicate records (simple normalized-string heuristic + hash)."""
        seen_hashes = set()
        seen_norm = set()
        out = []
        for r in records:
            txt = r.get("text", "")
            if not txt:
                continue
            h = hashlib.sha256(txt.encode("utf-8")).hexdigest()
            norm = re.sub(r'\s+', ' ', re.sub(r'[^\w\u0B80-\u0BFF]', '', txt)).strip()[:512]
            if h in seen_hashes or norm in seen_norm:
                continue
            seen_hashes.add(h)
            seen_norm.add(norm)
            out.append(r)
        return out

    def classify_topic(self, text: str) -> List[str]:
        """Keyword-based topic classification (returns Tamil labels)."""
        KEYWORDS = {
            "அரசியல்": ["அரசு", "மாநிலம்", "மந்திரி", "நேர்காணல்", "வாக்கு"],
            "செய்திகள்": ["செய்தி", "பதிவு", "நத்து", "தற்போது"],
            "விளையாட்டு": ["பந்து", "வெற்றி", "தோல்வி", "லீக்"],
            "இலக்கியம்": ["கவிதை", "கதை", "செந்தமிழ்", "இலக்கியம்", "பாடல்"],
            "தொழில்நுட்பம்": ["தொழில்நுட்ப", "யாண்டு", "எய்ஐ", "ஐடி", "மென்பொருள்"],
            "மதம்": ["திருச்சபை", "ஆன்மிக", "பழமொழி", "பரம்பரை"],
            "பண்பாடு": ["பண்பாடு", "கலை", "பண்டிகை", "பாடல்"]
        }
        labels = []
        lower = text.lower()
        for label_tamil, kws in KEYWORDS.items():
            for kw in kws:
                if kw in lower:
                    labels.append(label_tamil)
                    break
        return labels or ["புரியவில்லை"]

    def score_quality(self, text: str) -> Dict:
        """Heuristic quality scoring and simple flags."""
        char_count = len(text)
        word_count = len(text.split())
        tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', text))
        tamil_fraction = tamil_chars / max(1, char_count)
        length_score = min(1.0, word_count / 50)
        tamil_score = 1.0 if tamil_fraction > 0.6 else tamil_fraction
        quality_score = 0.6 * length_score + 0.4 * tamil_score
        flags = {
            "too_short": word_count < 5,
            "is_tamil": tamil_fraction > 0.5
        }
        return {"quality_score": round(quality_score, 3), "flags": flags, "char_count": char_count, "word_count": word_count}

    def filter_records(self, records: List[Dict], min_quality: float = 0.0, min_chars: int = 0, min_tamil_fraction: float = 0.0, require_tamil: bool = False, apply_fineweb: bool = False) -> List[Dict]:
        """Filter records using simple heuristics inspired by FineWeb.

        Heuristics implemented:
        - minimum `quality_score` (uses existing score_quality when missing)
        - minimum character length
        - minimum fraction of Tamil characters
        - boilerplate / navigation removal (copyright, contact, privacy, login)
        - optional `apply_fineweb` stronger checks (sentence count, repetition, punctuation ratios)
        """
        out = []
        removed = 0
        for r in records:
            txt = (r.get('text') or '').strip()
            if not txt:
                removed += 1
                continue

            # ensure meta fields available
            if 'quality_score' not in r:
                r.update(self.score_quality(txt))

            q = float(r.get('quality_score', 0.0))
            cc = int(r.get('char_count', len(txt)))

            # thresholds
            if min_quality and q < float(min_quality):
                removed += 1
                continue
            if min_chars and cc < int(min_chars):
                removed += 1
                continue

            # Tamil fraction check
            tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', txt))
            tamil_frac = tamil_chars / max(1, len(txt))
            if require_tamil and tamil_frac < max(0.5, float(min_tamil_fraction)):
                removed += 1
                continue
            if min_tamil_fraction and tamil_frac < float(min_tamil_fraction):
                removed += 1
                continue

            lowtxt = txt.lower()
            # remove obvious boilerplate / nav / policy pages
            boilerplate_tokens = ('copyright', 'all rights reserved', 'terms of use', 'privacy policy', 'contact us', 'login', 'sign in', 'cookie', 'sitemap')
            if any(tok in lowtxt for tok in boilerplate_tokens):
                removed += 1
                continue

            # remove pages with too high latin fraction when Tamil required
            latin_chars = len(re.findall(r'[A-Za-z]', txt))
            latin_frac = latin_chars / max(1, len(txt))
            if require_tamil and latin_frac > 0.25:
                removed += 1
                continue

            # optional FineWeb-like extra checks
            if apply_fineweb:
                # require at least 3 sentences
                sentences = re.split(r'[\.|\?|\!|\n]+', txt)
                sentences = [s for s in sentences if s.strip()]
                if len(sentences) < 3:
                    removed += 1
                    continue

                # reject extremely repetitive pages
                lines = [l.strip() for l in txt.splitlines() if l.strip()]
                if lines and (len(set(lines)) / len(lines) < 0.5):
                    removed += 1
                    continue

                # punctuation / letter ratio (noisy pages often have odd ratios)
                letters = len(re.findall(r'[\w\u0B80-\u0BFF]', txt))
                # Python `re` does not support \p{P}/\p{S}; use [^\w\s] as a practical fallback
                punctuation = len(re.findall(r'[^\w\s]', txt, flags=re.UNICODE))
                if letters > 0 and (punctuation / letters) > 0.5:
                    removed += 1
                    continue

            out.append(r)

        logger.info(f"Filtered records: kept={len(out)}, removed={removed}")
        return out

    def save_jsonl(self, records: List[Dict], filename: str = "tamil_corpus.jsonl"):
        """Save structured records as JSONL (UTF-8)."""
        filepath = Path(filename)
        if not filepath.is_absolute() and not str(filepath).startswith(str(self.base_dir)):
            filepath = self.base_dir / filepath
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(records)} records to {filepath}")

    def _extract_text_from_html(self, html: str) -> str:
        """Generic HTML -> cleaned Tamil text extractor."""
        soup = BeautifulSoup(html, 'html.parser')
        for element in soup.find_all(['script', 'style', 'img', 'noscript', 'svg', 'header', 'footer', 'nav']):
            element.decompose()
        paragraphs = [p.get_text(separator=' ', strip=True) for p in soup.find_all(['p', 'div', 'article', 'section'])]
        text = '\n'.join([t for t in paragraphs if t])
        return self._clean_tamil_text(text)

    def fetch_sitemap_urls(self, domain: str) -> List[str]:
        """Find and parse sitemap(s) for a domain and return all URLs found.

        Tries common sitemap locations and follows sitemap index files.
        """
        domain = domain.rstrip('/')
        candidates = [f"{domain}/sitemap.xml", f"{domain}/sitemap_index.xml", f"{domain}/sitemap/sitemap-index.xml"]
        found_urls = []
        visited_sitemaps = set()

        def parse_sitemap(content: bytes):
            try:
                root = ET.fromstring(content)
                for loc in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                    found_urls.append(loc.text.strip())
                # fallback: find any <loc> without namespace
                for loc in root.findall('.//loc'):
                    if loc.text and loc.text.strip() not in found_urls:
                        found_urls.append(loc.text.strip())
            except ET.ParseError:
                # last-resort: regex
                txt = content.decode('utf-8', errors='ignore')
                for m in re.findall(r'<loc>(.*?)</loc>', txt, flags=re.I):
                    if m.strip() not in found_urls:
                        found_urls.append(m.strip())

        for cand in candidates:
            try:
                resp = self.session.get(cand, timeout=15)
                if resp.status_code == 200 and len(resp.content) > 0:
                    visited_sitemaps.add(cand)
                    parse_sitemap(resp.content)
                    # If sitemap index, try to fetch nested sitemaps
                    for s in list(found_urls):
                        if s.endswith('.xml') and s not in visited_sitemaps:
                            try:
                                r2 = self.session.get(s, timeout=15)
                                if r2.status_code == 200:
                                    visited_sitemaps.add(s)
                                    parse_sitemap(r2.content)
                            except Exception:
                                continue
                    break
            except Exception:
                continue

        return list(dict.fromkeys(found_urls))

    def crawl_sitemap(self, domain: str, limit: int = 0, save_raw: bool = True, extract_text: bool = True, rate_limit_seconds: float = 1.0) -> List[Dict]:
        """Crawl all URLs discovered in a site's sitemap and return extracted records.

        - Respects `robots.txt` via urllib.robotparser.
        - Saves raw HTML under `data/raw_html/<domain>/...` when `save_raw` is True.
        - Extracts text and returns `make_record(...)` entries when `extract_text` is True.
        """
        if not domain.startswith('http'):
            domain = 'https://' + domain
        parsed = urlparse(domain)
        base = f"{parsed.scheme}://{parsed.netloc}"

        # robots.txt check
        rp = robotparser.RobotFileParser()
        rp.set_url(urljoin(base, '/robots.txt'))
        try:
            rp.read()
        except Exception:
            logger.warning(f"Could not read robots.txt for {base}; proceeding carefully")

        urls = self.fetch_sitemap_urls(base)
        if not urls:
            logger.info(f"No sitemap found for {base}; attempting to crawl root page only")
            urls = [base]

        logger.info(f"Found {len(urls)} URLs in sitemap for {base}")
        records = []
        count = 0

        for u in urls:
            if limit and count >= limit:
                break
            try:
                # robots check
                allowed = True
                try:
                    allowed = rp.can_fetch('*', u)
                except Exception:
                    allowed = True
                if not allowed:
                    logger.info(f"Skipping disallowed by robots.txt: {u}")
                    continue

                resp = self.session.get(u, timeout=20)
                if resp.status_code != 200:
                    logger.info(f"Skipping non-200 ({resp.status_code}): {u}")
                    continue

                # save raw HTML
                local_rel = urlparse(u).path.lstrip('/') or 'index.html'
                if local_rel.endswith('/'):
                    local_rel = local_rel + 'index.html'
                save_dir = self.base_dir / 'raw_html' / parsed.netloc / os.path.dirname(local_rel)
                save_dir.mkdir(parents=True, exist_ok=True)
                local_path = self.base_dir / 'raw_html' / parsed.netloc / local_rel
                with open(local_path, 'wb') as fh:
                    fh.write(resp.content)

                # extract text
                text = ''
                if extract_text:
                    text = self._extract_text_from_html(resp.text)
                    if not text:
                        # fallback: use visible text
                        text = BeautifulSoup(resp.content, 'html.parser').get_text(separator=' ', strip=True)

                rec = self.make_record(text, source=parsed.netloc, url=u)
                rec['raw_html_path'] = str(local_path)
                rec['status_code'] = resp.status_code
                records.append(rec)
                count += 1

                time.sleep(rate_limit_seconds)
            except Exception as e:
                logger.error(f"Error crawling {u}: {e}")
                continue

        logger.info(f"Crawled {len(records)} pages from {base}")
        return records

    def extract_pdfs_from_sitemap(self, domain: str, limit: int = 0, rate_limit_seconds: float = 1.0) -> List[Dict]:
        """Find PDF URLs from sitemap, download them, extract text and return records.

        This now logs progress for each PDF and attempts best-effort conversion from
        legacy encodings (TSCII, Mylai) when the extracted text looks like legacy.
        """
        try:
            from pdfminer.high_level import extract_text
        except Exception as e:
            raise RuntimeError("pdfminer.six is required for PDF extraction. Install with `pip install pdfminer.six`") from e

        if not domain.startswith('http'):
            domain = 'https://' + domain
        parsed = urlparse(domain)
        base = f"{parsed.scheme}://{parsed.netloc}"

        # robots.txt parser
        rp = robotparser.RobotFileParser()
        rp.set_url(urljoin(base, '/robots.txt'))
        try:
            rp.read()
        except Exception:
            logger.warning(f"Could not read robots.txt for {base}; proceeding carefully")

        urls = self.fetch_sitemap_urls(base)
        pdf_urls = [u for u in urls if u.lower().endswith('.pdf') or '/pdf/' in u.lower()]
        logger.info(f"Found {len(pdf_urls)} PDF URLs in sitemap for {base}")

        def _looks_like_legacy(txt: str) -> bool:
            if not txt:
                return False
            low = txt[:4096].lower()
            if 'tscii' in low or 'mylai' in low or 'mylai' in low or 'mylai font' in low:
                return True
            # heuristic: presence of high-latin glyphs commonly seen in TSCII/Mylai dumps
            if any(ch in txt for ch in ('¾', '¢', 'Õ', 'Ã', 'õ')):
                return True
            return False

        records = []
        count = 0
        for idx, u in enumerate(pdf_urls, start=1):
            if limit and count >= limit:
                break
            logger.info(f"[PDF {idx}/{len(pdf_urls)}] fetching: {u}")
            try:
                allowed = True
                try:
                    allowed = rp.can_fetch('*', u)
                except Exception:
                    allowed = True
                if not allowed:
                    logger.info(f"Skipping PDF disallowed by robots.txt: {u}")
                    continue

                resp = self.session.get(u, timeout=30, stream=True)
                if resp.status_code != 200:
                    logger.info(f"Skipping non-200 PDF ({resp.status_code}): {u}")
                    continue

                # save PDF
                local_rel = urlparse(u).path.lstrip('/') or 'file.pdf'
                save_dir = self.base_dir / 'raw_pdf' / parsed.netloc / os.path.dirname(local_rel)
                save_dir.mkdir(parents=True, exist_ok=True)
                local_pdf_path = self.base_dir / 'raw_pdf' / parsed.netloc / local_rel
                with open(local_pdf_path, 'wb') as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)

                # extract text
                logger.info(f"[PDF {idx}] extracting text from {local_pdf_path}")
                try:
                    extracted = extract_text(str(local_pdf_path))
                except Exception as e:
                    logger.error(f"Failed to extract PDF text for {u}: {e}")
                    extracted = ''

                # detect legacy encoding and convert (TSCII / Mylai)
                if _looks_like_legacy(extracted):
                    logger.info(f"[PDF {idx}] detected legacy encoding in extracted text; attempting conversion")
                    try:
                        import tamil
                        if 'mylai' in extracted.lower() and hasattr(tamil.txt2unicode, 'mylai2unicode'):
                            extracted = tamil.txt2unicode.mylai2unicode(extracted)
                            logger.info(f"[PDF {idx}] Mylai -> Unicode conversion applied")
                        elif hasattr(tamil, 'tscii') and hasattr(tamil.tscii, 'convert_to_unicode'):
                            extracted = tamil.tscii.convert_to_unicode(extracted)
                            logger.info(f"[PDF {idx}] TSCII -> Unicode conversion applied")
                        elif hasattr(tamil, 'txt2unicode') and hasattr(tamil.txt2unicode, 'tscii2unicode'):
                            extracted = tamil.txt2unicode.tscii2unicode(extracted)
                            logger.info(f"[PDF {idx}] TSCII (fallback) -> Unicode conversion applied")
                    except Exception as e:
                        logger.warning(f"[PDF {idx}] legacy conversion failed: {e}")

                text_path = self.base_dir / 'pdf_texts' / parsed.netloc / (Path(local_rel).stem + '.txt')
                text_path.parent.mkdir(parents=True, exist_ok=True)
                with open(text_path, 'w', encoding='utf-8') as tf:
                    tf.write(extracted)

                rec = self.make_record(extracted, source=parsed.netloc, url=u)
                rec['raw_pdf_path'] = str(local_pdf_path)
                rec['text_path'] = str(text_path)
                rec['status_code'] = resp.status_code
                records.append(rec)
                count += 1

                time.sleep(rate_limit_seconds)
            except Exception as e:
                logger.error(f"Error processing PDF {u}: {e}")
                continue

        logger.info(f"Processed {len(records)} PDF files from {base}")
        return records

    def build_manifests(self, records: List[Dict], out_dir: str = None) -> Dict[str, Dict]:
        """Create per-work manifests grouped by inferred work id.

        - Groups records by a work identifier derived from the URL/file stem.
        - Writes a manifest JSON and per-work JSONL under `out_dir` (defaults to `data/raw/projectmadurai_manifests`).
        - Returns the manifest mapping.
        """
        out_dir = Path(out_dir or (self.base_dir / 'projectmadurai_manifests'))
        out_dir.mkdir(parents=True, exist_ok=True)

        groups: Dict[str, List[Dict]] = {}
        for r in records:
            # coerce `url` into a safe string (handle None/bytes/Path)
            url = r.get('url', '')
            if isinstance(url, bytes):
                try:
                    url = url.decode('utf-8', errors='ignore')
                except Exception:
                    url = str(url)
            if url is None:
                url = ''
            url = str(url)

            stem = Path(urlparse(url).path).stem
            m = re.match(r'^([a-zA-Z]+\d+)', stem)
            work_id = m.group(1) if m else (stem or r.get('source', 'unknown'))
            groups.setdefault(work_id, []).append(r)

        manifest = {}
        for work_id, items in groups.items():
            combined_text = '\n\n'.join([it.get('text','') for it in items if it.get('text')])
            entry = {
                'work_id': work_id,
                'n_pages': len(items),
                'char_count': sum(it.get('char_count',0) for it in items),
                'word_count': sum(it.get('word_count',0) for it in items),
                'topics': list({t for it in items for t in it.get('topics',[]) if t}),
                'records': [{k: it.get(k) for k in ('id','url','raw_html_path','raw_pdf_path','text_path','char_count','word_count')} for it in items]
            }
            manifest[work_id] = entry

            # write per-work JSONL + combined text file
            work_jsonl = out_dir / f"{work_id}.jsonl"
            with open(work_jsonl, 'w', encoding='utf-8') as wf:
                for it in items:
                    wf.write(json.dumps(it, ensure_ascii=False) + '\n')
            with open(out_dir / f"{work_id}.txt", 'w', encoding='utf-8') as tf:
                tf.write(combined_text)

        # write master manifest
        with open(out_dir / 'manifest_index.json', 'w', encoding='utf-8') as mf:
            json.dump(manifest, mf, ensure_ascii=False, indent=2)

        logger.info(f"Wrote {len(manifest)} work manifests to {out_dir}")
        return manifest

    def convert_to_hf_dataset(self, records: List[Dict], out_dir: str = None, split=(0.8,0.1,0.1), min_quality: float = 0.0):
        """Export records into a simple HuggingFace-style JSONL dataset (train/val/test).

        - Filters by `min_quality` (quality_score), shuffles deterministically, and writes `train.jsonl`, `validation.jsonl`, `test.jsonl`.
        - Saves to `out_dir` (defaults to `data/raw/projectmadurai_hf`).
        """
        out_dir = Path(out_dir or (self.base_dir / 'projectmadurai_hf'))
        out_dir.mkdir(parents=True, exist_ok=True)

        # filter
        good = [r for r in records if r.get('quality_score', 0.0) >= min_quality and r.get('text')]
        # deterministic shuffle
        good.sort(key=lambda x: x.get('id'))

        n = len(good)
        n_train = int(n * split[0])
        n_val = int(n * split[1])
        train = good[:n_train]
        val = good[n_train:n_train+n_val]
        test = good[n_train+n_val:]

        def write_split(name, arr):
            path = out_dir / f"{name}.jsonl"
            with open(path, 'w', encoding='utf-8') as f:
                for r in arr:
                    # minimal schema for HF ingestion
                    obj = {"id": r.get('id'), "text": r.get('text'), "meta": {"source": r.get('source'), "url": r.get('url'), "topics": r.get('topics', []), "quality_score": r.get('quality_score', 0.0)}}
                    f.write(json.dumps(obj, ensure_ascii=False) + '\n')
            logger.info(f"Wrote {len(arr)} records to {path}")

        write_split('train', train)
        write_split('validation', val)
        write_split('test', test)

        # write README
        with open(out_dir / 'README.md', 'w', encoding='utf-8') as rf:
            rf.write(f"Project Madurai HF export — total={n}, train={len(train)}, val={len(val)}, test={len(test)}\n")

        return {'train': len(train), 'validation': len(val), 'test': len(test), 'total': n}
    
    def save_to_file(self, texts: List[str], filename: str = "tamil_corpus.txt"):
        """Save collected texts to file.

        Behavior:
        - If `filename` is an absolute path, use it as-is.
        - If `filename` is a relative path that already starts with `self.base_dir`, use it as-is.
        - Otherwise, place the file under `self.base_dir`.
        - Ensure parent directory exists before writing.
        """
        filepath = Path(filename)

        # If filename is relative and not already under base_dir, put it under base_dir
        if not filepath.is_absolute() and not str(filepath).startswith(str(self.base_dir)):
            filepath = self.base_dir / filepath

        # Make sure destination directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            for text in texts:
                if text:  # Skip empty strings
                    f.write(text + "\n\n")

        logger.info(f"Saved {len(texts)} texts to {filepath}")
    
    def get_language_packages(self) -> Dict[str, str]:
        """Check and install required language packages"""
        packages = {
            'tamil': 'pip install tamil',
            'indic-trans': 'pip install indic-trans',
            'indic-nlp-library': 'pip install indic-nlp-library',
            'sentencepiece': 'pip install sentencepiece',
            'fasttext': 'pip install fasttext'
        }
        
        installed = {}
        for pkg, install_cmd in packages.items():
            try:
                __import__(pkg)
                installed[pkg] = "installed"
            except ImportError:
                logger.info(f"Installing {pkg}...")
                os.system(install_cmd)
                try:
                    __import__(pkg)
                    installed[pkg] = "installed"
                except ImportError:
                    installed[pkg] = "failed"
        
        return installed
    
    def run_full_scrape(self):
        """Run complete scraping pipeline"""
        logger.info("Starting full Tamil corpus scraping...")
        
        all_texts = []
        
        # Check language packages
        packages = self.get_language_packages()
        logger.info(f"Language packages: {packages}")
        
        # Scrape different sources
        logger.info("Scraping Wikipedia...")
        all_texts.extend(self.scrape_wikipedia())
        
        logger.info("Scraping literature sites...")
        all_texts.extend(self.scrape_tamil_literature_sites())
        
        logger.info("Scraping news sites...")
        all_texts.extend(self.scrape_tamil_news_sites())
        
        # Save to file
        self.save_to_file(all_texts, "full_tamil_corpus.txt")
        
        logger.info(f"Scraping complete! Total texts: {len(all_texts)}")
        return all_texts

# Command line interface
def main():
    parser = argparse.ArgumentParser(description="Tamil Corpus Scraper for AADHAN")
    parser.add_argument("--output", type=str, default="tamil_corpus.txt",
                       help="Output file for scraped Tamil text (relative to scraper base_dir by default)")
    parser.add_argument("--format", choices=["text","jsonl"], default="text",
                       help="Save format: plain text or jsonl with metadata")
    parser.add_argument("--dedupe", action="store_true", help="Deduplicate collected items")
    parser.add_argument("--classify", action="store_true", help="Run topic classification and quality scoring")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of items saved (0 = all)")
    parser.add_argument("--crawl-sitemap", type=str, default=None,
                       help="Crawl sitemap for a domain (e.g. projectmadurai.org)")
    parser.add_argument("--basic", action="store_true",
                       help="Run a basic one-shot sitemap crawl across configured data sites")
    parser.add_argument("--basic-limit", type=int, default=50,
                       help="Per-site page limit when running `--basic` (0 = all)")
    parser.add_argument("--save-raw-html", action="store_true",
                       help="Save raw HTML files when crawling (under data/raw_html/<domain>)")
    parser.add_argument("--extract-pdfs", action="store_true",
                       help="Download PDF files found in sitemap and extract text (requires pdfminer)")
    parser.add_argument("--build-manifest", action="store_true",
                       help="Build per-work manifests from collected records")
    parser.add_argument("--to-hf", action="store_true",
                       help="Convert collected records to a simple HuggingFace JSONL dataset")
    parser.add_argument("--sources", nargs='+', 
                       choices=['wikipedia', 'literature', 'news', 'books', 'social'],
                       default=['wikipedia', 'literature', 'news'],
                       help="Sources to scrape from")
    parser.add_argument("--full", action="store_true", help="Run full scraping pipeline")
    parser.add_argument("--check_packages", action="store_true", 
                       help="Check and install language packages")
    parser.add_argument("--min-quality", type=float, default=0.0,
                       help="Minimum quality_score to keep a record (0.0-1.0)")
    parser.add_argument("--min-chars", type=int, default=0,
                       help="Minimum character length for records to keep")
    parser.add_argument("--require-tamil", action="store_true",
                       help="Require Tamil-dominant text (filters out pages with low Tamil fraction)")
    parser.add_argument("--apply-fineweb", action="store_true",
                       help="Apply additional FineWeb-style heuristics during filtering")
    
    args = parser.parse_args()
    
    scraper = TamilCorpusScraper()

    if args.check_packages:
        packages = scraper.get_language_packages()
        print(f"Language packages: {packages}")
        return

    # If crawl-sitemap is provided, run sitemap crawler first (specialized flow)
    # --basic: one-shot sitemap crawl across configured sites
    if args.basic:
        domains = ['projectmadurai.org', 'tamilvu.org', 'sangam.org']
        all_records: List[Dict] = []
        per_site_dir = scraper.base_dir / 'basic_crawl'
        per_site_dir.mkdir(parents=True, exist_ok=True)

        for d in domains:
            logger.info(f"[basic] crawling sitemap for {d} (limit={args.basic_limit})")
            recs = scraper.crawl_sitemap(d, limit=args.basic_limit or 0, save_raw=args.save_raw_html, extract_text=True)

            # fallback for literature sites when sitemap/robots block access
            if not recs and d in ('tamilvu.org', 'projectmadurai.org', 'sangam.org'):
                logger.info(f"No sitemap results for {d}; falling back to literature-site scraper")
                texts = scraper._scrape_literature_site(f"https://{d}")
                recs = [scraper.make_record(t, source=d) for t in texts]

            if args.extract_pdfs:
                recs.extend(scraper.extract_pdfs_from_sitemap(d, limit=args.basic_limit or 0))

            if args.dedupe:
                before = len(recs)
                recs = scraper.dedupe_records(recs)
                logger.info(f"[basic] dedupe {d}: {before} -> {len(recs)}")

            if args.classify:
                for r in recs:
                    r['topics'] = scraper.classify_topic(r.get('text',''))
                    r.update(scraper.score_quality(r.get('text','')))

            # optional FineWeb-style filtering per-site
            if (args.min_quality and args.min_quality > 0) or args.min_chars > 0 or args.require_tamil or args.apply_fineweb:
                before_f = len(recs)
                recs = scraper.filter_records(recs, min_quality=args.min_quality, min_chars=args.min_chars, min_tamil_fraction=0.6 if args.require_tamil else 0.0, require_tamil=args.require_tamil, apply_fineweb=args.apply_fineweb)
                logger.info(f"[basic] filtered {d}: {before_f} -> {len(recs)}")

            # save per-site jsonl
            site_out = per_site_dir / f"{d.replace('.', '_')}.jsonl"
            scraper.save_jsonl(recs, str(site_out))
            all_records.extend(recs)

        # combined processing
        if args.dedupe:
            all_before = len(all_records)
            all_records = scraper.dedupe_records(all_records)
            logger.info(f"[basic] combined dedupe: {all_before} -> {len(all_records)}")

        # combined FineWeb-style filtering
        if (args.min_quality and args.min_quality > 0) or args.min_chars > 0 or args.require_tamil or args.apply_fineweb:
            before_f = len(all_records)
            all_records = scraper.filter_records(all_records, min_quality=args.min_quality, min_chars=args.min_chars, min_tamil_fraction=0.6 if args.require_tamil else 0.0, require_tamil=args.require_tamil, apply_fineweb=args.apply_fineweb)
            logger.info(f"[basic] combined filtered: {before_f} -> {len(all_records)}")

        if args.build_manifest:
            scraper.build_manifests(all_records, out_dir=str(per_site_dir / 'manifests'))

        if args.to_hf:
            scraper.convert_to_hf_dataset(all_records, out_dir=str(per_site_dir / 'hf'), min_quality=args.min_quality) 

        # save combined
        if args.format == 'jsonl':
            scraper.save_jsonl(all_records, args.output)
        else:
            scraper.save_to_file([r.get('text','') for r in all_records], args.output)

        print(f"Basic crawl complete. Sites={len(domains)}, records={len(all_records)}")
        return

    # single-domain sitemap crawl (existing flow)
    if args.crawl_sitemap:
        domain = args.crawl_sitemap
        records = scraper.crawl_sitemap(domain, limit=args.limit or 0, save_raw=args.save_raw_html, extract_text=True)

        # fallback for known literature sites when sitemap returns nothing
        if not records and domain in ("tamilvu.org","projectmadurai.org","sangam.org"):
            logger.info(f"No sitemap results for {domain}; falling back to literature-site scraper")
            texts = scraper._scrape_literature_site(f"https://{domain}")
            records = [scraper.make_record(t, source=domain) for t in texts]

        # optionally extract PDFs listed in sitemap (separate flow)
        if args.extract_pdfs:
            pdf_recs = scraper.extract_pdfs_from_sitemap(domain, limit=args.limit or 0)
            # merge PDF records (avoid duplicates)
            records.extend(pdf_recs)

        initial_count = len(records)
        if args.dedupe:
            records = scraper.dedupe_records(records)
            logger.info(f"Deduplicated: {initial_count} -> {len(records)}")

        if args.classify:
            for r in records:
                r["topics"] = scraper.classify_topic(r["text"])
                r.update(scraper.score_quality(r["text"]))

        # optional FineWeb-style filtering for single-domain crawl
        if (args.min_quality and args.min_quality > 0) or args.min_chars > 0 or args.require_tamil or args.apply_fineweb:
            before_f = len(records)
            records = scraper.filter_records(records, min_quality=args.min_quality, min_chars=args.min_chars, min_tamil_fraction=0.6 if args.require_tamil else 0.0, require_tamil=args.require_tamil, apply_fineweb=args.apply_fineweb)
            logger.info(f"Filtered crawl {domain}: {before_f} -> {len(records)}")

        # build manifests if requested
        if args.build_manifest:
            manifest = scraper.build_manifests(records)

        # export HF dataset if requested
        if args.to_hf:
            hf_stats = scraper.convert_to_hf_dataset(records, min_quality=args.min_quality)
            logger.info(f"HF export: {hf_stats}")

        if args.format == "jsonl":
            scraper.save_jsonl(records, args.output)
        else:
            scraper.save_to_file([r["text"] for r in records], args.output)

        print(f"Crawl complete! Total records collected: {len(records)}")
        return

    # --default scraping flow (sources)
    records: List[Dict] = []

    def add_many(raw_texts: List[str], source_name: str):
        for t in raw_texts:
            records.append(scraper.make_record(t, source=source_name))

    if args.full:
        texts = scraper.run_full_scrape()
        add_many(texts, "full")
    else:
        if 'wikipedia' in args.sources:
            add_many(scraper.scrape_wikipedia(), "wikipedia")
        if 'literature' in args.sources:
            add_many(scraper.scrape_tamil_literature_sites(), "literature")
        if 'news' in args.sources:
            add_many(scraper.scrape_tamil_news_sites(), "news")
        if 'books' in args.sources:
            book_paths = []
            add_many(scraper.scrape_tamil_books(book_paths), "books")
        if 'social' in args.sources:
            hashtags = ['#' + tag for tag in TAMIL_HASHTAGS[:5]]  # Use core hashtags
            add_many(scraper.scrape_social_media(hashtags), "social")

    initial_count = len(records)

    if args.dedupe:
        records = scraper.dedupe_records(records)
        logger.info(f"Deduplicated: {initial_count} -> {len(records)}")

    if args.classify:
        for r in records:
            r["topics"] = scraper.classify_topic(r["text"])
            r.update(scraper.score_quality(r["text"]))

    # optional FineWeb-style filtering for default flow
    if (args.min_quality and args.min_quality > 0) or args.min_chars > 0 or args.require_tamil or args.apply_fineweb:
        before_f = len(records)
        records = scraper.filter_records(records, min_quality=args.min_quality, min_chars=args.min_chars, min_tamil_fraction=0.6 if args.require_tamil else 0.0, require_tamil=args.require_tamil, apply_fineweb=args.apply_fineweb)
        logger.info(f"Filtered default flow: {before_f} -> {len(records)}")

    if args.limit and args.limit > 0:
        records = records[:args.limit]

    # Save output
    if args.format == "jsonl":
        scraper.save_jsonl(records, args.output)
    else:
        # plain text: write texts only
        scraper.save_to_file([r["text"] for r in records], args.output)

    print(f"Scraping complete! Total records collected: {len(records)}")

if __name__ == "__main__":
    main()