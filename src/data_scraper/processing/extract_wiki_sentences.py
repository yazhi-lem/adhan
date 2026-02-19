#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract high-quality Tamil sentences from Tamil Wikipedia for pre-training.
Saves top-N sentences (by quality score) as JSONL and plain text.

Usage:
  python src/data_scraper/extract_wiki_sentences.py --n_sentences 5000 --output_dir data/pre_training/tamil_texts --format jsonl

"""
from __future__ import annotations
import argparse
import json
import hashlib
import logging
import re
from pathlib import Path
from typing import List, Dict

from tamil_corpus_scraper import TamilCorpusScraper

# Import core constants
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core import TAMIL_HASHTAGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SENT_SPLIT_RE = re.compile(r'[\n\.!?।]+')


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text) if p and p.strip()]
    # normalize whitespace
    parts = [re.sub(r"\s+", " ", p) for p in parts]
    # keep reasonably sized sentences
    parts = [p for p in parts if 10 <= len(p) <= 1000]
    return parts


def collect_articles(scraper: TamilCorpusScraper, category: str = "Tamil_language", max_articles: int = 500) -> List[Dict]:
    """Fetch article URLs from Tamil Wikipedia category and scrape each article (text + url).
    Returns list of dicts: {'url':..., 'text':...}
    """
    logger.info(f"Fetching up to {max_articles} articles from category: {category}")
    api = "https://ta.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'categorymembers',
        'cmtitle': f"Category:{category}",
        'cmlimit': min(max_articles, 500)
    }
    try:
        resp = scraper.session.get(api, params=params, timeout=30)
        data = resp.json()
        members = data.get('query', {}).get('categorymembers', [])
    except Exception as e:
        logger.error(f"Wikipedia API error: {e}")
        members = []

    articles = []
    for m in members:
        if m.get('ns') != 0:
            continue
        title = m.get('title')
        url = f"https://ta.wikipedia.org/wiki/{title.replace(' ', '_')}"
        try:
            txt = scraper._scrape_wikipedia_article(url)
            if txt:
                articles.append({'url': url, 'text': txt})
        except Exception:
            continue
        if len(articles) >= max_articles:
            break
    logger.info(f"Collected {len(articles)} articles from Wikipedia category {category}")
    return articles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_sentences', type=int, default=5000)
    parser.add_argument('--output_dir', type=str, default='data/pre_training/tamil_texts')
    parser.add_argument('--format', choices=['jsonl', 'txt'], default='jsonl')
    parser.add_argument('--category', type=str, default='Tamil_language')
    parser.add_argument('--max_articles', type=int, default=500)
    parser.add_argument('--include', type=str, default='wikipedia,literature,news,social,local',
                        help='Comma separated sources to include: wikipedia,literature,news,social,local')
    parser.add_argument('--to_hf', action='store_true', help='Also export HuggingFace-style train/val/test JSONL')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scraper = TamilCorpusScraper(base_dir='data/raw')

    # Collect articles
    articles = collect_articles(scraper, category=args.category, max_articles=args.max_articles)

    # Fallback: if API returned no articles, use the library method scrape_wikipedia()
    if not articles:
        logger.info("No category articles found via API; falling back to scraper.scrape_wikipedia()")
        fallback_texts = scraper.scrape_wikipedia()
        for t in fallback_texts:
            articles.append({'url': None, 'text': t})

    # Helper: collect local texts from workspace (pdf_texts, manifests, raw_html)
    def collect_local_texts(scraper: TamilCorpusScraper, max_files: int = 1000) -> List[Dict]:
        base = Path('data/raw')
        candidates: List[Dict] = []

        # PDF-extracted texts
        pdir = base / 'pdf_texts'
        if pdir.exists():
            for p in pdir.rglob('*.txt'):
                if len(candidates) >= max_files:
                    break
                try:
                    raw = p.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    continue
                if not raw.strip():
                    continue
                # split by paragraphs and keep longer chunks
                for part in re.split(r'\n{2,}', raw):
                    if len(candidates) >= max_files:
                        break
                    cleaned = scraper._clean_tamil_text(part)
                    if scraper._is_tamil_text(cleaned) and len(cleaned) > 40:
                        candidates.append({'url': str(p), 'text': cleaned})

        # Per-work manifests (.txt / .jsonl)
        mdir = base / 'projectmadurai_manifests'
        if mdir.exists():
            for p in list(mdir.rglob('*.txt')) + list(mdir.rglob('*.jsonl')):
                if len(candidates) >= max_files:
                    break
                try:
                    txt = p.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    continue
                if p.suffix == '.jsonl':
                    for line in txt.splitlines():
                        if len(candidates) >= max_files:
                            break
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        t = obj.get('text') or ''
                        cleaned = scraper._clean_tamil_text(t)
                        if scraper._is_tamil_text(cleaned) and len(cleaned) > 40:
                            candidates.append({'url': obj.get('url'), 'text': cleaned})
                else:
                    cleaned = scraper._clean_tamil_text(txt)
                    if scraper._is_tamil_text(cleaned) and len(cleaned) > 40:
                        candidates.append({'url': str(p), 'text': cleaned})

        # Raw HTML (extract visible text)
        rdir = base / 'raw_html'
        if rdir.exists():
            for p in rdir.rglob('*.html'):
                if len(candidates) >= max_files:
                    break
                try:
                    html = p.read_text(encoding='utf-8', errors='ignore')
                    txt = scraper._extract_text_from_html(html)
                    if txt and scraper._is_tamil_text(txt) and len(txt) > 40:
                        candidates.append({'url': str(p), 'text': txt})
                except Exception:
                    continue

        logger.info(f"Collected {len(candidates)} local text chunks for fallback")
        return candidates

    include = [x.strip() for x in (args.include or '').split(',') if x.strip()]

    # Split into sentences (primary from Wikipedia)
    sentences = []  # list of (sent, url)
    if 'wikipedia' in include:
        for art in articles:
            sents = split_sentences(art['text'])
            for s in sents:
                sentences.append({'text': s, 'source': 'wikipedia', 'url': art.get('url')})

    # If no Wikipedia content could be fetched, fall back to local extracted corpora
    if not sentences and 'local' in include:
        logger.info("No Wikipedia sentences found; falling back to local corpus (pdf_texts / manifests / raw_html)")
        local_articles = collect_local_texts(scraper, max_files=2000)
        for art in local_articles:
            sents = split_sentences(art['text'])
            for s in sents:
                sentences.append({'text': s, 'source': 'local', 'url': art.get('url')})

    # Also include literature, news, social sources if requested
    if 'literature' in include:
        try:
            texts = scraper.scrape_tamil_literature_sites()
            for t in texts:
                for s in split_sentences(t):
                    sentences.append({'text': s, 'source': 'literature', 'url': None})
        except Exception:
            logger.warning('Failed to include literature sites')

    if 'news' in include:
        try:
            texts = scraper.scrape_tamil_news_sites()
            for t in texts:
                for s in split_sentences(t):
                    sentences.append({'text': s, 'source': 'news', 'url': None})
        except Exception:
            logger.warning('Failed to include news sites')

    if 'social' in include:
        try:
            hashtags = TAMIL_HASHTAGS[:10]  # Use core hashtags
            texts = scraper.scrape_social_media(hashtags)
            for t in texts:
                for s in split_sentences(t):
                    sentences.append({'text': s, 'source': 'social', 'url': None})
        except Exception:
            logger.warning('Failed to include social samples')

    logger.info(f"Extracted {len(sentences)} candidate sentences from {len(articles)} articles (plus local fallback)")

    # Score and filter
    scored = []
    for s in sentences:
        score_info = scraper.score_quality(s['text'])
        s['quality_score'] = score_info.get('quality_score', 0.0)
        s['char_count'] = score_info.get('char_count', len(s['text']))
        s['word_count'] = score_info.get('word_count', len(s['text'].split()))
        # require Tamil-dominant
        tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', s['text']))
        tamil_frac = tamil_chars / max(1, len(s['text']))
        s['tamil_fraction'] = round(tamil_frac, 3)
        if tamil_frac < 0.5:
            continue
        scored.append(s)

    logger.info(f"Kept {len(scored)} Tamil-dominant sentences after filtering")

    # Deduplicate by normalized text
    seen = set()
    uniq = []
    for s in scored:
        norm = re.sub(r'\s+', ' ', re.sub(r'[^\w\u0B80-\u0BFF]', '', s['text'])).strip()[:512]
        if not norm:
            continue
        h = hashlib.sha256(norm.encode('utf-8')).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        s['id'] = sha256_hex(s['text'] + (s.get('url') or ''))
        uniq.append(s)

    logger.info(f"Deduplicated to {len(uniq)} unique sentences")

    # Sort by quality score desc and pick top N
    uniq.sort(key=lambda x: (x.get('quality_score', 0.0), x.get('char_count', 0)), reverse=True)
    top_n = uniq[:args.n_sentences]
    logger.info(f"Selected top {len(top_n)} sentences")

    # Save outputs
    if args.format == 'jsonl':
        out_file = out_dir / 'wiki_sentences.jsonl'
        with out_file.open('w', encoding='utf-8') as fh:
            for s in top_n:
                obj = {'id': s['id'], 'text': s['text'], 'source': s['source'], 'url': s['url'], 'quality_score': s['quality_score'], 'char_count': s['char_count'], 'word_count': s['word_count'], 'tamil_fraction': s['tamil_fraction']}
                fh.write(json.dumps(obj, ensure_ascii=False) + '\n')
        logger.info(f"Wrote {len(top_n)} records to {out_file}")
    else:
        out_file = out_dir / 'wiki_sentences.txt'
        with out_file.open('w', encoding='utf-8') as fh:
            for s in top_n:
                fh.write(s['text'].strip() + '\n')
        logger.info(f"Wrote {len(top_n)} sentences to {out_file}")

    # Also write a small manifest
    manifest = {'n_candidates': len(sentences), 'n_filtered': len(scored), 'n_unique': len(uniq), 'n_output': len(top_n), 'out_file': str(out_file)}
    with (out_dir / 'manifest.json').open('w', encoding='utf-8') as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)
    logger.info('Done. Manifest saved.')


if __name__ == '__main__':
    main()
