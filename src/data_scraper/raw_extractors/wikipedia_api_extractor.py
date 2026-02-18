#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch Tamil Wikipedia articles using `wikipedia-api` and extract high-quality Tamil sentences
for pretraining. Searches for Tamil and English keywords relevant to politics, history, education,
and society to assemble a diverse corpus.

Usage:
  python src/data_scraper/wikipedia_api_extractor.py --n_sentences 5000 --output_dir data/pre_training/tamil_texts

"""
from __future__ import annotations
import argparse
import json
import hashlib
import logging
import re
from pathlib import Path
from typing import List, Set

try:
    import wikipediaapi
except Exception:
    wikipediaapi = None

from tamil_corpus_scraper import TamilCorpusScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SENT_SPLIT_RE = re.compile(r'[\n\.!?।]+')


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text) if p and p.strip()]
    parts = [re.sub(r"\s+", " ", p) for p in parts]
    parts = [p for p in parts if 10 <= len(p) <= 1000]
    return parts


def fetch_articles_by_keywords(wiki, keywords: List[str], max_per_kw: int = 300) -> List[dict]:
    seen: Set[str] = set()
    articles = []
    for kw in keywords:
        try:
            logger.info(f"Searching for keyword: {kw}")
            results = wiki.search(kw, results=max_per_kw)
        except Exception as e:
            logger.warning(f"Search failed for {kw}: {e}")
            results = []
        for title in results:
            if title in seen:
                continue
            seen.add(title)
            try:
                page = wiki.page(title)
                if page and page.exists():
                    text = page.text or ''
                    if text.strip():
                        articles.append({'title': title, 'text': text, 'url': f"https://ta.wikipedia.org/wiki/{title.replace(' ', '_')}"})
            except Exception:
                continue
    logger.info(f"Fetched {len(articles)} articles from search keywords")
    return articles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_sentences', type=int, default=5000)
    parser.add_argument('--output_dir', type=str, default='data/pre_training/tamil_texts')
    parser.add_argument('--max_per_kw', type=int, default=300)
    parser.add_argument('--keywords', type=str, default='அரசியல்,வரலாறு,கல்வி,சமூகம்,கலாச்சாரம்,Politics,History,Education')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if wikipediaapi is None:
        logger.error('wikipedia-api package not found. Install with: pip install wikipedia-api')
        return

    wiki = wikipediaapi.Wikipedia(language='ta', extract_format=wikipediaapi.ExtractFormat.WIKI, user_agent='AdhanTamilCorpus/1.0')
    scraper = TamilCorpusScraper(base_dir='data/raw')

    kws = [k.strip() for k in args.keywords.split(',') if k.strip()]
    articles = fetch_articles_by_keywords(wiki, kws, max_per_kw=args.max_per_kw)

    # fallback: if no articles found, try category listing via page.categorymembers for main category
    if not articles:
        try:
            cat = wiki.page('Category:தமிழ்')
            if cat and cat.exists():
                for title, member in cat.categorymembers.items():
                    if member.ns == 0:
                        articles.append({'title': title, 'text': member.text or '', 'url': f"https://ta.wikipedia.org/wiki/{title.replace(' ', '_')}" } )
        except Exception:
            pass

    sentences = []
    for art in articles:
        sents = split_sentences(art.get('text',''))
        for s in sents:
            sentences.append({'text': scraper._clean_tamil_text(s), 'source': 'wikipedia', 'url': art.get('url')})

    logger.info(f'Extracted {len(sentences)} raw sentences from Wikipedia')

    # Score, filter and dedupe
    scored = []
    for s in sentences:
        score_info = scraper.score_quality(s['text'])
        s['quality_score'] = score_info.get('quality_score', 0.0)
        s['char_count'] = score_info.get('char_count', len(s['text']))
        tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', s['text']))
        tamil_frac = tamil_chars / max(1, len(s['text']))
        s['tamil_fraction'] = round(tamil_frac, 3)
        if tamil_frac < 0.5:
            continue
        scored.append(s)

    logger.info(f'Kept {len(scored)} Tamil-dominant sentences after filtering')

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

    logger.info(f'Deduplicated to {len(uniq)} unique sentences')

    uniq.sort(key=lambda x: (x.get('quality_score', 0.0), x.get('char_count', 0)), reverse=True)
    top_n = uniq[:args.n_sentences]

    out_file = out_dir / 'wiki_api_sentences.jsonl'
    with out_file.open('w', encoding='utf-8') as fh:
        for s in top_n:
            fh.write(json.dumps({'id': s['id'], 'text': s['text'], 'source': s['source'], 'url': s.get('url'), 'quality_score': s['quality_score'], 'tamil_fraction': s['tamil_fraction']}, ensure_ascii=False) + '\n')

    manifest = {'n_candidates': len(sentences), 'n_filtered': len(scored), 'n_unique': len(uniq), 'n_output': len(top_n), 'out_file': str(out_file)}
    with (out_dir / 'wiki_api_manifest.json').open('w', encoding='utf-8') as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

    logger.info('Done. Saved wiki_api_sentences and manifest')


if __name__ == '__main__':
    main()
