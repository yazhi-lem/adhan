#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build pre-training sentence corpus (Tamil) from multiple sources and validate colloquial coverage.

Writes:
- data/pre_training/tamil_texts/pretrain_sentences.jsonl
- data/pre_training/tamil_texts/pretrain_sentences.txt
- data/pre_training/tamil_texts/manifest.json
- data/pre_training/tamil_texts/sample_colloquial.txt

Usage:
  python src/data_scraper/build_pretraining_sentences.py --n 5000

"""
from __future__ import annotations
import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import List, Dict

from tamil_corpus_scraper import TamilCorpusScraper

SEED = 42
random.seed(SEED)

COLLOQUIAL_PAT = re.compile(r"\b(ன்னு|ச்சு|வச்சேன்|வச்சி|போயிட்ட[ோ]ன்|போயிட்றேன்|ஏங்க|யானா|யாரு|என்ன|எந்த|டா|யே|யா|யோ|ஹெய்|வா|வாங்க|சார்|சின்ன)\b", re.I)


def is_colloquial(text: str, source: str) -> bool:
    if not text:
        return False
    if source and source.lower() in ('social', 'tweet', 'telegram'):
        return True
    if COLLOQUIAL_PAT.search(text):
        return True
    # short informal sentences heuristic
    if len(text) < 60 and re.search(r'[!?]+$', text):
        return True
    return False


def collect_candidates(scraper: TamilCorpusScraper, sources: List[str], max_articles: int = 1000) -> List[Dict]:
    candidates: List[Dict] = []

    if 'wikipedia' in sources:
        # scrape via library (may be empty if network blocked)
        try:
            texts = scraper.scrape_wikipedia()
            for t in texts:
                candidates.append({'text': t, 'source': 'wikipedia', 'url': None})
        except Exception:
            pass

    if 'literature' in sources:
        try:
            texts = scraper.scrape_tamil_literature_sites()
            for t in texts:
                candidates.append({'text': t, 'source': 'literature', 'url': None})
        except Exception:
            pass

    if 'news' in sources:
        try:
            texts = scraper.scrape_tamil_news_sites()
            for t in texts:
                candidates.append({'text': t, 'source': 'news', 'url': None})
        except Exception:
            pass

    if 'social' in sources:
        try:
            hashtags = ['tamil', 'tamilculture', 'tamilpoetry']
            texts = scraper.scrape_social_media(hashtags)
            for t in texts:
                candidates.append({'text': t, 'source': 'social', 'url': None})
        except Exception:
            pass

    if 'local' in sources:
        # scan pdf_texts, projectmadurai_manifests, raw_html as fallback
        base = Path('data/raw')
        # pdf_texts
        pdir = base / 'pdf_texts'
        if pdir.exists():
            for p in pdir.rglob('*.txt'):
                try:
                    raw = p.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    continue
                if raw and len(raw) > 50:
                    candidates.append({'text': raw, 'source': 'local_pdf', 'url': str(p)})
        # manifests
        mdir = base / 'projectmadurai_manifests'
        if mdir.exists():
            for p in list(mdir.rglob('*.txt')) + list(mdir.rglob('*.jsonl')):
                try:
                    txt = p.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    continue
                if p.suffix == '.jsonl':
                    for line in txt.splitlines():
                        try:
                            obj = json.loads(line)
                            t = obj.get('text') or ''
                        except Exception:
                            continue
                        if t and len(t) > 40:
                            candidates.append({'text': t, 'source': 'local_manifest', 'url': obj.get('url')})
                else:
                    if txt and len(txt) > 40:
                        candidates.append({'text': txt, 'source': 'local_manifest', 'url': str(p)})
        # raw_html
        rdir = base / 'raw_html'
        if rdir.exists():
            for p in rdir.rglob('*.html'):
                try:
                    html = p.read_text(encoding='utf-8', errors='ignore')
                    t = scraper._extract_text_from_html(html)
                    if t and len(t) > 40:
                        candidates.append({'text': t, 'source': 'local_html', 'url': str(p)})
                except Exception:
                    continue

    return candidates[: max(1, max_articles)]


def split_sentences(text: str) -> List[str]:
    # basic split keeping Tamil punctuation in mind
    parts = [p.strip() for p in re.split(r'[\n\.!?।]+', text) if p and p.strip()]
    parts = [re.sub(r'\s+', ' ', p) for p in parts]
    return [p for p in parts if 6 <= len(p) <= 1000]


def score_and_filter(scraper: TamilCorpusScraper, candidates: List[Dict]) -> List[Dict]:
    outs = []
    for c in candidates:
        for s in split_sentences(c['text']):
            si = {'text': s, 'source': c.get('source', 'unknown'), 'url': c.get('url')}
            q = scraper.score_quality(s)
            si.update(q)
            tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', s))
            si['tamil_fraction'] = round(tamil_chars / max(1, len(s)), 3)
            if si['tamil_fraction'] < 0.5:
                continue
            si['is_colloquial'] = is_colloquial(s, si['source'])
            outs.append(si)
    return outs


def dedupe_and_select(records: List[Dict], n: int = 5000, colloq_ratio: float = 0.35) -> List[Dict]:
    # dedupe
    seen = set()
    uniq = []
    for r in records:
        norm = re.sub(r'\s+', ' ', re.sub(r"[^\w\u0B80-\u0BFF]", '', r['text'])).strip()[:512]
        h = None
        if norm:
            h = __import__('hashlib').sha256(norm.encode('utf-8')).hexdigest()
        if not h or h in seen:
            continue
        seen.add(h)
        uniq.append(r)

    # sort by quality_score then length
    uniq.sort(key=lambda x: (x.get('quality_score', 0.0), x.get('char_count', 0)), reverse=True)

    # stratified sampling to ensure colloquial coverage
    colloq = [r for r in uniq if r.get('is_colloquial')]
    noncol = [r for r in uniq if not r.get('is_colloquial')]

    want_col = min(len(colloq), int(n * colloq_ratio))
    selected = []
    selected.extend(colloq[:want_col])

    remaining = n - len(selected)
    selected.extend(noncol[:remaining])

    # if still short, top-up from colloq or noncol
    if len(selected) < n:
        need = n - len(selected)
        pool = colloq + noncol
        sel_extra = [r for r in pool if r not in selected][:need]
        selected.extend(sel_extra)

    # ensure ids
    for r in selected:
        if 'id' not in r:
            r['id'] = __import__('hashlib').sha256(r['text'].encode('utf-8')).hexdigest()
    return selected


def validate_sample(selected: List[Dict]) -> Dict:
    qualities = [r.get('quality_score', 0.0) for r in selected]
    tamil_fracs = [r.get('tamil_fraction', 0.0) for r in selected]
    lengths = [r.get('char_count', len(r['text'])) for r in selected]
    src_counts = Counter(r.get('source') for r in selected)
    colloq_frac = sum(1 for r in selected if r.get('is_colloquial')) / max(1, len(selected))
    sample_colloq = [r['text'] for r in selected if r.get('is_colloquial')][:20]
    sample_non = [r['text'] for r in selected if not r.get('is_colloquial')][:20]
    return {
        'n': len(selected),
        'mean_quality': round(mean(qualities), 3) if qualities else 0.0,
        'median_quality': round(median(qualities), 3) if qualities else 0.0,
        'mean_tamil_frac': round(mean(tamil_fracs), 3) if tamil_fracs else 0.0,
        'median_length': int(median(lengths)) if lengths else 0,
        'source_counts': dict(src_counts),
        'colloquial_fraction': round(colloq_frac, 3),
        'sample_colloquial': sample_colloq,
        'sample_non_colloquial': sample_non
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--output_dir', type=str, default='data/pre_training/tamil_texts')
    parser.add_argument('--sources', type=str, default='wikipedia,literature,news,social,local')
    parser.add_argument('--colloq_ratio', type=float, default=0.35)
    parser.add_argument('--max_candidates', type=int, default=20000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scraper = TamilCorpusScraper(base_dir='data/raw')

    sources = [s.strip() for s in args.sources.split(',') if s.strip()]
    print(f"Collecting from sources: {sources} (target n={args.n}, colloq_ratio={args.colloq_ratio})")

    candidates = collect_candidates(scraper, sources, max_articles=args.max_candidates)
    print(f"Collected {len(candidates)} raw candidate chunks")

    scored = score_and_filter(scraper, candidates)
    print(f"After sentence-splitting and filtering: {len(scored)} candidate sentences")

    selected = dedupe_and_select(scored, n=args.n, colloq_ratio=args.colloq_ratio)
    print(f"Selected {len(selected)} sentences (requested {args.n})")

    # save
    jsonl_out = out_dir / 'pretrain_sentences.jsonl'
    txt_out = out_dir / 'pretrain_sentences.txt'
    with jsonl_out.open('w', encoding='utf-8') as jf, txt_out.open('w', encoding='utf-8') as tf:
        for r in selected:
            obj = {'id': r['id'], 'text': r['text'], 'source': r.get('source'), 'url': r.get('url'), 'quality_score': r.get('quality_score', 0.0), 'tamil_fraction': r.get('tamil_fraction', 0.0), 'is_colloquial': bool(r.get('is_colloquial'))}
            jf.write(json.dumps(obj, ensure_ascii=False) + '\n')
            tf.write(r['text'].strip() + '\n')

    manifest = validate_sample(selected)
    manifest['requested_n'] = args.n
    manifest['colloq_ratio_target'] = args.colloq_ratio
    with (out_dir / 'manifest.json').open('w', encoding='utf-8') as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

    # save sample colloquial sentences for manual inspection
    with (out_dir / 'sample_colloquial.txt').open('w', encoding='utf-8') as sf:
        for s in manifest['sample_colloquial']:
            sf.write(s + '\n')

    print('Done. Manifest:')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
