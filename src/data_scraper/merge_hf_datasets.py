#!/usr/bin/env python3
# Merge HF JSONL outputs from basic + full scrapes, dedupe and re-export a single HF dataset
import json
from pathlib import Path
import sys
import os

# make local imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tamil_corpus_scraper import TamilCorpusScraper

BASIC = Path('data/raw/basic_crawl/combined_basic.jsonl')
FULL = Path('data/raw/full_combined.jsonl')
OUT_DIR = Path('data/raw/combined_hf')


def load_jsonl(p: Path):
    out = []
    if p.exists():
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    return out


print('Loading JSONL inputs...')
records = load_jsonl(BASIC) + load_jsonl(FULL)
print(f'Loaded: {len(records)} raw records')

scraper = TamilCorpusScraper()
# ensure quality/topics present
for r in records:
    if 'quality_score' not in r:
        r.update(scraper.score_quality(r.get('text','')))
    if 'topics' not in r:
        r['topics'] = scraper.classify_topic(r.get('text',''))

# dedupe
records = scraper.dedupe_records(records)
print(f'After dedupe: {len(records)} records')

# export HF dataset
stats = scraper.convert_to_hf_dataset(records, out_dir=str(OUT_DIR), split=(0.8,0.1,0.1), min_quality=0.0)
print('Exported combined HF dataset to', OUT_DIR)
print(stats)
