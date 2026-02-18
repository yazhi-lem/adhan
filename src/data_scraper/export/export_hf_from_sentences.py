#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPRECATED: Use export_unified_hf.py instead with --strategy standard

Export sentences JSONL to HuggingFace-style train/validation/test JSONL splits.

This script is deprecated. Use the unified exporter:
    python src/data_scraper/export/export_unified_hf.py --strategy standard
"""
import json
import warnings
from pathlib import Path

warnings.warn(
    "export_hf_from_sentences.py is deprecated. "
    "Use export_unified_hf.py with --strategy standard instead.",
    DeprecationWarning,
    stacklevel=2
)

IN_FILE = Path('data/pre_training/tamil_texts/all_sentences.jsonl')
OUT_DIR = Path('data/pre_training/tamil_texts/hf')
OUT_DIR.mkdir(parents=True, exist_ok=True)

records = []
if not IN_FILE.exists():
    print('Input file not found:', IN_FILE)
    raise SystemExit(1)

with IN_FILE.open('r', encoding='utf-8') as fh:
    for line in fh:
        try:
            obj = json.loads(line)
            records.append(obj)
        except Exception:
            continue

# deterministic ordering
records.sort(key=lambda x: x.get('id') or '')

n = len(records)
if n == 0:
    print('No records to export')
    raise SystemExit(1)

n_train = int(n * 0.8)
n_val = int(n * 0.1)
train = records[:n_train]
val = records[n_train:n_train + n_val]
test = records[n_train + n_val:]

for name, arr in (('train', train), ('validation', val), ('test', test)):
    path = OUT_DIR / f'{name}.jsonl'
    with path.open('w', encoding='utf-8') as f:
        for r in arr:
            obj = {'id': r.get('id'), 'text': r.get('text'), 'meta': {'source': r.get('source'), 'url': r.get('url'), 'quality_score': r.get('quality_score')}}
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print(f'Wrote {len(arr)} records to {path}')

# write README
with (OUT_DIR / 'README.md').open('w', encoding='utf-8') as rf:
    rf.write(f'HuggingFace-style export from merged Tamil corpus\n')
    rf.write(f'total={n}, train={len(train)}, validation={len(val)}, test={len(test)}\n')
    rf.write(f'Sources: wiki_sentences.jsonl + wiki_api_sentences.jsonl (Wikipedia via python-wikipedia-api)\n')

print('Export complete')

