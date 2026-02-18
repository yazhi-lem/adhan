#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebalance corpus: prioritize modern/news/social sources, filter archaic literary texts.
Rebuild with weighted selection to favor contemporary Tamil.
"""
import json
import hashlib
import re
from pathlib import Path
from collections import defaultdict

IN_FILE = Path('data/pre_training/tamil_texts/all_sentences.jsonl')
OUT_FILE = Path('data/pre_training/tamil_texts/all_sentences_rebalanced.jsonl')

records = []
with IN_FILE.open('r', encoding='utf-8') as fh:
    for line in fh:
        try:
            obj = json.loads(line)
            records.append(obj)
        except Exception:
            continue

print(f"Loaded {len(records)} records")

# Classify by source and apply weighting/filtering
by_source = defaultdict(list)
for r in records:
    src = r.get('source', 'unknown')
    by_source[src].append(r)

# Priority weighting (favor modern sources)
# news > social > wikipedia > literature > local
source_weights = {
    'news': 3.0,       # Modern contemporary
    'social': 2.5,     # Colloquial/conversational
    'wikipedia': 1.0,  # Formal but contemporary
    'literature': 0.5, # Old literary sites
    'local': 0.3,      # Classical/archaic (Project Madurai)
}

# Apply weights and quality filtering
selected = []
for src, items in by_source.items():
    weight = source_weights.get(src, 1.0)
    # add weight to items; prefer high-quality modern text
    for item in items:
        # boost weight if sentence has modern markers
        txt = item.get('text', '')
        if re.search(r'(என்றாள்|என்றான்|சொன்নாங்க|நீ|நான்|அவன்|அவள்)', txt):
            weight_adj = weight * 1.5
        else:
            weight_adj = weight
        
        item['_weight'] = weight_adj
        item['_source'] = src
        selected.append(item)

# Sort by weight and quality score, then pick top N
selected.sort(key=lambda x: (x.get('_weight', 1.0) * x.get('quality_score', 0.0), x.get('quality_score', 0.0)), reverse=True)

# Keep top 2900 (slight reduction to ensure quality)
final = selected[:2900]

# Deduplicate one more time
seen = set()
unique = []
for r in final:
    txt = r.get('text', '')
    if not txt:
        continue
    norm = txt.strip()[:512]
    h = hashlib.sha256(norm.encode('utf-8')).hexdigest()
    if h in seen:
        continue
    seen.add(h)
    unique.append(r)

# Write rebalanced output
with OUT_FILE.open('w', encoding='utf-8') as fh:
    for r in unique:
        # remove internal fields
        clean = {k: v for k, v in r.items() if not k.startswith('_')}
        fh.write(json.dumps(clean, ensure_ascii=False) + '\n')

print(f"Rebalanced to {len(unique)} sentences")
print(f"Source distribution after rebalancing:")
src_dist = defaultdict(int)
for r in unique:
    src = r.get('_source') or r.get('source', 'unknown')
    src_dist[src] += 1

for src, cnt in sorted(src_dist.items(), key=lambda x: -x[1]):
    pct = round(100 * cnt / len(unique), 1)
    print(f"  {src}: {cnt} ({pct}%)")

print(f"\n✅ Rebalanced corpus ready at {OUT_FILE}")
print(f"   Modern sources (news + social) now prioritized")
print(f"   Classical texts filtered for quality")
