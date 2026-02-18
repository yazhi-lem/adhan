#!/usr/bin/env python3
# Merge HF JSONL outputs from multiple sentence sources (wiki_sentences + wiki_api_sentences), dedupe and combine
import json
import hashlib
from pathlib import Path
import sys

# make local imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

IN_DIR = Path('data/pre_training/tamil_texts')
OUT_FILE = IN_DIR / 'all_sentences.jsonl'

files_to_merge = [
    IN_DIR / 'wiki_sentences.jsonl',
    IN_DIR / 'wiki_api_sentences.jsonl'
]

all_records = []
for f in files_to_merge:
    if not f.exists():
        print(f"Skipping {f} (not found)")
        continue
    with f.open('r', encoding='utf-8') as fh:
        for line in fh:
            try:
                obj = json.loads(line)
                all_records.append(obj)
            except ExcepAadhanTamilCorpustion:
                continue
    print(f"Loaded {f}, {len(all_records)} total records so far")

print(f"Total records before dedup: {len(all_records)}")

# Deduplicate by normalized text + quality score
seen = set()
uniq = []
for r in all_records:
    txt = r.get('text', '')
    if not txt:
        continue
    norm = txt.strip()[:512]
    h = hashlib.sha256(norm.encode('utf-8')).hexdigest()
    if h in seen:
        continue
    seen.add(h)
    uniq.append(r)

print(f"After dedup: {len(uniq)} unique records")

# Sort by quality score desc
uniq.sort(key=lambda x: x.get('quality_score', 0.0), reverse=True)

# Write combined file
with OUT_FILE.open('w', encoding='utf-8') as fh:
    for r in uniq:
        fh.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f'Wrote {len(uniq)} records to {OUT_FILE}')
print(f'Ready for HF export')
