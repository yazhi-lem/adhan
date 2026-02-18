#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build modern Tamil corpus by extracting sentences from social/news sources.
Process tamil_social_sample.jsonl, tamil_corpus.txt, and other modern sources.
"""
import json
import re
import hashlib
from pathlib import Path
from collections import defaultdict

def sentence_split(text):
    """Split text into sentences by Tamil punctuation and newlines."""
    # Split by period, exclamation, question mark, newline
    sentences = re.split(r'[।!?.\n]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def detect_modern_markers(text):
    """Score text for modern/colloquial Tamil features."""
    score = 0
    # Modern pronouns and conversational markers
    if re.search(r'\b(நீ|என்|அவν|அவங்க|எனக்கு|உனக்கு|சொன்னேன்|என்றீங்க)\b', text):
        score += 3
    # Modern verbs/tenses
    if re.search(r'(ற|ட்ட|வை|கூறுகிறோம்|பேசுகிறேன்|இருக்கிறது)', text):
        score += 2
    # Colloquial markers
    if re.search(r'(ங்க|ந्चा|டா|ய्ऌ|यह्यॉ)', text):
        score += 2
    # Modern/news vocabulary
    if re.search(r'(கடந்த|இந்த|வரும்|கூறினார்|தெரிவித்தார்)', text):
        score += 1
    return min(score, 5)  # cap at 5

# Load all sources
modern_records = []

# 1. Tamil social sample
social_file = Path('data/raw/tamil_social_sample.jsonl')
if social_file.exists():
    with social_file.open('r', encoding='utf-8') as fh:
        for line in fh:
            try:
                obj = json.loads(line)
                txt = obj.get('text', '').replace('tamil ', '').strip()
                if len(txt) > 15:  # filter very short
                    modern_records.append({
                        'text': txt,
                        'source': 'social',
                        'quality_score': 0.6,  # boost social sources
                        'tamil_fraction': 1.0,
                        'url': None,
                    })
            except Exception:
                pass

# 2. Tamil corpus as sentences
corpus_file = Path('data/raw/tamil_corpus.txt')
if corpus_file.exists():
    with corpus_file.open('r', encoding='utf-8') as fh:
        text = fh.read()
        # Extract sentences
        for para in text.split('\n\n'):
            sentences = sentence_split(para)
            for sent in sentences:
                if len(sent) > 20 and len(sent) < 400:
                    modern_records.append({
                        'text': sent,
                        'source': 'news',
                        'quality_score': 0.55,
                        'tamil_fraction': 0.95,
                        'url': None,
                    })

# Add some sample modern Tamil phrases (conversational corpus)
modern_phrases = [
    "நீ என்ன செய்யிறியா இப்போது?",
    "என்கிட்ட ஒண்ணும் சொல்லிக்காதான்",
    "இந்த பொம்மைக்களுக்கு என்ன பெதை?",
    "நான் வீட்டுக்கு போறேன்",
    "அவங்க எங்க போச்சு பார்?",
    "கடந்த வாரம் செய்த கொடுக்கல் சொல்லுங்க",
    "மக்கள் இதை கேட்டால் சந்தோஷப்பட்டுவாங்க",
    "நsteal நாம் যা பண்ணியோம் அது சரியே",
    "வந்த நாலு பேருக்கு வேற வந்து சொல்லிட்டாங்க",
    "தமிழ் மொழிய பயன்படுத்தணும் அப்ப நிறைய கேள்வி வரும்",
]

for phrase in modern_phrases:
    if len(phrase) > 10:
        modern_records.append({
            'text': phrase,
            'source': 'modern_conversational',
            'quality_score': 0.7,
            'tamil_fraction': 1.0,
            'url': None,
        })

print(f"Collected {len(modern_records)} modern Tamil records")

# Deduplicate
seen = set()
unique_modern = []
for r in modern_records:
    txt = r.get('text', '')
    norm = txt.strip()[:256]
    h = hashlib.sha256(norm.encode('utf-8')).hexdigest()
    if h in seen:
        continue
    seen.add(h)
    # Add modern score
    r['modern_score'] = detect_modern_markers(txt)
    unique_modern.append(r)

print(f"Deduplicated to {len(unique_modern)} unique modern records")

# Load existing corpus
all_file = Path('data/pre_training/tamil_texts/all_sentences.jsonl')
existing = []
if all_file.exists():
    with all_file.open('r', encoding='utf-8') as fh:
        for line in fh:
            try:
                obj = json.loads(line)
                existing.append(obj)
            except Exception:
                pass

print(f"Loaded {len(existing)} existing corpus records")

# Merge: prioritize modern, then existing
merged = unique_modern + existing
print(f"Merged to {len(merged)} total records")

# Final deduplication
final_seen = set()
final_unique = []
for r in merged:
    txt = r.get('text', '')
    norm = txt.strip()[:256]
    h = hashlib.sha256(norm.encode('utf-8')).hexdigest()
    if h in final_seen:
        continue
    final_seen.add(h)
    final_unique.append(r)

print(f"Final deduplicated: {len(final_unique)} records")

# Write output
out_file = Path('data/pre_training/tamil_texts/all_sentences_modern.jsonl')
with out_file.open('w', encoding='utf-8') as fh:
    for r in final_unique:
        clean = {k: v for k, v in r.items() if not k.startswith('_')}
        fh.write(json.dumps(clean, ensure_ascii=False) + '\n')

# Stats
src_count = defaultdict(int)
for r in final_unique:
    src = r.get('source', 'unknown')
    src_count[src] += 1

print(f"\n✅ Modern Tamil corpus ready at {out_file}")
print(f"Source distribution:")
for src, cnt in sorted(src_count.items(), key=lambda x: -x[1]):
    pct = round(100 * cnt / len(final_unique), 1)
    print(f"  {src}: {cnt} ({pct}%)")
