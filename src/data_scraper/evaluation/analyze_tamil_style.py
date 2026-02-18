#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze corpus for old vs modern Tamil patterns.
Classify sentences by archaic/formal vs colloquial/modern markers.
"""
import json
import re
from pathlib import Path
from collections import Counter

IN_FILE = Path('data/pre_training/tamil_texts/all_sentences.jsonl')

records = []
with IN_FILE.open('r', encoding='utf-8') as fh:
    for line in fh:
        try:
            obj = json.loads(line)
            records.append(obj)
        except Exception:
            continue

print(f"Total: {len(records)} sentences\n")

# Old Tamil markers (archaic, classical)
old_markers = {
    'archaic_verb_forms': r'(ஆல்|அந்தாய்|ஒழிந்து|அஞ்சிய)',  # old verb conjugations
    'archaic_pronouns': r'(அவர்|இவர்|உவர்)',  # old pronoun forms
    'classical_poetry': r'(வையம்|நிழல்|குழல்|சோலை|வண்)',  # poetry words
    'old_spellings': r'(தமிலு|தமிழalzl)',  # old/variant spellings
}

# Modern Tamil markers (colloquial, contemporary)
modern_markers = {
    'modern_dialogue': r'(என்றாள்|என்றான்|சொன்னாங்க|சொல்லலாம்)',  # modern speech
    'modern_pronouns': r'(நீ|நான்|அவன்|அவள்|நாம்)',  # modern pronouns
    'modern_slang': r'(ப்ரக்டிক్్|சூப்|கூல్|ஏன்கொறா)',  # informal
    'contemporary': r'(மொபைல்|கம্ப్యూటర్|ஆன்லைன్|வீடியो)',  # modern words
}

old_count = 0
modern_count = 0
mixed_count = 0

for r in records:
    txt = r.get('text', '')
    has_old = False
    has_modern = False

    for pattern in old_markers.values():
        if re.search(pattern, txt):
            has_old = True
            break

    for pattern in modern_markers.values():
        if re.search(pattern, txt):
            has_modern = True
            break

    if has_old and has_modern:
        mixed_count += 1
    elif has_old:
        old_count += 1
    elif has_modern:
        modern_count += 1

old_pct = round(100 * old_count / len(records), 1)
modern_pct = round(100 * modern_count / len(records), 1)
mixed_pct = round(100 * mixed_count / len(records), 1)

print(f"Old Tamil (classical/archaic): {old_count} ({old_pct}%)")
print(f"Modern Tamil (colloquial): {modern_count} ({modern_pct}%)")
print(f"Mixed: {mixed_count} ({mixed_pct}%)")
print(f"Neutral/other: {len(records) - old_count - modern_count - mixed_count}")

print(f"\n⚠️  Current corpus is {old_pct}% old Tamil")
print(f"    Need to add more modern/colloquial sources")
print(f"    Recommendation: Reweight to ~60% modern, 40% classical")
