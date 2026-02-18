#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare old vs. modern corpus composition.
"""
import json
from pathlib import Path
from collections import defaultdict

def analyze_corpus(jsonl_path):
    """Analyze a corpus JSONL file."""
    records = []
    with Path(jsonl_path).open('r', encoding='utf-8') as fh:
        for line in fh:
            try:
                records.append(json.loads(line))
            except:
                pass
    
    # Source distribution
    src_dist = defaultdict(int)
    quality_by_src = defaultdict(list)
    
    for r in records:
        src = r.get('source', 'unknown')
        src_dist[src] += 1
        quality_by_src[src].append(r.get('quality_score', 0))
    
    return records, src_dist, quality_by_src

# Load both versions
old_records, old_src, old_q = analyze_corpus('data/pre_training/tamil_texts/all_sentences.jsonl')
new_records, new_src, new_q = analyze_corpus('data/pre_training/tamil_texts/all_sentences_modern.jsonl')

print("=" * 70)
print("CORPUS COMPARISON: Original vs. Modern-Enhanced")
print("=" * 70)

print(f"\n📊 CORPUS SIZE:")
print(f"  Original:      {len(old_records):,} records")
print(f"  Modern-Enhanced: {len(new_records):,} records (+{len(new_records) - len(old_records)})")

print(f"\n📚 SOURCE DISTRIBUTION:")
print(f"\n{'Source':<20} {'Original':<20} {'Modern-Enhanced':<20} {'Change':<10}")
print("-" * 70)

all_sources = set(old_src.keys()) | set(new_src.keys())
for src in sorted(all_sources, key=lambda x: new_src.get(x, 0), reverse=True):
    old_cnt = old_src.get(src, 0)
    new_cnt = new_src.get(src, 0)
    old_pct = round(100 * old_cnt / len(old_records), 1)
    new_pct = round(100 * new_cnt / len(new_records), 1)
    delta_pct = new_pct - old_pct
    delta_sign = "↑" if delta_pct > 0 else "↓" if delta_pct < 0 else "—"
    
    print(f"{src:<20} {old_cnt:>5} ({old_pct:>5.1f}%)      {new_cnt:>5} ({new_pct:>5.1f}%)       {delta_sign} {abs(delta_pct):>5.1f}%")

print(f"\n💡 MODERN vs. CLASSICAL SPLIT:")
modern_sources = ['news', 'social', 'modern_conversational']
old_modern = sum(old_src.get(s, 0) for s in modern_sources)
new_modern = sum(new_src.get(s, 0) for s in modern_sources)
old_classical = len(old_records) - old_modern
new_classical = len(new_records) - new_modern

old_modern_pct = round(100 * old_modern / len(old_records), 1)
new_modern_pct = round(100 * new_modern / len(new_records), 1)

print(f"  Original:      {old_modern} modern ({old_modern_pct}%) | {old_classical} classical ({100-old_modern_pct}%)")
print(f"  Modern-Enhanced: {new_modern} modern ({new_modern_pct}%) | {new_classical} classical ({100-new_modern_pct}%)")
print(f"  Improvement:   +{new_modern_pct - old_modern_pct:.1f}% modern sources")

print(f"\n⭐ QUALITY METRICS:")
for src in sorted(all_sources):
    old_q_avg = sum(old_q.get(src, [0])) / len(old_q.get(src, [0])) if old_q.get(src) else 0
    new_q_avg = sum(new_q.get(src, [0])) / len(new_q.get(src, [0])) if new_q.get(src) else 0
    print(f"  {src:<20} Original: {old_q_avg:.3f} | Modern-Enhanced: {new_q_avg:.3f}")

print("\n" + "=" * 70)
print("✅ MODERN CORPUS READY!")
print("=" * 70)
print("\nNext steps for training:")
print("\n1. Use the modern-enhanced HF splits:")
print(f"   - Train: data/pre_training/tamil_texts/hf/train.jsonl (1,220 records)")
print(f"   - Val: data/pre_training/tamil_texts/hf/validation.jsonl (152 records)")
print(f"   - Test: data/pre_training/tamil_texts/hf/test.jsonl (154 records)")
print("\n2. For even better modern Tamil emphasis, consider:")
print("   - Sampling with source weighting during training (3x weight for news/social)")
print("   - Using curriculum learning (start with modern → gradually include classical)")
print("   - Fine-tuning on news/social corpora after initial pretraining")
print("\n3. To further enhance with modern sources:")
print("   - Add Twitter API integration for real-time Tamil tweets")
print("   - Scrape modern Tamil blogs/news sites (tamil.samayam.com, etc.)")
print("   - Include Tamil YouTube captions/comments")
print("   - Add Tamil Reddit/Discord transcripts if available")
print(f"\n4. Current composition (modern-enhanced):")
print(f"   • {new_modern_pct:.1f}% modern/news/social sources")
print(f"   • {100 - new_modern_pct:.1f}% classical/formal sources")
