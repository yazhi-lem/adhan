#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified corpus builder with configurable modern/classical balancing.
Consolidates build_modern_tamil_corpus.py and build_modern_tamil_sources.py.
"""
import json
import re
import hashlib
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sentence_split(text: str) -> List[str]:
    """Split text into sentences by Tamil punctuation and newlines."""
    sentences = re.split(r'[।!?.\n]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def detect_modern_markers(text: str) -> int:
    """Score text for modern/colloquial Tamil features."""
    score = 0
    
    # Modern pronouns and conversational markers
    if re.search(r'\b(நீ|என்|அவன்|அவங்க|எனக்கு|உனக்கு|சொன்னேன்|என்றீங்க)\b', text):
        score += 3
    
    # Modern verbs/tenses
    if re.search(r'(ற|ட்ட|வை|கூறுகிறோம்|பேசுகிறேன்|இருக்கிறது)', text):
        score += 2
    
    # Colloquial markers
    if re.search(r'(ங்க|ச்சா|டா|யா|யோ)', text):
        score += 2
    
    # Modern/news vocabulary
    if re.search(r'(கடந்த|இந்த|வரும்|கூறினார்|தெரிவித்தார்)', text):
        score += 1
    
    return min(score, 5)  # cap at 5


def load_social_sources(data_dir: Path) -> List[Dict]:
    """Load Tamil social media sources."""
    records = []
    social_file = data_dir / 'tamil_social_sample.jsonl'
    
    if social_file.exists():
        with social_file.open('r', encoding='utf-8') as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    txt = obj.get('text', '').replace('tamil ', '').strip()
                    if len(txt) > 15:
                        records.append({
                            'text': txt,
                            'source': 'social',
                            'quality_score': 0.6,
                            'tamil_fraction': 1.0,
                            'url': None,
                        })
                except Exception:
                    pass
    
    return records


def load_news_corpus(data_dir: Path) -> List[Dict]:
    """Load Tamil news corpus."""
    records = []
    corpus_file = data_dir / 'tamil_corpus.txt'
    
    if corpus_file.exists():
        with corpus_file.open('r', encoding='utf-8') as fh:
            text = fh.read()
            for para in text.split('\n\n'):
                sentences = sentence_split(para)
                for sent in sentences:
                    if 20 < len(sent) < 400:
                        records.append({
                            'text': sent,
                            'source': 'news',
                            'quality_score': 0.55,
                            'tamil_fraction': 0.95,
                            'url': None,
                        })
    
    return records


def get_conversational_phrases() -> List[Dict]:
    """Get sample modern conversational Tamil phrases from config file."""
    # Try to load from config file first
    config_file = Path(__file__).parent / 'conversational_phrases.yaml'
    
    if config_file.exists():
        try:
            import yaml
            with config_file.open('r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            phrases = config.get('phrases', [])
            source = config.get('source', 'modern_conversational')
            quality_score = config.get('quality_score', 0.7)
            tamil_fraction = config.get('tamil_fraction', 1.0)
            
            records = []
            for phrase in phrases:
                if len(phrase) > 10:
                    records.append({
                        'text': phrase,
                        'source': source,
                        'quality_score': quality_score,
                        'tamil_fraction': tamil_fraction,
                        'url': None,
                    })
            return records
        except Exception as e:
            logger.warning(f"Failed to load phrases from config: {e}. Using fallback.")
    
    # Fallback to hardcoded phrases if config not available
    phrases = [
        "நீ என்ன செய்கிறாய் இப்போது?",
        "என்கிட்ட ஒன்னும் சொல்லவில்லை",
        "இந்த பொருட்களுக்கு என்ன விலை?",
        "நான் வீட்டுக்கு போகிறேன்",
        "அவங்க எங்கு போனார்கள்?",
        "கடந்த வாரம் செய்த வேலை சொல்லுங்கள்",
        "மக்கள் இதை கேட்டால் சந்தோஷப்படுவார்கள்",
        "நாம் செய்தது சரியே",
        "வந்த பேருக்கு வேறு சொல்லிட்டாங்க",
        "தமிழ் மொழியை பயன்படுத்த வேண்டும்",
    ]
    
    records = []
    for phrase in phrases:
        if len(phrase) > 10:
            records.append({
                'text': phrase,
                'source': 'modern_conversational',
                'quality_score': 0.7,
                'tamil_fraction': 1.0,
                'url': None,
            })
    
    return records


def load_existing_corpus(corpus_file: Path) -> List[Dict]:
    """Load existing corpus from JSONL file."""
    records = []
    if corpus_file.exists():
        with corpus_file.open('r', encoding='utf-8') as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    records.append(obj)
                except Exception:
                    pass
    return records


def apply_source_weights(records: List[Dict], weight_map: Dict[str, float]) -> List[Dict]:
    """Apply source-based weights to records."""
    for r in records:
        src = r.get('source', 'unknown')
        weight = weight_map.get(src, 1.0)
        
        # Boost weight if sentence has modern markers
        txt = r.get('text', '')
        modern_score = detect_modern_markers(txt)
        
        if modern_score > 0:
            weight_adj = weight * (1.0 + modern_score * 0.1)
        else:
            weight_adj = weight
        
        r['_weight'] = weight_adj
        r['modern_score'] = modern_score
    
    return records


def deduplicate_records(records: List[Dict]) -> List[Dict]:
    """Deduplicate records based on text hash."""
    seen = set()
    unique = []
    
    for r in records:
        txt = r.get('text', '')
        if not txt:
            continue
        
        norm = txt.strip()[:256]
        h = hashlib.sha256(norm.encode('utf-8')).hexdigest()
        
        if h in seen:
            continue
        
        seen.add(h)
        unique.append(r)
    
    return unique


def select_top_records(records: List[Dict], max_count: int = None) -> List[Dict]:
    """Select top records by weight and quality score."""
    # Sort by weight * quality_score, then by quality_score
    records.sort(
        key=lambda x: (
            x.get('_weight', 1.0) * x.get('quality_score', 0.0),
            x.get('quality_score', 0.0)
        ),
        reverse=True
    )
    
    if max_count:
        return records[:max_count]
    return records


def print_source_distribution(records: List[Dict], title: str = "Distribution"):
    """Print source distribution statistics."""
    print(f"\n{title}:")
    src_dist = defaultdict(int)
    
    for r in records:
        src = r.get('source', 'unknown')
        src_dist[src] += 1
    
    for src, cnt in sorted(src_dist.items(), key=lambda x: -x[1]):
        pct = (cnt / len(records) * 100) if records else 0
        print(f"  {src}: {cnt} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Unified Tamil corpus builder')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                        help='Directory containing raw data files')
    parser.add_argument('--existing-corpus', type=str,
                        default='data/pre_training/tamil_texts/all_sentences.jsonl',
                        help='Path to existing corpus JSONL file')
    parser.add_argument('--output', type=str,
                        default='data/pre_training/tamil_texts/all_sentences_modern.jsonl',
                        help='Output JSONL file path')
    parser.add_argument('--strategy', choices=['balanced', 'modern', 'rebalanced'],
                        default='modern',
                        help='Corpus building strategy')
    parser.add_argument('--max-records', type=int, default=None,
                        help='Maximum number of records to keep (None = all)')
    parser.add_argument('--modern-only', action='store_true',
                        help='Only use modern sources (ignore existing corpus)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    existing_corpus_file = Path(args.existing_corpus)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Define weight maps for different strategies
    weight_maps = {
        'balanced': {
            'news': 1.0,
            'social': 1.0,
            'wikipedia': 1.0,
            'literature': 1.0,
            'local': 1.0,
            'modern_conversational': 1.0,
        },
        'modern': {
            'news': 3.0,
            'social': 2.5,
            'modern_conversational': 3.0,
            'wikipedia': 1.5,
            'literature': 0.5,
            'local': 0.3,
        },
        'rebalanced': {
            'news': 3.0,
            'social': 2.5,
            'modern_conversational': 2.5,
            'wikipedia': 1.0,
            'literature': 0.5,
            'local': 0.3,
        }
    }
    
    weight_map = weight_maps[args.strategy]
    
    # Load modern sources
    print("Loading modern sources...")
    modern_records = []
    
    social_records = load_social_sources(data_dir)
    print(f"  Loaded {len(social_records)} social media records")
    modern_records.extend(social_records)
    
    news_records = load_news_corpus(data_dir)
    print(f"  Loaded {len(news_records)} news corpus records")
    modern_records.extend(news_records)
    
    conversational = get_conversational_phrases()
    print(f"  Added {len(conversational)} conversational phrases")
    modern_records.extend(conversational)
    
    print(f"Total modern records collected: {len(modern_records)}")
    
    # Deduplicate modern records
    print("Deduplicating modern sources...")
    unique_modern = deduplicate_records(modern_records)
    print(f"Deduplicated to {len(unique_modern)} unique modern records")
    
    # Load existing corpus
    all_records = unique_modern.copy()
    
    if not args.modern_only:
        print(f"\nLoading existing corpus from {existing_corpus_file}...")
        existing = load_existing_corpus(existing_corpus_file)
        print(f"Loaded {len(existing)} existing corpus records")
        all_records.extend(existing)
        print(f"Merged to {len(all_records)} total records")
    
    # Apply weights
    print(f"\nApplying '{args.strategy}' weighting strategy...")
    weighted_records = apply_source_weights(all_records, weight_map)
    
    # Final deduplication
    print("Final deduplication...")
    final_unique = deduplicate_records(weighted_records)
    print(f"Deduplicated to {len(final_unique)} records")
    
    # Select top records if max_records specified
    if args.max_records:
        print(f"\nSelecting top {args.max_records} records by weight...")
        final_unique = select_top_records(final_unique, args.max_records)
    else:
        final_unique = select_top_records(final_unique)
    
    # Write output
    print(f"\nWriting output to {output_file}...")
    with output_file.open('w', encoding='utf-8') as fh:
        for r in final_unique:
            # Remove internal fields
            clean = {k: v for k, v in r.items() if not k.startswith('_')}
            fh.write(json.dumps(clean, ensure_ascii=False) + '\n')
    
    # Print statistics
    print(f"\n✅ Corpus ready at {output_file}")
    print(f"Total records: {len(final_unique)}")
    print_source_distribution(final_unique, "Source distribution")
    
    # Calculate quality metrics
    avg_quality = sum(r.get('quality_score', 0) for r in final_unique) / len(final_unique)
    avg_modern = sum(r.get('modern_score', 0) for r in final_unique) / len(final_unique)
    
    print(f"\n📊 Quality metrics:")
    print(f"  Average quality score: {avg_quality:.3f}")
    print(f"  Average modern score: {avg_modern:.2f}")
    
    modern_count = sum(1 for r in final_unique if r.get('modern_score', 0) > 0)
    modern_pct = (modern_count / len(final_unique) * 100) if final_unique else 0
    print(f"  Records with modern markers: {modern_count} ({modern_pct:.1f}%)")


if __name__ == '__main__':
    main()
