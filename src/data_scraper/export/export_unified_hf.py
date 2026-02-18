#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified HuggingFace export with configurable weighting strategies.
Consolidates export_hf_from_sentences.py and export_modern_hf.py into a single script.
"""
import json
import hashlib
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict


def load_records(input_file: Path) -> List[Dict]:
    """Load records from JSONL file."""
    records = []
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with input_file.open('r', encoding='utf-8') as fh:
        for line in fh:
            try:
                obj = json.loads(line)
                records.append(obj)
            except Exception:
                continue
    
    return records


def apply_weights(records: List[Dict], strategy: str = 'standard') -> List[Dict]:
    """Apply weighting strategy to records."""
    weighted_records = []
    
    for r in records:
        src = r.get('source', 'unknown')
        quality = r.get('quality_score', 0.5)
        
        if strategy == 'modern':
            # Boost weight for modern sources
            if src in ['social', 'news', 'modern_conversational']:
                weight = quality * 2.5  # 2.5x boost
            elif src == 'wikipedia':
                weight = quality * 1.5  # 1.5x boost for contemporary
            else:  # literature, local (classical)
                weight = quality * 0.8  # 0.8x reduction for archaic
        else:  # standard strategy - uniform weighting
            weight = quality
        
        r['_weight'] = weight
        weighted_records.append(r)
    
    return weighted_records


def stratified_selection(records: List[Dict], modern_ratio: float = 0.60) -> List[Dict]:
    """Select records with stratified sampling by source."""
    by_source = defaultdict(list)
    for r in records:
        src = r.get('source', 'unknown')
        by_source[src].append(r)
    
    target_total = len(records)
    modern_sources = ['news', 'social', 'modern_conversational']
    modern_target = int(target_total * modern_ratio)
    classical_target = target_total - modern_target
    
    # Select from modern sources
    modern_pool = []
    for src in modern_sources:
        modern_pool.extend(by_source.get(src, []))
    
    modern_selected = sorted(modern_pool, key=lambda x: x.get('_weight', 0), reverse=True)[:modern_target]
    
    # Fill rest with classical sources, high quality first
    classical_pool = []
    for src in ['wikipedia', 'local', 'literature']:
        classical_pool.extend(by_source.get(src, []))
    
    classical_selected = sorted(classical_pool, key=lambda x: x.get('_weight', 0), reverse=True)[:classical_target]
    
    selected = modern_selected + classical_selected
    print(f"Selected {len(selected)} records ({len(modern_selected)} modern + {len(classical_selected)} classical)")
    
    return selected


def deduplicate_records(records: List[Dict]) -> List[Dict]:
    """Deduplicate records based on text hash."""
    final_seen = set()
    final = []
    
    for r in records:
        txt = r.get('text', '')
        norm = txt.strip()[:256]
        h = hashlib.sha256(norm.encode('utf-8')).hexdigest()
        if h in final_seen:
            continue
        final_seen.add(h)
        final.append(r)
    
    return final


def split_data(records: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1, 
               seed: int = 42) -> tuple:
    """Split records into train/val/test sets."""
    random.seed(seed)
    random.shuffle(records)
    
    n_train = int(train_ratio * len(records))
    n_val = int(val_ratio * len(records))
    
    train = records[:n_train]
    val = records[n_train:n_train + n_val]
    test = records[n_train + n_val:]
    
    return train, val, test


def write_splits(train: List[Dict], val: List[Dict], test: List[Dict], output_dir: Path):
    """Write train/val/test splits to JSONL files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, records in [('train', train), ('validation', val), ('test', test)]:
        out_file = output_dir / f'{name}.jsonl'
        with out_file.open('w', encoding='utf-8') as fh:
            for r in records:
                # Remove internal fields
                clean = {k: v for k, v in r.items() if not k.startswith('_')}
                fh.write(json.dumps(clean, ensure_ascii=False) + '\n')
        print(f"Wrote {len(records)} records to {out_file}")


def write_readme(train: List[Dict], val: List[Dict], test: List[Dict], 
                 output_dir: Path, strategy: str):
    """Write dataset README with statistics."""
    total = len(train) + len(val) + len(test)
    
    # Calculate source distribution
    all_records = train + val + test
    src_dist = defaultdict(int)
    for r in all_records:
        src = r.get('source', 'unknown')
        src_dist[src] += 1
    
    # Calculate average quality
    avg_quality = sum(r.get('quality_score', 0) for r in all_records) / len(all_records)
    
    # Calculate modern markers
    modern_count = sum(1 for r in all_records if r.get('modern_score', 0) > 0)
    modern_pct = (modern_count / len(all_records) * 100) if all_records else 0
    
    content = f"""# Tamil Pretraining Dataset

**Strategy**: {strategy.upper()}

## Dataset Info

- **Total Records**: {total:,}
- **Train Split**: {len(train):,} ({len(train)/total*100:.1f}%)
- **Validation Split**: {len(val):,} ({len(val)/total*100:.1f}%)
- **Test Split**: {len(test):,} ({len(test)/total*100:.1f}%)

## Source Distribution

"""
    
    for src, cnt in sorted(src_dist.items(), key=lambda x: -x[1]):
        pct = (cnt / total * 100)
        content += f"- **{src}**: {cnt:,} ({pct:.1f}%)\n"
    
    content += f"""
## Quality Metrics

- **Avg Quality Score**: {avg_quality:.3f}
- **Modern Language Markers**: {modern_pct:.1f}%

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files={{
    "train": "train.jsonl",
    "validation": "validation.jsonl", 
    "test": "test.jsonl"
}})
```

## Export Strategy

This dataset was exported using the **{strategy}** strategy:
"""
    
    if strategy == 'modern':
        content += """
- Modern Tamil sources (social, news, conversational) are boosted by 2.5x
- Contemporary Wikipedia articles are boosted by 1.5x
- Classical/archaic literature is reduced to 40% of total
- Colloquial language is emphasized
"""
    else:
        content += """
- Uniform quality-based weighting
- Balanced representation across all sources
- No source-specific boosts applied
"""
    
    readme_file = output_dir / 'README.md'
    with readme_file.open('w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ README written to {readme_file}")


def print_split_stats(train: List[Dict], val: List[Dict], test: List[Dict]):
    """Print detailed statistics for each split."""
    print(f"\n📊 Split distribution:")
    
    for name, records in [('TRAIN', train), ('VALIDATION', val), ('TEST', test)]:
        src_dist = defaultdict(int)
        for r in records:
            src = r.get('source', 'unknown')
            src_dist[src] += 1
        
        print(f"\n{name} ({len(records)} records):")
        for src, cnt in sorted(src_dist.items(), key=lambda x: -x[1]):
            pct = (cnt / len(records) * 100) if records else 0
            print(f"  {src}: {cnt} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Unified HuggingFace dataset exporter')
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSONL file (e.g., all_sentences.jsonl)')
    parser.add_argument('--output', type=str, default='data/pre_training/tamil_texts/hf',
                        help='Output directory for HF splits')
    parser.add_argument('--strategy', choices=['standard', 'modern'], default='standard',
                        help='Weighting strategy: standard (uniform) or modern (boost contemporary)')
    parser.add_argument('--modern-ratio', type=float, default=0.60,
                        help='Target ratio for modern sources (only for modern strategy)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Train split ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling')
    parser.add_argument('--no-dedupe', action='store_true',
                        help='Skip deduplication step')
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_dir = Path(args.output)
    
    # Load records
    print(f"Loading records from {input_file}...")
    records = load_records(input_file)
    print(f"Loaded {len(records)} records")
    
    # Apply weighting strategy
    print(f"Applying {args.strategy} weighting strategy...")
    weighted_records = apply_weights(records, args.strategy)
    
    # Stratified selection (only for modern strategy)
    if args.strategy == 'modern':
        print(f"Performing stratified selection (modern ratio: {args.modern_ratio})...")
        selected = stratified_selection(weighted_records, args.modern_ratio)
    else:
        selected = weighted_records
    
    # Deduplicate
    if not args.no_dedupe:
        print("Deduplicating records...")
        final = deduplicate_records(selected)
        print(f"Deduplicated to {len(final)} records")
    else:
        final = selected
    
    # Sort for deterministic ordering
    final.sort(key=lambda x: x.get('id', '') or x.get('text', '')[:50])
    
    # Split data
    print(f"Splitting data (train: {args.train_ratio}, val: {args.val_ratio})...")
    train, val, test = split_data(final, args.train_ratio, args.val_ratio, args.seed)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Write splits
    print(f"Writing splits to {output_dir}...")
    write_splits(train, val, test, output_dir)
    
    # Write README
    write_readme(train, val, test, output_dir, args.strategy)
    
    # Print statistics
    print_split_stats(train, val, test)
    
    print(f"\n✅ Export complete! Dataset ready at {output_dir}")


if __name__ == '__main__':
    main()
