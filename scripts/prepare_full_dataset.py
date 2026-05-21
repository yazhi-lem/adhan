#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare consolidated and cleaned Tamil dataset for training.
Merges Wikipedia, News, and other sources, and cleans sticky headers.
"""

import json
import os
import re
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "pre_training" / "tamil_texts"
OUTPUT_DIR = PROJECT_ROOT / "data" / "final" / "tamil_texts" / "full_hf"

# Input files
INPUT_FILES = [
    RAW_DATA_DIR / "wiki_api_sentences.jsonl",
    RAW_DATA_DIR / "all_sentences_modern.jsonl",
    RAW_DATA_DIR / "all_sentences.jsonl",
]

# Patterns to strip (sticky headers)
STICKY_HEADERS = [
    r"^சினிமா செய்திகள்",
    r"^கல்வி செய்திகள்",
    r"^தமிழ்நாடு செய்திகள்",
    r"^இன்றைய ராசி பலன்",
    r"^இந்தியா செய்தி",
    r"^உலக செய்தி",
    r"^வீடு பராமரிப்பு",
    r"^தமிழக அரசு பணிகள்",
    r"^தேர்வு முடிவுகள்",
    r"^இந்து மதம்",
    r"^தமிழகம்",
    r"^இந்தியா",
    r"^விளையாட்டு",
]

def clean_text(text):
    if not text:
        return ""
    
    # Strip sticky headers
    for pattern in STICKY_HEADERS:
        text = re.sub(pattern, "", text).strip()
    
    # Clean up multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_output_file = RAW_DATA_DIR / "merged_corpus_raw.jsonl"
    
    records = []
    seen_texts = set()
    
    print(f"Loading and cleaning records...")
    for input_file in INPUT_FILES:
        if not input_file.exists():
            print(f"⚠️ Skipping missing file: {input_file}")
            continue
            
        print(f"  Processing {input_file.name}...")
        count = 0
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    raw_text = data.get("text", "")
                    cleaned_text = clean_text(raw_text)
                    
                    if not cleaned_text or len(cleaned_text) < 20:
                        continue
                        
                    # Deduplicate based on cleaned text
                    text_hash = cleaned_text[:200]
                    if text_hash in seen_texts:
                        continue
                        
                    seen_texts.add(text_hash)
                    data["text"] = cleaned_text
                    records.append(data)
                    count += 1
                except Exception as e:
                    continue
        print(f"    Done: {count} valid records.")

    print(f"Writing merged raw corpus to {merged_output_file}...")
    with open(merged_output_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nTotal unique records: {len(records)}")
    print(f"Now run the unified exporter to generate splits:")
    print(f"python src/data_scraper/export/export_unified_hf.py --input {merged_output_file} --output {OUTPUT_DIR} --strategy modern")

if __name__ == "__main__":
    main()
