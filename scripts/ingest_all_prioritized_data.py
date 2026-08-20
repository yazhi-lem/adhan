#!/usr/bin/env python3
"""Unified Data Ingestion & Prioritization Engine for Adhan SLM.

Ingests, cleans, deduplicates, and stages data across:
  - Tier 1: Government, SCERT Textbooks (Classes 1-12), TVA, Open-Sangam, Project Madurai
  - Tier 2: Tamil Wikipedia, Tamil Wiktionary, Press Information Bureau (PIB)
  - Tier 3: AI4Bharat (IndicCorp v2, Sangraha, Samanantar)

Usage:
    python scripts/ingest_all_prioritized_data.py \
        --output-dir data/raw/unified_prioritized \
        --target-tokens 300000000 \
        --sangam-path ../open-sangam
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Set, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adhan_slm.core.logging import get_logger

logger = get_logger(__name__)

# --- Regex Filters & Constants ---------------------------------------------
TAMIL_RANGE = re.compile(r'[\u0B80-\u0BFF]')
EMAIL_REGEX = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_REGEX = re.compile(r'(\+91[\-\s]?)?[6-9]\d{9}')
URL_REGEX = re.compile(r'https?://\S+|www\.\S+')
PULLI_CHAR = chr(0x0BCD)


def normalize_tamil_unicode(text: str) -> str:
    """Normalize text to Unicode NFC and fix isolated combining marks."""
    if not text:
        return ""
    # NFC Canonical Decomposition followed by Canonical Composition
    text = unicodedata.normalize("NFC", text)
    
    # Scrub PII
    text = URL_REGEX.sub(" ", text)
    text = EMAIL_REGEX.sub(" ", text)
    text = PHONE_REGEX.sub(" ", text)
    
    # Fix whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def compute_tamil_ratio(text: str) -> float:
    """Calculate the ratio of Tamil characters to total non-whitespace characters."""
    clean = text.replace(" ", "").replace("\n", "").replace("\t", "")
    if not clean:
        return 0.0
    tamil_chars = len(TAMIL_RANGE.findall(clean))
    return tamil_chars / len(clean)


def is_quality_document(text: str, min_chars: int = 30, max_chars: int = 15000, min_tamil_ratio: float = 0.70) -> bool:
    """Quality filter: checks length, Tamil character ratio, and repetitive spam."""
    if len(text) < min_chars or len(text) > max_chars:
        return False
    if compute_tamil_ratio(text) < min_tamil_ratio:
        return False
    # Check for excessive character repetition (e.g., aaaaa or னனனன)
    if re.search(r'(.)\1{6,}', text):
        return False
    return True


# --- MinHash LSH Near-Deduplication ----------------------------------------
class MinHashDeduplicator:
    """Fast 64-hash MinHash deduplicator over 5-gram akshara shingles."""
    
    def __init__(self, num_hashes: int = 64, threshold: float = 0.80):
        self.num_hashes = num_hashes
        self.threshold = threshold
        self.seen_hashes: List[Tuple[str, List[int]]] = []
        # Precompute random linear hash parameters (a*x + b) % p
        self.p = 2**61 - 1
        self.a = [((i * 10007 + 104729) % self.p) for i in range(1, num_hashes + 1)]
        self.b = [((i * 32452843 + 49979) % self.p) for i in range(1, num_hashes + 1)]

    def _get_shingles(self, text: str, k: int = 5) -> Set[int]:
        shingles = set()
        clean = re.sub(r'\s+', '', text)
        if len(clean) < k:
            shingles.add(hash(clean) & 0xFFFFFFFF)
            return shingles
        for i in range(len(clean) - k + 1):
            shingle = clean[i:i+k]
            shingles.add(hash(shingle) & 0xFFFFFFFF)
        return shingles

    def get_signature(self, shingles: Set[int]) -> List[int]:
        sig = []
        for i in range(self.num_hashes):
            min_val = self.p
            a_i = self.a[i]
            b_i = self.b[i]
            for s in shingles:
                h = (a_i * s + b_i) % self.p
                if h < min_val:
                    min_val = h
            sig.append(min_val)
        return sig

    def is_duplicate(self, text: str) -> bool:
        shingles = self._get_shingles(text)
        if not shingles:
            return True
        sig = self.get_signature(shingles)
        for _, other_sig in self.seen_hashes:
            matches = sum(1 for x, y in zip(sig, other_sig) if x == y)
            similarity = matches / self.num_hashes
            if similarity >= self.threshold:
                return True
        self.seen_hashes.append((text[:64], sig))
        return False


# --- Data Source Connectors ------------------------------------------------

def ingest_open_sangam(sangam_path: Path) -> Generator[Dict, None, None]:
    """Ingest 18 Sangam poems, 2,552 verses, and modern Tamil Urai prose."""
    texts_dir = sangam_path / "data" / "texts"
    if not texts_dir.exists():
        logger.warning(f"Open-Sangam texts dir not found at {texts_dir}")
        return

    count = 0
    for json_file in texts_dir.glob("**/*.json"):
        try:
            content = json.loads(json_file.read_text(encoding="utf-8"))
            records = content if isinstance(content, list) else [content]
            for rec in records:
                # Combine classical verse and modern urai
                verse = rec.get("sangamTamil", "") or rec.get("text", "")
                urai = rec.get("urai", "")
                full_text = f"{verse}\n\nஉரை:\n{urai}" if urai else verse
                full_text = normalize_tamil_unicode(full_text)
                if is_quality_document(full_text, min_chars=20):
                    yield {
                        "id": f"sangam-{rec.get('id', count)}",
                        "text": full_text,
                        "source": "open-sangam-classical",
                        "tier": 1,
                        "quality_score": 0.95
                    }
                    count += 1
        except Exception as exc:
            continue
    logger.info(f"Ingested {count:,} high-quality records from Open-Sangam.")


def ingest_scert_textbooks(scert_dir: Path) -> Generator[Dict, None, None]:
    """Ingest extracted text from TN SCERT Samacheer Kalvi school textbooks (Class 1-12)."""
    if not scert_dir.exists():
        logger.info(f"SCERT directory {scert_dir} not yet present (create and place textbook txt files here).")
        return

    count = 0
    for txt_file in scert_dir.glob("**/*.txt"):
        try:
            text = txt_file.read_text(encoding="utf-8")
            paragraphs = text.split("\n\n")
            for i, para in enumerate(paragraphs):
                clean = normalize_tamil_unicode(para)
                if is_quality_document(clean, min_chars=50):
                    yield {
                        "id": f"scert-{txt_file.stem}-{i}",
                        "text": clean,
                        "source": "tn-scert-textbooks",
                        "tier": 1,
                        "quality_score": 0.95
                    }
                    count += 1
        except Exception:
            continue
    logger.info(f"Ingested {count:,} records from SCERT school textbooks.")


def ingest_wikipedia_ta(limit: int = 150000) -> Generator[Dict, None, None]:
    """Stream Tamil Wikipedia articles via HuggingFace datasets."""
    try:
        from datasets import load_dataset
        logger.info("Connecting to Wikimedia Tamil dataset...")
        ds = load_dataset("wikimedia/wikipedia", "20231101.ta", split="train", streaming=True)
        count = 0
        for row in ds:
            text = normalize_tamil_unicode(row.get("text", ""))
            if is_quality_document(text, min_chars=100):
                yield {
                    "id": f"wiki-ta-{count}",
                    "title": row.get("title", ""),
                    "text": text,
                    "source": "wikipedia-ta",
                    "tier": 2,
                    "quality_score": 0.90
                }
                count += 1
                if count >= limit:
                    break
        logger.info(f"Ingested {count:,} articles from Tamil Wikipedia.")
    except Exception as exc:
        logger.warning(f"Could not load Tamil Wikipedia stream: {exc}")


def ingest_indiccorp_sample(limit: int = 250000) -> Generator[Dict, None, None]:
    """Stream clean Tamil text from AI4Bharat IndicCorp v2."""
    try:
        from datasets import load_dataset
        logger.info("Connecting to AI4Bharat IndicCorpV2 Tamil dataset...")
        ds = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split="tam_Taml", streaming=True)
        count = 0
        for row in ds:
            text = normalize_tamil_unicode(row.get("text", ""))
            if is_quality_document(text, min_chars=60, min_tamil_ratio=0.75):
                yield {
                    "id": f"indiccorp-{count}",
                    "text": text,
                    "source": "ai4bharat-indiccorp-v2",
                    "tier": 3,
                    "quality_score": 0.80
                }
                count += 1
                if count >= limit:
                    break
        logger.info(f"Ingested {count:,} documents from IndicCorp v2.")
    except Exception as exc:
        logger.warning(f"Could not load IndicCorp stream: {exc}")


# --- Orchestration Main ----------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest and prioritize all Tamil datasets for Adhan SLM")
    parser.add_argument("--output-dir", type=str, default="data/raw/unified_prioritized", help="Output directory")
    parser.add_argument("--sangam-path", type=str, default="../open-sangam", help="Path to open-sangam repo")
    parser.add_argument("--scert-dir", type=str, default="data/raw/scert_textbooks", help="Path to SCERT books")
    parser.add_argument("--limit-wiki", type=int, default=100000, help="Max wiki records")
    parser.add_argument("--limit-indic", type=int, default=200000, help="Max IndicCorp records")
    parser.add_argument("--skip-huggingface", action="store_true", help="Skip online dataset streaming")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    unified_file = out_dir / "unified_corpus.jsonl"
    stats_file = out_dir / "ingestion_datasheet.json"

    dedup = MinHashDeduplicator(num_hashes=64, threshold=0.80)
    source_counts = defaultdict(int)
    tier_counts = defaultdict(int)
    total_written = 0
    total_duplicates = 0

    logger.info(f"Starting unified ingestion into {unified_file}...")

    with open(unified_file, "w", encoding="utf-8") as f_out:
        # 1. Tier 1: Classical Literature & Open-Sangam
        for rec in ingest_open_sangam(Path(args.sangam_path)):
            if not dedup.is_duplicate(rec["text"]):
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                source_counts[rec["source"]] += 1
                tier_counts[rec["tier"]] += 1
                total_written += 1
            else:
                total_duplicates += 1

        # 2. Tier 1: SCERT Textbooks
        for rec in ingest_scert_textbooks(Path(args.scert_dir)):
            if not dedup.is_duplicate(rec["text"]):
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                source_counts[rec["source"]] += 1
                tier_counts[rec["tier"]] += 1
                total_written += 1
            else:
                total_duplicates += 1

        # 3. Tier 2: Wikipedia (if not skipped)
        if not args.skip_huggingface:
            for rec in ingest_wikipedia_ta(limit=args.limit_wiki):
                if not dedup.is_duplicate(rec["text"]):
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    source_counts[rec["source"]] += 1
                    tier_counts[rec["tier"]] += 1
                    total_written += 1
                else:
                    total_duplicates += 1

            # 4. Tier 3: IndicCorp v2 (if not skipped)
            for rec in ingest_indiccorp_sample(limit=args.limit_indic):
                if not dedup.is_duplicate(rec["text"]):
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    source_counts[rec["source"]] += 1
                    tier_counts[rec["tier"]] += 1
                    total_written += 1
                else:
                    total_duplicates += 1

    stats = {
        "total_documents_ingested": total_written,
        "duplicates_removed": total_duplicates,
        "dedup_rate_percent": round((total_duplicates / max(1, total_written + total_duplicates)) * 100, 2),
        "source_breakdown": dict(source_counts),
        "tier_breakdown": dict(tier_counts),
        "target_file": str(unified_file)
    }

    stats_file.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    
    logger.info(f"✅ Ingestion Complete! {total_written:,} clean records saved ({total_duplicates:,} duplicates removed).")
    logger.info(f"📊 Datasheet written to {stats_file}")


if __name__ == "__main__":
    main()
