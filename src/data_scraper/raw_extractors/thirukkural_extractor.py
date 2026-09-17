#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thirukkural Ingestion & Pure Tamil Dataset Builder.

Ingests all 1,330 Thirukkural verses across 133 Athikarams with couplet lines
and pure Tamil Urais (Mu. Varadarajan, Solomon Pappaiah, Mu. Karunanidhi).

Generates:
  1. `data/raw/thirukkural/thirukkural_corpus.jsonl` (Pretraining documents)
  2. `data/raw/thirukkural/thirukkural_harness_data.json` (Eval benchmark pairs)
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

THIRUKKURAL_URL = "https://raw.githubusercontent.com/tk120404/thirukkural/master/thirukkural.json"


def fetch_thirukkural_data() -> List[Dict[str, Any]]:
    """Fetch complete 1,330 Thirukkural dataset."""
    logger.info("Fetching Thirukkural dataset from upstream...")
    req = urllib.request.Request(
        THIRUKKURAL_URL,
        headers={"User-Agent": "YazhiAdhan/1.0 (+https://github.com/yazhi-lem/adhan)"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("kural", [])


def build_datasets(kurals: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_file = output_dir / "thirukkural_corpus.jsonl"
    harness_file = output_dir / "thirukkural_harness_data.json"

    corpus_records = []
    harness_records = []

    for k in kurals:
        num = k.get("Number")
        line1 = (k.get("Line1") or "").strip()
        line2 = (k.get("Line2") or "").strip()
        mv = (k.get("mv") or "").strip()
        sp = (k.get("sp") or "").strip()
        mk = (k.get("mk") or "").strip()

        couplet = f"{line1}\n{line2}"

        # Pure Tamil document format for SLM pretraining
        full_text = (
            f"திருக்குறள் {num}:\n"
            f"{couplet}\n\n"
            f"மு. வரதராசனார் உரை: {mv}\n\n"
            f"சாலமன் பாப்பையா உரை: {sp}\n\n"
            f"கலைஞர் உரை: {mk}"
        )

        corpus_records.append(
            {
                "id": f"thirukkural-{num}",
                "text": full_text,
                "source": "thirukkural-classical",
                "tier": 1,
                "quality_score": 1.0,
                "kural_num": num,
            }
        )

        # Benchmark pair for Thirukkural evaluation harness
        harness_records.append(
            {
                "kural_num": num,
                "line1": line1,
                "line2": line2,
                "couplet": couplet,
                "urai_mv": mv,
                "urai_sp": sp,
                "prompt_line2": f"திருக்குறள் முதல் அடி: {line1}\nஇரண்டாம் அடி:",
                "prompt_urai": f"திருக்குறள்:\n{couplet}\n\nஇதன் பொருள்:",
                "target_urai": mv,
            }
        )

    with corpus_file.open("w", encoding="utf-8") as f:
        for rec in corpus_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with harness_file.open("w", encoding="utf-8") as f:
        json.dump(harness_records, f, ensure_ascii=False, indent=2)

    logger.info(
        "Successfully generated %d Thirukkural pretraining records -> %s",
        len(corpus_records),
        corpus_file,
    )
    logger.info(
        "Successfully generated %d Thirukkural benchmark pairs -> %s",
        len(harness_records),
        harness_file,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Thirukkural & create evaluation benchmarks"
    )
    parser.add_argument("--output-dir", default="data/raw/thirukkural", help="Output directory")
    args = parser.parse_args()

    kurals = fetch_thirukkural_data()
    build_datasets(kurals, Path(args.output_dir))


if __name__ == "__main__":
    main()
