#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract the Tholkappiyam (தொல்காப்பியம்) moolam as clean, structured Tamil text.

The Tholkappiyam is the oldest extant Tamil grammar — its எழுத்ததிகாரம்
(Ezhuttatikaram) literally enumerates the Tamil letter/phoneme system the Swaram
tokenizer is built around, and its புணர்ச்சி (sandhi) rules are the grammatical
ground truth for the morpheme layer. It is public domain; this reads the Project
Madurai etext (pmuni0100), whose licence asks only that the attribution header be
kept — which this script preserves in the output.

Source (UTF-8, public domain): the Project Madurai GitHub mirror
`project-madurai/pm-repo-html`, file `html/pmuni0100.html`. The extractor accepts
either a local HTML path (`--html`) or a URL (`--url`, default: the mirror raw URL),
so it works offline once the HTML is saved.

Outputs (under --out):
  * `tholkappiyam_moolam.txt`  — one sutra per line, cleaned + NFC-normalized, with a
    retained Project Madurai attribution header (corpus / tokenizer training text)
  * `tholkappiyam.jsonl`       — structured records {book, iyal, sutra_no, text}

    python scripts/extract_tholkappiyam.py --out data/raw/classical/tholkappiyam
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

MIRROR_RAW = ("https://raw.githubusercontent.com/project-madurai/pm-repo-html/"
              "master/html/pmuni0100.html")

_TAMIL = re.compile(r"[஀-௿]")           # any Tamil-block char
_BARE_NUM = re.compile(r"^\d+$")                   # a lone sutra-number line
_IYAL_HEAD = re.compile(r"^\d+\.\s*\S")            # "2.   மொழி மரபு"
_LATIN_HEAVY = re.compile(r"[A-Za-z]")

# The three books (adhikarams); match on the distinctive Tamil name.
_BOOKS = [
    ("eluttatikaram", "எழுத்ததிகாரம்"),
    ("sollatikaram", "சொல்லதிகாரம்"),
    ("porulatikaram", "பொருளதிகாரம்"),
]

_ATTRIBUTION = (
    "# தொல்காப்பியம் — Tholkappiyam moolam (தொல்காப்பியர்)\n"
    "# Public-domain etext via Project Madurai (pmuni0100); etext prep: "
    "Dr. K. Kalyanasundaram; web version: N.D. Logasundaram.\n"
    "# Redistributed under Project Madurai's terms (attribution header kept intact).\n"
    "# Cleaned + NFC-normalized by scripts/extract_tholkappiyam.py — one sutra per line.\n"
)


def _load_html(html: Optional[str], url: str) -> str:
    if html:
        return Path(html).read_text(encoding="utf-8", errors="ignore")
    req = urllib.request.Request(url, headers={"User-Agent": "adhan-slm/extractor"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # honours HTTPS_PROXY env
        return resp.read().decode("utf-8", errors="ignore")


def _html_to_lines(html: str) -> List[str]:
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = text.replace("&nbsp;", " ")
    text = unicodedata.normalize("NFC", text)
    return [ln.strip() for ln in text.splitlines()]


def parse(html: str) -> List[Dict]:
    """Return structured sutra records with (book, iyal, sutra_no, text)."""
    lines = _html_to_lines(html)
    records: List[Dict] = []
    book = None
    iyal = None
    verse: List[str] = []
    started = False  # skip the English/prep header until the first book marker

    def flush(sutra_no: Optional[int]):
        nonlocal verse
        if verse and book:
            records.append({
                "book": book,
                "iyal": iyal,
                "sutra_no": sutra_no,
                "text": " ".join(verse).strip(),
            })
        verse = []

    for ln in lines:
        if not ln:
            continue
        # book boundary?
        matched_book = next((bid for bid, name in _BOOKS if name in ln), None)
        if matched_book and (_TAMIL.search(ln) and len(ln) < 80):
            flush(None)
            book, iyal, started = matched_book, None, True
            continue
        if not started:
            continue
        # iyal (chapter) header: "2.   மொழி மரபு"
        if _IYAL_HEAD.match(ln) and _TAMIL.search(ln):
            flush(None)
            iyal = ln
            continue
        # a lone number terminates the current sutra
        if _BARE_NUM.match(ln):
            flush(int(ln))
            continue
        # skip lines with no Tamil, or Latin-heavy residue from the header/footer
        if not _TAMIL.search(ln):
            continue
        if _LATIN_HEAVY.search(ln) and len(_LATIN_HEAVY.findall(ln)) > 3:
            continue
        verse.append(ln)
    flush(None)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract Tholkappiyam moolam as clean text")
    ap.add_argument("--url", default=MIRROR_RAW, help="source HTML URL (PM mirror)")
    ap.add_argument("--html", help="local HTML path (offline; overrides --url)")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] loading {'local ' + args.html if args.html else args.url} ...")
    html = _load_html(args.html, args.url)
    print(f"[2/3] parsing ({len(html):,} chars) ...")
    records = parse(html)
    if not records:
        sys.exit("no sutras parsed — source layout may have changed")

    by_book: Dict[str, int] = {}
    for r in records:
        by_book[r["book"]] = by_book.get(r["book"], 0) + 1

    txt_path = out / "tholkappiyam_moolam.txt"
    jsonl_path = out / "tholkappiyam.jsonl"
    txt_path.write_text(
        _ATTRIBUTION + "\n" + "\n".join(r["text"] for r in records) + "\n",
        encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_chars = sum(len(r["text"]) for r in records)
    print(f"[3/3] wrote {len(records)} sutras ({total_chars:,} chars)")
    for bid, _ in _BOOKS:
        print(f"        {bid:16s} {by_book.get(bid, 0)} sutras")
    print(f"      -> {txt_path}\n      -> {jsonl_path}")


if __name__ == "__main__":
    main()
