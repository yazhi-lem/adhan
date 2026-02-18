#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF Book Scraper for Tamil public-domain sources
- Targets domains (Project Madurai, TamilVu, Sangam.org by default)
- Reuses TamilCorpusScraper to find/download/extract PDFs from sitemaps
- Detects legacy encodings (TSCII / Mylai) and converts to Unicode
- Optionally performs machine translation (Tamil -> English) using googletrans (best-effort)
- Writes JSONL and per-work manifests under `data/raw/pdf_books_manifests`

Example:
  python src/data_scraper/pdf_book_scraper.py --domains projectmadurai.org,tamilvu.org --limit 20 --translate

"""
from __future__ import annotations
import argparse
import json
import os
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Optional

from tamil_corpus_scraper import TamilCorpusScraper

OUT_DIR = Path("data/raw/pdf_books_manifests")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def looks_like_tscii(s: str) -> bool:
    if not s:
        return False
    low = s[:4096].lower()
    if any(tok in low for tok in ('tscii', 'x-user-defined', 'x-tscii', 'mylai', 'mylai font', 'mylai format')):
        return True
    cnt = sum(1 for ch in s if 0x80 <= ord(ch) <= 0xFF)
    if len(s) > 200 and (cnt / len(s)) > 0.02:
        return True
    if any(ch in s for ch in ('¾', '¢', 'Õ', 'Ã', 'õ', 'þ', '±')):
        return True
    return False


def tscii_to_unicode(s: str) -> str:
    try:
        import tamil
        if hasattr(tamil, 'tscii') and hasattr(tamil.tscii, 'convert_to_unicode'):
            return tamil.tscii.convert_to_unicode(s)
        if hasattr(tamil, 'txt2unicode') and hasattr(tamil.txt2unicode, 'tscii2unicode'):
            return tamil.txt2unicode.tscii2unicode(s)
    except Exception:
        pass
    return s


def maybe_convert_tscii(s: str) -> str:
    return tscii_to_unicode(s) if looks_like_tscii(s) else s


def try_translate_ta_to_en(text: str) -> Optional[str]:
    """Best-effort translation (uses googletrans if installed)."""
    if not text or len(text.strip()) < 5:
        return None
    try:
        from googletrans import Translator
    except Exception:
        return None
    try:
        tr = Translator()
        out = tr.translate(text, src='ta', dest='en')
        return out.text
    except Exception:
        return None


def build_and_save(records: List[Dict], out_dir: Path, domain: str):
    # write domain JSONL
    domain_file = out_dir / f"{domain}.jsonl"
    with domain_file.open('w', encoding='utf-8') as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[INFO] Wrote {len(records)} records to {domain_file}")

    # group by inferred work id (use stem of pdf file or url path)
    groups: Dict[str, List[Dict]] = {}
    for r in records:
        stem = None
        if r.get('raw_pdf_path'):
            stem = Path(r['raw_pdf_path']).stem
        else:
            stem = Path(r.get('url','')).stem
        wid = stem or 'unknown'
        groups.setdefault(wid, []).append(r)

    manifest_index = {}
    for wid, items in groups.items():
        combined = "\n\n".join([it.get('text','') for it in items if it.get('text')])
        manifest_index[wid] = {
            'work_id': wid,
            'n_pages': len(items),
            'char_count': sum(it.get('char_count',0) for it in items),
            'records': [{ 'id': it.get('id'), 'raw_pdf_path': it.get('raw_pdf_path'), 'char_count': it.get('char_count', 0) } for it in items]
        }
        with (out_dir / f"{wid}.jsonl").open('w', encoding='utf-8') as wf:
            for it in items:
                wf.write(json.dumps(it, ensure_ascii=False) + "\n")
        with (out_dir / f"{wid}.txt").open('w', encoding='utf-8') as tf:
            tf.write(combined)

    # write per-domain manifest index
    with (out_dir / f"{domain}_manifest_index.json").open('w', encoding='utf-8') as mf:
        json.dump(manifest_index, mf, ensure_ascii=False, indent=2)

    return len(records), len(groups)


def scrape_domains(domains: List[str], limit: Optional[int] = None, translate: bool = False) -> Dict[str, Dict]:
    scraper = TamilCorpusScraper(base_dir='data/raw')
    summary = {}

    for d in domains:
        print(f"[START] domain={d} limit={limit} translate={translate}")
        try:
            recs = scraper.extract_pdfs_from_sitemap(d, limit=(limit or 0))
        except Exception as e:
            print(f"[ERROR] extract_pdfs_from_sitemap failed for {d}: {e}")
            recs = []

        processed = []
        for r in recs:
            text = r.get('text', '') or ''
            # detect/convert legacy encodings
            if looks_like_tscii(text):
                text_conv = tscii_to_unicode(text)
                if text_conv != text:
                    print(f"[INFO] Converted legacy encoding for {r.get('url')}")
                    text = text_conv
            # fallback: check text_path if present
            if not text and r.get('text_path'):
                try:
                    raw = Path(r['text_path']).read_text(encoding='utf-8', errors='ignore')
                    if looks_like_tscii(raw):
                        raw = tscii_to_unicode(raw)
                    text = raw
                except Exception:
                    pass

            r['text'] = text
            r['char_count'] = len(text)
            r['word_count'] = len(text.split())

            # translate if requested and translator available
            if translate:
                en = try_translate_ta_to_en(text)
                if en:
                    r['en_text'] = en
                else:
                    r['en_text'] = None

            # ensure id
            if not r.get('id'):
                r['id'] = sha256_hex((r.get('text') or '') + (r.get('url') or ''))

            processed.append(r)

        nrec, nworks = build_and_save(processed, OUT_DIR, d.replace('.', '_'))
        summary[d] = {'records': nrec, 'works': nworks}
        print(f"[DONE] domain={d} records={nrec} works={nworks}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="PDF book scraper for Tamil public-domain sites")
    parser.add_argument('--domains', type=str, default='projectmadurai.org', help='Comma-separated domains to scan')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of PDFs per-domain (0=no limit)')
    parser.add_argument('--translate', action='store_true', help='Attempt Tamil->English translation (best-effort)')
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(',') if d.strip()]
    res = scrape_domains(domains, limit=args.limit, translate=args.translate)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
