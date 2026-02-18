#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMWorks extractor — build a sanitized Project Madurai library from `pmworks.html`.
Streaming, minimal parsing for fast text extraction and manifest generation.

Outputs:
 - JSONL of per-page records: `data/raw/projectmadurai_manifests/pmworks_records.jsonl`
 - Per-work JSONL + combined `.txt` under `data/raw/projectmadurai_manifests/`

Usage:
    python src/data_scraper/pmworks_extractor.py

"""
import argparse
import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Optional

from bs4 import BeautifulSoup

BASE_RAW = Path("data/raw/raw_html/projectmadurai.org")
PMWORKS_HTML = BASE_RAW / "pmworks.html"
OUT_MANIFEST_DIR = Path("data/raw/projectmadurai_manifests")
OUT_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def parse_pmworks_table(html_path: Path) -> List[Dict]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    table = soup.find("table", {"id": "sortabletable"})
    if not table:
        raise RuntimeError(f"Could not find table with id=sortabletable in {html_path}")

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 6:
            continue
        work_no = tds[0].get_text(strip=True)
        title = tds[1].get_text(strip=True)
        author = tds[2].get_text(strip=True)
        genre = tds[3].get_text(strip=True)

        pdf_links = [a.get("href") for a in tds[4].find_all("a")]
        html_links = [a.get("href") for a in tds[5].find_all("a")]

        rows.append({
            "work_no": work_no,
            "title": title,
            "author": author,
            "genre": genre,
            "pdf_links": pdf_links,
            "html_links": html_links,
        })
    return rows


def resolve_local_path(href: str) -> Optional[Path]:
    if not href:
        return None
    rel = href.lstrip("/")
    candidates = [BASE_RAW / rel, BASE_RAW.parent / rel, Path("data/raw") / rel]
    for c in candidates:
        if c.exists():
            return c
    return Path("data/raw/raw_html/projectmadurai.org") / rel


def extract_pdf_text_if_possible(pdf_path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text
    except Exception:
        return ""
    try:
        txt = extract_text(str(pdf_path))
    except Exception:
        return ""

    # If PDF-extracted text looks like legacy TSCII, convert to Unicode
    try:
        if looks_like_tscii(txt):
            txt = tscii_to_unicode(txt)
    except Exception:
        pass

    return txt


# --- fast streaming HTML text extraction (avoid heavy parsing) ---
def fast_extract_text(html: str, max_chars: int = 50000) -> str:
    """Extract text from HTML with minimal parsing. Cap large documents."""
    if not html:
        return ""
    
    # Truncate if too large (avoid regex backtracking)
    if len(html) > max_chars * 10:
        html = html[:max_chars * 10]
    
    try:
        soup = BeautifulSoup(html[:max_chars * 10], 'html.parser', features='html.parser')
    except Exception:
        # fallback: raw strip
        return re.sub(r'<[^>]+>', ' ', html)
    
    # decompose script/style/etc
    for el in soup.find_all(['script', 'style', 'svg', 'noscript']):
        el.decompose()
    
    # extract text
    text = soup.get_text(separator=' ', strip=True)
    
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # cap final result
    if len(text) > max_chars:
        text = text[:max_chars]
    
    return text


# --- TSCII detection & conversion helpers -------------------------------------
def looks_like_tscii(s: str) -> bool:
    """Heuristic to detect legacy TSCII / Mylai content.

    Returns True if the input contains explicit legacy markers (e.g. 'tscii', 'mylai')
    or a suspicious fraction of high-Latin (0x80-0xFF) glyphs that indicate
    legacy encodings.
    """
    if not s:
        return False
    low = s[:4096].lower()
    # explicit markers
    if any(tok in low for tok in ('tscii', 'x-user-defined', 'x-tscii', 'tscii-encoding', 'mylai', 'mylai font', 'mylai format')):
        return True

    # quick high-latin / non-utf glyph ratio check
    cnt = sum(1 for ch in s if 0x80 <= ord(ch) <= 0xFF)
    if len(s) > 200 and (cnt / len(s)) > 0.02:
        return True

    # common glyphs that appear in TSCII/Mylai dumps
    if any(ch in s for ch in ('¾', '¢', 'Õ', 'Ã', 'õ', 'þ', '±')):
        return True

    return False


def tscii_to_unicode(s: str) -> str:
    """Convert TSCII-encoded text to Unicode using `tamil` if available."""
    try:
        import tamil
        # preferred: tamil.tscii.convert_to_unicode
        if hasattr(tamil, 'tscii') and hasattr(tamil.tscii, 'convert_to_unicode'):
            return tamil.tscii.convert_to_unicode(s)
        # fallback
        if hasattr(tamil, 'txt2unicode') and hasattr(tamil.txt2unicode, 'tscii2unicode'):
            return tamil.txt2unicode.tscii2unicode(s)
    except Exception as e:
        print(f"[WARN] TSCII->Unicode conversion failed: {e}", flush=True)
    return s


def maybe_convert_tscii(s: str) -> str:
    """Convert to Unicode only if text looks like TSCII."""
    return tscii_to_unicode(s) if looks_like_tscii(s) else s


def build_library(pmworks_html: Path, out_dir: Path, limit: Optional[int] = None, skip_pdf: bool = False) -> Dict:
    rows = parse_pmworks_table(pmworks_html)
    records = []

    for i, row in enumerate(rows):
        if limit and i >= limit:
            break
        if (i + 1) % 100 == 0:
            print(f"[INFO] Processing row {i+1}/{len(rows)}...", flush=True)
        
        work_no = row["work_no"]
        title = row["title"]
        author = row["author"]
        genre = row["genre"]

        pdf_paths = []
        for pdf in row["pdf_links"]:
            p = resolve_local_path(pdf)
            if p:
                pdf_paths.append(str(p))

        html_records = []
        for href in row["html_links"]:
            lp = resolve_local_path(href)
            if not lp or not lp.exists():
                rec = {
                    "work_no": work_no,
                    "title": title,
                    "author": author,
                    "genre": genre,
                    "html_href": href,
                    "html_path": None,
                    "text": "",
                }
                html_records.append(rec)
                continue

            try:
                # read raw bytes then decode according to declared charset (preserve TSCII bytes)
                raw_bytes = lp.read_bytes()
            except Exception as e:
                print(f"[WARN] Could not read {lp}: {e}", flush=True)
                html_records.append({
                    "work_no": work_no,
                    "title": title,
                    "author": author,
                    "genre": genre,
                    "html_href": href,
                    "html_path": str(lp),
                    "text": "",
                })
                continue

            # inspect declared charset in the HTML head
            head_snip = raw_bytes[:4096]
            declared = ''
            m = re.search(rb'charset\s*=\s*"?([^"\' >]+)', head_snip, flags=re.I)
            if m:
                try:
                    declared = m.group(1).decode('ascii', errors='ignore').lower()
                except Exception:
                    declared = ''

            # decode bytes: if TSCII/x-user-defined declared, use latin-1 to preserve 0x80-0xFF
            if 'x-user-defined' in declared or 'tscii' in declared or 'x-tscii' in declared:
                raw_html = raw_bytes.decode('latin-1', errors='ignore')
            else:
                try:
                    raw_html = raw_bytes.decode('utf-8')
                except Exception:
                    raw_html = raw_bytes.decode('latin-1', errors='ignore')

            # convert legacy TSCII to Unicode if detected
            raw_html = maybe_convert_tscii(raw_html)

            clean_text = fast_extract_text(raw_html, max_chars=50000)
            
            rec = {
                "work_no": work_no,
                "title": title,
                "author": author,
                "genre": genre,
                "html_href": href,
                "html_path": str(lp),
                "text": clean_text,
                "char_count": len(clean_text),
                "word_count": len(clean_text.split()),
                "id": sha256_hex(clean_text + str(lp))
            }
            html_records.append(rec)

        # Try extracting text from PDFs (best-effort, quick timeout)
        pdf_texts = {}
        for pstr in pdf_paths:
            ppath = Path(pstr)
            if ppath.exists():
                txt = extract_pdf_text_if_possible(ppath)
                if txt:
                    pdf_texts[str(ppath)] = txt

        # Flatten per-html-records into global records
        for hr in html_records:
            r = {
                "id": hr.get("id") or sha256_hex((hr.get("text") or hr.get("html_href", "")) + title + author),
                "work_no": work_no,
                "title": title,
                "author": author,
                "genre": genre,
                "pdf_paths": pdf_paths,
                "html_path": hr.get("html_path"),
                "html_href": hr.get("html_href"),
                "text": hr.get("text"),
                "char_count": hr.get("char_count", 0),
                "word_count": hr.get("word_count", 0),
                "source": "projectmadurai.org",
            }
            if pdf_texts:
                r["pdf_texts"] = pdf_texts
            records.append(r)

    # Save master JSONL
    out_file = out_dir / "pmworks_records.jsonl"
    print(f"[INFO] Writing {len(records)} records to {out_file}...", flush=True)
    with out_file.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Build per-work manifests + combined .txt
    groups = {}
    for rec in records:
        try:
            wid = f"pm{int(rec['work_no']):04d}"
        except (ValueError, TypeError):
            wid = Path(rec.get("html_path") or "").stem or rec.get("work_no") or "unknown"
        groups.setdefault(wid, []).append(rec)

    print(f"[INFO] Writing {len(groups)} per-work manifests...", flush=True)
    manifest_index = {}
    for j, (wid, items) in enumerate(groups.items()):
        if (j + 1) % 50 == 0:
            print(f"[INFO] Writing manifest {j+1}/{len(groups)}...", flush=True)
        
        combined = "\n\n".join([it.get("text", "") for it in items if it.get("text")])
        manifest_index[wid] = {
            "work_id": wid,
            "n_pages": len(items),
            "char_count": sum(it.get("char_count", 0) for it in items),
            "records": [{"id": it.get("id"), "html_path": it.get("html_path"), "char_count": it.get("char_count", 0)} for it in items]
        }
        # write per-work jsonl & text
        with (out_dir / f"{wid}.jsonl").open("w", encoding="utf-8") as wf:
            for it in items:
                wf.write(json.dumps(it, ensure_ascii=False) + "\n")
        with (out_dir / f"{wid}.txt").open("w", encoding="utf-8") as tf:
            tf.write(combined)

    # write manifest index
    with (out_dir / "manifest_index.json").open("w", encoding="utf-8") as mf:
        json.dump(manifest_index, mf, ensure_ascii=False, indent=2)

    print(f"[INFO] Done! Created {len(records)} records from {len(groups)} works.", flush=True)
    return {"records": len(records), "works": len(groups), "out_file": str(out_file)}


def main():
    parser = argparse.ArgumentParser(description="Extract Project Madurai works from pmworks.html and build sanitized library")
    parser.add_argument("--input", default=str(PMWORKS_HTML), help="Path to pmworks.html")
    parser.add_argument("--out", default=str(OUT_MANIFEST_DIR), help="Output manifest directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of works to process (for testing)")
    args = parser.parse_args()

    res = build_library(Path(args.input), Path(args.out), limit=args.limit)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
