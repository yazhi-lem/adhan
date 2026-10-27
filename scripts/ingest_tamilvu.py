#!/usr/bin/env python3
"""Ingest Tamil Virtual Academy (tamilvu.org) digital library.

tamilvu.org publishes a large HTML library: Sangam works with commentary,
grammar texts, dictionaries, encyclopedias, modern literature and folklore.
The catalogue lives at ``/ta/library-libcontnt-273141`` and each work is a
frameset whose left frame (``/slet/<code>/<code>lft.jsp``) lists every chapter,
each linking to ``<code>son.jsp`` (verses) and ``<code>uri.jsp`` (commentary).

This crawler:
  * starts from the catalogue and extracts all ``/ta/library-...`` work pages,
  * resolves each work's left-frame TOC and chapter/commentary URLs,
  * fetches pages politely (configurable delay), with a wall-clock budget,
  * strips site chrome and keeps Tamil prose,
  * appends JSONL records and resumes from a state file.

Usage:
    python scripts/ingest_tamilvu.py --out data/raw/web/tamilvu.jsonl \
        --state data/raw/web/tamilvu_state.json \
        --time-budget 1800 --max-pages-per-book 80 --delay 0.25
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Iterator, List, Set
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.core.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pip install requests beautifulsoup4") from exc

BASE = "https://www.tamilvu.org"
CATALOGUE = f"{BASE}/ta/library-libcontnt-273141"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AdhanBot/1.0; +https://yazhi.dev/bot)"}

TAMIL_RANGE = re.compile(r"[\u0B80-\u0BFF]")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_RE = re.compile(r"(\+91[\-\s]?)?[6-9]\d{9}")
WS_RE = re.compile(r"[ \t]+")

# Site chrome / boilerplate lines to drop from extracted pages.
BOILERPLATE = (
    "தமிழ் இணையக் கல்விக்கழகம்",
    "TAMIL VIRTUAL ACADEMY",
    "Skip to main content",
    "CONTACT INFO",
    "அண்ணா பல்கலைக்கழக வளாகம்",
    "காந்தி மண்டபச்",
    "தொலைபேசி",
    "இயக்குநர்",
    "தொடக்கம்",
    "த.இ.க. பற்றி",
    "தொடர்பு-கொள்ள",
    "தகவல் பெறும் உரிமை",
    "தள வரைபடம்",
    "இணையத்தள கொள்கைகள்",
    "CopyRight",
    "Powered by",
)

_CHAPTER_RE = re.compile(r"([A-Za-z0-9]+(?:son|uri|se\d|ind|hom)[^\"'\s]*\.jsp[^\"'\s]*)")
_WORK_RE = re.compile(r"/ta/library-[A-Za-z0-9]+-html-[A-Za-z0-9]+-\d+")


def normalize(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = WS_RE.sub(" ", text)
    return text.strip()


def tamil_ratio(text: str) -> float:
    clean = "".join(text.split())
    if not clean:
        return 0.0
    return len(TAMIL_RANGE.findall(clean)) / len(clean)


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()
    body = soup.body or soup
    lines: List[str] = []
    for raw in body.get_text("\n", strip=True).splitlines():
        line = normalize(raw)
        if len(line) < 2:
            continue
        if any(b in line for b in BOILERPLATE):
            continue
        if tamil_ratio(line) < 0.35:
            continue
        lines.append(line)
    return "\n".join(lines)


def fetch(session: requests.Session, url: str, timeout: int = 30) -> str:
    resp = session.get(url, timeout=timeout, headers=HEADERS)
    resp.raise_for_status()
    if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def discover_works(session: requests.Session) -> List[str]:
    html = fetch(session, CATALOGUE)
    paths: Set[str] = set(_WORK_RE.findall(html))
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        m = _WORK_RE.search(urljoin(BASE, a["href"]))
        if m:
            paths.add(m.group(0))
    return sorted({urljoin(BASE, p) for p in paths})


def work_code(work_url: str) -> str:
    m = re.search(r"library-([A-Za-z0-9]+)-html-", work_url)
    return m.group(1) if m else "work"


def discover_chapters(session: requests.Session, work_url: str) -> List[str]:
    """Resolve a work page's left-frame TOC into chapter and commentary URLs."""
    html = fetch(session, work_url)
    urls: Set[str] = set()

    # iframes: /slet/<code>/<code>lft.jsp and the simple content frame
    for src in re.findall(r'<iframe[^>]+src="([^"]+)"', html, flags=re.I):
        src = src.strip().rstrip("\r")
        full = urljoin(BASE, src)
        if "lft.jsp" in full:
            try:
                toc = fetch(session, full)
            except Exception as exc:  # noqa: BLE001
                logger.warning("TOC failed %s: %s", full, exc)
                continue
            for m in _CHAPTER_RE.findall(toc):
                urls.add(urljoin(full, m))
            # commentary links are inside window.open(...) on chapter pages
        elif "renderframe=simple" in full:
            urls.add(full)

    # direct chapter links in the page body (some works are not framed)
    for m in _CHAPTER_RE.findall(html):
        urls.add(urljoin(work_url, m))

    return sorted(urls)


def iter_records(
    session: requests.Session,
    works: List[str],
    state: dict,
    max_pages_per_book: int,
    delay: float,
    time_budget: float,
    with_commentary: bool = False,
    min_chars: int = 200,
) -> Iterator[dict]:
    fetched: Set[str] = set(state.get("fetched", []))
    started = time.time()
    seen_hashes = set(state.get("hashes", []))

    for wi, work in enumerate(works, 1):
        if time.time() - started > time_budget:
            logger.info("time budget reached at work %d/%d", wi, len(works))
            break
        code = work_code(work)
        try:
            chapters = discover_chapters(session, work)
        except Exception as exc:  # noqa: BLE001
            logger.warning("work failed %s: %s", work, exc)
            continue
        logger.info("work %d/%d %s: %d pages", wi, len(works), code, len(chapters))
        pages = 0
        queue = list(chapters)
        qi = 0
        while qi < len(queue) and pages < max_pages_per_book:
            if time.time() - started > time_budget:
                break
            url = queue[qi]
            qi += 1
            if url in fetched:
                continue
            try:
                html = fetch(session, url)
            except Exception as exc:  # noqa: BLE001
                logger.debug("fetch failed %s: %s", url, exc)
                continue
            fetched.add(url)
            time.sleep(delay)

            text = extract_text(html)
            if len(text) >= min_chars and tamil_ratio(text) >= 0.5:
                digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
                if digest not in seen_hashes:
                    seen_hashes.add(digest)
                    pages += 1
                    yield {
                        "text": text,
                        "source": f"tamilvu:{code}",
                        "tier": 1,
                        "quality_score": 0.93,
                        "url": url,
                    }

            # Commentary ("உரை") lives behind a multi-step popup flow
            # (uri.jsp -> author select -> sec.jsp) and is opt-in: on most works
            # it costs several requests per verse for little extracted text.
            if not with_commentary:
                continue
            for m in _CHAPTER_RE.findall(html):
                cand = urljoin(url, m)
                if "uri.jsp" not in cand or cand in fetched:
                    continue
                if pages >= max_pages_per_book or time.time() - started > time_budget:
                    break
                try:
                    uri_html = fetch(session, cand)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("uri fetch failed %s: %s", cand, exc)
                    continue
                fetched.add(cand)
                time.sleep(delay)
                uri_text = extract_text(uri_html)
                if len(uri_text) < 150 or tamil_ratio(uri_text) < 0.5:
                    continue
                d = hashlib.sha1(uri_text.encode("utf-8")).hexdigest()
                if d in seen_hashes:
                    continue
                seen_hashes.add(d)
                pages += 1
                yield {
                    "text": uri_text,
                    "source": f"tamilvu:{code}:urai",
                    "tier": 1,
                    "quality_score": 0.94,
                    "url": cand,
                }

        state["fetched"] = sorted(fetched)
        state["hashes"] = sorted(seen_hashes)[-200000:]


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest the tamilvu.org digital library")
    ap.add_argument("--out", default="data/raw/web/tamilvu.jsonl")
    ap.add_argument("--state", default="data/raw/web/tamilvu_state.json")
    ap.add_argument("--max-pages-per-book", type=int, default=80)
    ap.add_argument("--delay", type=float, default=0.25)
    ap.add_argument("--time-budget", type=float, default=1800.0, help="seconds")
    ap.add_argument("--max-books", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--only-codes",
        default="",
        help="comma-separated work codes to crawl (e.g. l1100,l2100,l3100)",
    )
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument(
        "--with-commentary",
        action="store_true",
        help="also crawl uri.jsp commentary pages (slower, flakier)",
    )
    ap.add_argument("--schema-scan-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    fetched_before = len(state.get("fetched", []))

    session = requests.Session()
    works = discover_works(session)
    if args.only_codes:
        wanted = {c.strip() for c in args.only_codes.split(",") if c.strip()}
        works = [w for w in works if work_code(w) in wanted]
        wanted_order = [c.strip() for c in args.only_codes.split(",") if c.strip()]
        works.sort(
            key=lambda w: wanted_order.index(work_code(w)) if work_code(w) in wanted_order else 999
        )
    if args.max_books:
        works = works[: args.max_books]
    logger.info("discovered %d works", len(works))
    if args.schema_scan_only:
        for w in works[:20]:
            print(w)
        return

    n = 0
    try:
        with out.open("a", encoding="utf-8") as fh:
            for rec in iter_records(
                session,
                works,
                state,
                args.max_pages_per_book,
                args.delay,
                args.time_budget,
                with_commentary=args.with_commentary,
                min_chars=args.min_chars,
            ):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n % 200 == 0:
                    fh.flush()
                    state_path.write_text(json.dumps(state, indent=2))
                    logger.info("  wrote %d new pages (%d urls fetched)", n, len(state["fetched"]))
    finally:
        state_path.write_text(json.dumps(state, indent=2))
    logger.info(
        "done: %d new pages; %d urls fetched this run (was %d)",
        n,
        len(state.get("fetched", [])) - fetched_before,
        fetched_before,
    )


if __name__ == "__main__":
    main()
