#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Twitter/X Tamil corpus collector.

Searches Twitter/X public search pages for Tamil-dominant tweets using
hashtags and keywords.  Uses only the public ``twitter.com/search`` endpoint
with a standard browser ``User-Agent``; **no API key or OAuth token required**.

Rate-limit safety:
- A configurable delay is inserted between every HTTP request.
- Exponential back-off on HTTP 429 / 5xx responses.
- Hard cap on the number of requests per run to avoid bans.

Output:
    <output-dir>/twitter_tamil.jsonl   – one JSON record per line
    <output-dir>/twitter_manifest.json – run statistics

Usage:
    python src/data_scraper/raw_extractors/twitter_scraper.py \
        --output-dir data/raw/twitter \
        --max-requests 100
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────

SEARCH_KEYWORDS = [
    "#தமிழ்",
    "#tamil",
    "#tamilnadu",
    "#kollywood",
    "site:twitter.com tamil",
    "lang:ta",
]

# Nitter is a privacy-respecting Twitter front-end with a more accessible
# search API.  We fall back gracefully if the instance is unavailable.
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
]
SEARCH_PATH = "/search?f=tweets&q={query}&lang=ta"

TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")

MIN_TAMIL_FRACTION = 0.25
MIN_TEXT_LEN = 15
MAX_TEXT_LEN = 1000

REQUEST_DELAY = 3.0  # seconds between requests
MAX_BACKOFF = 120    # seconds


# ── helpers ────────────────────────────────────────────────────────────────────

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def tamil_fraction(text: str) -> float:
    if not text:
        return 0.0
    return len(TAMIL_RE.findall(text)) / len(text)


def clean_text(text: str) -> str:
    # strip URLs
    text = re.sub(r"https?://\S+", "", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_acceptable(text: str) -> bool:
    if not text or len(text) < MIN_TEXT_LEN or len(text) > MAX_TEXT_LEN:
        return False
    return tamil_fraction(text) >= MIN_TAMIL_FRACTION


def fetch_html(url: str, retries: int = 4, delay: float = REQUEST_DELAY) -> Optional[str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "ta,en-US;q=0.9,en;q=0.8",
    }
    req = Request(url, headers=headers)
    backoff = delay
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=20) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                logger.warning("HTTP %s – backing off %.0fs …", exc.code, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
            else:
                logger.debug("HTTP %s for %s – skipping", exc.code, url)
                return None
        except URLError as exc:
            logger.warning("URL error: %s (attempt %d)", exc, attempt + 1)
            time.sleep(backoff)
    return None


# ── tweet extraction ───────────────────────────────────────────────────────────

# Pattern to extract tweet text from Nitter HTML.
# Nitter wraps tweet content in <div class="tweet-content …">…</div>.
_TWEET_CONTENT_RE = re.compile(
    r'<div class="tweet-content[^"]*"[^>]*>(.*?)</div>',
    re.DOTALL,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def extract_tweets_from_html(html: str, source_url: str) -> list[dict]:
    records = []
    for match in _TWEET_CONTENT_RE.finditer(html):
        raw = _HTML_TAG_RE.sub(" ", match.group(1))
        text = clean_text(raw)
        if is_acceptable(text):
            h = sha256_hex(text)
            records.append({
                "id": h,
                "text": text,
                "source": "twitter",
                "url": source_url,
                "tamil_fraction": round(tamil_fraction(text), 3),
                "record_type": "tweet",
            })
    return records


# ── main collection loop ───────────────────────────────────────────────────────

def collect(
    keywords: list[str],
    max_requests: int,
    output_dir: Path,
) -> list[dict]:
    seen: set[str] = set()
    records: list[dict] = []
    request_count = 0

    for instance in NITTER_INSTANCES:
        if request_count >= max_requests:
            break
        # Quick health check
        health_html = fetch_html(instance)
        if health_html is None:
            logger.info("Nitter instance %s is unavailable – trying next.", instance)
            continue
        logger.info("Using Nitter instance: %s", instance)

        for kw in keywords:
            if request_count >= max_requests:
                break
            query = quote_plus(kw)
            url = instance + SEARCH_PATH.format(query=query)
            logger.info("  Fetching: %s", url)
            html = fetch_html(url)
            request_count += 1
            time.sleep(REQUEST_DELAY)

            if not html:
                continue

            new_records = extract_tweets_from_html(html, url)
            for rec in new_records:
                if rec["id"] not in seen:
                    seen.add(rec["id"])
                    records.append(rec)

        # One instance is usually sufficient; break after success.
        break

    if not records:
        logger.warning(
            "No Tamil tweets collected. Nitter may be unavailable. "
            "Try again later or use the Twitter API with a bearer token."
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Tamil tweets via Nitter (no API key required)."
    )
    parser.add_argument("--output-dir", default="data/raw/twitter",
                        help="Directory to write output files.")
    parser.add_argument("--keywords", default=",".join(SEARCH_KEYWORDS),
                        help="Comma-separated search keywords.")
    parser.add_argument("--max-requests", type=int, default=100,
                        help="Hard cap on total HTTP requests per run.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kws = [k.strip() for k in args.keywords.split(",") if k.strip()]
    records = collect(kws, max_requests=args.max_requests, output_dir=out_dir)

    out_file = out_dir / "twitter_tamil.jsonl"
    with out_file.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "keywords": kws,
        "max_requests": args.max_requests,
        "total_records": len(records),
        "output_file": str(out_file),
    }
    with (out_dir / "twitter_manifest.json").open("w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

    logger.info("Done. Wrote %d records to %s", len(records), out_file)


if __name__ == "__main__":
    main()
