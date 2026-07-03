#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reddit Tamil corpus scraper.

Collects Tamil-dominant posts and comments from Tamil-relevant subreddits
(r/tamil, r/tamilnadu, r/kollywood, r/Chennai) using the public Reddit JSON
API (no credentials required).  Each record is deduplicated by SHA-256 hash
and written to a JSONL output file.

Usage:
    python src/data_scraper/raw_extractors/reddit_scraper.py \
        --output-dir data/raw/reddit \
        --max-posts 500

Output:
    <output-dir>/reddit_tamil.jsonl   – one JSON record per line
    <output-dir>/reddit_manifest.json – run statistics
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Generator, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────

SUBREDDITS = ["tamil", "tamilnadu", "kollywood", "Chennai"]

REDDIT_BASE = "https://www.reddit.com"
LISTING_URL = "{base}/r/{sub}/{sort}.json?limit={limit}&after={after}&t=all"

TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")

MIN_TAMIL_FRACTION = 0.30
MIN_TEXT_LEN = 20
MAX_TEXT_LEN = 2000

HEADERS = {
    "User-Agent": "AdhanTamilCorpus/1.0 (research; +https://github.com/yazhi-lem/adhan)",
}

REQUEST_DELAY = 2.0  # seconds between requests to stay within rate limits


# ── helpers ────────────────────────────────────────────────────────────────────

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def tamil_fraction(text: str) -> float:
    if not text:
        return 0.0
    tamil_chars = len(TAMIL_RE.findall(text))
    return tamil_chars / len(text)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def is_acceptable(text: str) -> bool:
    if not text or len(text) < MIN_TEXT_LEN or len(text) > MAX_TEXT_LEN:
        return False
    frac = tamil_fraction(text)
    return frac >= MIN_TAMIL_FRACTION


def fetch_json(url: str, retries: int = 3) -> Optional[dict]:
    req = Request(url, headers=HEADERS)
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 429:
                wait = 60 * (attempt + 1)
                logger.warning("Rate-limited; waiting %ds …", wait)
                time.sleep(wait)
            elif exc.code in (403, 404):
                logger.debug("HTTP %s for %s – skipping", exc.code, url)
                return None
            else:
                logger.warning("HTTP %s for %s (attempt %d)", exc.code, url, attempt + 1)
                time.sleep(REQUEST_DELAY * 2)
        except URLError as exc:
            logger.warning("URL error for %s: %s (attempt %d)", url, exc, attempt + 1)
            time.sleep(REQUEST_DELAY * 2)
    return None


# ── listing fetchers ───────────────────────────────────────────────────────────

def iter_posts(subreddit: str, sort: str = "top", batch: int = 100,
               max_posts: int = 500) -> Generator[dict, None, None]:
    """Yield raw post dicts from a subreddit listing."""
    after = ""
    fetched = 0
    while fetched < max_posts:
        limit = min(batch, max_posts - fetched)
        url = LISTING_URL.format(
            base=REDDIT_BASE, sub=subreddit, sort=sort,
            limit=limit, after=after,
        )
        data = fetch_json(url)
        if not data:
            break

        children = data.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            post = child.get("data", {})
            if post:
                yield post
                fetched += 1

        after = data.get("data", {}).get("after") or ""
        if not after:
            break
        time.sleep(REQUEST_DELAY)


def iter_comments(post_id: str, subreddit: str,
                  max_comments: int = 50) -> Generator[dict, None, None]:
    """Yield raw comment dicts for a single post."""
    url = f"{REDDIT_BASE}/r/{subreddit}/comments/{post_id}.json?limit={max_comments}"
    data = fetch_json(url)
    if not data or not isinstance(data, list) or len(data) < 2:
        return

    def walk(node: dict) -> Generator[dict, None, None]:
        kind = node.get("kind", "")
        if kind == "t1":
            yield node.get("data", {})
        replies = (node.get("data") or {}).get("replies") or {}
        if isinstance(replies, dict):
            for child in replies.get("data", {}).get("children", []):
                yield from walk(child)

    for child in data[1].get("data", {}).get("children", []):
        yield from walk(child)


# ── main extraction ────────────────────────────────────────────────────────────

def extract_records(
    subreddits: list[str],
    max_posts_per_sub: int,
    include_comments: bool,
    max_comments_per_post: int,
) -> list[dict]:
    seen: set[str] = set()
    records: list[dict] = []

    for sub in subreddits:
        logger.info("Scraping r/%s …", sub)
        post_count = 0

        for post in iter_posts(sub, sort="top", max_posts=max_posts_per_sub):
            # --- post title + selftext ---
            for field in ("title", "selftext"):
                raw = clean_text(post.get(field) or "")
                if is_acceptable(raw):
                    h = sha256_hex(raw)
                    if h not in seen:
                        seen.add(h)
                        records.append({
                            "id": h,
                            "text": raw,
                            "source": "reddit",
                            "subreddit": sub,
                            "post_id": post.get("id"),
                            "url": f"https://reddit.com{post.get('permalink', '')}",
                            "tamil_fraction": round(tamil_fraction(raw), 3),
                            "upvotes": post.get("score", 0),
                            "record_type": "post_" + field,
                        })

            # --- comments ---
            if include_comments:
                post_id = post.get("id")
                if post_id:
                    time.sleep(REQUEST_DELAY)
                    for comment in iter_comments(post_id, sub,
                                                 max_comments=max_comments_per_post):
                        body = clean_text(comment.get("body") or "")
                        if body in ("[deleted]", "[removed]", ""):
                            continue
                        if is_acceptable(body):
                            h = sha256_hex(body)
                            if h not in seen:
                                seen.add(h)
                                records.append({
                                    "id": h,
                                    "text": body,
                                    "source": "reddit",
                                    "subreddit": sub,
                                    "post_id": post_id,
                                    "url": f"https://reddit.com{post.get('permalink', '')}",
                                    "tamil_fraction": round(tamil_fraction(body), 3),
                                    "upvotes": comment.get("score", 0),
                                    "record_type": "comment",
                                })

            post_count += 1

        logger.info("  r/%s: processed %d posts, %d records so far",
                    sub, post_count, len(records))

    return records


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Tamil content from Reddit subreddits."
    )
    parser.add_argument("--output-dir", default="data/raw/reddit",
                        help="Directory to write output files.")
    parser.add_argument("--subreddits", default=",".join(SUBREDDITS),
                        help="Comma-separated list of subreddits to scrape.")
    parser.add_argument("--max-posts", type=int, default=500,
                        help="Maximum posts to fetch per subreddit.")
    parser.add_argument("--include-comments", action="store_true",
                        help="Also collect top comments for each post.")
    parser.add_argument("--max-comments-per-post", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subs = [s.strip() for s in args.subreddits.split(",") if s.strip()]

    records = extract_records(
        subreddits=subs,
        max_posts_per_sub=args.max_posts,
        include_comments=args.include_comments,
        max_comments_per_post=args.max_comments_per_post,
    )

    out_file = out_dir / "reddit_tamil.jsonl"
    with out_file.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "subreddits": subs,
        "max_posts_per_sub": args.max_posts,
        "total_records": len(records),
        "output_file": str(out_file),
    }
    with (out_dir / "reddit_manifest.json").open("w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

    logger.info("Done. Wrote %d records to %s", len(records), out_file)


if __name__ == "__main__":
    main()
