#!/usr/bin/env python3
"""Ingest Tamil Wikisource (ta.wikisource.org) via the MediaWiki API.

Wikisource hosts public-domain Tamil texts (Sangam works, bhakti, grammar,
prose) under CC-BY-SA. Unlike HTML scraping, the API is stable and gives clean
plain text through `prop=extracts&explaintext`.

Strategy: enumerate main-namespace pages with `list=allpages`, then fetch plain
text in batches of 20 titles with `prop=extracts`. Resume from a saved
`apcontinue` token and an already-seen page set.

Usage:
    python scripts/ingest_wikisource_ta.py --out data/raw/web/wikisource_ta.jsonl \
        --state data/raw/web/wikisource_ta_state.json --time-budget 1800
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterator, List, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.core.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pip install requests") from exc

API = "https://ta.wikisource.org/w/api.php"
HEADERS = {"User-Agent": "AdhanCorpusBot/1.0 (https://yazhi.dev; corpus@yazhi.dev)"}
TAMIL_RANGE = re.compile(r"[\u0B80-\u0BFF]")
WS_RE = re.compile(r"[ \t]+")


def tamil_ratio(text: str) -> float:
    clean = "".join(text.split())
    if not clean:
        return 0.0
    return len(TAMIL_RANGE.findall(clean)) / len(clean)


def api_get(params: dict, retries: int = 4) -> dict:
    params = {**params, "format": "json", "formatversion": "2", "maxlag": "5"}
    for attempt in range(retries):
        try:
            # POST, not GET: a batch of long Tamil titles exceeds the URL
            # length limit and returns HTTP 414.
            r = requests.post(API, data=params, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("API error (%d/%d): %s", attempt + 1, retries, exc)
            time.sleep(2 * (attempt + 1))
    return {}


def list_pages(apcontinue: str | None, limit: int = 500) -> tuple[List[str], str | None]:
    params = {
        "action": "query",
        "list": "allpages",
        "apnamespace": "0",
        "aplimit": str(limit),
        "apfilterredir": "nonredirects",
    }
    if apcontinue:
        params["apcontinue"] = apcontinue
    data = api_get(params)
    pages = data.get("query", {}).get("allpages", [])
    cont = data.get("continue", {}).get("apcontinue")
    return [p["title"] for p in pages], cont


def fetch_extracts(titles: List[str]) -> Iterator[tuple[str, str]]:
    data = api_get(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "plain",
            "titles": "|".join(titles),
            "redirects": "1",
        }
    )
    for page in data.get("query", {}).get("pages", []):
        title = page.get("title", "")
        text = page.get("extract", "") or ""
        if text:
            yield title, text


def clean(text: str) -> str:
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = WS_RE.sub(" ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest Tamil Wikisource")
    ap.add_argument("--out", default="data/raw/web/wikisource_ta.jsonl")
    ap.add_argument("--state", default="data/raw/web/wikisource_ta_state.json")
    ap.add_argument("--time-budget", type=float, default=1800.0)
    ap.add_argument("--max-pages", type=int, default=0, help="0 = no cap")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--delay", type=float, default=0.2)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    seen: Set[str] = set(state.get("titles", []))
    apcontinue = state.get("apcontinue")

    started = time.time()
    fetched = 0
    written = 0
    try:
        with out.open("a", encoding="utf-8") as fh:
            while True:
                if time.time() - started > args.time_budget:
                    logger.info("time budget reached")
                    break
                if args.max_pages and fetched >= args.max_pages:
                    logger.info("max pages reached")
                    break
                titles, apcontinue = list_pages(apcontinue)
                if not titles:
                    break
                new_titles = [t for t in titles if t not in seen][:200]
                fetched += len(titles)
                for i in range(0, len(new_titles), 20):
                    batch = new_titles[i : i + 20]
                    for title, raw in fetch_extracts(batch):
                        seen.add(title)
                        text = clean(raw)
                        if len(text) < args.min_chars or tamil_ratio(text) < 0.5:
                            continue
                        fh.write(
                            json.dumps(
                                {
                                    "text": text,
                                    "source": "wikisource-ta",
                                    "tier": 2,
                                    "quality_score": 0.90,
                                    "title": title,
                                    "url": "https://ta.wikisource.org/wiki/"
                                    + title.replace(" ", "_"),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        written += 1
                    time.sleep(args.delay)
                state.update({"apcontinue": apcontinue, "titles": sorted(seen)[-300000:]})
                state_path.write_text(json.dumps(state, indent=2))
                logger.info(
                    "scanned %d titles, wrote %d docs (continue=%s)", fetched, written, apcontinue
                )
                if not apcontinue:
                    logger.info("reached end of allpages")
                    break
    finally:
        state.update({"apcontinue": apcontinue, "titles": sorted(seen)[-300000:]})
        state_path.write_text(json.dumps(state, indent=2))
        logger.info("done: %d docs written", written)


if __name__ == "__main__":
    main()
