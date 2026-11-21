#!/usr/bin/env python3
"""Fetch the frozen `modern` Tamil source for the tokenizer benchmark (adhan#112, K6).

The modern register uses the dump-dated Tamil Wikipedia snapshot mirrored on HuggingFace
(see docs/reports/k6_modern_tamil_source.md). The snapshot config carries the dump date and
the file is pinned by revision, so the extracted JSONL is reproducible byte-for-byte — which
is what lets the corpus manifest record its SHA-256.

    pip install -e ".[eval-corpus]"        # pyarrow, only needed for this step
    python scripts/fetch_modern_tamil.py

    # second shard, or a different pinned snapshot
    python scripts/fetch_modern_tamil.py --shard 1
    python scripts/fetch_modern_tamil.py --config 20231101.ta --revision b04c8d1c

Writes ``data/raw/modern_tamil/wikipedia_ta.jsonl`` (git-ignored) with one
``{"text", "title", "url"}`` object per article, and prints a provenance summary that can be
copied into the manifest review.

The corpus builder stays standard-library only; ``pyarrow`` is imported here and nowhere else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "wikimedia/wikipedia"
DEFAULT_CONFIG = "20231101.ta"
DEFAULT_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
DEFAULT_OUT = "data/raw/modern_tamil/wikipedia_ta.jsonl"

#: Wikimedia dump terms. The 20231101 dump mixes revisions: edits since June 2023 are
#: CC BY-SA 4.0, older revisions are CC BY-SA 3.0, and all of it is additionally GFDL.
LICENSE = (
    "CC BY-SA 4.0 (edits since June 2023) and CC BY-SA 3.0 (older revisions); "
    "all also GFDL (Wikimedia dump terms)"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _import_pyarrow():
    try:
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError:  # pragma: no cover - environment dependent
        print(
            "FAIL: pyarrow is required to read the Wikipedia dump shards.\n"
            '      pip install -e ".[eval-corpus]"',
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    import pyarrow.parquet as pq

    return pq


def fetch_parquet(repo: str, config: str, revision: str, shard: int, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    filename = f"{config}/train-{shard:05d}-of-00002.parquet"
    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            revision=revision,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
    )


def extract(
    parquet_path: Path,
    out: Path,
    text_field: str = "text",
    snapshot: str = DEFAULT_CONFIG,
) -> dict:
    pq = _import_pyarrow()
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = kept = 0
    with pq.ParquetFile(parquet_path) as pf, out.open("w", encoding="utf-8") as fh:
        for batch in pf.iter_batches(batch_size=512):
            for rec in batch.to_pylist():
                rows += 1
                text = (rec.get(text_field) or "").strip()
                if not text:
                    continue
                fh.write(
                    json.dumps(
                        {
                            "text": text,
                            "title": rec.get("title") or "",
                            "url": rec.get("url") or "",
                            # Attribution material: title + page id + dump snapshot identify
                            # the exact revision a sentence came from.
                            "page_id": rec.get("id"),
                            "snapshot": snapshot,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                kept += 1
    return {"articles": rows, "written": kept, "bytes": out.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--shard", type=int, default=0, choices=(0, 1))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--cache-dir", default="data/raw/.hf")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    cache = Path(args.cache_dir)
    if not cache.is_absolute():
        cache = ROOT / cache

    shard_path = fetch_parquet(args.repo, args.config, args.revision, args.shard, cache)
    stats = extract(shard_path, out, snapshot=args.config)

    print(f"source:   {args.repo} config={args.config} revision={args.revision[:8]}")
    print(f"shard:    {shard_path.name}")
    print(f"licence:  {LICENSE}")
    print(f"articles: {stats['articles']:,} (written {stats['written']:,})")
    print(f"output:   {args.out} ({stats['bytes']:,} bytes)")
    print(f"sha256:   {sha256_file(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
