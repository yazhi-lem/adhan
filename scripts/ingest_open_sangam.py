#!/usr/bin/env python3
"""Extract the whole open-sangam world into training data + Tiṇai labels.

The generic JSON walker in `expand_tamil_corpus.py` treats every string field of a
Sangam poem record as its own document. That is how the archive ended up with ~80k
fragmentary `local:*` documents: a poem's `lines` array, its `english` gloss, its
title and its commentary all became separate "documents" of wildly different value.

This extractor is structured instead of generic:

  * `sangamTamil`  -> one document per poem, the verse itself
  * `urai`         -> one document per poem, the **traditional Tamil commentary** —
                     2.4x more tokens than the verses, and the only classical prose
                     in the corpus
  * `yazhi_urai`   -> model-generated commentary, kept but labelled separately
  * `title`/`heading`/`english`/`culturalNotes` -> **excluded**: they are metadata or
                     English, and fragment-level inclusion is what made the old
                     `local:*` sources noisy (see `docs/corpus/SANGAM_INGEST.md`)

**Songs are never merged.** One document is one (song, layer) pair: a verse is never
concatenated with its commentary or with another verse. Sangam poetics is a
four-dimensional label space — tiṇai (landscape), kalam (season), uripporul (essential
action/emotion: புணர்தல், இருத்தல், ஊடல், இரங்கல், பிரிதல்) and karupporul (the imagery:
deity, tree, flower, bird, beast, drum, water, people) — and merging songs destroys the
unit those labels describe. Documents are therefore also *partitioned* on disk by tiṇai
and layer (`--split-dirs`), so a mixture can sample kurinji-urai and palai-verse
independently without any document ever spanning two tiṇais.

It also emits a **Tiṇai label sidecar** (`data/final/tinai_labels.jsonl`): 73% of the
2,669 poems carry a `tinai` field, and the knowledge graph's `HAS_TINAI` edges fill in
some of the rest. Those labels are what the model's `use_tinai` conditioning layer
(`docs/architecture/TINAI_CONDITIONING.md`) needs — "Tamil intelligence" here means structured
grammar/landscape supervision, not only more tokens.

Usage:
    python scripts/ingest_open_sangam.py \
        --sangam-root /home/valav/Yazhi/repos/open-sangam \
        --out data/raw/sangam --dedup-against data/raw/expanded \
        --labels data/final/tinai_labels.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.core.logging import get_logger  # noqa: E402
from adhan_slm.data.sangam_tinai import labels_for, profile_for, quantise_kalam  # noqa: E402

logger = get_logger(__name__)

TAMIL = re.compile(r"[\u0B80-\u0BFF]")
#: poem fields that become documents, and the source label each one carries
TEXT_FIELDS = (
    ("sangamTamil", "verse"),
    ("urai", "urai"),
    ("yazhi_urai", "yazhi-urai"),
)
#: fields deliberately not ingested as text (metadata / English / fragments)
EXCLUDED_FIELDS = (
    "english",
    "englishMeta",
    "culturalNotes",
    "title",
    "heading",
    "lines",
    "audioUrl",
    "source",
    "licenses",
    "contributors",
    "keywords",
    "resources",
)


def tamil_ratio(text: str) -> float:
    return len(TAMIL.findall(text)) / max(1, len(text))


def is_quality(text: str, min_chars: int = 40, min_tamil: float = 0.70) -> bool:
    text = text.strip()
    if len(text) < min_chars:
        return False
    if tamil_ratio(text) < min_tamil:
        return False
    # reject records that are mostly digits/punctuation carrying a little Tamil
    letters = sum(1 for ch in text if ch.isalpha())
    return letters / max(1, len(text)) >= 0.5


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def iter_poem_records(sangam_root: Path) -> Iterable[Tuple[str, Dict]]:
    """Yield (work, record) for every poem JSON under data/texts/<work>/."""
    texts = sangam_root / "data" / "texts"
    if not texts.is_dir():
        raise SystemExit(f"no data/texts under {sangam_root}")
    for work_dir in sorted(p for p in texts.iterdir() if p.is_dir()):
        for path in sorted(work_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                logger.warning("skipping %s: %s", path.name, exc)
                continue
            records = payload if isinstance(payload, list) else [payload]
            for rec in records:
                if isinstance(rec, dict):
                    yield work_dir.name, rec


def graph_tinai(sangam_root: Path) -> Dict[str, str]:
    """work -> tinai from the knowledge graph's ``poem:<work> --HAS_TINAI-->`` edges."""
    path = sangam_root / "data" / "knowledge" / "graph.json"
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return out
    for edge in graph.get("edges", []):
        if edge.get("rel") != "HAS_TINAI":
            continue
        src, tgt = str(edge.get("source", "")), str(edge.get("target", ""))
        if src.startswith("poem:") and tgt.startswith("tinai:"):
            out[src.split(":", 1)[1]] = tgt.split(":", 1)[1]
    return out


def archive_hashes(shard_dir: Optional[Path], cap: int = 2_000_000) -> set:
    """Exact hashes of the existing archive, so overlap can be reported per source."""
    if not shard_dir or not shard_dir.exists():
        return set()
    hashes = set()
    for shard in sorted(shard_dir.glob("corpus-*.jsonl")):
        with shard.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = rec.get("text")
                if text:
                    hashes.add(sha1(text))
                    if len(hashes) >= cap:
                        logger.warning("archive hash set capped at %s", f"{cap:,}")
                        return hashes
    return hashes


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract all of open-sangam into corpus + labels")
    ap.add_argument("--sangam-root", default="/home/valav/Yazhi/repos/open-sangam")
    ap.add_argument("--out", default="data/raw/sangam")
    ap.add_argument("--labels", default="data/final/tinai_labels.jsonl")
    ap.add_argument("--dedup-against", default="data/raw/expanded")
    ap.add_argument(
        "--min-chars",
        type=int,
        default=24,
        help="classical verses are short; 40 rejects a third of them as 'too short'",
    )
    ap.add_argument("--min-tamil", type=float, default=0.70)
    ap.add_argument(
        "--split-dirs",
        action="store_true",
        help="write <out>/<tinai>/<layer>.jsonl partitions instead of one flat shard",
    )
    ap.add_argument(
        "--attested-only",
        action="store_true",
        help="only emit karupporul items the canon marks as attested",
    )
    args = ap.parse_args()

    root = Path(args.sangam_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    known = archive_hashes(Path(args.dedup_against) if args.dedup_against else None)
    logger.info("archive hashes loaded: %s", f"{len(known):,}")

    work_tinai = graph_tinai(root)
    logger.info("tinai labels inferred from the knowledge graph: %d works", len(work_tinai))

    seen_ids: set = set()
    seen_texts: set = set()
    keep_stats: Counter = Counter()  # (layer) -> docs
    keep_chars: Counter = Counter()  # (layer) -> chars
    overlap: Counter = Counter()  # (layer) -> already in archive (informational)
    rejected: Counter = Counter()  # (layer) -> filtered
    partition_stats: Counter = Counter()  # (tinai, layer) -> docs
    kalam_stats: Counter = Counter()
    uripporul_stats: Counter = Counter()
    labels: List[Dict] = []

    writers: Dict[str, object] = {}

    def writer_for(tinai: str, layer: str):
        key = f"{tinai}/{layer}" if args.split_dirs else "__flat__"
        if key not in writers:
            if args.split_dirs:
                d = out / tinai
                d.mkdir(parents=True, exist_ok=True)
                writers[key] = (d / f"{layer}.jsonl").open("w", encoding="utf-8")
            else:
                writers[key] = (out / "corpus-00000.jsonl").open("w", encoding="utf-8")
        return writers[key]

    for work, rec in iter_poem_records(root):
        poem_id = str(rec.get("id") or f"{work}_{rec.get('number', 'x')}")
        raw_tinai = str(rec.get("tinai") or "").strip().lower()
        prof = profile_for(raw_tinai) or profile_for(work_tinai.get(work))
        tinai = prof.name if prof else "unknown"
        # kalam: the canon's primary season for this tiṇai, with the raw value kept
        kalam_raw = rec.get("kalam") or rec.get("season")
        kalam = quantise_kalam(prof, str(kalam_raw)) if prof else "unknown"
        lab = labels_for(tinai, attested_only=args.attested_only)

        kalam_stats[kalam] += 1
        uripporul_stats[str(lab.get("uripporul"))] += 1
        labels.append(
            {
                "id": poem_id,
                "work": work,
                "layer": "song",
                **lab,
                "kalam_primary": kalam,
                "kalam_raw": str(kalam_raw) if kalam_raw else None,
                "poet": rec.get("poet"),
                "number": rec.get("number"),
            }
        )

        for field, layer in TEXT_FIELDS:
            raw = rec.get(field)
            if not isinstance(raw, str):
                continue
            text = normalize(raw)
            if not is_quality(text, args.min_chars, args.min_tamil):
                rejected[layer] += 1
                continue
            # Never merge: one document == one (song, layer). A repeated id would mean
            # two songs were concatenated into one record, so it is a hard error.
            doc_id = f"{layer}-{poem_id}"
            if doc_id in seen_ids:
                raise SystemExit(f"duplicate song id {doc_id}: songs must not be merged")
            seen_ids.add(doc_id)

            digest = sha1(text)
            if digest in known:
                overlap[layer] += 1  # informational: archive holds this exact fragment
            if digest in seen_texts:
                continue
            seen_texts.add(digest)

            record = {
                "text": text,
                "url": "",
                "source": f"sangam:{layer}:{work}",
                "tier": 1,
                "quality_score": 0.97 if layer == "urai" else 0.95,
                "id": doc_id,
                "chars": len(text),
                # the four poetics dimensions travel with the document
                "tinai": tinai,
                "kalam": kalam,
                "uripporul": lab.get("uripporul"),
                "karupporul": lab.get("karupporul"),
                "work": work,
                "poem_id": poem_id,
            }
            writer_for(tinai, layer).write(json.dumps(record, ensure_ascii=False) + "\n")
            keep_stats[layer] += 1
            keep_chars[layer] += len(text)
            partition_stats[f"{tinai}/{layer}"] += 1

    for fh in writers.values():
        fh.close()

    total_chars = sum(keep_chars.values())
    stats = {
        "sangam_root": str(root),
        "songs": len(labels),
        "documents": sum(keep_stats.values()),
        "characters": total_chars,
        "approx_tokens_swaram": int(total_chars * 0.311),
        "split_dirs": bool(args.split_dirs),
        "by_layer": {
            layer: {
                "docs": keep_stats[layer],
                "chars": keep_chars[layer],
                "tokens_est": int(keep_chars[layer] * 0.311),
                "already_in_archive": overlap[layer],
                "rejected": rejected[layer],
            }
            for _field, layer in TEXT_FIELDS
        },
        "by_partition": dict(partition_stats.most_common()),
        "poetics": {
            "tinai": dict(Counter(L["tinai"] for L in labels).most_common()),
            "kalam": dict(kalam_stats.most_common()),
            "uripporul": dict(uripporul_stats.most_common()),
        },
        "excluded_fields": list(EXCLUDED_FIELDS),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    labels_path = Path(args.labels)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", encoding="utf-8") as fh:
        for row in labels:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(
        "extracted %s documents from %s songs (%s chars, ~%s tokens) -> %s%s",
        f"{sum(keep_stats.values()):,}",
        f"{len(labels):,}",
        f"{total_chars:,}",
        f"{int(total_chars * 0.311):,}",
        out,
        " (partitioned by tinai)" if args.split_dirs else "",
    )
    for _field, layer in TEXT_FIELDS:
        logger.info(
            "  %-11s docs=%s chars=%s tokens~%s | already in archive=%s | rejected=%s",
            layer,
            f"{keep_stats[layer]:,}",
            f"{keep_chars[layer]:,}",
            f"{int(keep_chars[layer] * 0.311):,}",
            f"{overlap[layer]:,}",
            f"{rejected[layer]:,}",
        )
    logger.info("poetics: tinai=%s", stats["poetics"]["tinai"])
    logger.info("         kalam=%s", stats["poetics"]["kalam"])
    logger.info("         uripporul=%s", stats["poetics"]["uripporul"])
    logger.info(
        "labels -> %s (%s songs, 88%% tinai coverage expected)", labels_path, f"{len(labels):,}"
    )


if __name__ == "__main__":
    main()
