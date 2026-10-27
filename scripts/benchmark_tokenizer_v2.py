#!/usr/bin/env python3
"""Compare frozen tokenizer artifacts on the same held-out documents.

Answers the questions that decide whether an artifact is shippable:

* **lossless** — does ``decode(encode(text)) == text``? (byte fallback's job)
* **fertility** — tokens per akshara; the compression the model actually pays for
* **throughput** — kchar/s; the difference between a 2-minute and a 43-hour pack
* **morphology** — do token boundaries land on real stem/suffix breaks?
* **unk** — how often the vocabulary simply fails (the old artifact's failure mode)

Usage:
    python scripts/benchmark_tokenizer_v2.py \\
        --corpus data/raw/expanded \\
        --tokenizer data/final/tamil_slm:swaram-v1-python-12k \\
        --tokenizer data/final/tamil_slm_v2:swaram-v2-rust-32k \\
        --out docs/TOKENIZER_BENCHMARK_V2.md
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.core.logging import get_logger  # noqa: E402
from adhan_slm.tokenizer import morphology  # noqa: E402
from adhan_slm.tokenizer.fast_bpe import AksharaBPE  # noqa: E402
from adhan_slm.tokenizer.swaram_tokenizer import (  # noqa: E402
    SwaramTokenizer,
    default_akshara_inventory,
)

logger = get_logger(__name__)


def load_tokenizer(path: Path):
    """Load whichever artifact layout the directory has (Rust or pure python)."""
    if (path / "tokenizer.json").exists():
        return AksharaBPE.load(path, inventory=default_akshara_inventory()), "rust"
    if (path / "vocab.json").exists():
        return (
            SwaramTokenizer.from_files(str(path / "vocab.json"), str(path / "merges.txt")),
            "python",
        )
    raise FileNotFoundError(f"no tokenizer artifacts in {path}")


def sample_docs(corpus_dir: Path, n: int, seed: int = 42) -> List[str]:
    """Sparse, deterministic sample so the comparison sees the whole archive."""
    shards = sorted(corpus_dir.glob("*.jsonl"))
    docs: List[str] = []
    if not shards:
        raise FileNotFoundError(f"no shards in {corpus_dir}")
    total = 0
    for shard in shards:
        with shard.open(encoding="utf-8") as fh:
            for line in fh:
                total += 1
    stride = max(1, total // max(1, n * 3))
    for shard in shards:
        with shard.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i % stride:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = rec.get("text", "")
                if text:
                    docs.append(text)
    random.Random(seed).shuffle(docs)
    return docs[:n]


_TAMIL_RUN = re.compile(r"[\u0B80-\u0BFF]+")


def words_from(docs: Sequence[str], limit: int = 400) -> List[str]:
    """Tamil word forms for the morphology probe (punctuation stripped, deduped)."""
    seen, words = set(), []
    for doc in docs:
        for word in _TAMIL_RUN.findall(doc):
            if len(word) < 5:
                continue
            if word in seen:
                continue
            seen.add(word)
            words.append(word)
            if len(words) >= limit:
                return words
    return words


def run_one(name: str, path: Path, docs: Sequence[str], words: Sequence[str]) -> dict:
    tok, backend = load_tokenizer(path)
    vocab = tok.vocab if hasattr(tok, "vocab") else {}
    n_chars = sum(len(d) for d in docs)

    t0 = time.time()
    encoded = [tok.encode(d) for d in docs]
    elapsed = time.time() - t0

    lossless = sum(1 for d, ids in zip(docs, encoded) if tok.decode(ids) == d)
    unk_docs = sum(1 for ids in encoded if (tok.unk_id in ids) if hasattr(tok, "unk_id"))
    n_tokens = sum(len(ids) for ids in encoded)

    ferts = [tok.fertility(d) for d in docs if d.strip()]
    n_aks = sum(1 for d in docs for a in _aksharas(tok, d) if a.strip())
    report = morphology.boundary_agreement(tok, words)

    return {
        "name": name,
        "path": str(path),
        "backend": backend,
        "vocab": len(vocab),
        "docs": len(docs),
        "chars": n_chars,
        "tokens": n_tokens,
        "tokens_per_doc": n_tokens / len(docs) if docs else 0,
        "fertility": sum(ferts) / len(ferts) if ferts else 0,
        "lossless_rate": lossless / len(docs) if docs else 0,
        "unk_docs": unk_docs,
        "kchar_per_s": n_chars / elapsed / 1000 if elapsed else 0,
        "morph_agreement": report.agreement_rate,
        "morph_words": report.n_with_suffix,
        "n_aksharas": n_aks,
    }


def _aksharas(tok, text: str):
    if isinstance(tok, AksharaBPE):
        return tok._segment(text)
    return tok._segment(text)


def render(rows: Sequence[dict], docs: int, chars: int) -> str:
    def pct(x: float) -> str:
        return f"{100 * x:.1f}%"

    lines = [
        "# Tokenizer benchmark — frozen swaram artifacts on the same held-out documents",
        "",
        "Generated by `scripts/benchmark_tokenizer_v2.py` on a sparse sample of the",
        "packed archive, so every row sees the same documents.",
        "",
        f"- documents: **{docs:,}**  ·  characters: **{chars:,}**",
        "",
        "| artifact | backend | vocab | fertility (tok/akshara) | tokens/doc | lossless | `<unk>` docs | throughput | morph. agreement |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {name} | {backend} | {vocab:,} | {fert:.3f} | {tpd:.1f} | {ll} | {unk} | "
            "{kcs:,.0f} kchar/s | {morph} (n={nw}) |".format(
                name=r["name"],
                backend=r["backend"],
                vocab=r["vocab"],
                fert=r["fertility"],
                tpd=r["tokens_per_doc"],
                ll=pct(r["lossless_rate"]),
                unk=r["unk_docs"],
                kcs=r["kchar_per_s"],
                morph=pct(r["morph_agreement"]) if r["morph_words"] else "n/a",
                nw=r["morph_words"],
            )
        )
    lines += [
        "",
        "Notes:",
        "",
        "- **fertility** = tokens / akshara, using the tokenizer's own akshara layer as the",
        "  shared denominator; lower is more compression per token.",
        "- **lossless** is byte-exact `decode(encode(text)) == text`. Artifacts built with",
        "  byte fallback round-trip any script; an artifact without it collapses unknown",
        "  characters to `<unk>` and cannot round-trip.",
        "- **morph. agreement** = share of suffixed words whose token boundary lands exactly",
        "  on the stem/suffix break (`tokenizer/morphology.py`), i.e. whether Layer B merges",
        "  respect Tamil morphology rather than cutting through it.",
        "- **throughput** is single-process encode on this machine; it is the number that",
        "  decides whether packing the 1.06G-character archive takes minutes or a day.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark frozen tokenizer artifacts")
    ap.add_argument("--corpus", default="data/raw/expanded")
    ap.add_argument(
        "--tokenizer",
        action="append",
        default=None,
        help="DIR:label (repeatable); defaults compare data/final/tamil_slm and tamil_slm_v2",
    )
    ap.add_argument("--docs", type=int, default=300)
    ap.add_argument("--out", default="docs/TOKENIZER_BENCHMARK_V2.md")
    args = ap.parse_args()

    specs = args.tokenizer or [
        "data/final/tamil_slm:swaram-v1-python-12k",
        "data/final/tamil_slm_v2:swaram-v2-rust-32k",
    ]
    docs = sample_docs(Path(args.corpus), args.docs)
    words = words_from(docs)
    logger.info("benchmarking on %d docs / %d words", len(docs), len(words))

    rows = []
    for spec in specs:
        path_s, _, name = spec.rpartition(":")
        path = Path(path_s or spec)
        logger.info("  %s (%s) ...", name, path)
        rows.append(run_one(name, path, docs, words))

    md = render(rows, len(docs), sum(len(d) for d in docs))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(md)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
