#!/usr/bin/env python3
"""Fertility of candidate base-model tokenizers on one frozen Tamil sample.

Answers the question the base decision left open: *how expensive is each product
candidate's Tamil tokenizer on the same text, measured by committed code?*  The
earlier 2.016 / 0.538 tok/akshara figures had no committed sampling script, so this
one fixes the sample by rule (sha256(seed|id|text) ranking, first N) and re-measures
every tokenizer on it.

Rows: Swaram v3 (research reference), Qwen3 (product base), Sarvam-1 and Sarvam-M
(tokenizer references), Agaram 2.5B (from-scratch Tamil base), plus code-point and
UTF-8 byte baselines.

Usage:
    python scripts/benchmark_bases_tokenizers.py \\
        --corpus data/raw/expanded/corpus-00001.jsonl --docs 300 \\
        --out-json docs/reports/bases_tokenizer_bench.json \\
        --out-md docs/reports/bases_tokenizer_bench.md

Tokenizer locations can be overridden with --tokenizer NAME=PATH (repeatable).
Nothing leaves the machine: files are read from local paths only.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.eval.tokenizer_bench import evaluate, to_markdown  # noqa: E402

SEED = "adhan-bases-2026-11-05"
HF = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _snap(repo: str, filename: str) -> str | None:
    hits = sorted(
        glob.glob(str(HF / f"models--{repo.replace('/', '--')}" / "snapshots" / "*" / filename))
    )
    return hits[-1] if hits else None


def default_tokenizers() -> Dict[str, str]:
    """name -> tokenizer.json path (or a vocab/merges pair for Swaram)."""
    out: Dict[str, str] = {}
    agaram = os.environ.get("AGARAM_TOKENIZER")
    if agaram:
        out["agaram-2.5b"] = agaram
    for name, repo in (
        ("qwen3-1.7b", "Qwen/Qwen3-1.7B-Base"),
        ("sarvam-1", "sarvamai/sarvam-1"),
        ("sarvam-m", "sarvamai/sarvam-m"),
    ):
        p = _snap(repo, "tokenizer.json")
        if p:
            out[name] = p
    for name, d in (("swaram-v3", "tamil_slm_v3"), ("swaram-v2", "tamil_slm_v2")):
        v, m = (
            ROOT / "data" / "final" / d / "vocab.json",
            ROOT / "data" / "final" / d / "merges.txt",
        )
        if v.exists() and m.exists():
            out[name] = f"{v},{m}"
    return out


def sample_corpus(path: Path, n: int) -> List[Dict[str, str]]:
    docs: List[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            docs.append((str(rec.get("id", "")), unicodedata.normalize("NFC", text)))
    ranked = sorted(
        docs, key=lambda kv: hashlib.sha256(f"{SEED}|{kv[0]}|{kv[1]}".encode()).hexdigest()
    )
    if n > len(ranked):
        raise SystemExit(f"asked for {n} docs, corpus has {len(ranked)}")
    return [{"register": "modern", "text": t} for _, t in ranked[:n]]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--corpus", default="data/raw/expanded/corpus-00001.jsonl")
    ap.add_argument("--docs", type=int, default=300)
    ap.add_argument("--tokenizer", action="append", metavar="NAME=PATH")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    args = ap.parse_args(argv)

    specs = default_tokenizers()
    for spec in args.tokenizer or []:
        name, path = spec.split("=", 1)
        specs[name] = path

    records = sample_corpus(ROOT / args.corpus, args.docs)
    chars = sum(len(r["text"]) for r in records)

    from adhan_slm.tokenizer.swaram_tokenizer import segment_aksharas

    aksharas = sum(len(segment_aksharas(r["text"])) for r in records)

    from tokenizers import Tokenizer

    from adhan_slm.eval.tokenizer_bench import SwaramAdapter, TokenizersAdapter
    from adhan_slm.tokenizer.swaram_tokenizer import SwaramTokenizer

    adapters = []
    for name, path in specs.items():
        if "," in path:
            vocab, merges = path.split(",", 1)
            adapters.append(SwaramAdapter(SwaramTokenizer.from_files(vocab, merges), name))
        else:
            adapters.append(TokenizersAdapter(Tokenizer.from_file(path), name))

    results = [evaluate(a, records) for a in adapters]
    md = to_markdown(results)
    header = (
        f"# Candidate base tokenizers on one Tamil sample\n\n"
        f"- sample: `{args.corpus}`, **{len(records)} docs**, seed `{SEED}`\n"
        f"- size: **{chars:,} characters · {aksharas:,} aksharas**\n"
        f"- harness: `src/adhan_slm/eval/tokenizer_bench.py` (NFC, grapheme-safe offsets)\n\n"
    )
    body = header + md
    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(
                {
                    "sample": {
                        "corpus": args.corpus,
                        "docs": len(records),
                        "seed": SEED,
                        "chars": chars,
                        "aksharas": aksharas,
                    },
                    "results": [
                        {
                            "tokenizer": r.tokenizer,
                            "overall": r.overall,
                            "per_register": r.per_register,
                        }
                        for r in results
                    ],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.out_md:
        Path(args.out_md).write_text(body, encoding="utf-8")
    sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
