#!/usr/bin/env python3
"""Sweep Swaram vocabulary sizes and report fertility, OOV and round-trip.

Trains ``SwaramTokenizer`` at several vocab sizes, with and without byte fallback, on
the frozen Thirukkural benchmark corpus and measures each on the held-out split:

  * tokens per akshara (the pilot target is < 1.15; lower is better)
  * tokens per word
  * <unk> rate
  * round-trip: decode(encode(text)) == NFC(text)
  * training time

Usage:
    python scripts/swaram_vocab_sweep.py
    python scripts/swaram_vocab_sweep.py --sizes 4000 8000 12000

It writes both:
    docs/reports/swaram_vocab_sweep.md    the readable table
    docs/reports/swaram_vocab_sweep.json  the same numbers for downstream use

The corpus is small (about 1.3k couplets), so large vocabularies mostly learn whole
words from the training split. Read the numbers as a sanity check of the mechanism and
the trainer's speed, not as a final 12k validation; that needs a larger corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_tokenizers import deterministic_split, load_frozen_corpus  # noqa: E402

from adhan_slm.tokenizer import SwaramTokenizer  # noqa: E402
from adhan_slm.tokenizer.swaram_tokenizer import WORD_MARK  # noqa: E402

DEFAULT_MD = "docs/reports/swaram_vocab_sweep.md"
DEFAULT_JSON = "docs/reports/swaram_vocab_sweep.json"

# Fertility on the same 300-doc sample, from docs/reports/base_decision_2026-11-03.md
# and docs/reports/ab_swaram_vs_bpe.json. Context for the table, not a gate.
BASELINES: list[tuple[str, float, str]] = [
    ("Qwen3-1.7B-Base", 2.016, "product student"),
    ("Sarvam-1", 0.538, "licence-rejected"),
    ("Sarvam-M 24B", 0.804, "teacher"),
    ("Tamil-trained BPE", 0.346, "research baseline"),
]


def measure(tok: SwaramTokenizer, eval_docs: list[str]) -> dict:
    n_aks = sum(1 for d in eval_docs for a in tok.aksharas(d) if a.strip())
    n_words = sum(len(d.split()) for d in eval_docs)
    n_tok = n_unk = n_ids = 0
    exact = 0
    for d in eval_docs:
        ids = tok.encode(d)
        n_tok += sum(1 for t in tok.tokenize(d) if t != WORD_MARK)
        n_ids += len(ids)
        n_unk += sum(1 for i in ids if i == tok.unk_id)
        exact += tok.decode(ids) == unicodedata.normalize("NFC", d)
    return {
        "tok_per_akshara": n_tok / n_aks,
        "tok_per_word": n_tok / n_words,
        "unk_rate": n_unk / max(1, n_ids),
        "roundtrip": exact / len(eval_docs),
    }


def run(sizes: list[int], cache_dir: Path) -> list[dict]:
    docs = load_frozen_corpus(cache_dir)
    train_docs, eval_docs = deterministic_split(docs)
    rows = []
    for size in sizes:
        for byte_fb in (False, True):
            t0 = time.perf_counter()
            tok = SwaramTokenizer.train(train_docs, vocab_size=size, byte_fallback=byte_fb)
            secs = time.perf_counter() - t0
            print(f"  vocab {size:>6,} byte_fallback={byte_fb!s:<5} trained in {secs:.1f}s")
            rows.append(
                {
                    "requested": size,
                    "actual": len(tok),
                    "byte_fallback": byte_fb,
                    **measure(tok, eval_docs),
                }
            )
    return rows


def to_report(rows: list[dict], n_train: int, n_eval: int) -> dict:
    """Build the JSON report. Wall-clock timings are deliberately excluded (see the doc)."""
    return {
        "corpus": "thirukkural",
        "n_train_docs": n_train,
        "n_eval_docs": n_eval,
        "fertility_target_tok_per_akshara": 1.15,
        "note": (
            "Corpus is ~1.3k couplets; large vocabularies partly memorise training words. "
            "Tok/akshara below 1 measures merge rate, not a validated 12k vocabulary."
        ),
        "baselines": [
            {"name": name, "tok_per_akshara": val, "role": role} for name, val, role in BASELINES
        ],
        "rows": rows,
    }


def to_markdown(rows: list[dict], n_train: int, n_eval: int) -> str:
    lines = [
        "# Swaram vocabulary sweep",
        "",
        f"Frozen Thirukkural corpus: {n_train} training docs, {n_eval} held-out docs.",
        "Lower tokens/akshara is better (pilot target < 1.15).",
        "",
        "| Vocab | Byte fallback | Actual | Tokens/akshara | Tokens/word | `<unk>` rate "
        "| Round-trip |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['requested']} | {'yes' if r['byte_fallback'] else 'no'} | {r['actual']} "
            f"| {r['tok_per_akshara']:.3f} | {r['tok_per_word']:.2f} | {r['unk_rate']:.2%} "
            f"| {r['roundtrip']:.1%} |"
        )
    lines += [
        "",
        "Round-trip is `decode(encode(text)) == NFC(text)` on the held-out split.",
        "",
        "Training time is printed by the script but kept out of this report: it is per-machine",
        "wall clock, so including it made every re-run a spurious diff. The full sweep is about",
        "11s on a laptop.",
        "",
        "## How to read this",
        "",
        "Swaram emits one token per akshara, so tokens/akshara below 1 is really a",
        "*merge rate*: even the 1k vocabulary clears the 1.15 target. The number is",
        "therefore a mechanism check, not a discriminating gate. The comparison that",
        "matters is against other tokenizers measured on the same sample:",
        "",
        "| Tokenizer | Tokens/akshara | Role |",
        "|---|---:|---|",
    ]
    for name, val, role in BASELINES:
        lines.append(f"| {name} | {val:.3f} | {role} |")
    lines += [
        "",
        "Swaram only approaches the Tamil-trained BPE at 12k, and on this corpus that",
        "gain is mostly memorisation, not generalisation. The compression thesis is",
        "recorded as unsupported on the larger A/B",
        "([`ab_swaram_vs_bpe.md`](ab_swaram_vs_bpe.md)).",
        "",
        "Byte fallback is the honest lossless switch: +100% round-trip at every size,",
        "and from 2k up it costs almost nothing in compression. At 1k it costs ~13%",
        "because its 256 byte tokens occupy slots that would otherwise hold merges.",
        "",
        "Caveat: the corpus is small, so large vocabularies mostly memorise whole words",
        "from the training split. Treat this as a check of the mechanism and trainer",
        "speed, not a final 12k validation, which needs a larger corpus.",
        "",
        "Regenerate: `python scripts/swaram_vocab_sweep.py`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 4000, 8000, 12000])
    ap.add_argument("--cache-dir", default="data/raw/thirukkural")
    ap.add_argument("--out", default=DEFAULT_MD, help="markdown report path")
    ap.add_argument("--json", default=DEFAULT_JSON, help="JSON report path")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    docs = load_frozen_corpus(cache_dir)
    n_eval = len(deterministic_split(docs)[1])
    n_train = len(docs) - n_eval
    rows = run(args.sizes, cache_dir)

    md = to_markdown(rows, n_train, n_eval)
    report = to_report(rows, n_train, n_eval)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
