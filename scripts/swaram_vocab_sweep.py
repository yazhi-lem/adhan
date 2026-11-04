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
    python scripts/swaram_vocab_sweep.py --sizes 4000 8000 12000 --out docs/SWARAM_VOCAB_SWEEP.md

The corpus is small (about 1.3k couplets), so large vocabularies mostly learn whole
words from the training split. Read the numbers as a sanity check of the mechanism and
the trainer's speed, not as a final 12k validation; that needs a larger corpus.
"""

from __future__ import annotations

import argparse
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
            rows.append(
                {
                    "requested": size,
                    "actual": len(tok),
                    "byte_fallback": byte_fb,
                    "train_s": secs,
                    **measure(tok, eval_docs),
                }
            )
    return rows


def to_markdown(rows: list[dict], n_train: int, n_eval: int) -> str:
    lines = [
        "# Swaram vocabulary sweep",
        "",
        f"Frozen Thirukkural corpus: {n_train} training docs, {n_eval} held-out docs.",
        "Lower tokens/akshara is better (pilot target < 1.15).",
        "",
        "| Vocab | Byte fallback | Actual | Tokens/akshara | Tokens/word | `<unk>` rate "
        "| Round-trip | Train time |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['requested']} | {'yes' if r['byte_fallback'] else 'no'} | {r['actual']} "
            f"| {r['tok_per_akshara']:.3f} | {r['tok_per_word']:.2f} | {r['unk_rate']:.2%} "
            f"| {r['roundtrip']:.1%} | {r['train_s']:.1f}s |"
        )
    lines += [
        "",
        "Round-trip is `decode(encode(text)) == NFC(text)` on the held-out split.",
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
    ap.add_argument("--out", default="docs/SWARAM_VOCAB_SWEEP.md")
    args = ap.parse_args()

    docs = load_frozen_corpus(Path(args.cache_dir))
    n_eval = len(deterministic_split(docs)[1])
    rows = run(args.sizes, Path(args.cache_dir))
    md = to_markdown(rows, len(docs) - n_eval, n_eval)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
