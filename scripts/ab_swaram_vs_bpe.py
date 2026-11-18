#!/usr/bin/env python3
"""Swaram vs Tamil-trained BPE — the thesis A/B (adhan N14).

The whitepaper's central claim is that akshara-native tokenization beats
subword tokenization for Tamil. That claim has never been tested against the
one baseline that matters: a BPE trained on the *same Tamil corpus* at the
*same vocab size*. Comparing to GPT-4's English-fitted cl100k is table stakes.

This script:
  1. reserves a deterministic held-out set of raw documents;
  2. trains a byte-level BPE on the remaining corpus at the Swaram vocab size;
  3. tokenizes the held-out set with both tokenizers;
  4. reports tokens/akshara (fertility), tokens/word, chars/token, and the
     token-tax ratio.

The akshara count is computed once, tokenizer-independent, so the two
fertilities are directly comparable.

    python scripts/ab_swaram_vs_bpe.py \
        --corpus data/raw/expanded \
        --swaram-tokenizer data/final/tamil_slm_v2 \
        --out docs/reports/ab_swaram_vs_bpe.json

    # smaller/faster
    python scripts/ab_swaram_vs_bpe.py --corpus data/raw/expanded \
        --swaram-tokenizer data/final/tamil_slm_v2 --train-chars 50000000 --val-docs 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.core.logging import get_logger  # noqa: E402
from adhan_slm.tokenizer import indic_akshara  # noqa: E402
from adhan_slm.tokenizer.fast_bpe import AksharaBPE  # noqa: E402

logger = get_logger(__name__)


def iter_corpus(corpus_dir: Path):
    """Yield ``(index, text)`` from every ``*.jsonl`` shard, in shard order."""
    import json as _json

    shards = sorted(corpus_dir.glob("*.jsonl")) or [corpus_dir]
    i = 0
    for shard in shards:
        with shard.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                text = rec.get("text") or ""
                if text:
                    yield i, text
                    i += 1


def is_held_out(index: int, seed: int, val_frac: float) -> bool:
    return (hash((seed, index)) % 1000) / 1000.0 < val_frac


def akshara_count(text: str) -> int:
    """Tokenizer-independent akshara count (script-aware grapheme clusters)."""
    return sum(1 for a in indic_akshara.segment(text, [indic_akshara.TAMIL]) if a.strip())


def train_bpe(docs, vocab_size: int, style: str = "byte", min_frequency: int = 2):
    """Train a BPE baseline in one of two styles.

    ``byte``       — GPT-2-style byte-level BPE (what cl100k/o200k are). This is
                     the direct control for "the same tokenizer architecture,
                     retrained on Tamil".
    ``metaspace``  — SentencePiece-style BPE over Unicode with a ``▁`` word mark.
                     This is the *strong* baseline (IndicBERT/MuRIL-style) and
                     the one a reviewer will ask for.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    special = ["<unk>", "<bos>", "<eos>", "<pad>"]
    if style == "byte":
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
    elif style == "metaspace":
        tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
        tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special,
            show_progress=False,
        )
    else:
        raise ValueError(f"unknown BPE style {style!r}")
    tok.train_from_iterator(docs, trainer=trainer)
    return tok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default="data/raw/expanded", help="dir of *.jsonl shards")
    ap.add_argument("--swaram-tokenizer", default="data/final/tamil_slm_v2")
    ap.add_argument("--out", default="docs/reports/ab_swaram_vs_bpe.json")
    ap.add_argument(
        "--train-chars", type=int, default=100_000_000, help="characters of BPE training data"
    )
    ap.add_argument("--val-docs", type=int, default=2000, help="held-out documents to score")
    ap.add_argument("--val-frac", type=float, default=0.02, help="must match the shard split")
    ap.add_argument("--seed", type=int, default=0, help="must match the shard split")
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument(
        "--bpe-styles",
        default="byte,metaspace",
        help="comma-separated BPE baselines to train (byte, metaspace)",
    )
    args = ap.parse_args()

    corpus = Path(args.corpus)
    if not corpus.exists():
        logger.error("corpus not found: %s", corpus)
        return 2

    tok_dir = Path(args.swaram_tokenizer)
    if not tok_dir.is_absolute():
        tok_dir = ROOT / tok_dir
    swaram = AksharaBPE.load(tok_dir)
    vocab_size = len(swaram)
    logger.info("swaram vocab=%s; reserving held-out docs...", f"{vocab_size:,}")

    # Pass 1: collect held-out docs and a training sample.
    val_docs: list[str] = []
    train_sample: list[str] = []
    train_chars = 0
    n_seen = 0
    t0 = time.time()
    for index, text in iter_corpus(corpus):
        n_seen += 1
        if is_held_out(index, args.seed, args.val_frac):
            if len(val_docs) < args.val_docs:
                val_docs.append(text)
            continue
        if train_chars < args.train_chars:
            train_sample.append(text)
            train_chars += len(text)
        if len(val_docs) >= args.val_docs and train_chars >= args.train_chars:
            break
    logger.info(
        "scanned %s docs; held-out %s docs, BPE training sample %s docs / %s chars (%.0fs)",
        f"{n_seen:,}",
        f"{len(val_docs):,}",
        f"{len(train_sample):,}",
        f"{train_chars:,}",
        time.time() - t0,
    )
    if not val_docs:
        logger.error("no held-out docs found; check --val-frac/--seed")
        return 2

    # Pass 2: train each BPE baseline on the same corpus.
    styles = [s.strip() for s in args.bpe_styles.split(",") if s.strip()]
    bpes: dict[str, object] = {}
    train_secs: dict[str, float] = {}
    for style in styles:
        logger.info("training %s BPE (vocab=%s)...", style, f"{vocab_size:,}")
        t0 = time.time()
        bpes[style] = train_bpe(
            train_sample, vocab_size, style=style, min_frequency=args.min_frequency
        )
        train_secs[style] = time.time() - t0
        logger.info("%s BPE trained in %.1fs", style, train_secs[style])

    # Pass 3: score the held-out set with both tokenizers. Akshara counts are
    # computed once and shared, so the two fertilities are directly comparable.
    logger.info("scoring %s held-out docs...", f"{len(val_docs):,}")
    t0 = time.time()
    aksharas = [akshara_count(t) for t in val_docs]
    words = [len(t.split()) for t in val_docs]
    chars = [len(t) for t in val_docs]
    swaram_ids = swaram.encode_batch(val_docs)
    baseline_ids = {
        style: [e.ids for e in bpe.encode_batch(val_docs, add_special_tokens=False)]
        for style, bpe in bpes.items()
    }
    logger.info("scored in %.1fs", time.time() - t0)

    def summarize(name: str, ids: list[list[int]]) -> dict:
        total_tokens = sum(len(x) for x in ids)
        total_aks = sum(aksharas)
        total_words = sum(words)
        total_chars = sum(chars)
        return {
            "tokenizer": name,
            "docs": len(val_docs),
            "tokens": total_tokens,
            "aksharas": total_aks,
            "words": total_words,
            "chars": total_chars,
            "fertility_tok_per_akshara": round(total_tokens / total_aks, 4) if total_aks else None,
            "tokens_per_word": round(total_tokens / total_words, 4) if total_words else None,
            "chars_per_token": round(total_chars / total_tokens, 4) if total_tokens else None,
        }

    metrics = [summarize("swaram", swaram_ids)]
    taxes: dict[str, float] = {}
    verdicts: dict[str, dict] = {}
    sw_tokens = metrics[0]["tokens"]
    for style, ids in baseline_ids.items():
        m = summarize(f"bpe_{style}_32k", ids)
        metrics.append(m)
        if sw_tokens:
            taxes[style] = round(m["tokens"] / sw_tokens, 4)
        verdicts[style] = _verdict(metrics[0], m)

    overall = _overall_verdict(verdicts)

    report = {
        "corpus": str(corpus),
        "swaram_tokenizer": str(args.swaram_tokenizer),
        "vocab_size": vocab_size,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "held_out_docs": len(val_docs),
        "bpe_train_chars": train_chars,
        "bpe_train_seconds": {k: round(v, 1) for k, v in train_secs.items()},
        "metrics": metrics,
        "bpe_tax_vs_swaram": taxes,
        "verdict": overall,
        "note": (
            "Fertility uses a tokenizer-independent akshara count. A BPE tax > 1.0 means "
            "Swaram is more compressed; < 1.0 means the BPE wins on this sample. The "
            "byte-level baseline is the direct control for GPT-4-style tokenizers; the "
            "metaspace baseline is the stronger SentencePiece-style control."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %s", out)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _overall_verdict(verdicts: dict) -> dict:
    """Overall verdict keys off the *strongest* baseline.

    Swaram must beat the best control, not the weakest one. ``min`` of the
    per-baseline advantages is the advantage against the strongest baseline.
    """
    weakest_advantage = min(
        (v["swaram_advantage_relative"] for v in verdicts.values()), default=None
    )
    return {
        "status": (
            "swaram_supported"
            if weakest_advantage is not None and weakest_advantage >= 0.10
            else "thesis_not_supported_on_this_sample"
        ),
        "swaram_advantage_vs_strongest_baseline": weakest_advantage,
        "per_baseline": verdicts,
    }


def _verdict(swaram: dict, bpe: dict) -> dict:
    sw = swaram["fertility_tok_per_akshara"]
    bp = bpe["fertility_tok_per_akshara"]
    if not sw or not bp:
        return {"status": "inconclusive", "reason": "missing fertility"}
    # The thesis is "fewer tokens per akshara is better". Swaram wins if its
    # fertility is materially lower (>=10% relative) than the Tamil BPE's.
    rel = (bp - sw) / sw
    return {
        "swaram_fertility": sw,
        "bpe_fertility": bp,
        "swaram_advantage_relative": round(rel, 4),
        "swaram_wins_by_10pct": rel >= 0.10,
        "status": "swaram_supported" if rel >= 0.10 else "thesis_not_supported_on_this_sample",
    }


if __name__ == "__main__":
    raise SystemExit(main())
