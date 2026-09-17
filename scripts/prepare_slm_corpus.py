#!/usr/bin/env python3
"""Prepare a native-Tamil SLM corpus: freeze the swaram tokenizer + pack shards.

This closes the Phase 1 ("freeze vocab.json + merges", `adhan-tok-v1` artifact) and
Phase 2 ("tokenize → packed fixed-length sequences → sharded") gap in
ROADMAP_JAX_SLM.md. Given a corpus (txt / jsonl / directory), it:

  1. deduplicates (exact + near-duplicate) and scrubs PII (emails/phones/URLs),
  2. trains the Swaram (Tamil/Dravidian) or Aksharam (Hindi/Indic) tokenizer,
  3. freezes ``vocab.json`` + ``merges.txt``,
  4. measures fertility (tokens/akshara) on a held-out sample,
  5. tokenizes + packs the corpus into ``train.bin`` / ``val.bin`` shards, and
  6. writes a ``datasheet.json`` (sources, counts, fertility, dedup/PII stats,
     code SHA) — the data card the roadmap asks for.

Everything is pure-python (stdlib only); numpy just speeds up shard I/O if present.

    python scripts/prepare_slm_corpus.py \
        --corpus data/raw/tamil/ --out data/final/tamil_slm \
        --vocab-size 12000 --seq-len 1024 --val-frac 0.02
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.core.logging import get_logger  # noqa: E402
from adhan_slm.data import corpus as corpus_mod  # noqa: E402
from adhan_slm.data import packing  # noqa: E402
from adhan_slm.data.deduplicator import TextDeduplicator  # noqa: E402
from adhan_slm.data.filters import CorpusFilter  # noqa: E402
from adhan_slm.tokenizer import SwaramTokenizer  # noqa: E402
from adhan_slm.tokenizer.aksharam_tokenizer import AksharamTokenizer  # noqa: E402

logger = get_logger(__name__)

_TOKENIZERS = {"swaram": SwaramTokenizer, "aksharam": AksharamTokenizer}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _mean_fertility(tok, docs, cap: int = 500) -> float:
    sample = docs[:cap]
    vals = [tok.fertility(d) for d in sample if d.strip()]
    vals = [v for v in vals if v > 0]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze SLM tokenizer + pack shards")
    ap.add_argument("--corpus", required=True, help="txt/jsonl file, or directory of them")
    ap.add_argument("--out", required=True, help="output dir for tokenizer + shards")
    ap.add_argument("--tokenizer", choices=list(_TOKENIZERS), default="swaram")
    ap.add_argument("--vocab-size", type=int, default=12000)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=None, help="cap #documents (debug / dry runs)")
    ap.add_argument(
        "--whole-file-docs",
        action="store_true",
        help="treat each .txt file as one document (default: per line)",
    )
    ap.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.85,
        help="near-duplicate similarity threshold for TextDeduplicator (default: 0.85)",
    )
    ap.add_argument(
        "--pii-level",
        choices=["none", "standard", "aggressive"],
        default="standard",
        help="PII scrubbing level applied before packing (default: standard)",
    )
    ap.add_argument(
        "--skip-dedup",
        action="store_true",
        help="skip near-duplicate removal (not recommended for real training runs)",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    TokCls = _TOKENIZERS[args.tokenizer]

    logger.info("[1/6] reading corpus from %s ...", args.corpus)
    docs = corpus_mod.read_corpus(
        args.corpus, line_documents=not args.whole_file_docs, limit=args.limit
    )
    if not docs:
        sys.exit(f"no documents found under {args.corpus}")

    # Clean before splitting: dedup first (so exact/near-duplicates can't end up
    # split across train and val, which would leak eval data into training) then
    # scrub PII. Runs on the raw scraped text below every document eventually
    # traces back to (Reddit/Twitter/Wikipedia/etc.) since none of those upstream
    # sources currently filter or scrub before writing here.
    logger.info("[2/6] cleaning corpus (dedup + PII scrub) ...")
    dedup_stats: dict = {}
    pii_stats: dict = {}
    doc_dicts = [{"id": str(i), "text": t} for i, t in enumerate(docs)]

    if args.skip_dedup:
        logger.warning("      --skip-dedup set: near-duplicate documents will NOT be removed")
    else:
        deduper = TextDeduplicator(threshold=args.dedup_threshold)
        dedup_gen, dedup_stats = deduper.deduplicate(iter(doc_dicts))
        doc_dicts = list(dedup_gen)
        logger.info(
            "      dedup: %s -> %s docs (%s exact + %s near duplicates removed)",
            f"{dedup_stats.get('total_seen', 0):,}",
            f"{dedup_stats.get('kept', 0):,}",
            f"{dedup_stats.get('exact_duplicates', 0):,}",
            f"{dedup_stats.get('near_duplicates', 0):,}",
        )

    if args.pii_level == "none":
        logger.warning("      --pii-level=none: emails/phones/URLs will NOT be scrubbed")
    else:
        pii_filter = CorpusFilter()
        pii_gen, pii_stats = pii_filter.scrub_pii(iter(doc_dicts), anonymize_level=args.pii_level)
        doc_dicts = list(pii_gen)
        logger.info(
            "      PII scrub (%s): %s emails, %s phones, %s URLs anonymized",
            args.pii_level,
            f"{pii_stats.get('emails_removed', 0):,}",
            f"{pii_stats.get('phones_removed', 0):,}",
            f"{pii_stats.get('urls_anonymized', 0):,}",
        )

    docs = [d["text"] for d in doc_dicts]
    if not docs:
        sys.exit("no documents left after dedup/PII scrubbing (corpus too small or too repetitive)")

    # Shuffle before splitting: read_corpus() yields documents file-by-file in
    # sorted order, so an unshuffled prefix split would put whichever source
    # sorts first (e.g. one scraper's output) entirely into val_docs instead of
    # a representative sample. Fixed seed keeps the split reproducible.
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    n_val = max(1, int(len(docs) * args.val_frac))
    val_docs = docs[:n_val]
    train_docs = docs[n_val:]
    logger.info(
        "      %s docs -> %s train / %s val (shuffled, seed=%d)",
        f"{len(docs):,}",
        f"{len(train_docs):,}",
        f"{len(val_docs):,}",
        args.seed,
    )

    logger.info("[3/6] training %s tokenizer (vocab %d) ...", args.tokenizer, args.vocab_size)
    tok = TokCls.train(train_docs, vocab_size=args.vocab_size, min_freq=args.min_freq)
    vocab_path = out / "vocab.json"
    merges_path = out / "merges.txt"
    tok.save(str(vocab_path), str(merges_path))
    logger.info(
        "      froze vocab=%s merges=%s -> %s, %s",
        f"{len(tok):,}",
        f"{len(tok.merges):,}",
        vocab_path.name,
        merges_path.name,
    )

    logger.info("[4/6] measuring fertility on held-out sample ...")
    fert = _mean_fertility(tok, val_docs)
    flag = "OK" if fert < 1.15 else "ABOVE TARGET (<1.15)"
    log = logger.info if fert < 1.15 else logger.warning
    log("      mean fertility = %.3f tokens/akshara  [%s]", fert, flag)

    logger.info("[5/6] tokenizing + packing to seq_len=%d ...", args.seq_len)
    train_seqs = packing.pack_documents(train_docs, tok, seq_len=args.seq_len)
    val_seqs = packing.pack_documents(val_docs, tok, seq_len=args.seq_len)
    if not train_seqs:
        sys.exit(
            "corpus too small to fill even one packed sequence; add more text " "or lower --seq-len"
        )
    train_shard = packing.write_shard(
        train_seqs, out / "train.bin", seq_len=args.seq_len, vocab_size=len(tok)
    )
    val_shard = None
    if val_seqs:
        val_shard = packing.write_shard(
            val_seqs, out / "val.bin", seq_len=args.seq_len, vocab_size=len(tok)
        )
    logger.info(
        "      train.bin: %s seqs / %s tokens%s",
        f"{train_shard.n_sequences:,}",
        f"{train_shard.n_tokens:,}",
        (
            f"   val.bin: {val_shard.n_sequences:,} seqs"
            if val_shard
            else "   (val too small to pack)"
        ),
    )

    logger.info("[6/6] writing datasheet.json ...")
    datasheet = {
        "corpus_source": str(args.corpus),
        "tokenizer": args.tokenizer,
        "vocab_size": len(tok),
        "n_merges": len(tok.merges),
        "seq_len": args.seq_len,
        "n_documents": len(docs),
        "n_train_documents": len(train_docs),
        "n_val_documents": len(val_docs),
        "split_seed": args.seed,
        "dedup_threshold": None if args.skip_dedup else args.dedup_threshold,
        "dedup_stats": dedup_stats,
        "pii_level": args.pii_level,
        "pii_stats": pii_stats,
        "train_tokens": train_shard.n_tokens,
        "val_tokens": val_shard.n_tokens if val_shard else 0,
        "mean_fertility": round(fert, 4),
        "fertility_target": 1.15,
        "code_sha": _git_sha(),
    }
    (out / "datasheet.json").write_text(json.dumps(datasheet, indent=2), encoding="utf-8")
    logger.info("      -> %s", out / "datasheet.json")
    logger.info(
        "done. train with:  python -m adhan_slm.training.train_jax "
        "--config <config with data.shards: %s>",
        out,
    )


if __name__ == "__main__":
    main()
