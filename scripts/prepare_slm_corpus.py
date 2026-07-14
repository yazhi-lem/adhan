#!/usr/bin/env python3
"""Prepare a native-Tamil SLM corpus: freeze the swaram tokenizer + pack shards.

This closes the Phase 1 ("freeze vocab.json + merges", `adhan-tok-v1` artifact) and
Phase 2 ("tokenize → packed fixed-length sequences → sharded") gap in
ROADMAP_JAX_SLM.md. Given a corpus (txt / jsonl / directory), it:

  1. trains the Swaram (Tamil/Dravidian) or Aksharam (Hindi/Indic) tokenizer,
  2. freezes ``vocab.json`` + ``merges.txt``,
  3. measures fertility (tokens/akshara) on a held-out sample,
  4. tokenizes + packs the corpus into ``train.bin`` / ``val.bin`` shards, and
  5. writes a ``datasheet.json`` (sources, counts, fertility, code SHA) — the data
     card the roadmap asks for.

Everything is pure-python (stdlib only); numpy just speeds up shard I/O if present.

    python scripts/prepare_slm_corpus.py \
        --corpus data/raw/tamil/ --out data/final/tamil_slm \
        --vocab-size 12000 --seq-len 1024 --val-frac 0.02
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.data import corpus as corpus_mod
from adhan_slm.data import packing
from adhan_slm.tokenizer import SwaramTokenizer
from adhan_slm.tokenizer.aksharam_tokenizer import AksharamTokenizer

_TOKENIZERS = {"swaram": SwaramTokenizer, "aksharam": AksharamTokenizer}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _mean_fertility(tok, docs, cap: int = 500) -> float:
    sample = docs[:cap]
    vals = [tok.fertility(d) for d in sample if d.strip()]
    vals = [v for v in vals if v > 0]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze SLM tokenizer + pack shards")
    ap.add_argument("--corpus", required=True,
                    help="txt/jsonl file, or directory of them")
    ap.add_argument("--out", required=True, help="output dir for tokenizer + shards")
    ap.add_argument("--tokenizer", choices=list(_TOKENIZERS), default="swaram")
    ap.add_argument("--vocab-size", type=int, default=12000)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap #documents (debug / dry runs)")
    ap.add_argument("--whole-file-docs", action="store_true",
                    help="treat each .txt file as one document (default: per line)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    TokCls = _TOKENIZERS[args.tokenizer]

    print(f"[1/5] reading corpus from {args.corpus} ...")
    docs = corpus_mod.read_corpus(
        args.corpus, line_documents=not args.whole_file_docs, limit=args.limit)
    if not docs:
        sys.exit(f"no documents found under {args.corpus}")
    n_val = max(1, int(len(docs) * args.val_frac))
    val_docs = docs[:n_val]
    train_docs = docs[n_val:]
    print(f"      {len(docs):,} docs  ->  {len(train_docs):,} train / {len(val_docs):,} val")

    print(f"[2/5] training {args.tokenizer} tokenizer (vocab {args.vocab_size}) ...")
    tok = TokCls.train(train_docs, vocab_size=args.vocab_size, min_freq=args.min_freq)
    vocab_path = out / "vocab.json"
    merges_path = out / "merges.txt"
    tok.save(str(vocab_path), str(merges_path))
    print(f"      froze vocab={len(tok):,} merges={len(tok.merges):,} -> {vocab_path.name}, {merges_path.name}")

    print("[3/5] measuring fertility on held-out sample ...")
    fert = _mean_fertility(tok, val_docs)
    flag = "OK" if fert < 1.15 else "ABOVE TARGET (<1.15)"
    print(f"      mean fertility = {fert:.3f} tokens/akshara  [{flag}]")

    print(f"[4/5] tokenizing + packing to seq_len={args.seq_len} ...")
    train_seqs = packing.pack_documents(train_docs, tok, seq_len=args.seq_len)
    val_seqs = packing.pack_documents(val_docs, tok, seq_len=args.seq_len)
    if not train_seqs:
        sys.exit("corpus too small to fill even one packed sequence; add more text "
                 "or lower --seq-len")
    train_shard = packing.write_shard(
        train_seqs, out / "train.bin", seq_len=args.seq_len, vocab_size=len(tok))
    val_shard = None
    if val_seqs:
        val_shard = packing.write_shard(
            val_seqs, out / "val.bin", seq_len=args.seq_len, vocab_size=len(tok))
    print(f"      train.bin: {train_shard.n_sequences:,} seqs / {train_shard.n_tokens:,} tokens"
          + (f"   val.bin: {val_shard.n_sequences:,} seqs" if val_shard else "   (val too small to pack)"))

    print("[5/5] writing datasheet.json ...")
    datasheet = {
        "corpus_source": str(args.corpus),
        "tokenizer": args.tokenizer,
        "vocab_size": len(tok),
        "n_merges": len(tok.merges),
        "seq_len": args.seq_len,
        "n_documents": len(docs),
        "n_train_documents": len(train_docs),
        "n_val_documents": len(val_docs),
        "train_tokens": train_shard.n_tokens,
        "val_tokens": val_shard.n_tokens if val_shard else 0,
        "mean_fertility": round(fert, 4),
        "fertility_target": 1.15,
        "code_sha": _git_sha(),
    }
    (out / "datasheet.json").write_text(json.dumps(datasheet, indent=2), encoding="utf-8")
    print(f"      -> {out / 'datasheet.json'}")
    print(f"\ndone. train with:\n  python -m adhan_slm.training.train_jax "
          f"--config <config with data.shards: {out}>")


if __name__ == "__main__":
    main()
