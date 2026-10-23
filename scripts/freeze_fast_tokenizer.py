#!/usr/bin/env python3
"""Freeze a Rust-backed Swaram/Aksharam tokenizer and pack the corpus — streaming.

``prepare_slm_corpus.py`` is fine for small corpora but reads everything into
memory and uses the pure-Python encoder (~7 kchar/s, i.e. 43 h for the 1.06G
character Tamil archive). This script is the archive-scale path:

  1. train the akshara BPE on a *bounded, stratified sample* of the shard
     directory (merge statistics converge long before the corpus runs out, and
     ``BpeTrainer`` is single-pass, so memory stays flat),
  2. measure fertility and lossless round-trip on held-out documents,
  3. stream-pack every document into ``train.bin`` / ``val.bin`` using the Rust
     encoder (~3.5 Mchar/s, minutes instead of hours), writing in chunks so the
     packed corpus never has to fit in RAM.

Outputs, alongside ``tokenizer.json``/``vocab.json``/``merges.txt``:

  train.bin, train.bin.manifest.json
  val.bin,   val.bin.manifest.json
  datasheet.json      metrics + provenance for the frozen artifact

Usage:
    python scripts/freeze_fast_tokenizer.py \\
        --corpus data/raw/expanded --out data/final/tamil_slm_v2 \\
        --vocab-size 32000 --train-chars 300000000 --byte-fallback
"""

from __future__ import annotations

import argparse
import array
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.core.logging import get_logger  # noqa: E402
from adhan_slm.data.packing import PackedShard, pack_stream  # noqa: E402
from adhan_slm.tokenizer import indic_akshara  # noqa: E402
from adhan_slm.tokenizer.fast_bpe import AksharaBPE  # noqa: E402

logger = get_logger(__name__)

_SCRIPTS = {"swaram": indic_akshara.TAMIL, "aksharam": indic_akshara.DEVANAGARI}


def iter_corpus(corpus_dir: Path) -> Iterator[str]:
    """Yield documents from every ``*.jsonl`` shard, in shard order."""
    import json as _json

    shards = sorted(corpus_dir.glob("*.jsonl")) or [corpus_dir]
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
                    yield text


def stratified_sample(
    corpus_dir: Path,
    max_chars: int,
    stride: int,
) -> Iterator[str]:
    """Every ``stride``-th document until ``max_chars`` characters are yielded."""
    chars = 0
    for i, doc in enumerate(iter_corpus(corpus_dir)):
        if i % stride:
            continue
        yield doc
        chars += len(doc)
        if chars >= max_chars:
            return


class StreamingShardWriter:
    """Append fixed-length sequences to a ``.bin`` shard without buffering all of them."""

    def __init__(self, path: Path, seq_len: int, vocab_size: int) -> None:
        self.path = path
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.typecode = "H" if vocab_size <= 0xFFFF else "I"
        self.dtype = "<u2" if self.typecode == "H" else "<u4"
        self._fh = path.open("wb")
        self.n_sequences = 0
        self._buf: List[int] = []

    def add(self, seq: Sequence[int]) -> None:
        if len(seq) != self.seq_len:
            raise ValueError(f"sequence length {len(seq)} != seq_len {self.seq_len}")
        self._buf.extend(seq)
        self.n_sequences += 1
        if len(self._buf) >= self.seq_len * 2048:  # flush ~2048 sequences at a time
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        arr = array.array(self.typecode, self._buf)
        arr.tofile(self._fh)
        self._buf.clear()

    def close(self) -> PackedShard:
        self._flush()
        self._fh.close()
        shard = PackedShard(
            path=str(self.path),
            seq_len=self.seq_len,
            n_sequences=self.n_sequences,
            n_tokens=self.n_sequences * self.seq_len,
            vocab_size=self.vocab_size,
            dtype=self.dtype,
        )
        Path(str(self.path) + ".manifest.json").write_text(
            json.dumps(shard.as_dict(), indent=2), encoding="utf-8"
        )
        return shard


def code_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _train(args, corpus_dir: Path, inventory, script, stride: int):
    """Learn merges on a stratified sample; returns ``(tokenizer, seconds)``."""
    t0 = time.time()
    sample = stratified_sample(corpus_dir, args.train_chars or 10**18, stride)
    tok = AksharaBPE.train(
        sample,
        inventory,
        vocab_size=args.vocab_size,
        script=script,
        byte_fallback=args.byte_fallback,
        min_frequency=args.min_frequency,
    )
    secs = time.time() - t0
    logger.info(
        "froze tokenizer: %s tokens, %s merges (byte_fallback=%s) in %.0fs",
        f"{len(tok):,}",
        f"{len(tok.merges):,}",
        args.byte_fallback,
        secs,
    )
    return tok, secs


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze a fast akshara tokenizer + packed shards")
    ap.add_argument("--corpus", default="data/raw/expanded", help="dir of *.jsonl shards")
    ap.add_argument("--out", default="data/final/tamil_slm_v2")
    ap.add_argument("--tokenizer", choices=list(_SCRIPTS), default="swaram")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument(
        "--train-chars",
        type=int,
        default=300_000_000,
        help="characters of stratified sample used to learn merges (0 = whole corpus)",
    )
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--byte-fallback", dest="byte_fallback", action="store_true", default=True)
    ap.add_argument("--no-byte-fallback", dest="byte_fallback", action="store_false")
    ap.add_argument("--skip-pack", action="store_true", help="freeze the tokenizer only")
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="stop packing once this many tokens are written (0 = whole corpus); "
        "use it to emit a small shard set for benchmarking",
    )
    ap.add_argument(
        "--reuse-tokenizer",
        default=None,
        help="load tokenizer.json from this dir instead of training "
        "(re-pack an existing vocabulary at a different corpus size)",
    )
    ap.add_argument(
        "--morphology-guard",
        action="store_true",
        help="drop merges that cut through known Tamil suffixes; costs some "
        "compression, raises stem/suffix boundary agreement",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    corpus_dir = Path(args.corpus)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    script = _SCRIPTS[args.tokenizer]

    from adhan_slm.tokenizer.swaram_tokenizer import default_akshara_inventory

    if args.tokenizer == "swaram":
        inventory = default_akshara_inventory()
    else:
        from adhan_slm.tokenizer.aksharam_tokenizer import default_aksharam_inventory

        inventory = default_aksharam_inventory()

    # --- sample sizing --------------------------------------------------------
    stats_path = corpus_dir / "stats.json"
    n_docs = n_chars = 0
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        n_docs = int(stats.get("documents", 0))
        n_chars = int(stats.get("characters", 0))
    stride = 1
    if args.train_chars and n_chars and n_docs:
        avg = max(1, n_chars // max(1, n_docs))
        want_docs = max(1, args.train_chars // avg)
        stride = max(1, n_docs // want_docs)
    logger.info(
        "corpus: %s docs / %s chars | training sample: stride=%d, cap=%s chars",
        f"{n_docs:,}" if n_docs else "?",
        f"{n_chars:,}" if n_chars else "?",
        stride,
        f"{args.train_chars:,}" if args.train_chars else "none",
    )

    # --- tokenizer: train a new one, or reuse a frozen one ----------------------
    # Reuse exists so a smaller shard set (e.g. --max-tokens 30000000) can be packed
    # later against the exact same vocabulary; re-training on a sample could shift
    # merge ranks and make the two artifacts incomparable.
    if args.reuse_tokenizer:
        tok = AksharaBPE.load(args.reuse_tokenizer, inventory=inventory, script=script)
        train_secs = 0.0
        logger.info(
            "reused tokenizer %s: %s tokens, byte_fallback=%s",
            args.reuse_tokenizer,
            f"{len(tok):,}",
            tok.byte_fallback,
        )
    else:
        tok, train_secs = _train(args, corpus_dir, inventory, script, stride)
    if args.morphology_guard:
        before = len(tok.merges)
        tok = tok.prune_morphology()
        logger.info(
            "morphology guard: pruned %d merges (%d -> %d)",
            before - len(tok.merges),
            before,
            len(tok.merges),
        )
    tok.save(out)

    # --- evaluate on held-out docs -------------------------------------------
    import random

    rng = random.Random(args.seed)
    heldout: List[str] = []
    for i, doc in enumerate(iter_corpus(corpus_dir)):
        if i % 97:  # sparse, deterministic touch of the whole archive
            continue
        heldout.append(doc)
        if len(heldout) >= 300:
            break
    rng.shuffle(heldout)
    eval_docs = heldout[:200]

    t0 = time.time()
    ok = 0
    ferts: List[float] = []
    n_eval_chars = 0
    for doc in eval_docs:
        ids = tok.encode(doc)
        n_eval_chars += len(doc)
        if tok.decode(ids) == doc:
            ok += 1
        ferts.append(tok.fertility(doc))
    eval_secs = time.time() - t0
    mean_fert = sum(ferts) / len(ferts) if ferts else 0.0
    throughput = n_eval_chars / eval_secs if eval_secs else 0.0
    logger.info(
        "held-out: lossless %d/%d, mean fertility %.3f tok/akshara, %.0f kchar/s",
        ok,
        len(eval_docs),
        mean_fert,
        throughput / 1000,
    )

    datasheet = {
        "tokenizer": args.tokenizer,
        "backend": "rust",
        "script": script.name,
        "vocab_size": len(tok),
        "n_merges": len(tok.merges),
        "inventory_size": len(inventory),
        "byte_fallback": args.byte_fallback,
        "morphology_guard": args.morphology_guard,
        "train_chars": args.train_chars,
        "train_stride": stride,
        "train_seconds": round(train_secs, 1),
        "eval_docs": len(eval_docs),
        "eval_lossless_rate": round(ok / len(eval_docs), 4) if eval_docs else None,
        "mean_fertility": round(mean_fert, 4),
        "fertility_target": 1.15,
        "encode_kchar_per_s": round(throughput / 1000, 1),
        "corpus": str(corpus_dir),
        "code_sha": code_sha(),
    }

    # --- pack ---------------------------------------------------------------
    if not args.skip_pack:
        logger.info("packing corpus (seq_len=%d, val_frac=%.3f) ...", args.seq_len, args.val_frac)
        t0 = time.time()
        train_w = StreamingShardWriter(out / "train.bin", args.seq_len, len(tok))
        val_w = StreamingShardWriter(out / "val.bin", args.seq_len, len(tok))
        n_docs_seen = 0
        n_tokens = 0
        batch: List[Tuple[str, bool]] = []

        def flush_batch() -> None:
            nonlocal n_tokens
            if not batch:
                return
            docs_ = [d for d, _ in batch]
            is_val = [v for _, v in batch]
            for doc, ids, val in zip(docs_, tok.encode_batch(docs_), is_val):
                seq_ids = [tok.vocab.get("<bos>", 0), *ids, tok.vocab.get("<eos>", 0)]
                n_tokens += len(seq_ids)
                writer = val_w if val else train_w
                for seq in pack_stream(iter(seq_ids), args.seq_len):
                    writer.add(seq)
            batch.clear()

        for i, doc in enumerate(iter_corpus(corpus_dir)):
            # Stop cleanly at the token budget so a benchmark shard set can be
            # produced in seconds instead of packing the whole archive.
            if args.max_tokens and n_tokens >= args.max_tokens:
                logger.info("reached --max-tokens=%s; stopping", f"{args.max_tokens:,}")
                break
            # Deterministic, content-independent split: hash of the doc index.
            is_val = (hash((args.seed, i)) % 1000) / 1000.0 < args.val_frac
            batch.append((doc, is_val))
            n_docs_seen += 1
            if len(batch) >= 512:
                flush_batch()
                if n_docs_seen % (512 * 50) == 0:
                    logger.info("  %s docs, %s tokens...", f"{n_docs_seen:,}", f"{n_tokens:,}")
        flush_batch()

        train_shard = train_w.close()
        val_shard = val_w.close()
        pack_secs = time.time() - t0
        logger.info(
            "packed %s docs -> train %s seqs / val %s seqs, %s tokens in %.0fs",
            f"{n_docs_seen:,}",
            f"{train_shard.n_sequences:,}",
            f"{val_shard.n_sequences:,}",
            f"{n_tokens:,}",
            pack_secs,
        )
        datasheet.update(
            {
                "documents": n_docs_seen,
                "tokens": n_tokens,
                "train_sequences": train_shard.n_sequences,
                "val_sequences": val_shard.n_sequences,
                "seq_len": args.seq_len,
                "val_frac": args.val_frac,
                "pack_seconds": round(pack_secs, 1),
                "max_tokens": args.max_tokens or None,
                "tokenizer_reused_from": args.reuse_tokenizer,
            }
        )

    (out / "datasheet.json").write_text(json.dumps(datasheet, indent=2), encoding="utf-8")
    logger.info("datasheet -> %s", out / "datasheet.json")
    logger.info(
        "done: vocab=%s fert=%.3f lossless=%.1f%%",
        f"{len(tok):,}",
        mean_fert,
        100 * ok / len(eval_docs) if eval_docs else 0,
    )


if __name__ == "__main__":
    main()
