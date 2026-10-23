#!/usr/bin/env python3
"""Count (and optionally extract) *pure Tamil* tokens in the archive.

"328M tokens" is not the same claim as "328M Tamil tokens". A token can be Tamil,
Tamil mixed with Latin (code-switching), or not Tamil at all (digits, punctuation,
byte-fallback pieces for emoji and rare scripts). That distinction matters when
budgeting a mixture: a token that is a URL fragment or an English word teaches the
model nothing about Tamil morphology, but it costs exactly as much compute.

This tool classifies **tokens** (not documents) with the frozen v2 tokenizer:

  pure      every codepoint is Tamil script (0B80–0BFF) plus joiners/pulli
  mixed     contains Tamil *and* something else (கூல், AI, 2024)
  non-tamil no Tamil at all (Latin words, digits, punctuation, <0x..> bytes)
  special   <bos>/<eos>/▁ style structural tokens

Usage:
    # measure a sample (fast; stride-derived across every shard)
    python scripts/pure_tamil_tokens.py --corpus data/raw/expanded --docs 20000

    # full pass, and write a pure-Tamil-only corpus for a clean-subset run
    python scripts/pure_tamil_tokens.py --corpus data/raw/expanded --all \\
        --emit data/raw/pure_tamil --min-doc-purity 0.95
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adhan_slm.core.logging import get_logger  # noqa: E402
from adhan_slm.tokenizer.fast_bpe import AksharaBPE  # noqa: E402
from adhan_slm.tokenizer.swaram_tokenizer import (  # noqa: E402
    BYTE_TOKENS,
    SPECIAL_TOKENS,
    WORD_MARK,
    default_akshara_inventory,
)

logger = get_logger(__name__)

_BYTE_TOKEN_SET = set(BYTE_TOKENS)
_SPECIALS = set(SPECIAL_TOKENS) | {WORD_MARK}


def _is_tamil_cp(cp: int) -> bool:
    return 0x0B80 <= cp <= 0x0BFF or cp in (0x200C, 0x200D)


def classify_token(token: str) -> str:
    """Bucket one tokenizer piece: pure / mixed / non-tamil / special.

    Two corrections that a first version got wrong, both of which under-reported
    purity by ~2x:

    * ``▁`` is a *word marker*, not content. Almost every Tamil word token carries
      one, so counting it as non-Tamil classified most pure Tamil tokens as
      "mixed". It is stripped before classification.
    * structural tokens (``<bos>``, ``▁`` alone, byte-fallback pieces) carry no
      linguistic content and must not sit in the denominator — otherwise "pure
      Tamil share" is really "pure Tamil share × punctuation tax".
    """
    stripped = token.replace(WORD_MARK, "")
    if token in _SPECIALS or token in _BYTE_TOKEN_SET:
        return "special" if token in _SPECIALS else "non-tamil"
    if not stripped:
        return "special"
    has_tamil = False
    has_other = False
    for ch in stripped:
        if _is_tamil_cp(ord(ch)):
            has_tamil = True
        elif not ch.isspace():
            has_other = True
    if has_tamil and has_other:
        return "mixed"
    if has_tamil:
        return "pure"
    return "non-tamil"


@dataclass
class SourceStats:
    docs: int = 0
    tokens: int = 0
    pure: int = 0
    mixed: int = 0
    non_tamil: int = 0
    special: int = 0
    chars: int = 0

    def add(self, text: str, ids: Sequence[int], inv: Dict[int, str]) -> None:
        self.docs += 1
        self.chars += len(text)
        self.tokens += len(ids)
        for tid in ids:
            bucket = classify_token(inv.get(tid, ""))
            if bucket == "pure":
                self.pure += 1
            elif bucket == "mixed":
                self.mixed += 1
            elif bucket == "non-tamil":
                self.non_tamil += 1
            else:
                self.special += 1

    def as_dict(self) -> Dict[str, float]:
        content = max(1, self.tokens - self.special)
        return {
            "docs": self.docs,
            "chars": self.chars,
            "tokens": self.tokens,
            "content_tokens": self.tokens - self.special,
            "special_tokens": self.special,
            "pure_tamil_tokens": self.pure,
            # shares are of *content* tokens: the numbers that describe the text
            "pure_share": round(self.pure / content, 4),
            "mixed_share": round(self.mixed / content, 4),
            "non_tamil_share": round(self.non_tamil / content, 4),
            "special_share": round(self.special / max(1, self.tokens), 4),
            "tokens_per_char": round(self.tokens / max(1, self.chars), 4),
        }


@dataclass
class DocPurity:
    """Distribution of per-document purity, which is what a filter would threshold."""

    buckets: Counter = field(default_factory=Counter)

    def add(self, pure: int, total: int) -> None:
        share = pure / max(1, total)
        if share >= 0.99:
            self.buckets["0.99-1.00"] += 1
        elif share >= 0.95:
            self.buckets["0.95-0.99"] += 1
        elif share >= 0.90:
            self.buckets["0.90-0.95"] += 1
        elif share >= 0.70:
            self.buckets["0.70-0.90"] += 1
        else:
            self.buckets["<0.70"] += 1


def iter_docs(corpus_dir: Path, stride: int = 1) -> Iterator[Dict]:
    index = 0
    for shard in sorted(corpus_dir.glob("*.jsonl")):
        with shard.open(encoding="utf-8") as fh:
            for line in fh:
                index += 1
                if stride > 1 and index % stride:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = rec.get("text") or ""
                if text:
                    yield rec


def main() -> None:
    ap = argparse.ArgumentParser(description="Count / extract pure Tamil tokens")
    ap.add_argument("--corpus", default="data/raw/expanded")
    ap.add_argument("--tokenizer", default="data/final/tamil_slm_v2")
    ap.add_argument("--docs", type=int, default=20000, help="sample size (stride-derived)")
    ap.add_argument("--all", action="store_true", help="full pass instead of a sample")
    ap.add_argument("--batch", type=int, default=256, help="documents per encode batch")
    ap.add_argument("--emit", default=None, help="write documents above --min-doc-purity here")
    ap.add_argument("--min-doc-purity", type=float, default=0.95)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    stats_path = corpus / "stats.json"
    archive = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    total_docs = int(archive.get("documents", 0))
    stride = 1 if args.all else max(1, total_docs // max(1, args.docs))

    tok = AksharaBPE.load(args.tokenizer, inventory=default_akshara_inventory())
    inv = {i: t for t, i in tok.vocab.items()}
    logger.info(
        "tokenizer %s: vocab=%s byte_fallback=%s | corpus=%s stride=%s",
        args.tokenizer,
        f"{len(tok):,}",
        tok.byte_fallback,
        corpus,
        stride,
    )

    per_source: Dict[str, SourceStats] = {}
    overall = SourceStats()
    purity = DocPurity()
    emitted = 0
    out_fh = None
    if args.emit:
        out = Path(args.emit)
        out.mkdir(parents=True, exist_ok=True)
        out_fh = (out / "corpus-00000.jsonl").open("w", encoding="utf-8")

    batch: List[Dict] = []

    def flush() -> None:
        nonlocal emitted
        if not batch:
            return
        texts = [d["text"] for d in batch]
        for rec, ids in zip(batch, tok.encode_batch(texts)):
            src = rec.get("source", "?")
            st = per_source.setdefault(src, SourceStats())
            st.add(rec["text"], ids, inv)
            overall.add(rec["text"], ids, inv)
            buckets = [classify_token(inv.get(i, "")) for i in ids]
            pure = buckets.count("pure")
            content = sum(1 for b in buckets if b != "special")
            purity.add(pure, content)
            if out_fh is not None and pure / max(1, content) >= args.min_doc_purity:
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                emitted += 1
        batch.clear()

    for rec in iter_docs(corpus, stride):
        batch.append(rec)
        if len(batch) >= args.batch:
            flush()
    flush()
    if out_fh is not None:
        out_fh.close()

    total_pure = overall.pure
    report = {
        "corpus": str(corpus),
        "sample": {
            "all_documents": bool(args.all),
            "stride": stride,
            "docs": overall.docs,
            "chars": overall.chars,
            "tokenizer": args.tokenizer,
        },
        "overall": overall.as_dict(),
        "doc_purity_buckets": dict(purity.buckets),
        "per_source": {
            k: v.as_dict() for k, v in sorted(per_source.items(), key=lambda kv: -kv[1].tokens)
        },
    }

    # Scale the sample's purity to the archive so the number is usable in planning.
    if not args.all and archive.get("approx_tokens_swaram") and overall.tokens:
        content_share = 1 - overall.special / max(1, overall.tokens)
        scale = archive["approx_tokens_swaram"] / overall.tokens
        pure_share = total_pure / max(1, overall.tokens)
        report_content = total_pure / max(1, overall.tokens - overall.special)
        report["archive_projection"] = {
            "tokens_total": archive["approx_tokens_swaram"],
            "pure_tamil_tokens_est": int(total_pure * scale),
            "pure_share_of_content": round(report_content, 4),
            "pure_share_of_all_tokens": round(pure_share, 4),
            "content_token_share": round(content_share, 4),
            "non_tamil_tokens_est": int(overall.non_tamil * scale),
            "mixed_tokens_est": int(overall.mixed * scale),
        }

    content = max(1, overall.tokens - overall.special)
    print(
        f"\npure Tamil tokens: {overall.pure:,} of {content:,} content tokens "
        f"({100 * overall.pure / content:.2f}%), plus {overall.special:,} structural tokens"
    )
    if "archive_projection" in report:
        p = report["archive_projection"]
        print(
            f"archive projection: ~{p['pure_tamil_tokens_est']:,} pure of "
            f"{p['tokens_total']:,}  (mixed ~{p['mixed_tokens_est']:,}, "
            f"non-Tamil ~{p['non_tamil_tokens_est']:,})"
        )
    print("\nper-source purity:")
    for name, d in list(report["per_source"].items())[:10]:
        print(
            f"  {name:34} tokens={d['tokens']:>12,}  pure={100 * d['pure_share']:5.1f}%  "
            f"mixed={100 * d['mixed_share']:4.1f}%  non-tamil={100 * d['non_tamil_share']:4.1f}%"
        )
    print(f"\ndocument purity buckets: {report['doc_purity_buckets']}")
    if args.emit:
        print(f"emitted {emitted:,} documents (>={args.min_doc_purity:.0%} pure) -> {args.emit}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[written] {args.json}")


if __name__ == "__main__":
    main()
