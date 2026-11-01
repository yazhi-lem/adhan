"""Tokenizer benchmark harness for Tamil (adhan#74, part 2).

Measures, on a frozen eval corpus (see ``eval_corpus.py``), what a tokenizer costs on Tamil:

  fertility        tokens per word and tokens per akshara
  grapheme splits  share of Tamil grapheme clusters (aksharas) that more than one token
                   touches, i.e. a token boundary cuts the cluster, or a multi-byte
                   character is spread over several byte tokens
  cost             tokens per 1,000 sentences and, if a price is given, USD per 1,000
                   sentences

Works with the Swaram tokenizer and any tokenizer that can report character offsets:
Hugging Face fast tokenizers (``transformers``) and ``tokenizers.Tokenizer``. Two
dependency-free baselines (code point and UTF-8 byte) are included.

Everything runs locally. The harness never sends text anywhere. Hugging Face tokenizers
are loaded from a local path unless ``--allow-hub`` is given (which downloads the
tokenizer files, never your text).

Usage:
    python -m adhan_slm.eval.tokenizer_bench --corpus data/eval/tokenizer_bench.jsonl \\
        --swaram vocab.json merges.txt --hf path/to/tokenizer.json --baselines \\
        --price-per-million 0.5 --out-json results.json --out-md results.md

Grapheme clusters come from ``segment_aksharas`` (NFC text, vowel signs and pulli attach
to their consonant). Text is NFC-normalised before tokenizing, so offsets are consistent.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from adhan_slm.tokenizer.swaram_tokenizer import (
    BYTE_TOKENS,
    WORD_MARK,
    SwaramTokenizer,
    segment_aksharas,
)

Span = Tuple[int, int]

_TAMIL_LO, _TAMIL_HI = 0x0B80, 0x0BFF


class TokenizerAdapter(Protocol):
    """What the harness needs from a tokenizer."""

    name: str

    def spans(self, text: str) -> List[Span]:
        """One ``(start, end)`` character span per token, over the given NFC text.

        Tokens with no characters (special tokens) may be reported as ``(0, 0)``; they are
        counted as tokens but touch no cluster. Byte tokens of one character all report
        that character's span.
        """
        ...


# -- adapters -----------------------------------------------------------------------


class SwaramAdapter:
    """Swaram tokenizer. Span count always equals ``len(tok.encode(text))``."""

    def __init__(self, tok: SwaramTokenizer, name: str = "swaram") -> None:
        self.tok = tok
        self.name = name
        self._has_bytes = all(t in tok.vocab for t in BYTE_TOKENS)

    def spans(self, text: str) -> List[Span]:
        out: List[Span] = []
        pos = 0
        for piece in self.tok.tokenize(text):
            n = len(piece)
            if piece == WORD_MARK:  # stands for one space character
                out.append((pos, pos + 1))
                pos += 1
            elif piece in self.tok.vocab:
                out.append((pos, pos + n))
                pos += n
            elif self._has_bytes:
                for i, ch in enumerate(piece):
                    out.extend([(pos + i, pos + i + 1)] * len(ch.encode("utf-8")))
                pos += n
            else:  # <unk>
                out.append((pos, pos + n))
                pos += n
        return out


class CodepointAdapter:
    """One token per Unicode code point. Baseline: splits every multi-code-point akshara."""

    name = "codepoint"

    def spans(self, text: str) -> List[Span]:
        return [(i, i + 1) for i in range(len(text))]


class ByteAdapter:
    """One token per UTF-8 byte. Baseline: three tokens for every Tamil code point."""

    name = "utf8-byte"

    def spans(self, text: str) -> List[Span]:
        out: List[Span] = []
        for i, ch in enumerate(text):
            out.extend([(i, i + 1)] * len(ch.encode("utf-8")))
        return out


class TokenizersAdapter:
    """A ``tokenizers.Tokenizer`` (Hugging Face ``tokenizers`` library)."""

    def __init__(self, tok: Any, name: str) -> None:
        self.tok = tok
        self.name = name

    def spans(self, text: str) -> List[Span]:
        enc = self.tok.encode(text, add_special_tokens=False)
        return [(int(s), int(e)) for s, e in enc.offsets]


class TransformersAdapter:
    """A ``transformers`` fast tokenizer (needs ``return_offsets_mapping``)."""

    def __init__(self, tok: Any, name: str) -> None:
        if not getattr(tok, "is_fast", False):
            raise ValueError(
                f"{name}: a fast tokenizer is required for character offsets "
                "(use_fast=True, or pass a tokenizer.json)"
            )
        self.tok = tok
        self.name = name

    def spans(self, text: str) -> List[Span]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [(int(s), int(e)) for s, e in enc["offset_mapping"]]


# -- metrics ------------------------------------------------------------------------


def _is_tamil(cluster: str) -> bool:
    return any(_TAMIL_LO <= ord(c) <= _TAMIL_HI for c in cluster)


@dataclass
class Counts:
    sentences: int = 0
    words: int = 0
    aksharas: int = 0
    tokens: int = 0
    tamil_clusters: int = 0
    tamil_clusters_split: int = 0
    clusters: int = 0
    clusters_split: int = 0

    def add(self, other: "Counts") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


def measure_sentence(adapter: TokenizerAdapter, text: str) -> Counts:
    """Count tokens and split clusters for one sentence (text is NFC-normalised first)."""
    text = unicodedata.normalize("NFC", text)
    c = Counts(sentences=1, words=len(text.split()))

    clusters = segment_aksharas(text)
    owner: List[int] = []  # char index -> cluster index
    for ci, cl in enumerate(clusters):
        owner.extend([ci] * len(cl))
    is_real = [bool(cl.strip()) for cl in clusters]
    c.aksharas = sum(is_real)

    spans = adapter.spans(text)
    c.tokens = len(spans)
    touches = [0] * len(clusters)
    for s, e in spans:
        s, e = max(0, s), min(len(text), e)
        if e <= s:
            continue
        for ci in {owner[i] for i in range(s, e)}:
            touches[ci] += 1

    for ci, cl in enumerate(clusters):
        if not is_real[ci]:
            continue
        split = touches[ci] > 1
        c.clusters += 1
        c.clusters_split += split
        if _is_tamil(cl):
            c.tamil_clusters += 1
            c.tamil_clusters_split += split
    return c


def summarize(c: Counts, price_per_million: Optional[float] = None) -> Dict[str, Any]:
    """Turn raw counts into the reported metrics."""

    def ratio(a: int, b: int) -> Optional[float]:
        return a / b if b else None

    per_1k = ratio(c.tokens * 1000, c.sentences)
    out: Dict[str, Any] = {
        "sentences": c.sentences,
        "words": c.words,
        "aksharas": c.aksharas,
        "tokens": c.tokens,
        "tokens_per_word": ratio(c.tokens, c.words),
        "tokens_per_akshara": ratio(c.tokens, c.aksharas),
        "tokens_per_1k_sentences": per_1k,
        "tamil_clusters": c.tamil_clusters,
        "tamil_cluster_split_rate": ratio(c.tamil_clusters_split, c.tamil_clusters),
        "cluster_split_rate": ratio(c.clusters_split, c.clusters),
    }
    if price_per_million is not None and per_1k is not None:
        out["usd_per_1k_sentences"] = per_1k * price_per_million / 1_000_000
    return out


@dataclass
class Result:
    tokenizer: str
    overall: Dict[str, Any]
    per_register: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def evaluate(
    adapter: TokenizerAdapter,
    records: Sequence[Dict[str, str]],
    price_per_million: Optional[float] = None,
) -> Result:
    """Evaluate one tokenizer on corpus records (``{"register": ..., "text": ...}``)."""
    total = Counts()
    by_reg: Dict[str, Counts] = {}
    for rec in records:
        c = measure_sentence(adapter, rec["text"])
        total.add(c)
        by_reg.setdefault(rec.get("register", "all"), Counts()).add(c)
    return Result(
        tokenizer=adapter.name,
        overall=summarize(total, price_per_million),
        per_register={r: summarize(c, price_per_million) for r, c in sorted(by_reg.items())},
    )


# -- corpus, reporting --------------------------------------------------------------


def load_corpus(path: Path) -> List[Dict[str, str]]:
    """Read the frozen corpus (JSONL with ``register`` and ``text`` per line)."""
    records: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if not isinstance(rec.get("text"), str) or not isinstance(rec.get("register"), str):
                raise ValueError(f"{path}:{n}: each line needs string 'register' and 'text'")
            records.append(rec)
    if not records:
        raise ValueError(f"{path}: corpus is empty")
    return records


def _fmt(v: Optional[float], spec: str) -> str:
    return "n/a" if v is None else format(v, spec)


def to_markdown(results: Sequence[Result], price_per_million: Optional[float] = None) -> str:
    """Comparison tables: one overall table, then one per register."""
    cost_col = price_per_million is not None
    head = "| Tokenizer | Tokens/word | Tokens/akshara | Tamil clusters split | Tokens/1k sent. |"
    sep = "|---|---:|---:|---:|---:|"
    if cost_col:
        head += " USD/1k sent. |"
        sep += "---:|"

    def row(name: str, m: Dict[str, Any]) -> str:
        cells = [
            name,
            _fmt(m["tokens_per_word"], ".2f"),
            _fmt(m["tokens_per_akshara"], ".3f"),
            _fmt(
                (
                    None
                    if m["tamil_cluster_split_rate"] is None
                    else m["tamil_cluster_split_rate"] * 100
                ),
                ".1f",
            )
            + ("%" if m["tamil_cluster_split_rate"] is not None else ""),
            _fmt(m["tokens_per_1k_sentences"], ",.0f"),
        ]
        if cost_col:
            cells.append(_fmt(m.get("usd_per_1k_sentences"), ".5f"))
        return "| " + " | ".join(cells) + " |"

    lines = ["# Tamil tokenizer benchmark", "", "## All registers", "", head, sep]
    lines += [row(r.tokenizer, r.overall) for r in results]
    registers = sorted({reg for r in results for reg in r.per_register})
    for reg in registers:
        lines += ["", f"## {reg}", "", head, sep]
        lines += [row(r.tokenizer, r.per_register[reg]) for r in results if reg in r.per_register]
    lines += [
        "",
        "Lower is better for every column. 'Tamil clusters split' is the share of Tamil "
        "aksharas that more than one token touches (n/a when a register has no Tamil script).",
        "",
    ]
    return "\n".join(lines)


# -- CLI ----------------------------------------------------------------------------


def _load_adapters(args: argparse.Namespace) -> List[TokenizerAdapter]:
    adapters: List[TokenizerAdapter] = []
    if args.swaram:
        tok = SwaramTokenizer.from_files(args.swaram[0], args.swaram[1])
        adapters.append(SwaramAdapter(tok))
    for spec in args.hf or []:
        p = Path(spec)
        if p.suffix == ".json":
            from tokenizers import Tokenizer

            adapters.append(TokenizersAdapter(Tokenizer.from_file(str(p)), p.parent.name or p.stem))
        else:
            from transformers import AutoTokenizer

            hf = AutoTokenizer.from_pretrained(spec, local_files_only=not args.allow_hub)
            adapters.append(TransformersAdapter(hf, spec))
    if args.baselines:
        adapters += [CodepointAdapter(), ByteAdapter()]
    if not adapters:
        raise SystemExit("no tokenizers given: use --swaram, --hf and/or --baselines")
    return adapters


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--corpus", required=True, help="frozen eval corpus (JSONL)")
    ap.add_argument("--swaram", nargs=2, metavar=("VOCAB", "MERGES"))
    ap.add_argument(
        "--hf",
        action="append",
        help="tokenizer.json, or a local dir for transformers.AutoTokenizer (repeatable)",
    )
    ap.add_argument("--baselines", action="store_true", help="add codepoint and byte baselines")
    ap.add_argument("--allow-hub", action="store_true", help="let transformers download files")
    ap.add_argument(
        "--price-per-million", type=float, help="USD per 1M tokens, for the cost column"
    )
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    args = ap.parse_args(argv)

    records = load_corpus(Path(args.corpus))
    results = [evaluate(a, records, args.price_per_million) for a in _load_adapters(args)]

    md = to_markdown(results, args.price_per_million)
    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(
                [
                    {"tokenizer": r.tokenizer, "overall": r.overall, "per_register": r.per_register}
                    for r in results
                ],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.out_md:
        Path(args.out_md).write_text(md, encoding="utf-8")
    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
