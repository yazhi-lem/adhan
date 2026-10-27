#!/usr/bin/env python3
"""Token-tax benchmark: what does an English-centric tokenizer cost on Tamil?

"Token tax" is the extra number of tokens a tokenizer spends to represent the
same Tamil text compared with a script-native tokenizer. It is paid on every
training token, every API call, and every context window, so a 4x tax is a 4x
bill and a 4x smaller effective context.

This compares, on the *same* held-out Tamil documents:

  * swaram-v2  — our akshara-native tokenizer (the reference, tax = 1.00x)
  * swaram-v1  — the earlier Python artifact (if present)
  * cl100k_base — GPT-4 / text-embedding-3 family (tiktoken)
  * o200k_base  — GPT-4o / o-series (tiktoken)
  * char         — one token per Unicode code point (upper bound)
  * byte         — one token per UTF-8 byte (worst case)

Metrics:
  fertility        tokens per akshara (shared denominator from the swaram layer)
  tokens/word      tokens per Tamil word
  tax vs swaram    tokens_baseline / tokens_swaram  (the headline number)
  context words    Tamil words that fit in a 512-token window
  round-trip       decode(encode(text)) == text

Usage:
    python scripts/benchmark_token_tax.py --corpus data/raw/expanded --docs 400 \
        --out docs/tokenizer/TOKEN_TAX_BENCHMARK_V3.md
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
from adhan_slm.tokenizer.fast_bpe import AksharaBPE  # noqa: E402
from adhan_slm.tokenizer.swaram_tokenizer import (  # noqa: E402
    SwaramTokenizer,
    default_akshara_inventory,
)

logger = get_logger(__name__)

TAMIL_RUN = re.compile(r"[\u0B80-\u0BFF]+")
TAMIL_OR_SPACE = re.compile(r"[\u0B80-\u0BFF\s]+")
CONTEXT_WINDOW = 512


# --- corpus sampling --------------------------------------------------------


def sample_docs(corpus_dir: Path, n: int, max_chars: int, seed: int = 42) -> List[dict]:
    """Deterministic sparse sample across all shards, capped per document."""
    shards = sorted(corpus_dir.glob("*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"no shards in {corpus_dir}")
    docs: List[dict] = []
    # Aim for the requested total across all shards; take a contiguous slice per
    # shard so the sample is not capped at one document per shard on big archives.
    per_shard = max(1, (n + len(shards) - 1) // len(shards))
    rng = random.Random(seed)
    for shard in shards:
        rows: List[dict] = []
        with shard.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = rec.get("text", "")
                if text and len(TAMIL_OR_SPACE.findall(text)) >= 200:
                    rows.append({"text": text[:max_chars], "source": rec.get("source", "?")})
        rng.shuffle(rows)
        docs.extend(rows[:per_shard])
    rng.shuffle(docs)
    return docs[:n]


# --- tokenizers -------------------------------------------------------------


class CharTokenizer:
    name = "char"

    def encode(self, text: str) -> List[int]:
        return [ord(c) for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(i) for i in ids)


class ByteTokenizer:
    name = "byte"

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


class TiktokenTokenizer:
    def __init__(self, encoding_name: str):
        import tiktoken

        self.enc = tiktoken.get_encoding(encoding_name)
        self.name = encoding_name

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, disallowed_special=())

    def decode(self, ids: Sequence[int]) -> str:
        return self.enc.decode(list(ids))


def load_swaram(path: Path):
    if (path / "tokenizer.json").exists():
        return AksharaBPE.load(path, inventory=default_akshara_inventory()), "rust"
    if (path / "vocab.json").exists():
        return (
            SwaramTokenizer.from_files(str(path / "vocab.json"), str(path / "merges.txt")),
            "python",
        )
    raise FileNotFoundError(f"no tokenizer artifacts in {path}")


# --- measurement ------------------------------------------------------------


def akshara_count(ref_tok, text: str) -> int:
    """Akshara denominator from the native segmenter (shared across tokenizers)."""
    return sum(1 for a in ref_tok._segment(text) if a.strip())


def tamil_words(text: str) -> int:
    return sum(1 for w in TAMIL_RUN.findall(text) if len(w) > 0)


def measure(name: str, tok, ref_tok, docs: List[dict]) -> dict:
    texts = [d["text"] for d in docs]
    n_chars = sum(len(t) for t in texts)
    n_aks = sum(akshara_count(ref_tok, t) for t in texts)
    n_words = sum(tamil_words(t) for t in texts)

    t0 = time.perf_counter()
    encoded = [tok.encode(t) for t in texts]
    elapsed = time.perf_counter() - t0
    n_tokens = sum(len(ids) for ids in encoded)

    lossless = 0
    for t, ids in zip(texts, encoded):
        try:
            if tok.decode(ids) == t:
                lossless += 1
        except Exception:
            pass

    return {
        "name": name,
        "tokens": n_tokens,
        "chars": n_chars,
        "aksharas": n_aks,
        "words": n_words,
        "fertility": n_tokens / max(n_aks, 1),
        "tokens_per_word": n_tokens / max(n_words, 1),
        "chars_per_token": n_chars / max(n_tokens, 1),
        "context_words": CONTEXT_WINDOW / max(n_tokens / max(n_words, 1), 1e-9),
        "lossless_rate": lossless / max(len(texts), 1),
        "kchar_per_s": n_chars / max(elapsed, 1e-9) / 1000.0,
    }


def render(rows: Sequence[dict], docs: int) -> str:
    ref = next(r for r in rows if r["name"].startswith("swaram"))
    for r in rows:
        r["tax"] = r["tokens"] / max(ref["tokens"], 1)

    gpt4 = next((r for r in rows if "cl100k" in r["name"]), None)
    gpt4o = next((r for r in rows if "o200k" in r["name"]), None)
    byte = next((r for r in rows if r["name"].startswith("byte")), None)
    headline = [
        "# Token Tax — what English-centric tokenizers cost on Tamil",
        "",
        "> **Headline**: the swaram tokenizer is the reference at **1.00x**."
        + (f" GPT-4 (cl100k) spends **{gpt4['tax']:.2f}x** the tokens," if gpt4 else "")
        + (f" GPT-4o (o200k) **{gpt4o['tax']:.2f}x**," if gpt4o else "")
        + (f" and raw UTF-8 bytes **{byte['tax']:.2f}x**." if byte else ""),
        "> For a fixed Tamil corpus that is the same multiple in training compute,",
        "> API cost, generation latency, and context the model can actually see.",
        "",
    ]
    lines = headline + [
        "Generated by `scripts/benchmark_token_tax.py` on a deterministic sparse",
        f"sample of the packed Tamil archive ({docs:,} documents). Every row sees the",
        "same text; the swaram tokenizer is the 1.00x reference.",
        "",
        "| tokenizer | fertility (tok/akshara) | tok / word | chars / token | tax vs swaram | words in 512 ctx | round-trip | throughput |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {name} | {f:.3f} | {tpw:.3f} | {cpt:.2f} | **{tax:.2f}x** | "
            "{cw:,.0f} | {ll:.1%} | {kcs:,.0f} kchar/s |".format(
                name=r["name"],
                f=r["fertility"],
                tpw=r["tokens_per_word"],
                cpt=r["chars_per_token"],
                tax=r["tax"],
                cw=r["context_words"],
                ll=r["lossless_rate"],
                kcs=r["kchar_per_s"],
            )
        )
    lines += [
        "",
        "## Reading the table",
        "",
        "- **fertility** — tokens per akshara (Tamil grapheme cluster). Lower is better;",
        "  A script-native tokenizer can sit below 1.0 because Layer B merges whole morphemes;\n  the tax is what matters, not the absolute value.",
        "- **tax vs swaram** — the headline. `4.0x` means an English tokenizer spends four",
        "  tokens for the same Tamil text, i.e. four times the training compute, four times",
        "  the API bill, and a context window that holds a quarter as much Tamil.",
        "- **words in 512 ctx** — how much Tamil actually fits in a 512-token window.",
        "  This is the number users feel: a 4x tax means a quarter of the dialogue.",
        "- **round-trip** — `decode(encode(text)) == text`; a tokenizer that silently",
        "  loses characters cannot be trusted for training targets or user output.",
        "",
        "## Cost implications",
        "",
        "| effect | formula | meaning |",
        "|---|---|---|",
        "| training compute | tax x tokens | pretraining a fixed Tamil corpus costs `tax`x more |",
        "| inference bill | tax x output tokens | every generated Tamil response costs `tax`x more |",
        "| effective context | window / tax | a 8k window holds `1/tax` of the Tamil |",
        "| latency | tax x decode steps | generation takes `tax`x longer for the same text |",
        "",
        "The swaram tokenizer removes this tax by making the akshara the atomic unit",
        "instead of an English-motivated byte pair.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Token-tax benchmark")
    ap.add_argument("--corpus", default="data/raw/expanded")
    ap.add_argument("--docs", type=int, default=400)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--swaram", default="data/final/tamil_slm_v2")
    ap.add_argument("--swaram-v1", default="data/final/tamil_slm")
    ap.add_argument("--out", default="docs/tokenizer/TOKEN_TAX_BENCHMARK_V3.md")
    args = ap.parse_args()

    docs = sample_docs(Path(args.corpus), args.docs, args.max_chars)
    logger.info("sampled %d docs", len(docs))

    ref_tok, _ = load_swaram(Path(args.swaram))
    ref_label = f"swaram ({Path(args.swaram).name})"
    candidates: List[tuple] = [(ref_label, ref_tok)]

    v1 = Path(args.swaram_v1)
    if v1.exists():
        try:
            v1_tok, _ = load_swaram(v1)
            candidates.append((f"swaram ({v1.name})", v1_tok))
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not load swaram-v1: %s", exc)

    from importlib.util import find_spec

    if find_spec("tiktoken") is not None:
        for enc_name, label in (
            ("cl100k_base", "GPT-4 (cl100k)"),
            ("o200k_base", "GPT-4o (o200k)"),
        ):
            try:
                candidates.append((label, TiktokenTokenizer(enc_name)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not load %s: %s", enc_name, exc)
    else:
        logger.warning("tiktoken not installed — skipping GPT baselines")

    candidates.append(("char (codepoint)", CharTokenizer()))
    candidates.append(("byte (UTF-8)", ByteTokenizer()))

    rows = []
    for name, tok in candidates:
        logger.info("measuring %s ...", name)
        rows.append(measure(name, tok, ref_tok, docs))

    md = render(rows, len(docs))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(md)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
