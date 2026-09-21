"""Benchmark the Swaram tokenizer against BPE and SentencePiece.

Compares the three on tokens-per-word, vocabulary coverage, and fertility
(tokens/akshara). All three are trained and evaluated on the *same* frozen
corpus so the comparison is meaningful:

  * Source: the Thirukkural corpus (1,330 couplets + 3 classical
    commentaries each), fetched deterministically from a fixed public URL
    via src/data_scraper/raw_extractors/thirukkural_extractor.py. Public,
    freely available, reproducible without committing any raw data file.
  * Split: deterministic by document index, not random -- every 10th
    document (index 9, 19, 29, ...) is held out for eval, the rest is
    train. No shuffling, no seed needed: same input always gives the
    same split.

Usage:
    python scripts/benchmark_tokenizers.py --vocab-size 4000 --out docs/SWARAM_BENCHMARK.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))  # noqa: E402

from adhan_slm.core.logging import get_logger  # noqa: E402
from adhan_slm.data import corpus as corpus_mod  # noqa: E402
from adhan_slm.tokenizer.swaram_tokenizer import SwaramTokenizer, segment_aksharas  # noqa: E402
from data_scraper.raw_extractors.thirukkural_extractor import (  # noqa: E402
    THIRUKKURAL_URL,
    build_datasets,
    fetch_thirukkural_data,
)

logger = get_logger(__name__)


def load_frozen_corpus(cache_dir: Path) -> list[str]:
    """Fetch (or reuse a cached) Thirukkural corpus and return it as a list
    of document strings, in stable kural-number order.
    """
    corpus_file = cache_dir / "thirukkural_corpus.jsonl"
    if not corpus_file.exists():
        logger.info("Fetching frozen benchmark corpus from %s ...", THIRUKKURAL_URL)
        kurals = fetch_thirukkural_data()
        build_datasets(kurals, cache_dir)
    else:
        logger.info("Reusing cached frozen corpus at %s", corpus_file)
    return corpus_mod.read_corpus(str(corpus_file))


def deterministic_split(docs: list[str], eval_every: int = 10) -> tuple[list[str], list[str]]:
    """Every `eval_every`-th document (by stable index) is held out for eval."""
    eval_docs = [d for i, d in enumerate(docs) if (i + 1) % eval_every == 0]
    train_docs = [d for i, d in enumerate(docs) if (i + 1) % eval_every != 0]
    return train_docs, eval_docs


# --------------------------------------------------------------------------- #
# Per-tokenizer adapters: each exposes .encode_ids(text) -> List[int] and
# .is_unk(id) -> bool, so the metric functions below can treat all three
# tokenizers identically.
# --------------------------------------------------------------------------- #


class SwaramAdapter:
    name = "Swaram"

    def __init__(self, train_docs: list[str], vocab_size: int):
        self.tok = SwaramTokenizer.train(train_docs, vocab_size=vocab_size, min_freq=2)

    def encode_ids(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def is_unk(self, tid: int) -> bool:
        return tid == self.tok.unk_id

    def vocab_size(self) -> int:
        return len(self.tok)


class BPEAdapter:
    name = "BPE (tokenizers, byte-level)"

    def __init__(self, train_docs: list[str], vocab_size: int):
        from tokenizers import ByteLevelBPETokenizer

        self.tok = ByteLevelBPETokenizer()
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("\n".join(train_docs))
            train_path = f.name
        try:
            self.tok.train([train_path], vocab_size=vocab_size, min_frequency=2)
        finally:
            Path(train_path).unlink(missing_ok=True)

    def encode_ids(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def is_unk(self, tid: int) -> bool:
        # Byte-level BPE has no <unk> -- any byte sequence is representable
        # by construction. This is a real, reportable property of the
        # approach, not a benchmark gap.
        return False

    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()


class SentencePieceAdapter:
    name = "SentencePiece (BPE)"

    def __init__(self, train_docs: list[str], vocab_size: int):
        import sentencepiece as spm

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("\n".join(train_docs))
            train_path = f.name
        model_prefix = train_path.replace(".txt", "")
        try:
            spm.SentencePieceTrainer.train(
                input=train_path,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type="bpe",
                character_coverage=0.9995,
                byte_fallback=False,  # deliberate: see report methodology note
                unk_id=0,
                pad_id=1,
                bos_id=2,
                eos_id=3,
            )
            self.sp = spm.SentencePieceProcessor(model_file=model_prefix + ".model")
        finally:
            Path(train_path).unlink(missing_ok=True)
            Path(model_prefix + ".model").unlink(missing_ok=True)
            Path(model_prefix + ".vocab").unlink(missing_ok=True)

    def encode_ids(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def is_unk(self, tid: int) -> bool:
        return tid == self.sp.unk_id()

    def vocab_size(self) -> int:
        return self.sp.vocab_size()


def compute_metrics(adapter, eval_docs: list[str]) -> dict:
    total_tokens = 0
    total_unk = 0
    total_words = 0
    total_aksharas = 0
    per_doc_fertility = []

    for doc in eval_docs:
        if not doc.strip():
            continue
        ids = adapter.encode_ids(doc)
        n_tok = len(ids)
        n_unk = sum(1 for i in ids if adapter.is_unk(i))
        n_words = len(doc.split())
        n_aks = sum(1 for a in segment_aksharas(doc) if a.strip())

        total_tokens += n_tok
        total_unk += n_unk
        total_words += n_words
        total_aksharas += n_aks
        if n_aks > 0:
            per_doc_fertility.append(n_tok / n_aks)

    return {
        "vocab_size": adapter.vocab_size(),
        "tokens_per_word": total_tokens / total_words if total_words else float("nan"),
        "vocab_coverage_pct": 100.0 * (1 - total_unk / total_tokens) if total_tokens else 0.0,
        "fertility_mean": statistics.mean(per_doc_fertility) if per_doc_fertility else float("nan"),
        "fertility_median": (
            statistics.median(per_doc_fertility) if per_doc_fertility else float("nan")
        ),
        "total_eval_tokens": total_tokens,
        "total_eval_docs": len(eval_docs),
    }


def write_report(out_path: Path, results: dict, meta: dict) -> None:
    lines = []
    lines.append("# Swaram Tokenizer Benchmark: vs BPE and SentencePiece")
    lines.append("")
    lines.append(
        "One-page comparison of the Swaram (akshara-native) tokenizer against a "
        "byte-level BPE tokenizer and a SentencePiece (BPE mode) tokenizer, "
        "trained and evaluated on the same frozen corpus."
    )
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(f"- **Corpus:** Thirukkural ({meta['n_docs']} documents: couplet + 3 classical")
    lines.append(
        f"  commentaries each), fetched from `{THIRUKKURAL_URL}` via "
        "`src/data_scraper/raw_extractors/thirukkural_extractor.py`."
    )
    lines.append(
        f"- **Split:** deterministic by document index -- every 10th document "
        f"held out for eval ({meta['n_eval']} docs), the rest for training "
        f"({meta['n_train']} docs). No randomness, no seed needed."
    )
    lines.append(
        f"- **Target vocab size:** {meta['vocab_size']} for all three tokenizers. "
        "BPE's *actual* vocab (see table) can plateau below target -- the "
        "`tokenizers` library stops merging once no more frequent adjacent "
        "pairs exist in the training data, which can happen before the "
        "target on a small corpus. Swaram and SentencePiece both fill the "
        "full requested budget by falling back to lower-frequency "
        "merges/pieces, so they typically reach the target exactly."
    )
    lines.append(
        "- **Fertility** = tokens / akshara, using Swaram's own rule-based akshara "
        "segmenter (`segment_aksharas`) as the shared denominator for all three "
        "tokenizers -- akshara is the linguistically correct atomic unit for "
        "Tamil, so this keeps the comparison apples-to-apples regardless of "
        "how each tokenizer internally splits text."
    )
    lines.append(
        "- **Vocabulary coverage** = % of eval-set tokens that are NOT the "
        "tokenizer's `<unk>` token. Byte-level BPE has no `<unk>` by "
        "construction (any byte sequence is representable), so it always "
        "reports 100% here -- this is a real property of that approach, not "
        "a benchmark artifact. SentencePiece was trained with "
        "`byte_fallback=False` deliberately, so its coverage number reflects "
        "what its *learned* vocabulary actually covers, not a byte-level "
        "safety net."
    )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| Tokenizer | Vocab size | Tokens/word | Vocab coverage | Fertility (mean) | Fertility (median) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for name, m in results.items():
        lines.append(
            f"| {name} | {m['vocab_size']:,} | {m['tokens_per_word']:.3f} | "
            f"{m['vocab_coverage_pct']:.2f}% | {m['fertility_mean']:.3f} | "
            f"{m['fertility_median']:.3f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    swaram = results.get("Swaram")
    others = [(n, m) for n, m in results.items() if n != "Swaram"]
    if swaram and others:
        best_name, best_other = min(others, key=lambda kv: kv[1]["fertility_mean"])
        if swaram["fertility_mean"] < best_other["fertility_mean"]:
            lines.append(
                f"Swaram achieves the lowest mean fertility "
                f"({swaram['fertility_mean']:.3f} tokens/akshara), beating the "
                f"best general-purpose baseline, {best_name} "
                f"({best_other['fertility_mean']:.3f})."
            )
        else:
            lines.append(
                f"On this run, {best_name} achieves lower mean fertility "
                f"({best_other['fertility_mean']:.3f} tokens/akshara) than Swaram "
                f"({swaram['fertility_mean']:.3f}). This is a genuine result, not "
                "a bug in the benchmark -- see the caveat below for the most "
                "likely reason, and re-running at a larger vocab size / on a "
                "less repetitive corpus is the natural next step to check "
                "whether it holds up."
            )
        lines.append("")
        lines.append(
            "A fertility near or below 1.0 means the tokenizer spends about one "
            "token (or fewer, once merges kick in) per akshara -- the "
            "theoretical floor for a lossless akshara-aware scheme -- rather "
            "than fragmenting Tamil script into multiple sub-akshara pieces "
            "the way byte-level tokenizers with no Tamil-specific prior tend to."
        )
        lines.append("")
    lines.append("## Caveat: this corpus is unusually repetitive")
    lines.append("")
    lines.append(
        "Every Thirukkural document repeats the exact same boilerplate labels "
        '(`"மு. வரதராசனார் உரை:"`, `"சாலமன் பாப்பையா உரை:"`, `"கலைஞர் உரை:"`, '
        '`"திருக்குறள் <N>:"`) 1,330 times. A frequency-driven learner (BPE, '
        "SentencePiece) can dedicate a large slice of a *small* vocabulary "
        "budget to memorizing these exact recurring phrases as single "
        "efficient merges, which flatters tokens-per-word and fertility in a "
        "way that may not hold on more varied, less boilerplate-heavy Tamil "
        "text. Swaram's vocabulary is built akshara-first (247 base units) "
        "before merges, so a larger share of its budget goes to general "
        "coverage rather than corpus-specific memorization -- expected to "
        "matter more at larger vocab sizes and on more varied corpora, but "
        "not tested here. Treat this report as a first, reproducible "
        "data point, not a final verdict; a follow-up with a more varied "
        "corpus and multiple vocab sizes is the natural next step."
    )
    lines.append("")
    lines.append(
        "Numbers above are from a single run on the frozen split described in "
        "Methodology; re-running `scripts/benchmark_tokenizers.py` reproduces "
        "them exactly (same corpus source, same split rule, same vocab size)."
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote report to %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab-size", type=int, default=4000)
    ap.add_argument("--cache-dir", default="data/raw/thirukkural")
    ap.add_argument("--out", default="docs/SWARAM_BENCHMARK.md")
    args = ap.parse_args()

    docs = load_frozen_corpus(Path(args.cache_dir))
    if not docs:
        sys.exit("no documents loaded -- check network access / cache dir")
    train_docs, eval_docs = deterministic_split(docs)
    logger.info("%d total docs -> %d train / %d eval", len(docs), len(train_docs), len(eval_docs))

    adapters = [
        SwaramAdapter(train_docs, args.vocab_size),
        BPEAdapter(train_docs, args.vocab_size),
        SentencePieceAdapter(train_docs, args.vocab_size),
    ]

    results = {}
    for adapter in adapters:
        logger.info("Evaluating %s ...", adapter.name)
        results[adapter.name] = compute_metrics(adapter, eval_docs)
        logger.info("  %s -> %s", adapter.name, results[adapter.name])

    meta = {
        "n_docs": len(docs),
        "n_train": len(train_docs),
        "n_eval": len(eval_docs),
        "vocab_size": args.vocab_size,
    }
    write_report(Path(args.out), results, meta)

    # Also dump raw numbers as JSON for anything that wants to consume them
    # programmatically (e.g. a future CI regression check).
    json_path = Path(args.out).with_suffix(".json")
    json_path.write_text(json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")
    logger.info("Wrote raw metrics to %s", json_path)


if __name__ == "__main__":
    main()
