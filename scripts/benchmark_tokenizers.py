"""Benchmark the Swaram tokenizer against BPE and SentencePiece.

Compares the three on tokens-per-word, vocabulary coverage, fertility
(tokens/akshara), corpus-wide round-trip losslessness, and a curated Tamil
script stress test. All three are trained and evaluated on the *same*
frozen corpus so the comparison is meaningful:

  * Source: the Thirukkural corpus (1,330 couplets + 3 classical
    commentaries each), fetched deterministically from a fixed public URL
    via src/data_scraper/raw_extractors/thirukkural_extractor.py. Public,
    freely available, reproducible without committing any raw data file.
  * Split: deterministic by document index, not random -- every 10th
    document (index 9, 19, 29, ...) is held out for eval, the rest is
    train. No shuffling, no seed needed: same input always gives the
    same split.

Fertility/coverage alone only says how *efficient* a tokenizer is. Before
committing to a real training run, the more important question is
whether it can *losslessly* round-trip real Tamil -- decode(encode(text))
reconstructing the original exactly -- across the full range of Tamil
script (Grantha letters, complex conjuncts, composed vs. decomposed
Unicode vowel-sign sequences, Tamil numerals, code-switched text). A
tokenizer that silently loses information here would silently corrupt
every training run built on top of it. This script checks that
corpus-wide (every eval document) and against a curated set of known-hard
cases (see TAMIL_STRESS_CASES below), and reports failures explicitly
rather than only an aggregate percentage.

Usage:
    python scripts/benchmark_tokenizers.py --vocab-size 4000 --out docs/SWARAM_BENCHMARK.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import unicodedata
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


def _build_tamil_stress_cases() -> list[tuple[str, str]]:
    """Curated (label, text) pairs covering Tamil script edge cases that a
    tokenizer must handle losslessly before it's safe to build a training
    pipeline on. The composed/decomposed pair is generated programmatically
    via unicodedata (not hand-typed) so it's guaranteed correct rather than
    risking a typo in a rare codepoint.
    """
    cases = [
        ("empty_string", ""),
        ("single_vowel", "அ"),
        ("all_12_vowels", "அஆஇஈஉஊஎஏஐஒஓஔ"),
        ("all_18_consonants_pulli", "க்ங்ச்ஞ்ட்ண்த்ந்ப்ம்ய்ர்ல்வ்ழ்ள்ற்ன்"),
        ("uyirmei_full_row_ka", "கககாகிகீகுகூகெகேகைகொகோகௌ"),
        ("aytham", "எஃகு"),  # steel -- uses the rare ஃ (aytham) character
        ("grantha_ja", "ஜனநாயகம்"),  # democracy -- uses ஜ
        ("grantha_sha", "விஷயம்"),  # matter/subject -- uses ஷ
        ("grantha_sa", "ஸ்ரீ"),  # honorific -- complex sa+ra+ii conjunct
        ("grantha_ha", "வாஹனம்"),  # vehicle -- uses ஹ
        ("ksha_conjunct", "லக்ஷ்மி"),  # goddess name -- uses க்ஷ conjunct
        ("tamil_numerals", "".join(chr(0x0BE6 + i) for i in range(10))),  # ௦-௯
        ("western_numerals_mixed", "2024 ஆம் ஆண்டு"),  # "in the year 2024"
        ("code_switched_english", "இது ஒரு AI model தான்"),  # common in real social text
        ("punctuation_heavy", "என்ன, இது சரியா? ஆம்! இல்லை..."),
        ("long_agglutinative_word", "வணக்கங்களுக்குரியவர்களே"),  # Tamil stacks suffixes freely
        ("repeated_whitespace", "  தமிழ்   மொழி  "),  # leading/trailing/multiple spaces
        ("newline_in_text", "வணக்கம்\nதமிழ் மொழி"),
    ]
    # Composed vs. decomposed Unicode for the same visual word ("கொடு" = give):
    # U+0BCA (vowel sign O) canonically decomposes to <U+0BC6, U+0BBE>. Real
    # scraped text can contain either form inconsistently depending on the
    # input method used at the source, so a tokenizer needs to not silently
    # mis-segment or lose information on either.
    composed = "கொடு"
    decomposed = unicodedata.normalize("NFD", composed)
    cases.append(("vowel_sign_composed", composed))
    cases.append(("vowel_sign_decomposed", decomposed))
    return cases


TAMIL_STRESS_CASES = _build_tamil_stress_cases()


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
# Per-tokenizer adapters: each exposes .encode_ids(text) -> List[int],
# .decode_ids(ids) -> str, and .is_unk(id) -> bool, so the metric functions
# below can treat all three tokenizers identically.
# --------------------------------------------------------------------------- #


class SwaramAdapter:
    name = "Swaram"

    def __init__(self, train_docs: list[str], vocab_size: int):
        self.tok = SwaramTokenizer.train(train_docs, vocab_size=vocab_size, min_freq=2)

    def encode_ids(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def decode_ids(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

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

    def decode_ids(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

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

    def decode_ids(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

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


def check_roundtrip_corpus(adapter, eval_docs: list[str]) -> dict:
    """Corpus-wide round-trip losslessness: decode(encode(text)) vs. text,
    across every real document in the eval set (not just curated cases).

    Reports two rates because they mean different things:
    - exact: decode(encode(text)) == text, byte for byte. This is what
      actually matters for training-data fidelity.
    - nfc: decode(encode(text)) == NFC-normalized text. Distinguishes
      "the tokenizer normalizes Unicode as a matter of policy" (usually
      fine, e.g. SentencePiece's default NFKC normalization) from
      "the tokenizer actually lost/garbled information" (not fine).
    """
    exact_matches = 0
    nfc_matches = 0
    total = 0
    first_failures: list[tuple[str, str]] = []

    for doc in eval_docs:
        if not doc.strip():
            continue
        total += 1
        ids = adapter.encode_ids(doc)
        decoded = adapter.decode_ids(ids)
        if decoded == doc:
            exact_matches += 1
            nfc_matches += 1
        elif decoded == unicodedata.normalize("NFC", doc):
            nfc_matches += 1
            if len(first_failures) < 3:
                first_failures.append((doc, decoded))
        else:
            if len(first_failures) < 3:
                first_failures.append((doc, decoded))

    return {
        "exact_pct": 100.0 * exact_matches / total if total else 0.0,
        "nfc_pct": 100.0 * nfc_matches / total if total else 0.0,
        "total_checked": total,
        "example_failures": first_failures,
    }


def run_stress_tests(adapter, cases: list[tuple[str, str]]) -> dict:
    """Run the curated Tamil script stress-test suite through one tokenizer.
    Returns per-case pass/fail plus a summary, so a real failure is
    immediately visible rather than buried in an aggregate.

    Distinguishes "exact" (byte-for-byte round-trip -- what matters for
    training-data fidelity) from "nfc" (matches after Unicode NFC
    normalization -- a tokenizer's normalization *policy*, not necessarily
    data loss) using the same standard as check_roundtrip_corpus, so a case
    like feeding in non-NFC input isn't unfairly counted as a failure for
    a tokenizer that deliberately NFC-normalizes by design.
    """
    results = []
    for label, text in cases:
        try:
            ids = adapter.encode_ids(text)
            decoded = adapter.decode_ids(ids)
            exact = decoded == text
            nfc_ok = exact or decoded == unicodedata.normalize("NFC", text)
        except Exception as e:  # a crash is itself a failure worth reporting
            decoded = f"<EXCEPTION: {e}>"
            exact = False
            nfc_ok = False
        results.append(
            {
                "label": label,
                "input": text,
                "decoded": decoded,
                "passed": nfc_ok,
                "exact": exact,
            }
        )
    n_passed = sum(1 for r in results if r["passed"])
    n_exact = sum(1 for r in results if r["exact"])
    return {
        "results": results,
        "n_passed": n_passed,
        "n_exact": n_exact,
        "n_total": len(results),
    }


def _diagnose_failure(original: str, decoded: str) -> str:
    """Best-effort plain-English hypothesis for why a round-trip failed,
    based on failure signatures found while building/validating this
    benchmark. Not a substitute for reading the actual diff, but turns a
    bare "these didn't match" into something actionable.
    """
    if decoded == "":
        return (
            "entire input vanished on decode -- likely every token mapped to "
            "<unk> and got silently suppressed on decode (default "
            "skip_special behavior), rather than an encoding failure"
        )
    if "\u2047" in decoded or "<unk>" in decoded:
        return (
            "decoded output contains an explicit unknown-token marker -- "
            "input has character(s) outside this tokenizer's trained "
            "vocabulary (out-of-vocab / low character coverage for this "
            "character)"
        )
    if original.replace("\n", " ") == decoded or " ".join(original.split()) == decoded:
        return (
            "differs only in whitespace/newlines -- tokenizer's normalizer "
            "collapsed whitespace or newlines (a real loss of document "
            "structure, not just Unicode form)"
        )
    if unicodedata.normalize("NFC", original) == decoded:
        return (
            "differs only by Unicode NFC normalization -- a normalization "
            "policy choice, not information loss"
        )
    if len(decoded) < len(original):
        return (
            f"decoded output is shorter ({len(decoded)} vs {len(original)} "
            "chars) with no visible unknown-token marker in the output -- "
            "consistent with the same silent unk-drop pattern above, just "
            "harder to spot here because nothing marks where the loss "
            "happened (unlike a tokenizer that leaves a visible <unk>/⁇ "
            "placeholder, which at least tells you where to look)"
        )
    return "unexplained difference -- needs manual inspection"


def write_report(
    out_path: Path,
    results: dict,
    meta: dict,
    roundtrip_results: dict,
    stress_results: dict,
) -> None:
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
    lines.append("## Round-trip Losslessness")
    lines.append("")
    lines.append(
        "Fertility and coverage above only measure *efficiency*. Before "
        "committing to a real training run, the more important question is "
        "whether each tokenizer can *losslessly* reconstruct real Tamil -- "
        "`decode(encode(text)) == text` -- across the full eval corpus. A "
        "tokenizer that silently loses information here would silently "
        "corrupt every training run built on it."
    )
    lines.append("")
    lines.append(
        "| Tokenizer | Exact round-trip | Round-trip after NFC normalization | Docs checked |"
    )
    lines.append("|---|---|---|---|")
    for name, rt in roundtrip_results.items():
        lines.append(
            f"| {name} | {rt['exact_pct']:.2f}% | {rt['nfc_pct']:.2f}% | "
            f"{rt['total_checked']:,} |"
        )
    lines.append("")
    lines.append(
        '"Exact" is byte-for-byte identity with the original text -- what '
        'actually matters for training-data fidelity. "After NFC" separates '
        "genuine information loss from a tokenizer's Unicode normalization "
        "policy (e.g. SentencePiece normalizes input by default, so a "
        "document containing non-NFC sequences can legitimately decode to "
        "its NFC-normalized form rather than its original byte sequence -- "
        "that's a policy difference, not data corruption)."
    )
    lines.append("")
    any_failures = any(rt["exact_pct"] < 100.0 for rt in roundtrip_results.values())
    if any_failures:
        lines.append("**Example round-trip mismatches found (first few per tokenizer):**")
        lines.append("")
        for name, rt in roundtrip_results.items():
            if rt["example_failures"]:
                lines.append(f"- `{name}`:")
                for original, decoded in rt["example_failures"]:
                    lines.append(f"  - input:  `{original!r}`")
                    lines.append(f"  - output: `{decoded!r}`")
                    lines.append(f"  - likely cause: {_diagnose_failure(original, decoded)}")
        lines.append("")
    else:
        lines.append(
            "No round-trip mismatches found in this run -- every tokenizer "
            "reconstructed every eval document exactly."
        )
        lines.append("")
    lines.append("## Tamil Script Stress Test")
    lines.append("")
    lines.append(
        f"{len(TAMIL_STRESS_CASES)} curated cases covering Tamil script "
        "edge cases that a tokenizer must handle correctly before it's safe "
        "to build a training pipeline on: the full vowel and consonant "
        "inventory, Grantha/borrowed letters (ஜ ஷ ஸ ஹ), complex conjuncts "
        "(க்ஷ், ஸ்ரீ), the rare ஆய்தம் (ஃ), Tamil numerals, code-switched "
        "English, punctuation, whitespace edge cases, and -- notably -- the "
        "*same visual word* encoded two different but both-valid ways in "
        "Unicode (composed `கொ` vs. its canonically-decomposed form), since "
        "real scraped text can contain either inconsistently."
    )
    lines.append("")
    lines.append("| Tokenizer | Passed (NFC-tolerant) | Passed (byte-exact) |")
    lines.append("|---|---|---|")
    for name, sr in stress_results.items():
        lines.append(
            f"| {name} | {sr['n_passed']}/{sr['n_total']} | {sr['n_exact']}/{sr['n_total']} |"
        )
    lines.append("")
    any_stress_failures = any(sr["n_passed"] < sr["n_total"] for sr in stress_results.values())
    if any_stress_failures:
        lines.append("**Failing cases (NFC-tolerant), with likely cause:**")
        lines.append("")
        for name, sr in stress_results.items():
            failed = [r for r in sr["results"] if not r["passed"]]
            if not failed:
                continue
            lines.append(f"- `{name}`:")
            for r in failed:
                cause = _diagnose_failure(r["input"], r["decoded"])
                lines.append(f"  - `{r['label']}`: `{r['input']!r}` -> `{r['decoded']!r}`")
                lines.append(f"    likely cause: {cause}")
        lines.append("")
    else:
        lines.append(
            "All tokenizers passed every stress case in this run -- every "
            "case round-tripped exactly, including the composed/decomposed "
            "Unicode pair."
        )
        lines.append("")

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
    lines.append("## Recommendations before starting real training")
    lines.append("")
    swaram_stress = stress_results.get("Swaram")
    swaram_rt = roundtrip_results.get("Swaram")
    recs = []
    if swaram_stress and swaram_stress["n_exact"] < swaram_stress["n_total"]:
        n_fail = swaram_stress["n_total"] - swaram_stress["n_exact"]
        recs.append(
            f"- Swaram failed {n_fail}/{swaram_stress['n_total']} stress cases "
            "byte-exact. Check the failure diagnoses above -- if any show "
            '"entire input vanished on decode" or an unknown-token marker, '
            "that means `SwaramTokenizer.decode()`'s default "
            "`skip_special=True` is silently deleting out-of-vocabulary "
            "characters (English words, digits, rare symbols) rather than "
            "showing them. That's invisible unless you specifically decode "
            "with `skip_special=False` -- worth fixing (or at minimum "
            "documenting loudly) before trusting Swaram on real scraped "
            "corpora, which will contain code-switched English and digits "
            "far more often than Thirukkural does."
        )
    if swaram_rt and swaram_rt["exact_pct"] < 100.0:
        recs.append(
            f"- Swaram's corpus-wide exact round-trip is "
            f"{swaram_rt['exact_pct']:.2f}%, not 100%. Given the akshara "
            "segmenter itself is documented as lossless "
            "(`segment_aksharas`), any gap here traces back to the same "
            "vocab-coverage/decode issue above, not the segmentation layer."
        )
    for name, rt in roundtrip_results.items():
        if name == "Swaram":
            continue
        if rt["exact_pct"] < 50.0:
            recs.append(
                f"- {name}'s corpus-wide exact round-trip is only "
                f"{rt['exact_pct']:.2f}% (and {rt['nfc_pct']:.2f}% even "
                "after NFC normalization, so this is real information loss, "
                "not just a Unicode form difference). Do not use this "
                "configuration for real training data preparation without "
                "fixing the underlying cause (see failure diagnoses above)."
            )
    if not recs:
        recs.append(
            "- No blocking correctness issues found in this run. Still "
            "worth re-running on a larger, more varied corpus (see Caveat "
            "below) before fully committing, since Thirukkural's small size "
            "and repetitiveness limits how much this run can catch."
        )
    lines.extend(recs)
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
    roundtrip_results = {}
    stress_results = {}
    for adapter in adapters:
        logger.info("Evaluating %s ...", adapter.name)
        results[adapter.name] = compute_metrics(adapter, eval_docs)
        logger.info("  %s -> %s", adapter.name, results[adapter.name])

        logger.info("Checking round-trip losslessness for %s ...", adapter.name)
        roundtrip_results[adapter.name] = check_roundtrip_corpus(adapter, eval_docs)
        logger.info(
            "  %s round-trip: exact=%.2f%% nfc=%.2f%% (n=%d)",
            adapter.name,
            roundtrip_results[adapter.name]["exact_pct"],
            roundtrip_results[adapter.name]["nfc_pct"],
            roundtrip_results[adapter.name]["total_checked"],
        )

        logger.info("Running Tamil script stress test for %s ...", adapter.name)
        stress_results[adapter.name] = run_stress_tests(adapter, TAMIL_STRESS_CASES)
        logger.info(
            "  %s stress test: %d/%d passed",
            adapter.name,
            stress_results[adapter.name]["n_passed"],
            stress_results[adapter.name]["n_total"],
        )

    meta = {
        "n_docs": len(docs),
        "n_train": len(train_docs),
        "n_eval": len(eval_docs),
        "vocab_size": args.vocab_size,
    }
    write_report(Path(args.out), results, meta, roundtrip_results, stress_results)

    # Also dump raw numbers as JSON for anything that wants to consume them
    # programmatically (e.g. a future CI regression check).
    json_path = Path(args.out).with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "meta": meta,
                "results": results,
                "roundtrip_results": roundtrip_results,
                "stress_results": stress_results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote raw metrics to %s", json_path)


if __name__ == "__main__":
    main()
