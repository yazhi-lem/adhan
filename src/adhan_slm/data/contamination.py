"""Benchmark contamination detection and removal.

Adapted from Agaram 2.5B's data pipeline (Murai-Labs/Agaram2.5b).
Detects documents in a pretraining corpus that overlap with benchmark
or evaluation test sets, then removes or flags them.

Three detection methods (in order of strictness):
  1. Exact match (character-level identity)
  2. N-gram containment (word n-gram overlap ratio)
  3. Character n-gram similarity (character 5-gram Jaccard)

Usage::

    from adhan_slm.data.contamination import (
        BenchmarkIndex, ContaminationConfig, detect_contamination
    )

    # Build index from benchmark test/dev files
    index = BenchmarkIndex.from_files([
        "data/benchmarks/flores/ta-en.test.jsonl",
        "data/benchmarks/in22/ta-en.test.jsonl",
    ])

    # Detect contaminated documents in corpus
    contaminated, report = detect_contamination(
        corpus_records, index, cfg=ContaminationConfig()
    )

Or CLI::

    python -m adhan_slm.data.contamination \
        --build-index data/benchmarks/*.jsonl \
        --index-path benchmarks.idx.json \
        --corpus data/raw/corpus.jsonl \
        --output data/clean/corpus.jsonl \
        --report contamination_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from adhan_slm.core.logging import get_logger

logger = get_logger(__name__)

# Default benchmark IDs for Tamil NLLP
DEFAULT_BENCHMARKS = [
    "flores",
    "in22",
    "belebele",
    "xcopa",
    "hellaswag",
    "arc",
    "mmlu",
    "include",
    "multiblimp",
    "milu",
]


@dataclass
class ContaminationConfig:
    """Overlap detection thresholds.

    Agaram 2.5B used exact-match plus word 8-gram and character 5-gram
    similarity filters. For Adhan's smaller scale we default to
    slightly stricter settings.
    """

    # Exact match: any document identical to a benchmark entry is removed
    exact_match: bool = True

    # Word n-gram overlap: if a document shares more than this fraction
    # of its word n-grams with any benchmark entry, it is flagged.
    word_ngram_size: int = 8
    word_ngram_threshold: float = 0.80

    # Character n-gram Jaccard similarity threshold
    char_ngram_size: int = 5
    char_ngram_threshold: float = 0.80

    # Partial-match: if a document contains a benchmark sentence as a
    # substring (after normalisation), flag it.
    substring_match: bool = True


@dataclass
class ContaminationReport:
    """Summary of contamination scan results."""

    total_docs: int = 0
    exact_matches: int = 0
    word_ngram_flags: int = 0
    char_ngram_flags: int = 0
    substring_flags: int = 0
    total_contaminated: int = 0
    unique_contaminated: int = 0
    by_benchmark: Dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.by_benchmark is None:
            self.by_benchmark = {}


def _normalise(text: str) -> str:
    """Normalise text for comparison: lowercase, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _word_ngrams(text: str, n: int) -> Set[str]:
    words = _normalise(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _char_ngrams(text: str, n: int) -> Set[str]:
    text = _normalise(text)
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: Set, b: Set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class BenchmarkIndex:
    """Pre-computed index of benchmark entries for fast contamination checks.

    Holds three representations per benchmark entry:
      - exact normalised text
      - word n-gram sets
      - character n-gram sets
    """

    def __init__(self, entries: Sequence[dict], cfg: Optional[ContaminationConfig] = None):
        self.cfg = cfg or ContaminationConfig()
        self.entries = list(entries)
        self._exact: Set[str] = set()
        self._word_grams: Dict[str, Set[str]] = {}
        self._char_grams: Dict[str, Set[str]] = {}
        self._build()

    def _build(self):
        logger.info("Building benchmark index from %d entries …", len(self.entries))
        for entry in self.entries:
            text = entry.get("text", "")
            if not text:
                continue
            norm = _normalise(text)
            self._exact.add(norm)
            wid = entry.get("id", hashlib.sha256(norm.encode()).hexdigest()[:16])
            self._word_grams[wid] = _word_ngrams(text, self.cfg.word_ngram_size)
            self._char_grams[wid] = _char_ngrams(text, self.cfg.char_ngram_size)
        logger.info(
            "Index ready: %d exact texts, %d n-gram profiles",
            len(self._exact),
            len(self._word_grams),
        )

    @classmethod
    def from_files(
        cls, paths: Sequence[str], text_key: str = "text", cfg: Optional[ContaminationConfig] = None
    ) -> "BenchmarkIndex":
        entries = []
        for path in paths:
            p = Path(path)
            if not p.exists():
                logger.warning("Benchmark file not found: %s", p)
                continue
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    # Handle both flat {"text": "…"} and structured formats
                    text = rec.get(text_key, "")
                    if not text and "translation" in rec:
                        text = rec["translation"].get("ta", "") or rec["translation"].get(
                            "tamil", ""
                        )
                    if not text and "sentence" in rec:
                        text = rec["sentence"]
                    rec["text"] = text
                    entries.append(rec)
        return cls(entries, cfg)

    def save(self, path: str | Path):
        """Serialise index to JSON for reuse."""
        data = {
            "config": asdict(self.cfg),
            "entries": [
                {
                    "id": e.get("id", i),
                    "text": (
                        e.get("text", "")[:500] + "…"
                        if len(e.get("text", "")) > 500
                        else e.get("text", "")
                    ),
                }
                for i, e in enumerate(self.entries)
            ],
            "n_entries": len(self.entries),
        }
        Path(path).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved index metadata to %s", path)

    def check(self, text: str) -> Tuple[bool, Optional[str], List[str]]:
        """Check if ``text`` contaminates any benchmark entry.

        Returns ``(is_contaminated, match_type, benchmark_ids)``.
        """
        flags: List[str] = []
        norm = _normalise(text)

        if self.cfg.exact_match and norm in self._exact:
            return True, "exact", ["exact_match"]

        if self.cfg.substring_match:
            for entry in self.entries:
                bench_text = _normalise(entry.get("text", ""))
                if bench_text and bench_text in norm:
                    flags.append(f"substring:{entry.get('id', '?')}")
                    if len(flags) >= 3:
                        break

        # Word n-gram overlap
        doc_word = _word_ngrams(text, self.cfg.word_ngram_size)
        for wid, bench_word in self._word_grams.items():
            if _jaccard(doc_word, bench_word) >= self.cfg.word_ngram_threshold:
                flags.append(f"word_ngram:{wid}")
                break

        # Character n-gram overlap
        doc_char = _char_ngrams(text, self.cfg.char_ngram_size)
        for wid, bench_char in self._char_grams.items():
            if _jaccard(doc_char, bench_char) >= self.cfg.char_ngram_threshold:
                flags.append(f"char_ngram:{wid}")
                break

        if flags:
            return True, flags[0].split(":")[0], flags
        return False, None, []


def detect_contamination(
    corpus: Iterable[dict],
    index: BenchmarkIndex,
    text_key: str = "text",
    id_key: str = "id",
) -> Tuple[List[dict], ContaminationReport]:
    """Filter contaminated documents from a corpus.

    Returns ``(clean_records, report)``.
    """
    report = ContaminationReport()
    clean: List[dict] = []

    for rec in corpus:
        report.total_docs += 1
        text = rec.get(text_key, "")
        if not text:
            clean.append(rec)
            continue

        contaminated, match_type, flags = index.check(text)
        if contaminated:
            report.total_contaminated += 1
            if match_type == "exact":
                report.exact_matches += 1
            elif match_type == "word_ngram":
                report.word_ngram_flags += 1
            elif match_type == "char_ngram":
                report.char_ngram_flags += 1
            elif match_type == "substring":
                report.substring_flags += 1
            for flag in flags:
                bench = flag.split(":")[0]
                report.by_benchmark[bench] = report.by_benchmark.get(bench, 0) + 1
            continue
        clean.append(rec)

    # Every removed document is counted once, so unique == total here. The two
    # fields stay separate because a report consumed downstream may later be
    # merged with another scan's output, where duplicates are possible.
    report.unique_contaminated = report.total_contaminated
    logger.info(
        "Contamination scan: %d docs → %d clean (removed %d: exact=%d word_ngram=%d char_ngram=%d substring=%d)",
        report.total_docs,
        len(clean),
        report.total_contaminated,
        report.exact_matches,
        report.word_ngram_flags,
        report.char_ngram_flags,
        report.substring_flags,
    )
    return clean, report


def _cli():
    ap = argparse.ArgumentParser(description="Benchmark contamination detection")
    sub = ap.add_subparsers(dest="cmd")

    # Build index
    build_ap = sub.add_parser("build-index", help="Build benchmark index from files")
    build_ap.add_argument("files", nargs="+", help="Benchmark JSONL files")
    build_ap.add_argument("--output", "-o", required=True, help="Output index JSON")
    build_ap.add_argument("--text-key", default="text")

    # Scan corpus
    scan_ap = sub.add_parser("scan", help="Scan corpus for contamination")
    scan_ap.add_argument("--index", "-i", required=True, help="Benchmark index JSON or raw files")
    scan_ap.add_argument("--corpus", "-c", required=True, help="Corpus JSONL")
    scan_ap.add_argument("--output", "-o", required=True, help="Clean output JSONL")
    scan_ap.add_argument("--report", "-r", required=True, help="Report JSON")
    scan_ap.add_argument("--text-key", default="text")
    scan_ap.add_argument("--word-threshold", type=float, default=0.80)
    scan_ap.add_argument("--char-threshold", type=float, default=0.80)
    scan_ap.add_argument("--max-docs", type=int, default=None)

    args = ap.parse_args()

    if args.cmd == "build-index":
        cfg = ContaminationConfig()
        index = BenchmarkIndex.from_files(args.files, text_key=args.text_key, cfg=cfg)
        index.save(args.output)
        return 0

    if args.cmd == "scan":
        cfg = ContaminationConfig(
            word_ngram_threshold=args.word_threshold, char_ngram_threshold=args.char_threshold
        )
        # Load index — if it's a directory, treat it as raw files; if JSON, reload metadata
        idx_path = Path(args.index)
        if idx_path.is_dir() or (idx_path.suffix not in (".json", ".jsonl")):
            # Treat as raw benchmark files pattern
            import glob

            files = list(glob.glob(args.index))
            if not files:
                logger.error("No benchmark files matched: %s", args.index)
                return 1
            index = BenchmarkIndex.from_files(files, text_key=args.text_key, cfg=cfg)
        else:
            # For JSON metadata, rebuild from raw entries (simplification)
            data = json.loads(idx_path.read_text(encoding="utf-8"))
            index = BenchmarkIndex(data.get("entries", []), cfg=cfg)

        def _stream():
            count = 0
            with Path(args.corpus).open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)
                    count += 1
                    if args.max_docs and count >= args.max_docs:
                        break

        clean, report = detect_contamination(_stream(), index, text_key=args.text_key)

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for rec in clean:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Wrote %d clean records to %s", len(clean), out_path)

        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(asdict(report), indent=1, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Wrote report to %s", report_path)
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
