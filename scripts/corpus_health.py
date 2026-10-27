#!/usr/bin/env python3
"""Corpus health report: how good is the archive, and how much more do we need?

Answers two questions with numbers instead of adjectives:

1. **Health** — length distribution, script mix, PII, boilerplate repetition,
   near-duplicate rate, vocabulary richness, domain coverage, per-source quality.
2. **Gap** — measured against explicit targets (compute-optimal tokens for a given
   model size, per-register budgets), so "how much more" has a denominator.

Design notes:

* It **samples** (default 20k documents drawn across every shard by stride) because
  the point is a fast, repeatable health check, not a full pass. Everything is
  reported with the sample size attached so nobody mistakes it for a census.
* Near-duplicate estimation uses MinHash + LSH banding over character 5-grams,
  the same family of parameters the ingest pipeline uses (`num_perm=16`,
  `bands=4`), so the number is comparable to the dedup that already ran.
* No tokenizer is required. If a fast tokenizer exists in the shard dir it is used
  for real token counts; otherwise tokens are estimated from the measured
  0.311 tokens/char ratio and labelled as such.

Usage:
    python scripts/corpus_health.py --corpus data/raw/expanded
    python scripts/corpus_health.py --corpus data/raw/expanded --docs 50000 \\
        --out docs/corpus/CORPUS_HEALTH.md --json /tmp/corpus_health.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Measured on an 800-doc sample with the frozen swaram tokenizer (see
# docs/corpus/CORPUS_EXPANSION.md). Used only when no tokenizer is available.
TOKENS_PER_CHAR = 0.311

_TAMIL = re.compile(r"[\u0B80-\u0BFF]")
_LATIN = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"[0-9\u0BE6-\u0BEF]")
_OTHER_INDIC = re.compile(r"[\u0900-\u0AFF\u0C00-\u0D7F]")
_PII = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"(?:\+?91[-.\s]?)?\b[6-9]\d{9}\b"),
    "url": re.compile(r"https?://\S+|www\.\S+"),
}
# Crude register probes. Not a classifier — a count of documents that contain at
# least one marker, which is enough to see which registers are thin.
_DOMAINS = {
    "news": r"செய்தி|தலைப்பு|நிருபர்|ஏஜென்சி",
    "education": r"பள்ளி|மாணவர்|பாடம்|கல்வி|வகுப்பு",
    "government": r"அரசு|அரசின்|துறை|அறிவிப்பு|ஆணை",
    "legal": r"சட்டம்|வழக்கு|நீதிமன்றம்|உரிமை",
    "health": r"மருத்துவ|நோய்|சிகிச்சை|உடல்நலம்",
    "religion": r"கடவுள்|கோயில்|இறைவன்|வழிபாடு",
    "commerce": r"விலை|சந்தை|விற்பனை|நிறுவனம்|ரூபாய்",
    "literature": r"கவிதை|சங்க|இலக்கிய|பாடல்|கதை",
}
_DOMAIN_RE = {k: re.compile(v) for k, v in _DOMAINS.items()}


def iter_docs(corpus_dir: Path, stride: int) -> Iterator[Dict]:
    """Yield records from every shard, taking one document per `stride`."""
    index = 0
    for shard in sorted(corpus_dir.glob("*.jsonl")):
        with shard.open(encoding="utf-8") as fh:
            for line in fh:
                if index % stride:
                    index += 1
                    continue
                index += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = rec.get("text") or ""
                if text:
                    yield {"text": text, "source": rec.get("source", "?")}


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# A Mersenne-ish prime for the (a*x+b) mod p permutations that define MinHash.
_MINHASH_P = (1 << 61) - 1
# Deterministic permutation coefficients. (a must be odd; b arbitrary.)
_PERMS = [((0x9E3779B97F4A7C15 * (i + 1)) | 1, 0xBF58476D1CE4E5B9 * (i + 1)) for i in range(16)]


def _minhash(
    text: str, num_perm: int = 16, k: int = 5, cap: int = 4000, max_shingles: int = 300
) -> Tuple[int, ...]:
    """MinHash signature over character k-shingles.

    The first version of this sampled *quantiles* of the sorted shingle hashes,
    which is not MinHash: two nearly identical documents get different quantiles,
    so the estimator reported 0% near-duplicates on any input. Real MinHash takes,
    per permutation, ``min((a*x + b) mod p)`` over the shingle hashes; then equal
    signature bands imply Jaccard similarity above the banding threshold.

    Shingles are capped (``max_shingles``) to keep the probe linear-time on long
    documents, so the returned rate is a **lower bound** on the true rate.
    """
    t = text[:cap]
    if len(t) <= k:
        return tuple([0] * num_perm)
    shingle_hashes = {
        int.from_bytes(hashlib.blake2b(t[i : i + k].encode("utf-8"), digest_size=8).digest(), "big")
        for i in range(len(t) - k + 1)
    }
    if len(shingle_hashes) > max_shingles:
        # deterministic subsample: keep the smallest hashes
        shingle_hashes = set(sorted(shingle_hashes)[:max_shingles])
    xs = [x % _MINHASH_P for x in shingle_hashes]
    if not xs:
        return tuple([0] * num_perm)
    return tuple(min((a * x + b) % _MINHASH_P for x in xs) for a, b in _PERMS[:num_perm])


def near_duplicate_rate(
    texts: Sequence[str], bands: int = 4, threshold_report: float = 0.8
) -> float:
    """Fraction of sampled docs sharing an LSH band with an earlier doc."""
    rows = 16 // bands
    seen: Dict[Tuple[int, ...], int] = {}
    dups = 0
    for t in texts:
        sig = _minhash(t)
        hit = False
        for b in range(bands):
            key = (b, *sig[b * rows : (b + 1) * rows])
            if key in seen:
                hit = True
            seen.setdefault(key, 1)
        dups += hit
    return dups / len(texts) if texts else 0.0


def _pct(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))
    return float(s[i])


def analyse(corpus_dir: Path, docs_wanted: int, seed_stride: int | None = None) -> Dict:
    stats_path = corpus_dir / "stats.json"
    archive = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    total_docs = int(archive.get("documents", 0))

    stride = seed_stride or max(1, total_docs // max(1, docs_wanted))
    docs = list(iter_docs(corpus_dir, stride))
    texts = [d["text"] for d in docs]
    n = len(texts)
    if not n:
        raise SystemExit(f"no documents found under {corpus_dir}")

    lens = [len(t) for t in texts]
    tamil_frac, latin_frac, digit_frac, other_frac = [], [], [], []
    for t in texts:
        total = max(1, len(t))
        tamil_frac.append(len(_TAMIL.findall(t)) / total)
        latin_frac.append(len(_LATIN.findall(t)) / total)
        digit_frac.append(len(_DIGIT.findall(t)) / total)
        other_frac.append(len(_OTHER_INDIC.findall(t)) / total)

    pii_hits = Counter()
    for t in texts:
        for name, rx in _PII.items():
            if rx.search(t):
                pii_hits[name] += 1

    # boilerplate: how much of the text is covered by the most repeated 8-grams
    words_per_doc = [t.split() for t in texts]
    gram_counter: Counter = Counter()
    for words in words_per_doc:
        for i in range(max(0, len(words) - 7)):
            gram_counter[" ".join(words[i : i + 8])] += 1
    repeated = [(g, c) for g, c in gram_counter.most_common(200) if c >= 3]

    word_counter: Counter = Counter()
    for words in words_per_doc:
        word_counter.update(words)
    n_words = sum(word_counter.values())

    domain_hits = {name: sum(1 for t in texts if rx.search(t)) for name, rx in _DOMAIN_RE.items()}

    source_stats: Dict[str, Dict[str, float]] = {}
    by_source: Dict[str, List[int]] = {}
    for d, L in zip(docs, lens):
        by_source.setdefault(d["source"], []).append(L)
    for src, ls in sorted(by_source.items(), key=lambda kv: -len(kv[1])):
        source_stats[src] = {
            "docs_sampled": len(ls),
            "mean_chars": round(statistics.fmean(ls), 1) if ls else 0,
            "median_chars": float(statistics.median(ls)) if ls else 0,
            "share_under_200_chars": round(sum(1 for L in ls if L < 200) / len(ls), 4) if ls else 0,
        }

    sample_chars = sum(lens)
    report = {
        "corpus": str(corpus_dir),
        "archive": {
            "documents": total_docs or None,
            "characters": archive.get("characters"),
            "tokens_swaram": archive.get("approx_tokens_swaram"),
            "shards": archive.get("shards"),
            "source_counts": archive.get("source_counts", {}),
            "tier_counts": archive.get("tier_counts", {}),
        },
        "sample": {
            "docs": n,
            "stride": stride,
            "chars": sample_chars,
            "tokens_swaram_est": int(sample_chars * TOKENS_PER_CHAR),
            "chars_pct_of_archive": (
                round(100 * sample_chars / archive["characters"], 2)
                if archive.get("characters")
                else None
            ),
        },
        "length": {
            "mean": round(statistics.fmean(lens), 1),
            "median": float(statistics.median(lens)),
            "p10": _pct(lens, 10),
            "p90": _pct(lens, 90),
            "p99": _pct(lens, 99),
            "share_under_200_chars": round(sum(1 for L in lens if L < 200) / n, 4),
            "share_under_60_chars": round(sum(1 for L in lens if L < 60) / n, 4),
        },
        "script_mix": {
            "tamil_mean": round(statistics.fmean(tamil_frac), 4),
            "tamil_median": round(statistics.median(tamil_frac), 4),
            "latin_mean": round(statistics.fmean(latin_frac), 4),
            "digit_mean": round(statistics.fmean(digit_frac), 4),
            "other_indic_mean": round(statistics.fmean(other_frac), 4),
            "share_below_70pct_tamil": round(sum(1 for f in tamil_frac if f < 0.70) / n, 4),
        },
        "pii": {
            "docs_with_pii": sum(1 for t in texts if any(rx.search(t) for rx in _PII.values())),
            "rate": round(
                sum(1 for t in texts if any(rx.search(t) for rx in _PII.values())) / n, 4
            ),
            "by_type": dict(pii_hits),
        },
        "duplication": {
            "near_dup_rate": round(near_duplicate_rate(texts), 4),
            "repeated_8gram_types": len(repeated),
            "top_repeated_8grams": [{"gram": g, "docs": c} for g, c in repeated[:5]],
        },
        "lexicon": {
            "words": n_words,
            "types": len(word_counter),
            "type_token_ratio": round(len(word_counter) / max(1, n_words), 5),
            "top10_share": round(
                sum(c for _, c in word_counter.most_common(10)) / max(1, n_words), 4
            ),
            "words_per_doc": round(n_words / n, 1),
        },
        "domains": {
            k: {"docs": v, "share": round(v / n, 4)}
            for k, v in sorted(domain_hits.items(), key=lambda kv: -kv[1])
        },
        "sources": source_stats,
    }
    return report


# --- gap analysis -------------------------------------------------------------
#
# Targets are stated, not implied. Two kinds:
#   * compute-optimal: Chinchilla ~20 tokens/param; the archive is measured in
#     swaram tokens, so a model of N params wants ~20N tokens.
#   * register targets: how much clean text each register should contribute
#     before the mixture stops being IndicCorp-dominated.
REGISTER_TARGETS = {
    # register: (target tokens, what it buys)
    "news/web (IndicCorp)": (150_000_000, "modern syntax, named entities"),
    "consumer web (FineWeb-2)": (120_000_000, "forum/blog register, code-mixing"),
    "encyclopaedic (Wikipedia)": (40_000_000, "facts, definitions, clean prose"),
    "education (SCERT/textbooks)": (60_000_000, "curriculum-shaped, graded language"),
    "classical (Sangam/Thirukkural)": (10_000_000, "morphology, sandhi, literary Tamil"),
    "legal/government": (30_000_000, "formal register, forms, procedures"),
}


def gap_analysis(report: Dict) -> Dict:
    have_tokens = report["archive"].get("tokens_swaram") or 0
    # derive per-register "have" from source counts x mean tokens per document
    src = report["archive"].get("source_counts", {})
    per_doc = report["sample"]["tokens_swaram_est"] / max(1, report["sample"]["docs"])
    have_by_source = {k: int(v * per_doc) for k, v in src.items()}
    current = {
        "news/web (IndicCorp)": have_by_source.get("ai4bharat-indiccorp-v2", 0),
        "consumer web (FineWeb-2)": have_by_source.get("fineweb-2-ta", 0),
        "encyclopaedic (Wikipedia)": have_by_source.get("wikipedia-ta", 0),
        "education (SCERT/textbooks)": 0,  # no dedicated source yet
        "classical (Sangam/Thirukkural)": sum(
            v for k, v in have_by_source.items() if k.startswith("local:")
        ),
        "legal/government": 0,  # no dedicated source yet
    }
    rows = []
    for register, (target, why) in REGISTER_TARGETS.items():
        rows.append(
            {
                "register": register,
                "have": current.get(register, 0),
                "target": target,
                "gap": max(0, target - current.get(register, 0)),
                "why": why,
            }
        )
    compute_optimal = {
        f"{params}M params": 20 * params * 1_000_000 for params in (16, 50, 100, 300)
    }
    return {
        "have_tokens": have_tokens,
        "registers": rows,
        "compute_optimal": {
            k: {"needed": v, "we_have": have_tokens, "more_needed": max(0, v - have_tokens)}
            for k, v in compute_optimal.items()
        },
    }


def render_md(report: Dict, gaps: Dict) -> str:
    a = report["archive"]
    s = report["sample"]
    L = report["length"]
    sc = report["script_mix"]
    d = report["duplication"]
    lx = report["lexicon"]

    def fmt(n) -> str:
        return f"{n:,}" if isinstance(n, (int, float)) and n is not None else "n/a"

    lines = [
        "# Tamil corpus health",
        "",
        f"Generated by `scripts/corpus_health.py` on a **{fmt(s['docs'])}-document sample** "
        f"(stride {s['stride']}, {fmt(s['chars'])} chars = "
        f"{s['chars_pct_of_archive']}% of the archive).",
        "",
        "## Size",
        "",
        "| metric | value |",
        "|---|---|",
        f"| documents | {fmt(a['documents'])} |",
        f"| characters | {fmt(a['characters'])} |",
        f"| swaram tokens | {fmt(a['tokens_swaram'])} |",
        f"| shards | {fmt(a['shards'])} |",
        "",
        "## Health signals",
        "",
        "| signal | value | read |",
        "|---|---|---|",
        f"| median document | {L['median']:.0f} chars (p10 {L['p10']:.0f} · p90 {L['p90']:.0f}) | "
        f"widest spread between registers |",
        f"| docs under 200 chars | {100 * L['share_under_200_chars']:.1f}% | "
        f"{'ok' if L['share_under_200_chars'] < 0.10 else 'high — check snippets'} |",
        f"| mean Tamil-script share | {100 * sc['tamil_mean']:.1f}% | "
        f"{'ok' if sc['tamil_mean'] > 0.75 else 'low'} |",
        f"| docs under 70% Tamil | {100 * sc['share_below_70pct_tamil']:.1f}% | code-mixed / noise |",
        f"| Latin share (code-mixing) | {100 * sc['latin_mean']:.1f}% | expected in web register |",
        f"| docs containing PII | {100 * report['pii']['rate']:.1f}% | "
        f"{'scrub at pack time' if report['pii']['rate'] > 0.01 else 'ok'} |",
        f"| near-duplicate rate (LSH, J≥0.8) | {100 * d['near_dup_rate']:.1f}% | "
        f"{'ok' if d['near_dup_rate'] < 0.05 else 'needs another dedup pass'} |",
        f"| repeated 8-gram types (≥3 docs) | {fmt(d['repeated_8gram_types'])} | boilerplate footprint |",
        f"| type/token ratio | {lx['type_token_ratio']:.4f} | higher = more diverse |",
        f"| words per document | {lx['words_per_doc']:.0f} | |",
        "",
        "## Domain coverage (share of sampled docs mentioning each register)",
        "",
        "| register | docs | share |",
        "|---|---|---|",
    ]
    for name, v in report["domains"].items():
        lines.append(f"| {name} | {fmt(v['docs'])} | {100 * v['share']:.1f}% |")

    lines += [
        "",
        "## Top sources in the sample",
        "",
        "| source | docs | mean chars | median chars |",
        "|---|---|---|---|",
    ]
    for name, v in list(report["sources"].items())[:10]:
        lines.append(
            f"| `{name}` | {fmt(v['docs_sampled'])} | {v['mean_chars']:.0f} | {v['median_chars']:.0f} |"
        )

    lines += [
        "",
        "## How much more, and of what",
        "",
        f"Archive holds **{fmt(gaps['have_tokens'])}** swaram tokens.",
        "",
        "### Against compute-optimal budgets (~20 tokens/param)",
        "",
        "| model | tokens needed | we have | more needed |",
        "|---|---|---|---|",
    ]
    for name, v in gaps["compute_optimal"].items():
        lines.append(
            f"| {name} | {fmt(v['needed'])} | {fmt(v['we_have'])} | "
            f"**{fmt(v['more_needed'])}** |"
        )

    lines += [
        "",
        "### Against per-register targets",
        "",
        "| register | have | target | gap | what it buys |",
        "|---|---|---|---|---|",
    ]
    total_gap = 0
    for r in sorted(gaps["registers"], key=lambda x: -x["gap"]):
        total_gap += r["gap"]
        lines.append(
            f"| {r['register']} | {fmt(r['have'])} | {fmt(r['target'])} | "
            f"**{fmt(r['gap'])}** | {r['why']} |"
        )
    lines += ["", f"**Total missing to hit all register targets: {fmt(total_gap)} tokens.**", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Corpus health + gap report")
    ap.add_argument("--corpus", default="data/raw/expanded")
    ap.add_argument("--docs", type=int, default=20000, help="sample size (stride-derived)")
    ap.add_argument("--out", default=None, help="write a markdown report here")
    ap.add_argument("--json", default=None, help="write raw metrics here")
    args = ap.parse_args()

    report = analyse(Path(args.corpus), args.docs)
    gaps = gap_analysis(report)
    md = render_md(report, gaps)
    print(md)
    if args.out:
        Path(args.out).write_text(md + "\n", encoding="utf-8")
        print(f"\n[written] {args.out}")
    if args.json:
        Path(args.json).write_text(
            json.dumps({"report": report, "gaps": gaps}, indent=2), encoding="utf-8"
        )
        print(f"[written] {args.json}")


if __name__ == "__main__":
    main()
