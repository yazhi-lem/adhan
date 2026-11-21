"""Tamil text normalisation and grapheme-safety checks.

One small, dependency-free module that every Adhan data step (corpus reading,
dedup, tokenizer benchmark, data freeze) can share so that "the same Tamil text"
always means the same code points:

* :func:`normalise` strips invisible format characters that break matching
  (zero-width space, BOM, soft hyphen, word joiner) and then applies Unicode
  NFC. NFC is what composes a split vowel sign such as ``ெ`` + ``ா`` into
  ``ொ``, or ``ஒ`` + ``ௗ`` into ``ஔ``.
* :func:`graphemes` splits text into user-perceived Tamil letters (a base
  character plus its dependent vowel sign, pulli and joiners), following the
  UAX #29 rule that combining marks attach to the preceding character.
* :func:`find_issues` reports text that is not NFC and clusters that no Tamil
  writer can produce: a vowel sign or pulli with no base ("orphan"), a vowel
  sign on an independent vowel, or two vowel signs stacked on one consonant.
* :func:`safe_truncate` cuts text to a length budget without splitting a
  letter, for packing and display.

The Tamil strings used in this module's tests are drafts and still need sign-off
from a native reviewer.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

__all__ = [
    "INVISIBLE_CHARS",
    "IssueKind",
    "TextIssue",
    "normalise",
    "graphemes",
    "grapheme_count",
    "find_issues",
    "is_clean",
    "safe_truncate",
    "summarise_issues",
    "iter_text_lines",
    "main",
]

#: Format characters that are invisible in Tamil text and only break matching.
#: ZWJ (U+200D) and ZWNJ (U+200C) are kept: they are rare in Tamil but can be
#: deliberate rendering hints, and they never change the letters.
INVISIBLE_CHARS: frozenset[str] = frozenset(
    {
        "\u200b",  # zero-width space
        "\ufeff",  # byte-order mark / zero-width no-break space
        "\u00ad",  # soft hyphen
        "\u2060",  # word joiner
    }
)

_ZWNJ = "\u200c"
_ZWJ = "\u200d"
_PULLI = "\u0bcd"
_ANUSVARA = "\u0b82"

_TAMIL_VOWEL_SIGNS: frozenset[str] = frozenset(
    chr(cp)
    for cp in (*range(0x0BBE, 0x0BC3), *range(0x0BC6, 0x0BC9), *range(0x0BCA, 0x0BCD), 0x0BD7)
)
#: Dependent marks that need a Tamil consonant as their base.
_TAMIL_DEPENDENT: frozenset[str] = _TAMIL_VOWEL_SIGNS | {_PULLI}


def _is_tamil_consonant(ch: str) -> bool:
    return "\u0b95" <= ch <= "\u0bb9" and unicodedata.category(ch) == "Lo"


def _is_tamil_independent_vowel(ch: str) -> bool:
    return "\u0b85" <= ch <= "\u0b94" and unicodedata.category(ch) == "Lo"


def _attaches(ch: str) -> bool:
    """True if ``ch`` extends the previous grapheme (combining mark or joiner)."""
    return ch in (_ZWJ, _ZWNJ) or unicodedata.category(ch) in ("Mn", "Mc", "Me")


class IssueKind(str, Enum):
    """Kinds of problems :func:`find_issues` reports."""

    NOT_NFC = "not_nfc"
    INVISIBLE_CHAR = "invisible_char"
    ORPHAN_MARK = "orphan_mark"
    MARK_ON_VOWEL = "mark_on_vowel"
    STACKED_MARKS = "stacked_marks"


@dataclass(frozen=True)
class TextIssue:
    """One problem in a piece of text.

    ``offset`` is a code-point index. For :attr:`IssueKind.NOT_NFC` and
    :attr:`IssueKind.INVISIBLE_CHAR` it points into the original text; for the
    cluster kinds it points into ``normalise(text)``, which is what the cluster
    checks inspect.
    """

    kind: IssueKind
    offset: int
    snippet: str


def normalise(text: str) -> str:
    """Return ``text`` without invisible format characters, in NFC.

    Invisible characters are removed *before* NFC so that a zero-width space
    between ``ெ`` and ``ா`` cannot stop them composing into ``ொ``. The function
    is idempotent: ``normalise(normalise(t)) == normalise(t)``.
    """
    cleaned = "".join(ch for ch in text if ch not in INVISIBLE_CHARS)
    return unicodedata.normalize("NFC", cleaned)


def graphemes(text: str) -> List[str]:
    """Split ``text`` into graphemes: a base character plus attached marks.

    ``"தமிழ்"`` gives ``["த", "மி", "ழ்"]``. A mark with nothing before it
    becomes its own grapheme, which :func:`find_issues` reports as an orphan.
    """
    clusters: List[str] = []
    for ch in text:
        if clusters and _attaches(ch) and not clusters[-1].isspace():
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def grapheme_count(text: str) -> int:
    """Number of graphemes in ``text`` (see :func:`graphemes`)."""
    return len(graphemes(text))


def _cluster_issue(cluster: str) -> IssueKind | None:
    base, marks = cluster[0], [m for m in cluster[1:] if m not in (_ZWJ, _ZWNJ)]
    if base in _TAMIL_DEPENDENT:
        return IssueKind.ORPHAN_MARK
    tamil_marks = [m for m in marks if m in _TAMIL_DEPENDENT]
    if not tamil_marks:
        return None
    if _is_tamil_consonant(base):
        return IssueKind.STACKED_MARKS if len(tamil_marks) > 1 else None
    if _is_tamil_independent_vowel(base) or base == "\u0b83":
        return IssueKind.MARK_ON_VOWEL
    return IssueKind.ORPHAN_MARK


def find_issues(text: str) -> List[TextIssue]:
    """List every normalisation and grapheme problem in ``text``.

    An empty list means the text is already normalised and every Tamil letter
    is well formed. The anusvara (U+0B82) and non-Tamil marks are not judged.
    """
    issues: List[TextIssue] = []
    for i, ch in enumerate(text):
        if ch in INVISIBLE_CHARS:
            issues.append(TextIssue(IssueKind.INVISIBLE_CHAR, i, f"U+{ord(ch):04X}"))
    if unicodedata.normalize("NFC", text) != text:
        first = next(
            (i for i, (a, b) in enumerate(zip(text, unicodedata.normalize("NFC", text))) if a != b),
            0,
        )
        issues.append(TextIssue(IssueKind.NOT_NFC, first, text[max(0, first - 2) : first + 3]))

    offset = 0
    for cluster in graphemes(normalise(text)):
        kind = _cluster_issue(cluster)
        if kind is not None:
            issues.append(TextIssue(kind, offset, cluster))
        offset += len(cluster)
    return issues


def is_clean(text: str) -> bool:
    """True if :func:`find_issues` finds nothing."""
    return not find_issues(text)


def safe_truncate(text: str, max_chars: int) -> str:
    """Longest prefix of ``text`` with at most ``max_chars`` code points that
    does not end in the middle of a grapheme."""
    if max_chars < 0:
        raise ValueError(f"max_chars must be >= 0, got {max_chars}")
    out: List[str] = []
    used = 0
    for cluster in graphemes(text):
        if used + len(cluster) > max_chars:
            break
        out.append(cluster)
        used += len(cluster)
    return "".join(out)


def summarise_issues(lines: Iterable[str]) -> Dict[str, int]:
    """Count issues by kind over many lines, plus ``lines`` and ``dirty_lines``.

    Used by :func:`main` to gate a corpus before a data freeze.
    """
    counts: Dict[str, int] = {kind.value: 0 for kind in IssueKind}
    counts["lines"] = 0
    counts["dirty_lines"] = 0
    for line in lines:
        counts["lines"] += 1
        found = find_issues(line)
        if found:
            counts["dirty_lines"] += 1
        for issue in found:
            counts[issue.kind.value] += 1
    return counts


def iter_text_lines(path: Path, text_key: str = "text") -> Iterator[str]:
    """Yield raw text lines from a ``.txt`` file or ``text_key`` of a ``.jsonl`` file.

    The text is *not* normalised: the point is to see what is on disk.
    """
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if path.suffix == ".jsonl":
                obj = json.loads(line)
                value = obj.get(text_key) if isinstance(obj, dict) else None
                if isinstance(value, str):
                    yield value
            else:
                yield line


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Check corpus files and exit non-zero if too many lines are dirty.

    Usage::

        python -m adhan_slm.data.tamil_normalise data/corpus/*.txt --max-dirty 0
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("paths", nargs="+", type=Path, help=".txt or .jsonl files")
    parser.add_argument("--text-key", default="text", help="JSONL field holding the text")
    parser.add_argument(
        "--max-dirty",
        type=float,
        default=0.0,
        help="largest allowed fraction of dirty lines (0 = every line must be clean)",
    )
    args = parser.parse_args(argv)

    total: Dict[str, int] = {}
    for path in args.paths:
        counts = summarise_issues(iter_text_lines(path, args.text_key))
        print(json.dumps({"path": str(path), **counts}, ensure_ascii=False))
        for key, value in counts.items():
            total[key] = total.get(key, 0) + value

    fraction = total["dirty_lines"] / total["lines"] if total.get("lines") else 0.0
    print(json.dumps({"path": "TOTAL", **total, "dirty_fraction": round(fraction, 6)}))
    return 0 if fraction <= args.max_dirty else 1


if __name__ == "__main__":  # pragma: no cover - exercised via main() in tests
    sys.exit(main())
