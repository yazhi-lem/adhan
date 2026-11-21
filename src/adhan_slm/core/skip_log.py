"""Structured logging and counting for records a pipeline deliberately skips (adhan#124).

The audit behind #124 found 88 ``except`` handlers in ``src/`` that neither re-raised nor logged.
In a corpus pipeline that is the worst failure mode available: a malformed shard line, an
unreadable file or a bad record simply disappears, the build reports success, and the resulting
corpus is smaller than anyone can explain. Reproducibility dies quietly.

Two pieces, deliberately small:

* :func:`log_skipped` — one warning per skipped item, with a stable set of fields
  (``action``, ``reason``, ``path``, ``error``) so logs are greppable and machine-readable.
* :class:`SkipCounter` — the same information as numbers, so a build can report
  "12,431 records skipped: 12,400 malformed_json, 31 unreadable_file" instead of nothing.

Both are cheap enough to call per record. Neither swallows anything: the caller still decides
whether to continue, and the count is what makes that decision visible.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

#: Reasons used across the pipeline. Free-form strings are allowed; these keep the common ones
#: spelled the same way so a metric can be compared between builds.
MALFORMED_JSON = "malformed_json"
UNREADABLE_FILE = "unreadable_file"
WRONG_SHAPE = "wrong_shape"
EMPTY_TEXT = "empty_text"
QUALITY_GATE = "quality_gate"
DUPLICATE = "duplicate"
MISSING_FIELD = "missing_field"
UNSUPPORTED_FORMAT = "unsupported_format"
SOURCE_FAILED = "source_failed"
CONVERSION_FAILED = "conversion_failed"


def log_skipped(
    logger: logging.Logger,
    action: str,
    *,
    reason: str,
    path: Optional[Union[str, Path]] = None,
    error: Optional[BaseException] = None,
    level: int = logging.WARNING,
    **context: Any,
) -> None:
    """Log one skipped item with structured context.

    ``action`` says what was being attempted (``"read shard"``), ``reason`` says why the item was
    dropped (use the constants above where they fit), and ``path``/``error`` carry the location and
    the exception. Extra keyword arguments are appended as ``key=value``.
    """
    parts = [f"skipped {action}", f"reason={reason}"]
    if path is not None:
        parts.append(f"path={path}")
    if error is not None:
        parts.append(f"error={type(error).__name__}: {error}")
    for key, value in context.items():
        parts.append(f"{key}={value}")
    logger.log(level, " | ".join(parts))


@dataclass
class SkipCounter:
    """Counts skipped records by reason, with optional per-source breakdown.

    ``record`` is called where a record is dropped; ``summary`` and ``as_dict`` are what a build
    script writes into its stats file, so a silent drop becomes a number in the output.
    """

    by_reason: Counter = field(default_factory=Counter)
    by_source: Dict[str, Counter] = field(default_factory=dict)
    examples: Dict[str, str] = field(default_factory=dict)
    max_examples: int = 3

    def record(
        self,
        reason: str,
        *,
        source: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        self.by_reason[reason] += 1
        if source is not None:
            self.by_source.setdefault(source, Counter())[reason] += 1
        if detail and len(self.examples) < self.max_examples and reason not in self.examples:
            self.examples[reason] = detail

    @property
    def total(self) -> int:
        return sum(self.by_reason.values())

    def summary(self) -> str:
        """One-line human summary, e.g. ``12,431 skipped: malformed_json 12,400, ...``."""
        if not self.total:
            return "0 skipped"
        parts = ", ".join(f"{reason} {count:,}" for reason, count in self.by_reason.most_common())
        return f"{self.total:,} skipped: {parts}"

    def as_dict(self) -> Dict[str, Any]:
        """JSON-ready form for a stats file."""
        return {
            "total": self.total,
            "by_reason": dict(self.by_reason),
            "by_source": {src: dict(counts) for src, counts in self.by_source.items()},
            "examples": dict(self.examples),
        }

    def merge(self, other: "SkipCounter") -> None:
        self.by_reason.update(other.by_reason)
        for source, counts in other.by_source.items():
            self.by_source.setdefault(source, Counter()).update(counts)
        for reason, detail in other.examples.items():
            self.examples.setdefault(reason, detail)


def skipped_reasons(counter: SkipCounter) -> Iterable[str]:
    """Reasons in descending count order, for stable report output."""
    return [reason for reason, _ in counter.by_reason.most_common()]
