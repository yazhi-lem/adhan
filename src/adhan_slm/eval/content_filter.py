"""Versioned content exclusion filter for the public eval corpus (adhan#112, K6).

The public tokenizer benchmark must not ship a random inflammatory, political or sexual
sentence scraped from social media. The N3 scope report
(``docs/reports/n3_tanglish_source_scope.md``) specifies the remedy: a deterministic,
high-recall gazetteer pass applied before the sample is frozen, with the filter version
recorded in the manifest so a reviewer can see exactly what was excluded.

The gazetteer lives in JSON, not in code, so a native reviewer can extend it without a
code change:

    eval_manifests/content_blocklist.v1.json

Matching rules (deliberately simple and reproducible):

  * Latin terms match case-insensitively as whole words, so ``vote`` does not fire inside
    ``devote``. Internal whitespace matches any run of whitespace.
  * Tamil terms match as plain substrings, because Tamil does not use ASCII word
    boundaries.
  * Categories are checked in file order and the first hit wins, so the same sentence
    always reports the same category.

Pure standard library. No network, no model, no floating point.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: Filter applied when a spec does not name one: nothing.
NO_FILTER = "none"

#: Where the shipped gazetteers live, relative to the repository root.
BLOCKLIST_DIR = "eval_manifests"


class ContentFilterError(ValueError):
    """Raised for a malformed or missing gazetteer file."""


@dataclass(frozen=True)
class Category:
    """One exclusion category, e.g. ``politics``."""

    name: str
    latin: Tuple[str, ...]
    tamil: Tuple[str, ...]

    def _latin_regex(self) -> Optional["re.Pattern[str]"]:
        pattern = _latin_alternation(self.latin)
        return re.compile(pattern, re.IGNORECASE) if pattern else None

    def matches(self, text: str, latin_re: Optional["re.Pattern[str]"] = None) -> bool:
        if any(term in text for term in self.tamil):
            return True
        if latin_re is None:
            latin_re = self._latin_regex()
        return bool(latin_re and latin_re.search(text))


@dataclass(frozen=True)
class ContentFilter:
    """A loaded, immutable gazetteer."""

    version: str
    categories: Tuple[Category, ...]
    source_path: Optional[Path] = None
    source_sha256: Optional[str] = None
    native_review: str = "pending"

    @property
    def active(self) -> bool:
        return bool(self.categories)

    def category_names(self) -> List[str]:
        return [c.name for c in self.categories]

    def blocked(self, text: str) -> Optional[str]:
        """Return the first category that vetoes ``text``, or ``None`` if it is clean."""
        for cat in self.categories:
            if cat.matches(text):
                return cat.name
        return None

    def blocked_category(self, text: str) -> Optional[str]:
        """Alias kept for readability at call sites that read ``blocked_category(...)``."""
        return self.blocked(text)

    def describe(self) -> Dict[str, Any]:
        """Manifest-ready description of what ran, so the exclusion is auditable."""
        return {
            "version": self.version,
            "path": str(self.source_path) if self.source_path else None,
            "sha256": self.source_sha256,
            "native_review": self.native_review,
            "categories": self.category_names(),
        }

    @classmethod
    def none(cls) -> "ContentFilter":
        return cls(version=NO_FILTER, categories=())

    @classmethod
    def load(cls, path: Path) -> "ContentFilter":
        path = Path(path)
        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            raise ContentFilterError(f"cannot read blocklist {path}: {exc}") from exc
        try:
            raw = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContentFilterError(f"blocklist {path} is not valid JSON: {exc}") from exc
        return cls.from_dict(
            raw,
            source_path=path,
            source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )

    @classmethod
    def from_dict(
        cls,
        raw: Dict[str, Any],
        source_path: Optional[Path] = None,
        source_sha256: Optional[str] = None,
    ) -> "ContentFilter":
        version = str(raw.get("version", "")).strip()
        if not version:
            raise ContentFilterError("blocklist is missing a version")
        categories_raw = raw.get("categories")
        if not isinstance(categories_raw, dict) or not categories_raw:
            raise ContentFilterError(f"blocklist {version}: categories must be a non-empty object")
        categories = tuple(
            Category(
                name=str(name),
                latin=_terms(raw_cat.get("latin", []), version, name, "latin"),
                tamil=_terms(raw_cat.get("tamil", []), version, name, "tamil"),
            )
            for name, raw_cat in categories_raw.items()
        )
        return cls(
            version=version,
            categories=categories,
            source_path=source_path,
            source_sha256=source_sha256,
            native_review=str(raw.get("native_review", "pending")),
        )


def _terms(items: Iterable[Any], version: str, category: str, kind: str) -> Tuple[str, ...]:
    out: List[str] = []
    for item in items:
        term = str(item).strip()
        if not term:
            raise ContentFilterError(f"blocklist {version}: empty {kind} term in {category!r}")
        out.append(term)
    return tuple(out)


def _latin_alternation(terms: Tuple[str, ...]) -> str:
    """Whole-word alternation for Latin terms; longest first so substrings cannot win."""
    parts = []
    for term in sorted(terms, key=len, reverse=True):
        escaped = re.escape(term).replace(r"\ ", r"\s+")
        parts.append(escaped)
    if not parts:
        return ""
    return r"(?<![A-Za-z0-9])(?:" + "|".join(parts) + r")(?![A-Za-z0-9])"


def repo_root() -> Path:
    """Repository root, found by walking up from this module to ``pyproject.toml``."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise ContentFilterError("cannot locate the repository root from the package")


def default_blocklist_path(version: str = "v1") -> Path:
    return repo_root() / BLOCKLIST_DIR / f"content_blocklist.{version}.json"


def load_named(version: str = "v1") -> ContentFilter:
    """Load a shipped gazetteer by version, e.g. ``load_named("v1")``."""
    return ContentFilter.load(default_blocklist_path(version))
