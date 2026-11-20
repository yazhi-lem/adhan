"""Versioned content exclusion filter for the public eval corpus (adhan#112, K6).

The public tokenizer benchmark must not ship a random inflammatory, political or sexual
sentence scraped from social media. The N3 scope report
(``docs/reports/n3_tanglish_source_scope.md``) specifies the remedy: a deterministic,
high-recall gazetteer pass applied before the sample is frozen, with the filter version
recorded in the manifest so a reviewer can see exactly what was excluded.

The gazetteer lives in JSON, not in code, so a native reviewer can extend it without a
code change:

    eval_manifests/content_blocklist.v2.json

Matching rules (deliberately simple, reproducible, and tuned against both misses and false
positives — see the tests):

  * **Latin exact terms** match case-insensitively as whole words, so ``vote`` does not fire
    inside ``devote``. Internal whitespace matches any run of whitespace.
  * **Latin stems** also allow a short closed set of Tanglish case suffixes
    (``pundai`` + ``ku``, ``mayir`` + ``um``, ``arasiyal`` + ``uku``), which is how Tanglish
    inflects. The suffix list is closed on purpose: an open ``\\w*`` would drag in
    ``warning`` for ``war``.
  * **Tamil terms** match at a word start and may be followed only by case/plural markers
    (``தேர்தல்`` → ``தேர்தலில்``, ``அமைச்சர்`` → ``அமைச்சரை``). A term that ends in pulli is
    indexed both with and without it, because Tamil drops the pulli before a suffix.
    Requiring the word boundary and a known suffix is what stops ``சாதி`` from firing inside
    ``சாதிக்க`` and ``புண்ட`` inside ``புண்டரீகாட்சன்``.
  * Categories are checked in file order and the first hit wins, so the same sentence always
    reports the same category.
  * All terms and all input are NFC-normalised, so decomposed text cannot slip past.

Patterns are compiled once, when the gazetteer is loaded, not per sentence.

Pure standard library. No network, no model, no floating point.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: Filter applied when a spec does not name one: nothing.
NO_FILTER = "none"

#: Where the shipped gazetteers live, relative to the repository root.
BLOCKLIST_DIR = "eval_manifests"

#: Gazetteer version used by default.
DEFAULT_VERSION = "v2"

#: Tamil pulli (virama), which Tamil drops before a case suffix.
PULLI = "\u0bcd"

#: Tamil case/number markers a blocked term may be followed by.
TAMIL_SUFFIXES: Tuple[str, ...] = (
    "",
    "ல்",
    "ில்",
    "யில்",
    "இல்",
    "ை",
    "ையே",
    "ையும்",
    "ைப்",
    "ால்",
    "ஆல்",
    "க்கு",
    "ுக்கு",
    "க்கும்",
    "ின்",
    "இன்",
    "ோடு",
    "ுடன்",
    "ோடும்",
    "ும்",
    "உம்",
    "ாக",
    "ஆக",
    "ாகவும்",
    "்கள்",
    "கள்",
    "்களை",
    "களை",
    "்களில்",
    "களில்",
)

#: Tanglish case suffixes a stem may take. Closed set: an open suffix would over-block.
LATIN_SUFFIXES: Tuple[str, ...] = (
    "ku",
    "kku",
    "ukku",
    "uku",
    "la",
    "le",
    "il",
    "um",
    "a",
    "aa",
    "ai",
    "in",
    "inu",
    "gal",
    "ngal",
    "ungal",
)

_TAMIL_RANGE = "\u0b80-\u0bff"


class ContentFilterError(ValueError):
    """Raised for a malformed or missing gazetteer file."""


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


@dataclass(frozen=True)
class Category:
    """One exclusion category, e.g. ``politics``, with its patterns compiled once."""

    name: str
    latin: Tuple[str, ...] = ()
    latin_stems: Tuple[str, ...] = ()
    tamil: Tuple[str, ...] = ()
    _latin_re: Optional["re.Pattern[str]"] = field(
        default=None, init=False, repr=False, compare=False
    )
    _tamil_re: Optional["re.Pattern[str]"] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_latin_re", _compile_latin(self.latin, self.latin_stems))
        object.__setattr__(self, "_tamil_re", _compile_tamil(self.tamil))

    def matches(self, text: str) -> bool:
        if self._tamil_re is not None and self._tamil_re.search(text):
            return True
        return bool(self._latin_re is not None and self._latin_re.search(text))


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
        normalised = nfc(text)
        for cat in self.categories:
            if cat.matches(normalised):
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
                latin_stems=_terms(raw_cat.get("latin_stems", []), version, name, "latin_stems"),
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
        term = nfc(str(item)).strip()
        if not term:
            raise ContentFilterError(f"blocklist {version}: empty {kind} term in {category!r}")
        out.append(term)
    return tuple(out)


def _alternation(patterns: Iterable[str]) -> str:
    """Longest-first alternation so a short term cannot shadow a longer one."""
    return "|".join(sorted(patterns, key=len, reverse=True))


def _compile_latin(exact: Tuple[str, ...], stems: Tuple[str, ...]) -> Optional["re.Pattern[str]"]:
    parts: List[str] = []
    if exact:
        parts.append(_alternation(re.escape(t).replace(r"\ ", r"\s+") for t in exact))
    if stems:
        suffix = _alternation(LATIN_SUFFIXES)
        parts.append(rf"(?:{_alternation(re.escape(s) for s in stems)})(?:{suffix})?")
    if not parts:
        return None
    # Whole-word on both sides: `vote` must not fire inside `devote`, `war` not inside `warning`.
    return re.compile(rf"(?<![A-Za-z0-9])(?:{'|'.join(parts)})(?![A-Za-z0-9])", re.IGNORECASE)


def _compile_tamil(terms: Tuple[str, ...]) -> Optional["re.Pattern[str]"]:
    if not terms:
        return None
    variants: List[str] = []
    for term in terms:
        variants.append(re.escape(term))
        if term.endswith(PULLI):
            # Tamil drops the pulli before a case suffix: தேர்தல் -> தேர்தலில்.
            variants.append(re.escape(term[: -len(PULLI)]))
    suffix = _alternation(re.escape(s) for s in TAMIL_SUFFIXES if s)
    # Word start, then a known case suffix, then a non-Tamil boundary. Without the trailing
    # lookahead a bare prefix match is enough and சாதி fires inside சாதிக்க, புண்ட inside
    # புண்டரீகாட்சன்; with it the suffix has to account for the rest of the word.
    return re.compile(
        rf"(?<![{_TAMIL_RANGE}])(?:{_alternation(variants)})(?:{suffix})?(?![{_TAMIL_RANGE}])"
    )


def repo_root() -> Path:
    """Repository root, found by walking up from this module to ``pyproject.toml``."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise ContentFilterError("cannot locate the repository root from the package")


def default_blocklist_path(version: str = DEFAULT_VERSION) -> Path:
    return repo_root() / BLOCKLIST_DIR / f"content_blocklist.{version}.json"


def load_named(version: str = DEFAULT_VERSION) -> ContentFilter:
    """Load a shipped gazetteer by version, e.g. ``load_named("v2")``."""
    return ContentFilter.load(default_blocklist_path(version))
