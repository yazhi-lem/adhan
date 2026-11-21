"""Canonical text hygiene, hashing and near-duplicate logic for corpus building.

Before this module existed, the same helpers were copy-pasted across the corpus scripts and
extractors (adhan#125): ``_h64``, ``_shingles``, ``_signature``, ``_sig_similarity``,
``is_duplicate`` and ``_iter_strings`` in both ``expand_tamil_corpus.py`` and
``expand_tamil_corpus_v2.py``; ``tamil_ratio`` in six files; ``sha256_hex`` in six. Copies drift,
and a drifted copy silently changes which documents enter the corpus — which is exactly the kind of
thing that makes published numbers irreproducible.

Everything here is pure standard library except an optional ``xxhash`` fast path. The v1/v2
differences that used to be separate function bodies are now :class:`QualityConfig` presets, so the
behaviour is explicit and testable instead of implied by which file you imported.

Note on ``tamil_ratio``: the old copies disagreed. Two stripped all whitespace, one stripped only
spaces/newlines/tabs, and ``ingest_open_sangam.py`` divided by the *unstripped* length. The
canonical function strips all whitespace by default (the majority and the sane definition) and
takes ``strip_whitespace=False`` for the one caller that measured the other way.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterator, List, Set, Tuple

#: Tamil Unicode block.
TAMIL_RANGE = re.compile(r"[\u0B80-\u0BFF]")

#: PII / noise patterns used by :func:`normalize`.
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_RE = re.compile(r"(\+91[\-\s]?)?[6-9]\d{9}")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
REPEAT_RE = re.compile(r"(.)\1{6,}")
WS_RE = re.compile(r"[ \t]+")
NL_RE = re.compile(r"\n{3,}")

#: Boilerplate markers used by the v2 quality gate.
BOILERPLATE: Tuple[str, ...] = (
    "javascript",
    "cookie",
    "privacy policy",
    "terms of use",
    "all rights reserved",
    "click here to",
    "subscribe now",
    "பதிவு செய்ய",
    "விளம்பரம்",
    "advertisement",
)


# -- hashing ------------------------------------------------------------------------


def h64(data: bytes) -> int:
    """64-bit hash of ``data``: xxhash when installed, blake2b otherwise.

    Both paths are deterministic for a given input, but they are *different* functions, so a
    corpus built with xxhash and one built without will not agree on shingle hashes. That is
    pre-existing behaviour, kept as-is; the fallback exists so the scripts run without the extra.
    """
    try:
        import xxhash  # type: ignore
    except ImportError:  # pragma: no cover - depends on the environment
        return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")
    return int(xxhash.xxh64_intdigest(data))


def sha256_hex(text: str, *, errors: str = "strict") -> str:
    """SHA-256 of ``text`` as hex.

    ``errors`` is exposed because one extractor hashes text that may contain lone surrogates and
    used ``errors="ignore"``; silently changing that would change its document ids.
    """
    return hashlib.sha256(text.encode("utf-8", errors=errors)).hexdigest()


# -- text hygiene -------------------------------------------------------------------


def normalize(text: str) -> str:
    """NFC-normalise, scrub PII (URLs, emails, phones), collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = WS_RE.sub(" ", text)
    text = NL_RE.sub("\n\n", text)
    return text.strip()


def tamil_ratio(text: str, *, strip_whitespace: bool = True) -> float:
    """Share of characters in the Tamil block.

    ``strip_whitespace=True`` (default) measures over non-whitespace characters, which is what
    most callers meant. ``False`` divides by the raw length, preserving the one caller that
    counted spaces against the ratio.
    """
    clean = "".join(text.split()) if strip_whitespace else text
    if not clean:
        return 0.0
    return len(TAMIL_RANGE.findall(clean)) / len(clean)


@dataclass(frozen=True)
class QualityConfig:
    """Tunable quality gate. The v1/v2 scripts differ only in these values."""

    min_chars: int = 40
    max_chars: int = 20000
    min_tamil_ratio: float = 0.60
    max_markup: int = 20
    max_pipe_ratio: float = 0.02
    boilerplate: Tuple[str, ...] = ()
    min_boilerplate_hits: int = 3

    def with_overrides(self, **kwargs: Any) -> "QualityConfig":
        return replace(self, **kwargs)


#: ``scripts/expand_tamil_corpus.py`` (v1) gate.
V1_QUALITY = QualityConfig()

#: ``scripts/expand_tamil_corpus_v2.py`` gate: looser length/ratio, stricter markup, boilerplate.
V2_QUALITY = QualityConfig(
    max_chars=50000,
    min_tamil_ratio=0.55,
    max_markup=30,
    max_pipe_ratio=0.03,
    boilerplate=BOILERPLATE,
)


def is_quality(text: str, config: QualityConfig = V1_QUALITY) -> bool:
    """True when ``text`` passes the configured length, script, repetition and markup gates."""
    if not text:
        return False
    n = len(text)
    if n < config.min_chars or n > config.max_chars:
        return False
    if tamil_ratio(text) < config.min_tamil_ratio:
        return False
    if REPEAT_RE.search(text):
        return False
    if text.count("<") > config.max_markup or text.count("|") > n * config.max_pipe_ratio:
        return False
    if config.boilerplate:
        lower = text.lower()
        hits = sum(1 for marker in config.boilerplate if marker in lower)
        if hits >= config.min_boilerplate_hits:
            return False
    return True


# -- near-duplicate detection -------------------------------------------------------


class LshDedup:
    """Exact + near-duplicate filtering with banded LSH.

    Shingle hashes are cached so each document costs one pass over its text; candidates are only
    pulled from matching LSH bands, then verified against the true signatures (no false positives
    from band luck). Only signatures are retained, not raw shingle sets: at 1M documents the raw
    sets cost >20 GiB, while 16-permille signatures cost ~1.2 GiB.
    """

    def __init__(
        self,
        num_perm: int = 16,
        bands: int = 4,
        shingle_k: int = 5,
        threshold: float = 0.80,
        max_shingle_chars: int = 4000,
    ) -> None:
        if num_perm % bands != 0:
            raise ValueError(f"num_perm {num_perm} must be divisible by bands {bands}")
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.k = shingle_k
        self.threshold = threshold
        self.max_chars = max_shingle_chars
        self.exact: Set[str] = set()
        self.buckets: List[Dict[Tuple[int, ...], List[int]]] = [{} for _ in range(bands)]
        self.sigs: Dict[int, Tuple[int, ...]] = {}
        self._next_id = 0
        # deterministic permutation coefficients (odd multipliers)
        self._a = [(i * 2 + 1) * 0x9E3779B97F4A7C15 % (2**64) for i in range(num_perm)]
        self._b = [(i * 0xBF58476D1CE4E5B9 + 1) % (2**64) for i in range(num_perm)]

    def _shingles(self, text: str) -> Set[int]:
        clean = text[: self.max_chars]
        if len(clean) < self.k:
            return {h64(clean.encode("utf-8"))} if clean else set()
        return {h64(clean[i : i + self.k].encode("utf-8")) for i in range(len(clean) - self.k + 1)}

    def _signature(self, shingles: Set[int]) -> Tuple[int, ...]:
        if not shingles:
            return tuple([0] * self.num_perm)
        return tuple(
            min((a * s + b) & 0xFFFFFFFFFFFFFFFF for s in shingles)
            for a, b in zip(self._a, self._b)
        )

    @staticmethod
    def _sig_similarity(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
        matches = sum(1 for x, y in zip(a, b) if x == y)
        return matches / len(a)

    def seed_exact(self, text: str) -> None:
        """Register an already-accepted document by exact hash only.

        Used on resume/extension so a second run does not re-emit documents from a previous run.
        Signatures are not rebuilt because that costs a full minhash pass over the archive; exact
        hashing already catches the dominant duplication vector, re-reading the same source.
        """
        self.exact.add(hashlib.sha1(text.encode("utf-8")).hexdigest())

    def is_duplicate(self, text: str) -> bool:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if digest in self.exact:
            return True

        shingles = self._shingles(text)
        sig = self._signature(shingles)

        candidates: Set[int] = set()
        keys: List[Tuple[int, Tuple[int, ...]]] = []
        for band in range(self.bands):
            start = band * self.rows
            key = sig[start : start + self.rows]
            keys.append((band, key))
            bucket = self.buckets[band].get(key)
            if bucket:
                candidates.update(bucket)

        for cand in candidates:
            if self._sig_similarity(sig, self.sigs[cand]) >= self.threshold:
                return True

        doc_id = self._next_id
        self._next_id += 1
        self.exact.add(digest)
        self.sigs[doc_id] = sig
        for band, key in keys:
            self.buckets[band].setdefault(key, []).append(doc_id)
        return False


# -- misc ---------------------------------------------------------------------------


def iter_strings(obj: Any) -> Iterator[str]:
    """Yield every string inside a nested JSON-like structure, in document order."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from iter_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_strings(value)
