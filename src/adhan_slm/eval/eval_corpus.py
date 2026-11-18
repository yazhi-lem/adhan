"""Frozen evaluation corpus for the public Tamil tokenizer benchmark (adhan#74, part 1).

Builds a small, deterministic, auditable eval set in three registers:

  classical   Sangam / Thirukkural-style Tamil
  modern      contemporary Tamil prose (news, Wikipedia, ...)
  tanglish    Tamil written in Latin script

Design rules (from the issue):

  * Every number must come from committed code, so sampling is a pure function of the
    source text and a seed: candidates are ranked by ``sha256(seed|source|text)`` and the
    first N kept. Input order, file order and the platform do not matter.
  * Tamil handling is grapheme-safe: all text is NFC-normalised before anything else.
  * Every source must carry a licence and a URL. A source whose licence is empty or
    ``TODO`` is rejected, so an unlicensed source cannot end up in the public set.
  * The output is described by a manifest (per-source counts, input and output SHA-256,
    the exact spec) and ``verify`` rebuilds the corpus and compares it to the manifest.

Pure Python, standard library only. No text ever leaves the machine; the module does no
network I/O. Fetching the raw sources is a separate step (see the extractors under
``src/data_scraper/raw_extractors``).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from adhan_slm.eval.content_filter import NO_FILTER, ContentFilter, ContentFilterError

REGISTERS = ("classical", "modern", "tanglish")
SCHEMA_VERSION = 1

_TAMIL = re.compile(r"[஀-௿]")
_LATIN = re.compile(r"[A-Za-z]")
_SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")
_WS = re.compile(r"\s+")

#: Minimum share of letters that must be in the expected script for a sentence to count.
MIN_SCRIPT_SHARE = 0.8


class EvalCorpusError(ValueError):
    """Raised for an invalid spec, an unlicensed source or a failed verification."""


@dataclass(frozen=True)
class Source:
    id: str
    register: str
    path: str
    license: str
    url: str
    format: str = "jsonl"  # "jsonl" (uses text_field) or "txt" (one document per line)
    text_field: str = "text"

    def validate(self) -> None:
        if not self.id:
            raise EvalCorpusError("source is missing an id")
        if self.register not in REGISTERS:
            raise EvalCorpusError(
                f"source {self.id!r}: register {self.register!r} not in {REGISTERS}"
            )
        if self.format not in ("jsonl", "txt"):
            raise EvalCorpusError(f"source {self.id!r}: unknown format {self.format!r}")
        for name in ("license", "url", "path"):
            value = getattr(self, name).strip()
            if not value or value.upper().startswith("TODO"):
                raise EvalCorpusError(
                    f"source {self.id!r}: {name} is empty or TODO; every source needs a "
                    "licence, a URL and a path before it can enter the public eval set"
                )


@dataclass(frozen=True)
class Spec:
    sources: Sequence[Source]
    per_register: int = 500
    seed: int = 13
    min_chars: int = 15
    max_chars: int = 300
    #: ``"none"`` or a path to a content blocklist, resolved like a source path.
    content_filter: str = NO_FILTER

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Spec":
        srcs = [Source(**s) for s in raw.get("sources", [])]
        spec = cls(
            sources=srcs,
            per_register=int(raw.get("per_register", 500)),
            seed=int(raw.get("seed", 13)),
            min_chars=int(raw.get("min_chars", 15)),
            max_chars=int(raw.get("max_chars", 300)),
            content_filter=str(raw.get("content_filter", NO_FILTER)),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.sources:
            raise EvalCorpusError("spec has no sources")
        ids = [s.id for s in self.sources]
        if len(set(ids)) != len(ids):
            raise EvalCorpusError("source ids must be unique")
        for s in self.sources:
            s.validate()
        missing = [r for r in REGISTERS if not any(s.register == r for s in self.sources)]
        if missing:
            raise EvalCorpusError(f"no source for register(s): {', '.join(missing)}")
        if self.per_register < 1 or self.min_chars < 1 or self.max_chars <= self.min_chars:
            raise EvalCorpusError("per_register/min_chars/max_chars are inconsistent")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sources": [s.__dict__ for s in self.sources],
            "per_register": self.per_register,
            "seed": self.seed,
            "min_chars": self.min_chars,
            "max_chars": self.max_chars,
            "content_filter": self.content_filter,
        }


# -- text helpers -------------------------------------------------------------------


def normalize(text: str) -> str:
    """NFC-normalise and collapse whitespace."""
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def split_sentences(text: str) -> List[str]:
    """Split a document into sentence-like lines (full stop, ``?``, ``!``, ``।``, newline)."""
    return [p for p in (normalize(x) for x in _SENT_SPLIT.split(text)) if p]


def script_ok(sentence: str, register: str) -> bool:
    """True if the sentence is in the script its register requires.

    classical / modern: at least ``MIN_SCRIPT_SHARE`` of the letters are Tamil script.
    tanglish: no Tamil-script characters at all, and mostly Latin letters.
    """
    tamil = len(_TAMIL.findall(sentence))
    latin = len(_LATIN.findall(sentence))
    letters = tamil + latin
    if letters == 0:
        return False
    if register == "tanglish":
        return tamil == 0 and latin / letters >= MIN_SCRIPT_SHARE
    return tamil / letters >= MIN_SCRIPT_SHARE


def _rank_key(seed: int, source_id: str, text: str) -> str:
    return hashlib.sha256(f"{seed}|{source_id}|{text}".encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -- reading sources ----------------------------------------------------------------


def _read_documents(source: Source, base: Path) -> Iterable[str]:
    path = (
        (base / source.path).resolve() if not Path(source.path).is_absolute() else Path(source.path)
    )
    if not path.is_file():
        raise EvalCorpusError(f"source {source.id!r}: file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if source.format == "txt":
                yield line
            else:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvalCorpusError(f"source {source.id!r}: bad JSON line: {exc}") from exc
                text = rec.get(source.text_field)
                if isinstance(text, str):
                    yield text


# -- build --------------------------------------------------------------------------


def load_content_filter(spec: Spec, base: Path) -> ContentFilter:
    """Resolve and load the spec's content filter, or the inert one when unset."""
    name = (spec.content_filter or NO_FILTER).strip()
    if name == NO_FILTER:
        return ContentFilter.none()
    try:
        return ContentFilter.load(_resolve_path(name, base))
    except ContentFilterError as exc:
        raise EvalCorpusError(f"content_filter {name!r}: {exc}") from exc


def _describe_filter(content_filter: ContentFilter, base: Path) -> Optional[Dict[str, Any]]:
    """Manifest description with a repo-relative path, so the manifest is portable."""
    if not content_filter.active:
        return None
    desc = content_filter.describe()
    path = desc.get("path")
    if path:
        try:
            desc["path"] = Path(path).resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            pass  # outside the repo: keep the absolute path rather than hide it
    return desc


def build(spec: Spec, base: Path) -> Dict[str, Any]:
    """Build the corpus. Returns ``{"records": [...], "manifest": {...}}``.

    Each record is ``{"id", "register", "source", "text"}``. Records are ordered by
    register, then by rank, so the output bytes are identical on every run.
    """
    spec.validate()
    content_filter = load_content_filter(spec, base)
    seen: Set[str] = set()
    per_source: Dict[str, Dict[str, Any]] = {}
    ranked: Dict[str, List[Tuple[str, str, str]]] = {r: [] for r in REGISTERS}

    for source in spec.sources:
        stats: Dict[str, Any] = {
            "documents": 0,
            "sentences": 0,
            "kept_candidates": 0,
            "blocked": {name: 0 for name in content_filter.category_names()},
        }
        for doc in _read_documents(source, base):
            stats["documents"] += 1
            for sent in split_sentences(doc):
                stats["sentences"] += 1
                if not (spec.min_chars <= len(sent) <= spec.max_chars):
                    continue
                if not script_ok(sent, source.register):
                    continue
                if sent in seen:
                    continue
                category = content_filter.blocked(sent)
                if category is not None:
                    stats["blocked"][category] += 1
                    continue
                seen.add(sent)
                stats["kept_candidates"] += 1
                ranked[source.register].append(
                    (_rank_key(spec.seed, source.id, sent), source.id, sent)
                )
        per_source[source.id] = stats

    records: List[Dict[str, str]] = []
    selected: Dict[str, int] = {}
    for register in REGISTERS:
        chosen = sorted(ranked[register])[: spec.per_register]
        selected[register] = len(chosen)
        for i, (_, source_id, sent) in enumerate(chosen):
            records.append(
                {
                    "id": f"{register}-{i:05d}",
                    "register": register,
                    "source": source_id,
                    "text": sent,
                }
            )

    for source in spec.sources:
        per_source[source.id]["selected"] = sum(1 for r in records if r["source"] == source.id)

    body = serialize(records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "unicode_normalization": "NFC",
        "spec": spec.to_dict(),
        "content_filter": _describe_filter(content_filter, base),
        "counts": {
            "total": len(records),
            "per_register": selected,
            "per_source": per_source,
        },
        "inputs": {
            s.id: {
                "path": s.path,
                "sha256": sha256_file(_resolve(s, base)),
                "license": s.license,
                "url": s.url,
            }
            for s in spec.sources
        },
        "output_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }
    return {"records": records, "manifest": manifest}


def _resolve_path(path: str, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (base / p).resolve()


def _resolve(source: Source, base: Path) -> Path:
    return _resolve_path(source.path, base)


def serialize(records: Sequence[Dict[str, str]]) -> str:
    """Canonical JSONL: stable key order, no ASCII escaping, trailing newline."""
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)


def shortfalls(manifest: Dict[str, Any]) -> List[str]:
    """Registers that ended up with fewer sentences than ``per_register`` asked for."""
    want = manifest["spec"]["per_register"]
    return [
        f"{reg}: {got} of {want}"
        for reg, got in manifest["counts"]["per_register"].items()
        if got < want
    ]


# -- verify -------------------------------------------------------------------------


def _filter_signature(desc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The part of a content-filter description that must match on verify.

    The stored path is dropped: it is absolute on the machine that built the manifest
    and may differ in a clean clone, while the version and hash identify the gazetteer.
    """
    if not desc:
        return None
    return {
        "version": desc.get("version"),
        "sha256": desc.get("sha256"),
        "categories": desc.get("categories"),
    }


def verify(corpus_path: Path, manifest: Dict[str, Any], base: Path) -> List[str]:
    """Check an existing corpus file against its manifest. Returns a list of problems.

    Rebuilds from the spec and the raw inputs, then compares input hashes, counts, the
    output hash and the bytes on disk. An empty list means everything matches.
    """
    problems: List[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version {manifest.get('schema_version')} != {SCHEMA_VERSION}")
    spec = Spec.from_dict(manifest["spec"])
    try:
        content_filter = load_content_filter(spec, base)
    except EvalCorpusError as exc:
        problems.append(str(exc))
        return problems
    want_filter = manifest.get("content_filter")
    got_filter = _describe_filter(content_filter, base)
    if _filter_signature(got_filter) != _filter_signature(want_filter):
        problems.append("content_filter differs from the manifest version/hash")
    for sid, meta in manifest["inputs"].items():
        src = next((s for s in spec.sources if s.id == sid), None)
        if src is None:
            problems.append(f"manifest input {sid!r} is not in the spec")
            continue
        actual = sha256_file(_resolve(src, base))
        if actual != meta["sha256"]:
            problems.append(f"input {sid!r} changed (sha256 differs)")
    if problems:
        return problems

    rebuilt = build(spec, base)
    if rebuilt["manifest"]["counts"] != manifest["counts"]:
        problems.append("counts differ from the manifest")
    if rebuilt["manifest"]["output_sha256"] != manifest["output_sha256"]:
        problems.append("rebuilt output hash differs from the manifest")
    on_disk = corpus_path.read_text(encoding="utf-8")
    if hashlib.sha256(on_disk.encode("utf-8")).hexdigest() != manifest["output_sha256"]:
        problems.append("corpus file on disk does not match the manifest hash")
    for line_no, line in enumerate(on_disk.splitlines(), 1):
        rec = json.loads(line)
        if rec["text"] != unicodedata.normalize("NFC", rec["text"]):
            problems.append(f"line {line_no}: text is not NFC")
            break
    return problems


def load_spec(path: Path) -> Spec:
    return Spec.from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_outputs(result: Dict[str, Any], corpus_path: Path, manifest_path: Optional[Path]) -> None:
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(serialize(result["records"]), encoding="utf-8")
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
