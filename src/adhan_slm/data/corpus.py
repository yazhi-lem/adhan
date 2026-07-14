"""Corpus reading: normalize heterogeneous raw text into a stream of documents.

Accepts the formats the existing scrapers/exporters in `src/data_scraper/` produce
as well as ad-hoc dumps:

  * ``.txt``   — one document per non-empty line, OR the whole file as one document
                 when ``line_documents=False``.
  * ``.jsonl`` — one JSON object per line; the document text is taken from the first
                 present key in ``text_keys`` (default: text/content/body/sentence).
  * a directory — every ``.txt``/``.jsonl`` under it, sorted for determinism.

Everything is NFC-normalized and whitespace-trimmed; empty docs are dropped. This is
deliberately dependency-free (pure stdlib) so it runs anywhere.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

DEFAULT_TEXT_KEYS: Sequence[str] = ("text", "content", "body", "sentence", "tamil")
_SUFFIXES = (".txt", ".jsonl", ".json")


def _clean(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def _iter_jsonl(path: Path, text_keys: Sequence[str]) -> Iterator[str]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, str):
                doc = _clean(obj)
                if doc:
                    yield doc
                continue
            if isinstance(obj, dict):
                for key in text_keys:
                    if key in obj and isinstance(obj[key], str):
                        doc = _clean(obj[key])
                        if doc:
                            yield doc
                        break


def _iter_txt(path: Path, line_documents: bool) -> Iterator[str]:
    if line_documents:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                doc = _clean(line)
                if doc:
                    yield doc
    else:
        doc = _clean(path.read_text(encoding="utf-8"))
        if doc:
            yield doc


def _iter_file(path: Path, text_keys: Sequence[str], line_documents: bool) -> Iterator[str]:
    if path.suffix in (".jsonl", ".json"):
        yield from _iter_jsonl(path, text_keys)
    else:
        yield from _iter_txt(path, line_documents)


def iter_documents(
    source,
    text_keys: Sequence[str] = DEFAULT_TEXT_KEYS,
    line_documents: bool = True,
) -> Iterator[str]:
    """Yield cleaned document strings from a file, directory, or list of paths.

    ``source`` may be a path (str/Path) to a file or directory, or an iterable of
    paths. Directories are walked recursively; files are visited in sorted order so
    the document stream is deterministic across machines.
    """
    if isinstance(source, (str, Path)):
        paths: List[Path] = [Path(source)]
    else:
        paths = [Path(p) for p in source]

    files: List[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(
                sorted(f for f in p.rglob("*") if f.is_file() and f.suffix in _SUFFIXES)
            )
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"corpus source not found: {p}")

    for f in files:
        yield from _iter_file(f, text_keys, line_documents)


def read_corpus(
    source,
    text_keys: Sequence[str] = DEFAULT_TEXT_KEYS,
    line_documents: bool = True,
    limit: int | None = None,
) -> List[str]:
    """Eagerly collect documents into a list (convenience for small corpora / tests)."""
    out: List[str] = []
    for i, doc in enumerate(iter_documents(source, text_keys, line_documents)):
        if limit is not None and i >= limit:
            break
        out.append(doc)
    return out
