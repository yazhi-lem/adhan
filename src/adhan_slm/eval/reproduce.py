"""One-command reproduction of the Tamil tokenizer benchmark (adhan#74, part 3).

Ties parts 1 and 2 together:

  1. check the pinned library versions
  2. fetch any raw source that is missing (only sources with a known fetcher)
  3. rebuild the frozen eval corpus and compare it with the committed manifest
  4. run the harness on every configured tokenizer
  5. write the results tables, or in ``--check`` mode compare them with the committed ones

Everything is driven by one config file (see ``eval_manifests/tokenizer_bench.reproduce.example.json``).
The frozen corpus is protected: if the rebuilt corpus differs from the committed manifest
(an input file changed, the spec changed), the run fails unless ``refreeze`` is set, so a
silent change of the evaluation data cannot slip into the published numbers.

No text is sent anywhere. Fetchers download public source files; tokenizers run locally.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from adhan_slm.eval import eval_corpus as ec
from adhan_slm.eval import tokenizer_bench as tb
from adhan_slm.tokenizer.swaram_tokenizer import SwaramTokenizer

SCHEMA_VERSION = 1
Fetcher = Callable[[Path], None]


class ReproduceError(RuntimeError):
    """A step failed; the message says what to do."""


@dataclass(frozen=True)
class Config:
    spec: str
    manifest: str
    corpus: str
    results_json: str
    results_md: str
    tokenizers: List[Dict[str, Any]]
    pins: Dict[str, str]
    fetch: Dict[str, str]
    price_per_million: Optional[float] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        need = ("spec", "manifest", "corpus", "results_json", "results_md", "tokenizers")
        missing = [k for k in need if k not in raw]
        if missing:
            raise ReproduceError(f"config is missing: {', '.join(missing)}")
        if not raw["tokenizers"]:
            raise ReproduceError("config lists no tokenizers")
        price = raw.get("price_per_million")
        return cls(
            spec=str(raw["spec"]),
            manifest=str(raw["manifest"]),
            corpus=str(raw["corpus"]),
            results_json=str(raw["results_json"]),
            results_md=str(raw["results_md"]),
            tokenizers=list(raw["tokenizers"]),
            pins={str(k): str(v) for k, v in raw.get("pins", {}).items()},
            fetch={str(k): str(v) for k, v in raw.get("fetch", {}).items()},
            price_per_million=None if price is None else float(price),
        )


def load_config(path: Path) -> Config:
    return Config.from_dict(json.loads(path.read_text(encoding="utf-8")))


# -- step 1: pinned versions --------------------------------------------------------


def check_pins(pins: Mapping[str, str]) -> List[str]:
    """Problems with pinned library versions (empty list means all match)."""
    problems: List[str] = []
    for name, want in sorted(pins.items()):
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            problems.append(f"{name} is not installed (pinned {want})")
            continue
        if have != want:
            problems.append(f"{name} is {have}, pinned {want}")
    return problems


def environment() -> Dict[str, str]:
    """Versions to stamp next to results (informational; not compared)."""
    env = {"python": sys.version.split()[0]}
    for pkg in ("tokenizers", "transformers"):
        try:
            env[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return env


# -- step 2: fetch missing sources --------------------------------------------------


def _fetch_thirukkural(dest: Path) -> None:
    from data_scraper.raw_extractors.thirukkural_extractor import (
        build_datasets,
        fetch_thirukkural_data,
    )

    build_datasets(fetch_thirukkural_data(), dest.parent)


DEFAULT_FETCHERS: Dict[str, Fetcher] = {"thirukkural": _fetch_thirukkural}


def fetch_missing(
    spec: ec.Spec,
    cfg: Config,
    base: Path,
    fetchers: Mapping[str, Fetcher] = DEFAULT_FETCHERS,
) -> List[str]:
    """Fetch every source file that is not on disk. Returns the source ids fetched."""
    fetched: List[str] = []
    for src in spec.sources:
        path = Path(src.path)
        path = path if path.is_absolute() else base / path
        if path.is_file():
            continue
        name = cfg.fetch.get(src.id)
        if name is None or name not in fetchers:
            raise ReproduceError(
                f"source {src.id!r} is missing at {path} and has no fetcher. Put the file "
                "there (see docs/EVAL_CORPUS.md) or add a 'fetch' entry for it."
            )
        fetchers[name](path)
        if not path.is_file():
            raise ReproduceError(f"fetcher {name!r} did not create {path}")
        fetched.append(src.id)
    return fetched


# -- step 3: frozen corpus ----------------------------------------------------------


def build_frozen_corpus(spec: ec.Spec, cfg: Config, base: Path, refreeze: bool) -> Dict[str, Any]:
    """Build the corpus and check it against the committed manifest.

    Returns the build result. Raises ``ReproduceError`` if the corpus no longer matches
    the committed manifest, unless ``refreeze`` is true.
    """
    result = ec.build(spec, base)
    short = ec.shortfalls(result["manifest"])
    if short:
        raise ReproduceError("not enough usable sentences: " + "; ".join(short))
    manifest_path = base / cfg.manifest
    if manifest_path.is_file() and not refreeze:
        committed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if committed != json.loads(json.dumps(result["manifest"], sort_keys=True)):
            raise ReproduceError(
                "the rebuilt corpus differs from the committed manifest "
                f"({cfg.manifest}): an input file or the spec changed. Fix the input, or "
                "re-run with --refreeze if the change is intended."
            )
    return result


# -- step 4: run the harness --------------------------------------------------------


def build_adapters(cfg: Config, base: Path) -> List[tb.TokenizerAdapter]:
    adapters: List[tb.TokenizerAdapter] = []
    for entry in cfg.tokenizers:
        kind = entry.get("kind")
        if kind == "baselines":
            adapters += [tb.CodepointAdapter(), tb.ByteAdapter()]
        elif kind == "swaram":
            tok = SwaramTokenizer.from_files(
                str(base / entry["vocab"]), str(base / entry["merges"])
            )
            adapters.append(tb.SwaramAdapter(tok, name=entry.get("name", "swaram")))
        elif kind == "tokenizers-json":
            from tokenizers import Tokenizer

            adapters.append(
                tb.TokenizersAdapter(Tokenizer.from_file(str(base / entry["path"])), entry["name"])
            )
        elif kind == "transformers-local":
            from transformers import AutoTokenizer

            hf = AutoTokenizer.from_pretrained(str(base / entry["path"]), local_files_only=True)
            adapters.append(tb.TransformersAdapter(hf, entry["name"]))
        else:
            raise ReproduceError(f"unknown tokenizer kind {kind!r}")
    names = [a.name for a in adapters]
    if len(set(names)) != len(names):
        raise ReproduceError(f"tokenizer names must be unique, got {names}")
    return adapters


def _round(obj: Any, ndigits: int = 8) -> Any:
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, ndigits) for v in obj]
    return obj


def run_harness(
    cfg: Config, records: List[Dict[str, str]], base: Path, corpus_sha256: str
) -> Dict[str, Any]:
    """Results document. Contains no timestamps or timings, so it is reproducible."""
    results = [tb.evaluate(a, records, cfg.price_per_million) for a in build_adapters(cfg, base)]
    return {
        "schema_version": SCHEMA_VERSION,
        "corpus_sha256": corpus_sha256,
        "price_per_million": cfg.price_per_million,
        "tokenizers": [
            _round({"tokenizer": r.tokenizer, "overall": r.overall, "per_register": r.per_register})
            for r in results
        ],
    }


def results_markdown(doc: Mapping[str, Any]) -> str:
    results = [
        tb.Result(t["tokenizer"], t["overall"], t["per_register"]) for t in doc["tokenizers"]
    ]
    return tb.to_markdown(results, doc["price_per_million"])


def diff_json(a: Any, b: Any, path: str = "$") -> List[str]:
    """Human-readable differences between two JSON values."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: List[str] = []
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: only in regenerated")
            elif k not in b:
                out.append(f"{path}.{k}: only in committed")
            else:
                out += diff_json(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff_json(x, y, f"{path}[{i}]")]
    return [] if a == b else [f"{path}: regenerated {a!r} vs committed {b!r}"]


# -- orchestration ------------------------------------------------------------------


def run(
    cfg: Config,
    base: Path,
    *,
    check: bool = False,
    refreeze: bool = False,
    allow_version_drift: bool = False,
    fetchers: Mapping[str, Fetcher] = DEFAULT_FETCHERS,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Run every step. In ``check`` mode nothing is written and a mismatch raises."""
    problems = check_pins(cfg.pins)
    if problems and not allow_version_drift:
        raise ReproduceError(
            "pinned versions do not match: " + "; ".join(problems) + ". Install the pinned "
            "versions or pass --allow-version-drift (the numbers may then differ)."
        )
    for p in problems:
        log(f"warning: {p}")

    spec = ec.load_spec(base / cfg.spec)
    fetched = fetch_missing(spec, cfg, base, fetchers)
    if fetched:
        log(f"fetched: {', '.join(fetched)}")

    built = build_frozen_corpus(spec, cfg, base, refreeze)
    manifest = built["manifest"]
    counts = manifest["counts"]["per_register"]
    log(f"corpus: {manifest['counts']['total']} sentences {counts}")

    doc = run_harness(cfg, built["records"], base, manifest["output_sha256"])
    md = results_markdown(doc)

    if check:
        problems_out: List[str] = []
        committed_json = base / cfg.results_json
        committed_md = base / cfg.results_md
        for committed in (committed_json, committed_md):
            if not committed.is_file():
                problems_out.append(f"committed file missing: {committed}")
        if not problems_out:
            old = json.loads(committed_json.read_text(encoding="utf-8"))
            old.pop("environment", None)  # informational, not compared
            problems_out += diff_json(json.loads(json.dumps(doc)), old)
            if committed_md.read_text(encoding="utf-8") != md:
                problems_out.append(f"{cfg.results_md} differs from the regenerated table")
        if problems_out:
            raise ReproduceError(
                "results do not match the committed tables:\n  " + "\n  ".join(problems_out)
            )
        log("OK: regenerated results match the committed tables")
        return doc

    ec.write_outputs(built, base / cfg.corpus, base / cfg.manifest)
    out_json = base / cfg.results_json
    out_md = base / cfg.results_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {**doc, "environment": environment()}, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    out_md.write_text(md, encoding="utf-8")
    log(f"wrote {cfg.results_json} and {cfg.results_md}")
    return doc
