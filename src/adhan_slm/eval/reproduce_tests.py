"""Tests for the one-command benchmark reproduction (adhan#74, part 3).

Synthetic fixtures only; the "fetchers" write fake files, nothing is downloaded.

Run: PYTHONPATH=src python -m adhan_slm.eval.reproduce_tests
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on path

from adhan_slm.core.selftest import run_module_tests  # noqa: E402
from adhan_slm.eval import reproduce as rp  # noqa: E402

_TA = ["அன்பு", "அறிவு", "நல்ல", "மனிதன்", "கடல்", "மலை", "வானம்", "பூமி", "நீர்", "காற்று"]
_TG = ["naan", "inge", "irukken", "nee", "enna", "pannura", "romba", "nalla", "iruku", "vanga"]


def _ta(n, prefix=""):
    return [
        prefix + " ".join(_TA[(i + k) % 10] for k in range(4)) + f" {_TA[i % 10]}{_TA[i * 3 % 10]}."
        for i in range(n)
    ]


def _tg(n):
    return [
        " ".join(_TG[(i + k) % 10] for k in range(4)) + f" {_TG[i * 3 % 10]}{i}." for i in range(n)
    ]


def _fake_classical(dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        "".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in _ta(40)),
        encoding="utf-8",
    )


def _setup(base, with_classical=True):
    (base / "raw").mkdir()
    if with_classical:
        _fake_classical(base / "raw" / "classical.jsonl")
    (base / "raw" / "modern.txt").write_text("\n".join(_ta(40, "இது புதிய ")), encoding="utf-8")
    (base / "raw" / "tg.txt").write_text("\n".join(_tg(40)), encoding="utf-8")
    spec = {
        "per_register": 10,
        "seed": 3,
        "min_chars": 8,
        "max_chars": 200,
        "sources": [
            {"id": "c", "register": "classical", "path": "raw/classical.jsonl",
             "license": "CC0", "url": "https://example.org/c"},
            {"id": "m", "register": "modern", "path": "raw/modern.txt", "format": "txt",
             "license": "CC0", "url": "https://example.org/m"},
            {"id": "t", "register": "tanglish", "path": "raw/tg.txt", "format": "txt",
             "license": "CC0", "url": "https://example.org/t"},
        ],
    }  # fmt: skip
    (base / "spec.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    return rp.Config.from_dict(
        {
            "spec": "spec.json",
            "manifest": "out/manifest.json",
            "corpus": "out/corpus.jsonl",
            "results_json": "out/results.json",
            "results_md": "out/results.md",
            "tokenizers": [{"kind": "baselines"}],
            "fetch": {"c": "fake"},
        }
    )


FETCHERS = {"fake": _fake_classical}
QUIET = {"fetchers": FETCHERS, "log": lambda _m: None}


def test_run_writes_corpus_manifest_and_tables():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        cfg = _setup(base)
        doc = rp.run(cfg, base, **QUIET)
        for name in ("corpus.jsonl", "manifest.json", "results.json", "results.md"):
            assert (base / "out" / name).is_file(), name
        assert {t["tokenizer"] for t in doc["tokenizers"]} == {"codepoint", "utf8-byte"}
        assert "timestamp" not in json.dumps(doc)


def test_second_run_is_byte_identical():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        cfg = _setup(base)
        rp.run(cfg, base, **QUIET)
        first = {n: (base / "out" / n).read_bytes() for n in ("corpus.jsonl", "results.md")}
        rp.run(cfg, base, **QUIET)
        for n, b in first.items():
            assert (base / "out" / n).read_bytes() == b


def test_check_passes_after_a_run_and_writes_nothing():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        cfg = _setup(base)
        rp.run(cfg, base, **QUIET)
        before = (base / "out" / "results.json").read_bytes()
        rp.run(cfg, base, check=True, **QUIET)
        assert (base / "out" / "results.json").read_bytes() == before


def test_check_detects_a_changed_table():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        cfg = _setup(base)
        rp.run(cfg, base, **QUIET)
        path = base / "out" / "results.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["tokenizers"][0]["overall"]["tokens"] += 1
        path.write_text(json.dumps(doc), encoding="utf-8")
        try:
            rp.run(cfg, base, check=True, **QUIET)
        except rp.ReproduceError as exc:
            assert "tokens" in str(exc)
            return
        raise AssertionError("check accepted a modified table")


def test_check_fails_when_committed_tables_are_missing():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        cfg = _setup(base)
        try:
            rp.run(cfg, base, check=True, **QUIET)
        except rp.ReproduceError as exc:
            assert "missing" in str(exc)
            return
        raise AssertionError("check passed without committed tables")


def test_missing_source_is_fetched_and_fetch_is_not_repeated():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        cfg = _setup(base, with_classical=False)
        logs = []
        rp.run(cfg, base, fetchers=FETCHERS, log=logs.append)
        assert any("fetched: c" in m for m in logs)
        logs.clear()
        rp.run(cfg, base, fetchers=FETCHERS, log=logs.append)
        assert not any("fetched" in m for m in logs)


def test_missing_source_without_fetcher_gives_a_clear_error():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        cfg = _setup(base)
        (base / "raw" / "tg.txt").unlink()
        try:
            rp.run(cfg, base, **QUIET)
        except rp.ReproduceError as exc:
            assert "no fetcher" in str(exc) and "'t'" in str(exc)
            return
        raise AssertionError("missing source was not reported")


def test_changed_input_is_blocked_until_refreeze():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        cfg = _setup(base)
        rp.run(cfg, base, **QUIET)
        (base / "raw" / "tg.txt").write_text("\n".join(_tg(60)), encoding="utf-8")
        try:
            rp.run(cfg, base, **QUIET)
        except rp.ReproduceError as exc:
            assert "differs from the committed manifest" in str(exc)
        else:
            raise AssertionError("a changed corpus was accepted silently")
        rp.run(cfg, base, refreeze=True, **QUIET)
        rp.run(cfg, base, check=True, **QUIET)


def test_pin_mismatch_fails_unless_drift_allowed():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        cfg = _setup(base)
        pinned = rp.Config.from_dict({**_cfg_dict(cfg), "pins": {"pytest": "0.0.0"}})
        try:
            rp.run(pinned, base, **QUIET)
        except rp.ReproduceError as exc:
            assert "pinned" in str(exc)
        else:
            raise AssertionError("pin mismatch was accepted")
        logs = []
        rp.run(pinned, base, allow_version_drift=True, fetchers=FETCHERS, log=logs.append)
        assert any(m.startswith("warning:") for m in logs)
        assert rp.check_pins({"definitely-not-installed-pkg": "1"})[0].endswith("(pinned 1)")


def _cfg_dict(cfg):
    return {
        "spec": cfg.spec, "manifest": cfg.manifest, "corpus": cfg.corpus,
        "results_json": cfg.results_json, "results_md": cfg.results_md,
        "tokenizers": cfg.tokenizers, "fetch": cfg.fetch,
    }  # fmt: skip


def test_config_validation():
    for bad in ({}, {**_cfg_dict(rp.Config.from_dict(_cfg_dict_min())), "tokenizers": []}):
        try:
            rp.Config.from_dict(bad)
        except rp.ReproduceError:
            continue
        raise AssertionError("accepted a bad config")


def _cfg_dict_min():
    return {
        "spec": "s", "manifest": "m", "corpus": "c", "results_json": "j", "results_md": "d",
        "tokenizers": [{"kind": "baselines"}],
    }  # fmt: skip


def test_unknown_tokenizer_kind_and_duplicate_names():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        cfg = _setup(base)
        for toks in ([{"kind": "nope"}], [{"kind": "baselines"}, {"kind": "baselines"}]):
            bad = rp.Config.from_dict({**_cfg_dict(cfg), "tokenizers": toks})
            try:
                rp.build_adapters(bad, base)
            except rp.ReproduceError:
                continue
            raise AssertionError(f"accepted {toks}")


def test_diff_json_reports_paths():
    a = {"x": {"y": 1, "z": [1, 2]}, "only_a": 1}
    b = {"x": {"y": 2, "z": [1, 3]}, "only_b": 1}
    out = rp.diff_json(a, b)
    assert any("$.x.y" in m for m in out) and any("$.x.z[1]" in m for m in out)
    assert any("only_a" in m for m in out) and any("only_b" in m for m in out)
    assert rp.diff_json(a, a) == []


if __name__ == "__main__":
    run_module_tests(globals(), "reproduce")
