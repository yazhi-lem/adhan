"""Tests for the frozen tokenizer-benchmark eval corpus builder (adhan#74, part 1).

Uses small synthetic fixtures only; no real corpus text is read or sent anywhere.

Run: PYTHONPATH=src python -m adhan_slm.eval.eval_corpus_tests
"""

import json
import sys
import tempfile
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on path

from adhan_slm.core.selftest import run_module_tests  # noqa: E402
from adhan_slm.eval import eval_corpus as ec  # noqa: E402

_TA = ["அன்பு", "அறிவு", "நல்ல", "மனிதன்", "கடல்", "மலை", "வானம்", "பூமி", "நீர்", "காற்று"]
_TG = ["naan", "inge", "irukken", "nee", "enna", "pannura", "romba", "nalla", "iruku", "vanga"]


def _tamil_sentences(n):
    return [
        " ".join(_TA[(i + k) % len(_TA)] for k in range(4)) + f" {_TA[i % 10]}{_TA[(i * 3) % 10]}."
        for i in range(n)
    ]


def _tanglish_sentences(n):
    return [
        " ".join(_TG[(i + k) % len(_TG)] for k in range(4)) + f" {_TG[(i * 3) % 10]}{i}."
        for i in range(n)
    ]


def _write_jsonl(path, texts):
    path.write_text(
        "".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts),
        encoding="utf-8",
    )


def _make_spec(base, per_register=10, **overrides):
    _write_jsonl(base / "classical.jsonl", _tamil_sentences(40))
    _write_jsonl(base / "modern.jsonl", ["இது புதிய உரை " + t for t in _tamil_sentences(40)])
    (base / "tg.txt").write_text("\n".join(_tanglish_sentences(40)), encoding="utf-8")
    raw = {
        "per_register": per_register,
        "seed": 7,
        "min_chars": 8,
        "max_chars": 200,
        "sources": [
            {
                "id": "c1",
                "register": "classical",
                "path": "classical.jsonl",
                "license": "CC0",
                "url": "https://example.org/c",
            },
            {
                "id": "m1",
                "register": "modern",
                "path": "modern.jsonl",
                "license": "CC0",
                "url": "https://example.org/m",
            },
            {
                "id": "t1",
                "register": "tanglish",
                "path": "tg.txt",
                "format": "txt",
                "license": "CC0",
                "url": "https://example.org/t",
            },
        ],
    }
    raw.update(overrides)
    return ec.Spec.from_dict(raw)


def test_build_has_all_registers_and_requested_counts():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        res = ec.build(_make_spec(base), base)
        assert res["manifest"]["counts"]["per_register"] == {
            "classical": 10,
            "modern": 10,
            "tanglish": 10,
        }
        assert {r["register"] for r in res["records"]} == set(ec.REGISTERS)
        assert ec.shortfalls(res["manifest"]) == []


def test_build_is_deterministic_and_input_order_independent():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        a = ec.build(_make_spec(base), base)
        lines = (base / "classical.jsonl").read_text(encoding="utf-8").splitlines()
        (base / "classical.jsonl").write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
        b = ec.build(_make_spec(base), base)
        assert [r["text"] for r in a["records"] if r["register"] == "classical"] == [
            r["text"] for r in b["records"] if r["register"] == "classical"
        ]


def test_seed_changes_the_sample():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        a = ec.build(_make_spec(base), base)
        b = ec.build(_make_spec(base, seed=8), base)
        assert a["manifest"]["output_sha256"] != b["manifest"]["output_sha256"]


def test_output_is_nfc_and_script_checked():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        res = ec.build(_make_spec(base), base)
        for r in res["records"]:
            assert r["text"] == unicodedata.normalize("NFC", r["text"])
            assert ec.script_ok(r["text"], r["register"])
        assert not ec.script_ok("naan inge irukken", "classical")
        assert not ec.script_ok("அன்பு அறிவு naan", "tanglish")
        assert not ec.script_ok("1234 !!!", "tanglish")


def test_nfd_input_is_normalised():
    nfd = unicodedata.normalize("NFD", "கொடு கொடு கொடு கொடு கொடு")
    assert ec.split_sentences(nfd) == [unicodedata.normalize("NFC", nfd)]


def test_duplicates_across_sources_are_dropped():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        spec = _make_spec(base, per_register=1000)
        _write_jsonl(base / "modern.jsonl", _tamil_sentences(40))  # same text as classical
        res = ec.build(spec, base)
        texts = [r["text"] for r in res["records"]]
        assert len(texts) == len(set(texts))
        assert res["manifest"]["counts"]["per_register"]["modern"] == 0


def test_shortfall_is_reported():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        res = ec.build(_make_spec(base, per_register=10_000), base)
        assert len(ec.shortfalls(res["manifest"])) == 3


def test_unlicensed_or_incomplete_sources_are_rejected():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for field, value in (("license", ""), ("license", "TODO"), ("url", "TODO: later")):
            spec = _make_spec(base)
            raw = spec.to_dict()
            raw["sources"][0][field] = value
            try:
                ec.Spec.from_dict(raw)
            except ec.EvalCorpusError:
                continue
            raise AssertionError(f"accepted a source with {field}={value!r}")


def test_spec_requires_every_register_and_unique_ids():
    with tempfile.TemporaryDirectory() as d:
        raw = _make_spec(Path(d)).to_dict()
        raw["sources"] = raw["sources"][:2]
        for bad in (raw, {**raw, "sources": [raw["sources"][0], raw["sources"][0]]}):
            try:
                ec.Spec.from_dict(bad)
            except ec.EvalCorpusError:
                continue
            raise AssertionError("accepted an invalid spec")


def test_verify_passes_then_detects_tampering():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        res = ec.build(_make_spec(base), base)
        out = base / "out" / "bench.jsonl"
        ec.write_outputs(res, out, base / "out" / "manifest.json")
        manifest = json.loads((base / "out" / "manifest.json").read_text(encoding="utf-8"))
        assert ec.verify(out, manifest, base) == []

        out.write_text(out.read_text(encoding="utf-8") + '{"text": "x"}\n', encoding="utf-8")
        assert any("on disk" in p for p in ec.verify(out, manifest, base))

        ec.write_outputs(res, out, None)  # restore
        (base / "tg.txt").write_text("changed input\n", encoding="utf-8")
        assert any("changed" in p for p in ec.verify(out, manifest, base))


def test_manifest_records_licence_url_and_hashes():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        m = ec.build(_make_spec(base), base)["manifest"]
        assert m["unicode_normalization"] == "NFC"
        for sid in ("c1", "m1", "t1"):
            meta = m["inputs"][sid]
            assert meta["license"] == "CC0" and meta["url"].startswith("https://")
            assert len(meta["sha256"]) == 64


if __name__ == "__main__":
    run_module_tests(globals(), "eval corpus")
