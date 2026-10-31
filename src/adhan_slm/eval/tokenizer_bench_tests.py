"""Tests for the Tamil tokenizer benchmark harness (adhan#74, part 2).

Uses a tiny synthetic Tamil/Tanglish fixture; no real corpus is read.

Run: PYTHONPATH=src python -m adhan_slm.eval.tokenizer_bench_tests
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on path

from adhan_slm.core.selftest import run_module_tests, skip_unless  # noqa: E402
from adhan_slm.eval import tokenizer_bench as tb  # noqa: E402
from adhan_slm.tokenizer import SwaramTokenizer  # noqa: E402

try:
    import tokenizers  # noqa: F401

    HAS_TOKENIZERS = True
except ImportError:
    HAS_TOKENIZERS = False

# "கா" is two code points (க + ா) but ONE akshara: the known grapheme-split case.
KAA = "கா"
FIXTURE = [
    {"register": "classical", "text": "அன்பு அறிவு நல்ல மனிதன்"},
    {"register": "classical", "text": "கடல் மலை வானம் பூமி"},
    {"register": "modern", "text": "தமிழ் கற்போம் வணக்கம் நண்பர்களே"},
    {"register": "tanglish", "text": "naan inge irukken nee enna pannura"},
]


def _swaram(byte_fallback=False, vocab_size=600):
    corpus = [r["text"] for r in FIXTURE if r["register"] != "tanglish"] * 5
    return SwaramTokenizer.train(corpus, vocab_size=vocab_size, byte_fallback=byte_fallback)


def test_known_grapheme_split_codepoint_baseline_splits_kaa():
    c = tb.measure_sentence(tb.CodepointAdapter(), KAA)
    assert c.aksharas == 1 and c.tokens == 2
    assert c.tamil_clusters == 1 and c.tamil_clusters_split == 1


def test_byte_baseline_splits_a_single_code_point():
    c = tb.measure_sentence(tb.ByteAdapter(), "க")  # one code point, three UTF-8 bytes
    assert c.tokens == 3 and c.tamil_clusters_split == 1


def test_swaram_never_splits_tamil_aksharas():
    adapter = tb.SwaramAdapter(_swaram())
    res = tb.evaluate(adapter, FIXTURE)
    assert res.overall["tamil_cluster_split_rate"] == 0.0
    for text in (KAA, "படித்துக்கொண்டிருந்தேன்"):
        assert tb.measure_sentence(adapter, text).tamil_clusters_split == 0


def test_swaram_span_count_matches_encode():
    adapter = tb.SwaramAdapter(_swaram(byte_fallback=True))
    for text in [r["text"] for r in FIXTURE] + ["நல்ல €5 😀 வணக்கம்"]:
        assert len(adapter.spans(text)) == len(adapter.tok.encode(text)), text


def test_swaram_byte_fallback_splits_non_tamil_but_not_tamil():
    adapter = tb.SwaramAdapter(_swaram(byte_fallback=True))
    c = tb.measure_sentence(adapter, "நல்ல 😀")
    assert c.clusters_split == 1  # the emoji is spread over four byte tokens
    assert c.tamil_clusters_split == 0


def test_fertility_counts():
    c = tb.measure_sentence(tb.CodepointAdapter(), "தமிழ் வணக்கம்")
    assert c.words == 2
    assert c.aksharas == 8  # த மி ழ் (3) + வ ண க் க ம் (5)
    assert c.tokens == len("தமிழ் வணக்கம்")
    m = tb.summarize(c)
    assert m["tokens_per_word"] == c.tokens / 2
    assert m["tokens_per_akshara"] == c.tokens / 8


def test_cost_per_1k_sentences():
    res = tb.evaluate(tb.CodepointAdapter(), FIXTURE, price_per_million=2.0)
    per_1k = res.overall["tokens_per_1k_sentences"]
    assert per_1k == res.overall["tokens"] / len(FIXTURE) * 1000
    assert abs(res.overall["usd_per_1k_sentences"] - per_1k * 2.0 / 1e6) < 1e-12
    assert "usd_per_1k_sentences" not in tb.evaluate(tb.CodepointAdapter(), FIXTURE).overall


def test_per_register_and_tanglish_has_no_tamil_split_rate():
    res = tb.evaluate(tb.CodepointAdapter(), FIXTURE)
    assert set(res.per_register) == {"classical", "modern", "tanglish"}
    assert res.per_register["tanglish"]["tamil_cluster_split_rate"] is None
    assert res.per_register["classical"]["tamil_cluster_split_rate"] > 0


def test_text_is_nfc_normalised_before_measuring():
    import unicodedata

    nfd = unicodedata.normalize("NFD", "கொடு")
    a = tb.measure_sentence(tb.CodepointAdapter(), nfd)
    b = tb.measure_sentence(tb.CodepointAdapter(), unicodedata.normalize("NFC", nfd))
    assert a == b


def test_load_corpus_and_validation():
    with tempfile.TemporaryDirectory() as d:
        good = Path(d) / "c.jsonl"
        good.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in FIXTURE), encoding="utf-8"
        )
        assert len(tb.load_corpus(good)) == len(FIXTURE)
        for name, body in (("bad.jsonl", '{"text": "x"}\n'), ("empty.jsonl", "\n")):
            p = Path(d) / name
            p.write_text(body, encoding="utf-8")
            try:
                tb.load_corpus(p)
            except ValueError:
                continue
            raise AssertionError(f"accepted {name}")


def test_markdown_report_lists_every_tokenizer_and_register():
    results = [
        tb.evaluate(a, FIXTURE, 1.0)
        for a in (tb.SwaramAdapter(_swaram()), tb.CodepointAdapter(), tb.ByteAdapter())
    ]
    md = tb.to_markdown(results, 1.0)
    for needle in ("swaram", "codepoint", "utf8-byte", "## tanglish", "USD/1k sent.", "n/a"):
        assert needle in md


def test_cli_end_to_end_with_swaram_files_and_baselines():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        corpus = base / "c.jsonl"
        corpus.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in FIXTURE), encoding="utf-8"
        )
        _swaram().save(str(base / "vocab.json"), str(base / "merges.txt"))
        rc = tb.main(
            [
                "--corpus", str(corpus),
                "--swaram", str(base / "vocab.json"), str(base / "merges.txt"),
                "--baselines",
                "--out-json", str(base / "r.json"),
                "--out-md", str(base / "r.md"),
            ]
        )  # fmt: skip
        assert rc == 0
        data = json.loads((base / "r.json").read_text(encoding="utf-8"))
        assert {x["tokenizer"] for x in data} == {"swaram", "codepoint", "utf8-byte"}
        assert (base / "r.md").read_text(encoding="utf-8").startswith("# Tamil tokenizer benchmark")


@skip_unless(HAS_TOKENIZERS, "tokenizers not installed")
def test_huggingface_byte_level_bpe_splits_tamil_clusters():
    from tokenizers import ByteLevelBPETokenizer

    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "train.txt"
        f.write_text("\n".join(r["text"] for r in FIXTURE), encoding="utf-8")
        hf = ByteLevelBPETokenizer()
        hf.train([str(f)], vocab_size=300, min_frequency=1)
    adapter = tb.TokenizersAdapter(hf, "bytelevel-bpe")
    res = tb.evaluate(adapter, FIXTURE)
    assert res.overall["tokens"] > 0
    # Byte-level BPE trained on a tiny corpus cannot keep every akshara in one token.
    assert res.per_register["classical"]["tamil_cluster_split_rate"] > 0


if __name__ == "__main__":
    run_module_tests(globals(), "tokenizer bench")
