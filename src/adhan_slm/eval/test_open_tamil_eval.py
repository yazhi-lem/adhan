"""Tests for the open-tamil-backed eval modules (morphology, kid-level
prompts, n-gram baseline). Skip cleanly if open-tamil isn't installed.

Run: python -m pytest (or python this_file.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on path

from adhan_slm.external.open_tamil_bridge import HAS_OPEN_TAMIL  # noqa: E402

INFLECTED_WORDS = [
    "படித்துக்கொண்டிருந்தேன்", "எழுதினேன்", "பேசினோம்", "வந்தார்கள்",
    "சொன்னாள்", "ஓடுகிறான்", "படிக்கிறாள்", "வருவேன்", "செய்தார்",
    "நடந்தது", "பார்த்தேன்", "கேட்டான்",
]

GRAMMATICAL_PHRASES = [
    "மரம் வளர்ந்தது",
    "நான் பள்ளிக்கு போனேன்",
    "அவன் நன்றாக பேசுகிறான்",
]

CORPUS = [
    "தமிழ் ஒரு அழகான மொழி", "நான் தமிழில் படிக்கிறேன்",
    "இது ஒரு நல்ல நாள்", "அவன் பள்ளிக்கு சென்றான்",
    "நாய் ஓடுகிறது", "பூனை பாலைக் குடிக்கிறது",
]


def test_stemmer_boundary_agreement_on_trained_tokenizer():
    if not HAS_OPEN_TAMIL:
        print("SKIP (open-tamil not installed)")
        return
    from adhan_slm.tokenizer import SwaramTokenizer, default_akshara_inventory
    from adhan_slm.eval.morphology import stemmer_boundary_agreement

    tok = SwaramTokenizer.train(INFLECTED_WORDS, vocab_size=len(default_akshara_inventory()) + 20)
    report = stemmer_boundary_agreement(tok, INFLECTED_WORDS)
    assert report.n_with_suffix > 0, "expected the stemmer to find suffixes in inflected words"
    assert 0.0 <= report.agreement_rate <= 1.0


def test_sandhi_correctness_rate_on_grammatical_phrases():
    if not HAS_OPEN_TAMIL:
        print("SKIP (open-tamil not installed)")
        return
    from adhan_slm.eval.morphology import sandhi_correctness_rate

    report = sandhi_correctness_rate(GRAMMATICAL_PHRASES)
    assert report.n_words == sum(len(p.split()) for p in GRAMMATICAL_PHRASES)
    # already-grammatical phrases should need no correction
    assert report.word_correctness_rate == 1.0, report.corrections


def test_kid_level_prompts_are_well_formed():
    if not HAS_OPEN_TAMIL:
        print("SKIP (open-tamil not installed)")
        return
    from adhan_slm.eval.kid_level_prompts import build_kid_level_prompts

    prompts = build_kid_level_prompts(n=50, seed=0)
    assert len(prompts) == 50
    assert len({p.word for p in prompts}) == 50, "expected 50 distinct seed words"
    for p in prompts:
        assert p.word in p.prompt


def test_kid_level_prompts_deterministic():
    if not HAS_OPEN_TAMIL:
        print("SKIP (open-tamil not installed)")
        return
    from adhan_slm.eval.kid_level_prompts import build_kid_level_prompts

    a = build_kid_level_prompts(n=20, seed=42)
    b = build_kid_level_prompts(n=20, seed=42)
    assert [p.prompt for p in a] == [p.prompt for p in b]


def test_ngram_baseline_perplexity_is_finite_and_positive():
    if not HAS_OPEN_TAMIL:
        print("SKIP (open-tamil not installed)")
        return
    from adhan_slm.eval.ngram_baseline import AksharaUnigramBaseline

    baseline = AksharaUnigramBaseline(CORPUS)
    ppl = baseline.perplexity("நான் தமிழில் படிக்கிறேன்")
    assert ppl > 0 and ppl == ppl, f"expected finite positive perplexity, got {ppl}"  # ppl==ppl false for NaN


def _run():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")


if __name__ == "__main__":
    _run()
