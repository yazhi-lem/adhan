"""Tamil-first evaluation for Adhan SLM (roadmap Phase 4).

Modules:
  morphology.py         stemmer-boundary agreement (Layer B vs open-tamil TamilStemmer),
                         sandhi (புணர்ச்சி) correctness rate via open-tamil tamilsandhi
  kid_level_prompts.py  50-prompt kid-level (5-7y/o) eval set, seeded from open-tamil
                         solthiruthi lexicons
  ngram_baseline.py     classical per-akshara unigram perplexity floor, via
                         open-tamil ngram.LetterModels

Planned:
  perplexity.py   per-akshara / per-word perplexity (comparable across vocabs)
  codeswitch.py   Tamil-English mixed robustness

All open-tamil-backed modules require `pip install -r requirements-jax.txt`
(or `pip install open-tamil`) and skip cleanly in tests if it's absent — see
src/adhan_slm/external/open_tamil_bridge.py. See docs/ARCHITECTURE_SWARAM_SLM.md §5/§10.
"""
