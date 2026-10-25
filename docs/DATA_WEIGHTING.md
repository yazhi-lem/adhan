# Pure Tamil tokens, and how to weigh the mixture

Two questions that sound like one: *how much Tamil do we actually have?* and *what
should the model train on?* The first is a measurement; the second is a policy.

Tools: `scripts/pure_tamil_tokens.py` (measure / extract) and
`scripts/build_mixture.py` (build a weighted, reproducible mixture).

---

## 1. How many *pure Tamil* tokens do we have?

"328M tokens" is not "328M Tamil tokens". Measured on a **30,319-document sample**
(stride-derived across all 71 shards, real v2 tokenizer):

| class | share of content tokens | projected count |
|---|---|---|
| **pure Tamil** (every codepoint Tamil-script, word marker ignored) | **81.4%** | **~164.2M** |
| mixed (Tamil + Latin/digits: `AI`, `2024`, `கூல்`) | 8.6% | ~17.4M |
| non-Tamil (Latin words, digits, punctuation, `<0x..>` bytes) | 10.0% | ~20.2M |

And the denominator surprise:

| | count |
|---|---|
| all tokens | 328.4M |
| content tokens | **202.0M** (61.5%) |
| structural tokens (`▁` word markers) | **126.3M** (38.5%) |

**So the archive holds ~164M pure Tamil tokens, plus ~17M mixed and ~20M non-Tamil
content tokens.** Nearly 4 in 10 tokens are the `▁` word marker — real tokens the
model pays for, but they carry no Tamil.

### Purity by source (this is what mixture weights should key on)

| source | pure | mixed | non-Tamil |
|---|---|---|---|
| `local:akananooru` (classical) | **88.6%** | 4.8% | 6.6% |
| `local:open-sangam-classical` | 87.3% | 4.5% | 8.1% |
| `local:thirukkural-classical` | 85.3% | 8.4% | 6.3% |
| `fineweb-2-ta` | 83.8% | 4.6% | 8.2% |
| `ai4bharat-indiccorp-v2` | 82.4% | 9.9% | 7.8% |
| `wikipedia-ta` | 76.0% | 7.6% | 16.2% |
| `local:wikipedia-ta` | 72.8% | 6.7% | 21.0% |
| `local:maduraikanchi` | 71.3% | 11.7% | 17.0% |

Classical poetry is the purest text in the corpus (short lines, no URLs), which is
convenient: it is also the most morphology-dense.

### What that does to the token budget

The earlier "nano has enough data" conclusion counted *all* tokens:

| budget basis | tokens | for 16.7M params at 20 tok/param |
|---|---|---|
| all tokens | 328.4M | **20 tok/param — satisfied** |
| content tokens | 202.0M | 12 tok/param |
| **pure Tamil tokens** | **164.2M** | **~10 tok/param — thin** |

Both numbers are honest; they answer different questions. Structural and non-Tamil
tokens are cheap to learn (in fact 38% of the training signal is "predict the word
marker"), so per-param counting overstates how much Tamil the model actually sees.
**Treat ~164M pure Tamil tokens as the real budget**: enough for `adhan-nano`, thin
for anything larger, and the reason the mixture strategy below matters more than
another 50M scraped documents.

---

## 2. Strategy: what the mixture should be

The archive's *implicit* mixture is ingest order: **72% IndicCorp**. Nobody chose
that. Weights turn it into a decision.

### Principles

1. **Deduplicate before weighting.** 7.3% of the archive is near-duplicate
   (`docs/CORPUS_HEALTH.md`); weighting a duplicated source up multiplies its
   duplicates. `run_dedup.py --num-perm 128` first.
2. **Cap the dominant source.** IndicCorp's register (news wire, median 237 chars)
   should not be 72% of what a model learns about Tamil. ~45% keeps its syntax and
   named-entity value without letting it define the language.
3. **Up-weight morphology, not volume.** Classical texts are ~6% of tokens but the
   only source that exercises sandhi, agglutination and literary vocabulary. They
   also happen to be the purest. Give them more weight than their size implies.
4. **Keep one web-register source.** FineWeb-2 is where code-mixing, forum syntax and
   real-world messiness live; a model trained only on edited prose breaks on Reddit.
5. **No document repeats.** One epoch per document; repetition should come from a
   deliberately re-weighted mixture, never from looping shards.
6. **Longer documents carry more context per window.** With 32% of docs under 200
   chars, weighting by *tokens* (not documents) is what actually controls the mixture.

### v1 mixture (what we can build today, from the archive)

```bash
python scripts/build_mixture.py \
    --corpus data/raw/expanded --out data/final/mix_v1 \
    --max-tokens 30000000 --seq-len 512 \
    --weights "ai4bharat-indiccorp-v2:0.45,fineweb-2-ta:0.25,wikipedia-ta:0.15,local:0.15"
```

Measured result (`mixture.json`, 30M budget): **28.5M tokens packed** (95% of target;
the gap is hash-thinning against *estimated* availability), realized
**IndicCorp 41.5% · FineWeb 24.6% · Wikipedia 14.8% · classical/local 19.1%**.

Why 45/25/15/15 and not something fancier: with only four usable source families, a
temperature flattening (`w_i ∝ n_i^α`) adds complexity without changing the ordering.
Once education and legal sources exist, revisit with α≈0.7 to keep five sources in
the 10–40% band.

### v2 mixture (after the data work)

| register | v1 | v2 | why |
|---|---|---|---|
| IndicCorp | 45% | 35% | still the biggest single source, less dominant |
| FineWeb-2 | 25% | 25% | resume the run that was OOM-killed at 157k/400k docs |
| Wikipedia | 15% | 12% | near its ceiling |
| classical/local | 15% | 13% | purest + most morphology-dense |
| education (SCERT) | — | 10% | curriculum register, needs extraction |
| legal/government | — | 5% | formal register, needs extraction |

### Curriculum (cheap, no code changes)

Two stages, each a mixture directory the trainer points at:

| stage | tokens | mixture | rationale |
|---|---|---|---|
| 1 — shape | 60–70% of budget | v1 as above | learn orthography, syntax, vocabulary |
| 2 — anneal | last 30–40% | drop IndicCorp to ~25%, raise classical + Wikipedia, add education | high-quality tail is what generations imitate |

Implementation is just `data.shards` in the config: build `mix_v1_stage1/` and
`mix_v1_stage2/`, train on the first, then `--resume` against the second with the
same checkpoint dir. `max_steps` splits accordingly.

---

## 3. How to know the weights are right

Weights are a hypothesis; ablate them on the 30M shard set (which exists precisely
so this is cheap):

1. Build 3 mixtures: `v1`, `indiccorp-heavy` (65/15/10/10), `classical-heavy`
   (35/20/10/35).
2. Train `adhan-nano` for **915 updates (one epoch of 30M)** on each — same seed,
   same LR, ~1.4 h each on this CPU, ~30 min total on a T4.
3. Compare: **bits/akshara** on held-out (headline), fertility parity (tokenizer is
   fixed, so this only moves if packing changes), morphology boundary agreement,
   sandhi correctness, and kid-prompt readability.
4. Accept `v1` if it is within 2% of the best bits/akshara *and* wins on at least one
   Tamil-specific probe (sandhi or boundary agreement).

Report the loss too. A mixture that improves Tamil morphology while regressing 10% on
general next-token loss is a trade, not a win — say so explicitly.

## 4. Risks

| risk | why it bites | mitigation |
|---|---|---|
| Over-weighting classical | literary Tamil ≠ modern usage; the model starts writing 7th-century syntax | cap classical at ~15%; stage 2 only |
| Wikipedia "AI slop" | machine-translated articles inflate volume and flatten style | purity filter (`--min-doc-purity`) + prefer human-edited articles; audit samples |
| Short-document sources | 32% of docs <200 chars; per-doc packing silently drops them (a 30M "target" delivered 10.5M before this was fixed) | pack across document boundaries (now the default in `build_mixture.py`) |
| Val leakage | a doc in both train and val makes every gate meaningless | val is hash-sampled from the same stream, so a document lands on exactly one side |
| Weighting before dedup | duplicates get weighted up too | dedup first, always |

---

## 5. Reproducing

```bash
# 1. health + purity (fast samples; add --all for a full pass)
python scripts/corpus_health.py --corpus data/raw/expanded --docs 20000 --out docs/CORPUS_HEALTH.md
python scripts/pure_tamil_tokens.py --corpus data/raw/expanded --docs 30000

# 2. optional: extract a pure-Tamil-only corpus (>=95% pure documents)
python scripts/pure_tamil_tokens.py --corpus data/raw/expanded --all \
    --emit data/raw/pure_tamil --min-doc-purity 0.95

# 3. dedup, then build the weighted mixture
python scripts/run_dedup.py --corpus data/raw/expanded --num-perm 128 \
    --output data/raw/expanded-dedup
python scripts/build_mixture.py --corpus data/raw/expanded-dedup \
    --out data/final/mix_v1 --max-tokens 30000000 \
    --weights "ai4bharat-indiccorp-v2:0.45,fineweb-2-ta:0.25,wikipedia-ta:0.15,local:0.15"

# 4. train against the mixture
adhan_slm.configs/adhan_slm_nano_t4.yaml  ->  data.shards: data/final/mix_v1/
```

`mixture.json` in the output directory records requested weights, the per-group plan,
realized shares, document counts and packed token totals — so a run can be audited
against the mixture it claims to have used.
