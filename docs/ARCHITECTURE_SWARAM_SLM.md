# Adhan Native Tamil SLM — Architecture (Swaram Tokens + JAX)

> **Swaram** — Dravidian-language tokenizer prototype (Tamil first, then Kannada,
> Telugu, Malayalam). A parallel project, **Aksharam**, does the same for Hindi/Indic
> scripts. Both feed native SLMs via **yazhi-api** (internal deployment platform).

This document specifies the design of Adhan's from-scratch Tamil small language
model: the **swaram tokenizer**, the **agglutination-aware model**, and the
**JAX/Flax + MLflow** training stack. It complements `ROADMAP_JAX_SLM.md`.

---

## 1. Tamil script primer (what "swaram as token" means)

Tamil is an **abugida**. Text is a sequence of **grapheme clusters (aksharas)**,
each rendered from Unicode codepoints in the block **U+0B80–U+0BFF**:

| Class | Tamil | Members | Unicode |
|---|---|---|---|
| உயிர் **uyir** (vowels / *swaram*) | அ ஆ இ ஈ உ ஊ எ ஏ ஐ ஒ ஓ ஔ | 12 | U+0B85–U+0B94 |
| மெய் **mey** (consonant + pulli) | க் ங் ச் … ன் | 18 | base U+0B95–U+0BB9 + ் U+0BCD |
| உயிர்மெய் **uyirmey** | க கா கி … | 18×12 = 216 | base + matra U+0BBE–U+0BCC |
| ஆய்தம் **aytham** | ஃ | 1 | U+0B83 |
| Grantha (loanwords) | ஜ ஷ ஸ ஹ ஶ | 5 | U+0B9C, U+0BB7–U+0BB9, U+0BB6 |

**Akshara formation rule** used by the tokenizer:
```
akshara := uyir                       # standalone vowel        (அ, இ, உ …)
         | consonant + pulli          # pure consonant / mey    (க் , ண்)
         | consonant + matra?         # uyirmey (bare = inherent 'a')  (க, கி, கூ)
         | aytham                     # ஃ
```
Combining marks (matra U+0BBE–U+0BCC, pulli U+0BCD) **never** start an akshara;
they attach to the preceding base. This is exactly the boundary a naive BPE
tokenizer destroys.

**Why it matters:** a multilingual BPE tokenizer emits ~3–6 sub-tokens per Tamil
akshara. The swaram tokenizer emits **1 token per akshara** at the base layer, then
merges frequent morpheme runs. Result: shorter sequences, aligned boundaries, and a
small closed vocabulary a tiny model can actually learn.

---

## 2. Swaram tokenizer design

Two layers, both lossless and reversible:

### Layer A — akshara segmentation (deterministic, closed)
Regex/state-machine over Unicode that groups codepoints into aksharas per the rule
above. Handles: NFC normalization, Tamil digits (௦–௯) and ASCII digits, punctuation,
whitespace, and **code-switch Latin runs** (kept as byte/char tokens so English words
in Tamil text degrade gracefully). Output is a stream of akshara strings.

Base vocabulary (closed, ~250):
- 12 uyir + 18 mey + 216 uyirmey + aytham + Grantha set
- digits, punctuation, `▁` word-boundary marker, and control tokens
  `<pad> <bos> <eos> <unk> <mask>`

### Layer B — morpheme merges (learned, bounded)
A **BPE trained over aksharas** (not bytes), capped so the final vocab is 8k–16k.
Because Tamil agglutinates with a small, high-frequency suffix inventory, a few
thousand merges capture the productive suffix chains:
`-கிற- -ந்த- -ப்ப- -ஏன் -ஆய் -ஆன் -இல் -உக்கு -ஆக -கள் -இன்` …
So *படித்துக்கொண்டிருந்தேன்* collapses toward `[root] [aspect] [tense] [person]`
rather than a dozen fragments.

### Optional Layer C — morphological tags (agglutination signal)
A rule-based **sandhi (புணர்ச்சி) splitter** annotates morpheme boundaries and
suffix roles. These become an **auxiliary boundary embedding** the model reads
alongside token embeddings (see §3). Layer C is *additive*: if segmentation is
uncertain, Layer A/B output is still complete and lossless.

**Interface** (`src/adhan_slm/tokenizer/swaram_tokenizer.py`):
```python
tok = SwaramTokenizer.from_files("vocab.json", "merges.txt")  # or .train(corpus)
ids  = tok.encode("தமிழ் கற்போம்")          # -> List[int]
text = tok.decode(ids)                        # lossless round-trip
aks  = tok.aksharas("கற்போம்")               # -> ['க','ற்','போ','ம்']  (Layer A)
```

**Quality metric — fertility:** mean tokens per akshara on held-out text.
Target **< 1.15** (Layer B should merge, not split). Must be **lossless** round-trip.

---

## 3. Model architecture (agglutination-aware)

Decoder-only transformer (GPT-style), Flax, three sizes (`nano/tiny/mini` in §1 of
the roadmap). Tamil-specific choices:

- **Token embedding** over the 8k–16k swaram vocab (tied to output head).
- **Rotary position embeddings (RoPE)** — extrapolates context, no learned pos table.
- **Boundary embedding (optional):** a small learned vector added when Layer C marks
  a morpheme boundary, so the model perceives root/suffix structure explicitly.
- **RMSNorm + SwiGLU MLP**, pre-norm blocks — strong quality-per-parameter for SLMs.
- **Weight tying** input/output embeddings to save parameters (matters at nano scale).
- Causal LM objective (next-token / next-akshara prediction).

```
tokens ─embed─┐
              ├─(+boundary emb)─► [ RMSNorm→RoPE-Attn→+ ]×L ─►RMSNorm─►tied head─►logits
positions ────┘ (RoPE in attn)
```

Factored-embedding variant (research option, Phase 7): represent a token as
`root_emb ⊕ suffix_role_emb` to share parameters across the combinatorial suffix
space — potentially large savings for an agglutinative language.

---

## 4. Training stack (JAX/Flax + MLflow)

| Piece | Tool | Detail |
|---|---|---|
| Params/model | Flax | `jax.jit`-ed forward; bf16 compute, fp32 master weights |
| Optimizer | Optax | AdamW, cosine decay, linear warmup, grad clip 1.0, weight decay 0.1 |
| Sharding | `jax.sharding` / pjit | nano/tiny = single device; mini = data+tensor parallel |
| Data | Grain / tf.data | stream sharded tokenized TFRecord, sequence packing |
| Checkpoint | Orbax | async, resumable, keeps best-by-val |
| Tracking | **MLflow** | see below |

**MLflow contract** (`src/adhan_slm/training/mlflow_utils.py`):
- **Params:** full config (model size, LR schedule, batch, seq len, vocab, seed),
  code git SHA, dataset version/hash.
- **Metrics (per step/epoch):** train loss, val loss, per-akshara perplexity,
  tokens/sec, learning rate, grad norm.
- **Artifacts:** `tokenizer.json`, resolved config, eval report, sample generations,
  final checkpoint. Register the shippable model in the **MLflow Model Registry**
  (`adhan-nano-base` → `adhan-nano-instruct` → `…-int4`).
- **Reproducibility:** every run is fully re-derivable from its logged params +
  data version + code SHA. No un-tracked runs.

**Training loop skeleton** (`train_jax.py`):
```
init params → for step in schedule:
    batch = next(packed_stream)
    loss, grads = value_and_grad(loss_fn)(params, batch)   # jit
    params, opt_state = optax.update(...)
    mlflow.log_metric("train_loss", loss, step)
    if step % eval_every == 0: eval + mlflow.log_metric(val ppl); orbax.save()
```

---

## 5. Evaluation (Tamil-first, not just perplexity)

### Automatic metrics
- **Per-akshara & per-word perplexity** on held-out corpus (comparable across vocabs).
- **Classical baseline** (`eval/ngram_baseline.py`): add-one smoothed unigram-over-akshara
  perplexity via open-tamil's `ngram.LetterModels` — the floor `adhan-nano` must clear.
- **Morphology probe** (`eval/morphology.py`): agreement between Layer B (BPE) merge
  boundaries and open-tamil's `TamilStemmer` suffix splits on inflected words.
- **Sandhi accuracy** (`eval/morphology.py`): word-level correctness rate of model
  generations under open-tamil's `tamilsandhi` ~40-rule புணர்ச்சி checker.
- **Agglutination stress set:** long derived word-forms; measure token fertility + ppl.
- **Code-switch robustness:** Tamil–English mixed sentences.

### Human rubric (kid-level fluency bar)
Quality target: **age 5–7 Tamil speaker level** — the model should:
- Read & understand simple Tamil text (basic vocabulary, 1–3 sentence paragraphs)
- Answer basic questions (who, what, where, what color, favorite food)
- Generate 2–3 coherent sentences on familiar topics (family, school, animals)
- Use correct grammatical case endings and tense agreement
- Handle familiar code-switch (Tamil + English numerals/tech words)

Evaluated by small Tamil-fluent panel (min. 5 raters) on a 20-prompt holdout.

See `docs/EVAL_TAMIL.md` (Phase 4) and MLflow evaluation runs for the live table.

---

## 6. Serving & compression (ship light)
Reuse the existing optimization path (`scripts/export_onnx.py`,
`scripts/quantize_model.py`): export → INT8 → INT4 weight-only → benchmark on
Raspberry Pi 5 / Android. Targets for the first launch: **INT4 nano ≤ 30 MB,
> 10 tok/s on-device, fully offline.** Also emit GGUF/TFLite for broader runtimes.

---

## 7. Integration with yazhi-api (deployment)

**yazhi-api** (private: `yazhi-lem/yazhi-api`) is the internal platform for deploying
language models. Adhan ships through it:

- **Model registry:** checkpoint uploaded and versioned
- **REST API:** `/models/adhan-nano/infer` endpoint (streaming or batch)
- **Python SDK:** `from yazhi_api import AdhanClient; client = AdhanClient("adhan-nano-instruct-int4")`
- **Usage tracking:** request/response logging for feedback loops
- **Version control:** A/B testing across model variants (nano, tiny, instruct, int4)

This keeps production separate from research; the repo stays a library.

## 8. Swaram as Dravidian prototype

Swaram is not Tamil-specific — it's a **Dravidian-language tokenizer** that respects
akshara boundaries across the script family:
- **Tamil** (தமிழ்) — Phase 0/1 reference implementation
- **Kannada** (ಕನ್ನಡ), **Telugu** (తెలుగు), **Malayalam** (മലയാളം) — Phase 7+ road
- Shared morphological rules (agglutination patterns) where they overlap
- Language-specific sandhi / case rules per Dravidian grammar

Parallel: **Aksharam** (Indic scripts, Hindi-first) solves the same for matra-based scripts.

## 9. How this coexists with the current repo
- The PyTorch corpus pipeline (`src/data_scraper/*`, scrapers, HF export) is **reused
  as-is** to build and clean the Tamil corpus.
- The from-scratch model, tokenizer, and JAX training live under `src/adhan_slm/`
  and `requirements-jax.txt`, isolated from the existing fine-tuning code.
- Gemma/XLM-R/distilgpt2 fine-tunes remain as **baselines** to beat, not the product.
- Phase 6+: checkpoint moves to yazhi-api for production serving; adhan repo stays research.

## 10. open-tamil as the base Tamil-NLP layer

Rather than reimplementing every mature-but-tedious piece of Tamil NLP, Adhan builds
on **open-tamil** (Ezhil Language Foundation, MIT-licensed, `pip install open-tamil`)
wherever it already solves the problem well. The integration point is
`src/adhan_slm/external/open_tamil_bridge.py`; it is imported by `eval/` and by
tests, **never by the tokenizer's encode/decode hot path** — the swaram/aksharam
cores stay pure-python and dependency-free (§2 design goal).

| open-tamil piece | Adhan use | Where |
|---|---|---|
| `tamil.utf8.get_letters` | Reference akshara segmenter — differential test against Layer A on every commit | `tokenizer/test_open_tamil_crosscheck.py` |
| `tamilstemmer.TamilStemmer` | Suffix-stripping stemmer — morpheme-boundary signal for scoring Layer B (BPE) merges | `eval/morphology.py::stemmer_boundary_agreement` |
| `tamilsandhi.check_sandhi` | Rule-based ~40-rule சந்தி grammar checker — scores model generations, satisfies the Phase 1 "morphological analyzer" requirement | `eval/morphology.py::sandhi_correctness_rate` |
| `ngram.LetterModels.Unigram` | Classical per-akshara frequency baseline — the floor `adhan-nano` perplexity must clear | `eval/ngram_baseline.py::AksharaUnigramBaseline` |
| `solthiruthi` categorized lexicons (animals, objects, verbs, pronouns, ...) | Seed vocabulary for the kid-level (5–7 y/o register) 50-prompt eval set | `eval/kid_level_prompts.py` |
| `tamil.txt2unicode` (25 legacy encodings) | Normalizes scraped text (TSCII, TAB, Bamini, dinamani, ...) to Unicode before corpus ingestion | `external/open_tamil_bridge.py::normalize_encoding` |
| `transliterate.azhagi` | Recovers Tamil meaning from romanized/Tanglish social text before it's discarded as non-Tamil during filtering | `external/open_tamil_bridge.py::transliterate_tanglish` |
| `tamil.numeral` | Number → Tamil-word rendering for synthetic kid-level counting/date data | `external/open_tamil_bridge.py::number_to_tamil` |

**License note:** open-tamil itself is MIT. Its bundled `tamilsandhi` submodule is
GPLv3 (Nithya Duraisamy's sandhi checker) — Adhan only *calls* it as a separately
installed package via `pip install open-tamil`, never vendors its source, so this
repo's licensing is unaffected.
