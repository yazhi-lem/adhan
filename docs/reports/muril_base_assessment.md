# MuRIL assessment — political filtering, encoder baseline, and the Kutty base decision

**Date:** 2026-11-03
**Model:** [`google/muril-base-cased`](https://huggingface.co/google/muril-base-cased) @ `afd9f36c7923d54e97903922ff1b260d091d202f`
**Probe:** `scripts/probe_muril.py` · **Raw result:** [`muril_probe.json`](muril_probe.json)
**Related:** `docs/adr/0001-kutty-v1.md` · `docs/reports/n3_tanglish_source_scope.md`

---

## 1. What MuRIL actually is

| Property | Value |
|---|---|
| Architecture | BERT-base **encoder** (`BertForMaskedLM`), 12 layers, 768 hidden, 12 heads |
| Parameters | ~237.6M (197,285-token vocab dominates the embedding matrix) |
| Context | 512 tokens |
| Languages | 17 Indian languages + transliterated counterparts, including Tamil |
| Pretraining | Wikipedia, Common Crawl, PMINDIA, Dakshina; translation + transliteration pairs |
| Licence | **Apache-2.0** |
| Output | contextual embeddings / MLM logits — **no text-generation head** |

The important fact for Adhan: **MuRIL is not a generative language model.** It cannot
answer a prompt, write Tamil, or serve as the Kutty chat/Sangam/QA base. It is an
encoder: classification, tagging, retrieval, sentence representation.

## 2. Can MuRIL replace the Kutty base?

**No.** It fails the product requirement at the architecture level, not on quality:

- no autoregressive decoder, so no `/generate`, chat, SFT targets, or Sangam answers;
- 512-token context, while the Kutty roadmap assumes at least 1–2k for dialogue;
- 237M encoder parameters with no vocabulary/tokenizer alignment to Swaram;
- Apache-2.0 is fine, but the licence does not create a decoder.

**Decision impact:** none on `D1`. The recommendation in `docs/adr/0001-kutty-v1.md`
still holds — Kutty v1 is a fine-tune of an open **decoder** base (option c), keeping
`adhan-nano` + Swaram as the research artifact. MuRIL is an **auxiliary encoder**, not a
candidate product base.

Where MuRIL *can* help:

1. **Political/sensitive-content classifier** for the Tanglish eval source (with fine-tuning).
2. **Tamil/Tanglish retrieval encoder** for citation/QA retrieval, if a small encoder is needed.
3. **Encoder baseline** on XNLI-style classification, to sanity-check Adhan's Tamil representations.
4. **Teacher/feature extractor** for data quality and clustering — not for generation.

## 3. Probe 1 — Tamil understanding on IndicXNLI

Method: frozen MuRIL, `concat([CLS], mean-pool)` (1536-d), logistic regression on
20,000 Tamil XNLI train rows, evaluated on the full 5,010-row Tamil test split.

| Metric | MuRIL frozen + linear probe |
|---|---|
| Test accuracy | **42.77%** |
| Majority baseline | 33.33% |
| 3-way chance | 33.33% |
| Validation accuracy | 42.50% |

Per-class F1 is roughly balanced (0.417 / 0.446 / 0.416), so the gain is not a
majority-class artefact. But ~43% is far from a fine-tuned MuRIL result and far from
what a generative model can do on the same task.

**Reading:** MuRIL carries real Tamil sentence-pair signal out of the box, but a frozen
linear probe is not competitive with a fine-tuned encoder. If Adhan wants an XNLI
encoder baseline, fine-tune the classification head; do not quote the frozen probe as
"MuRIL on tamil-bench".

### Why this is *not* a tamil-bench run

`bench/tamil-bench/bench.py` is an **OpenAI-compatible generative** runner. MuRIL cannot
be plugged into it: `milu`, `indicqa`, `sangam`, and `chat` all require generation, and
the current `xnli` task asks a chat model to emit A/B/C. A frozen encoder probe on the
same underlying XNLI data is a different experiment. If we want an encoder track, it
belongs in a separate `encoder` table — never mixed into the generative leaderboard.

## 4. Probe 2 — Tanglish political content

Method: filter the `vishnu-n/Tanglish-Corpus-185k` rows to Latin-only sentences of
15–300 characters, then build a **weak label** from a high-recall political/sensitive
regex (parties, elections, offices, known politicians, corruption, religion/caste).
Sample 1,000 keyword-positive + 1,000 keyword-negative sentences, embed with frozen
MuRIL, and run 5-fold cross-validated logistic regression.

| Metric | Value |
|---|---|
| Candidate pool after script/length filters | 181,378 |
| Keyword-positive (weak) | 9,929 (5.5%) |
| Keyword-negative | 171,449 |
| Political sample / neutral sample | 1,000 / 1,000 |
| CV accuracy | **67.75% ± 1.57%** |
| CV AUC | **0.754** |

**Reading:** the representations separate the keyword-defined groups better than chance,
but this is **not** a political-opinion detector. The labels are keyword-derived, so the
probe is partly learning the keywords. Manual inspection of the highest-scoring
keyword-negative rows found mostly cricket, film, education, and personal chatter — not
hidden political argument. The lowest-scoring keyword-positive rows included ordinary
"government laptop" support questions. **A frozen MuRIL probe is not good enough to
ship as a sanitizer.**

Also note: the source itself is 5.5% keyword-positive, so even a crude high-recall
keyword filter leaves ~171k neutral sentences — more than enough for the 500-sentence
benchmark without any model at all.

## 5. Recommended political sanitization pipeline

The goal for N3 is a **neutral 500-sentence tokenizer eval set**, not a political-science
classifier. Optimise for auditability and high precision on what remains.

### Policy first

Write down what "political" means for this benchmark, e.g. exclude sentences whose main
content is:

- parties, candidates, elections, voting, alliances, office-holders;
- government policy, schemes, protests, corruption/scams;
- partisan identity, religion/caste when used politically.

Keep: technology, sport, film, food, education, personal life, language, culture.
Record the policy in the manifest so the filter is reviewable.

### Stage 1 — deterministic gazetteer + regex (primary)

- Party/coalition names and abbreviations (`TVK`, `DMK`, `ADMK`, `BJP`, `NTK`, …).
- Offices and election terms (`CM`, `MLA`, `MP`, `governor`, `election`, `vote`, …).
- Known politician names and constituency names.
- High-recall, low-cost, fully reproducible; drop matches before sampling.
- This alone removes the obvious 5.5% and preserves ~171k candidates.

### Stage 2 — human-labeled MuRIL classifier (only if Stage 1 misses too much)

Do **not** ship the frozen probe. If paraphrase recall matters:

1. Label 500–1,000 sentences stratified across keyword hits, random negatives, and
   sentences the frozen probe is uncertain about.
2. Fine-tune a MuRIL classification head (or a smaller DistilMuRIL/XLM-R student).
3. Tune the threshold for **high precision on the kept set**, not maximum F1 — a false
   positive only costs one candidate; a false negative puts politics in the public set.
4. Freeze model revision + threshold + label policy in the manifest.
5. Audit 100 kept + 100 dropped sentences by a native reviewer.

### Stage 3 — abuse/sensitive-content auxiliary

`Hate-speech-CNERG/tamil-codemixed-abusive-MuRIL` (AFL-3.0) is a fine-tuned MuRIL for
Tamil code-mixed abusive language. It can serve as an optional abuse filter, but it is
not a political classifier, and its licence needs review before use. Do not conflate
"abusive" with "political".

### Stage 4 — output and audit

- Write `data/raw/tanglish/tanglish_neutral.jsonl` (local only; git-ignored).
- Commit only the manifest: source pin, SHA-256, policy version, filter version,
  counts, and the 500 selected sentence hashes.
- Native review of the 500 before freezing.
- Never redistribute the raw vishnu corpus.

## 6. Bottom line

| Question | Answer |
|---|---|
| Can MuRIL be the Kutty base? | **No** — encoder-only, no generation. |
| Does it change D1? | **No** — Kutty v1 remains a fine-tune of an open decoder base. |
| Can it help Adhan? | **Yes**, as an auxiliary encoder for political/abuse filtering and retrieval. |
| Is frozen MuRIL a good political sanitizer? | **No** — 67.75% CV accuracy on weak labels; needs fine-tuning + human labels. |
| Do we need MuRIL to sanitize vishnu for N3? | **Not necessarily** — the gazetteer/regex pass leaves ~171k neutral candidates. Start there. |
| Can MuRIL run tamil-bench? | **Not the generative tasks.** It can be an encoder-only XNLI baseline (42.77% frozen probe) on a separate table. |
