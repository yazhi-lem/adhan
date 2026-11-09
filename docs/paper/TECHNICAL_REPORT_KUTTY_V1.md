# ஆதன் / Adhan Kutty v1 — a Tamil post-training and evaluation playbook on a multilingual base

**Status:** draft skeleton, 2026-11-05. Every section has its evidence; numbers that do not exist yet are
marked `⏳` and named in [Appendix A](#appendix-a--fill-in-ledger) with the task and file that will produce them.
Nothing in this document may be published with a `⏳` still in it.
**Authors:** Valavan (Yazhi) · **Model:** `yazhi/adhan-kutty-v1` (working name)
**Companion docs:** [`factory/CURRENT.md`](../../factory/CURRENT.md) · [`docs/reports/base_decision_2026-11-03.md`](../reports/base_decision_2026-11-03.md)
· [`docs/reports/agaram_reading_2026-11-05.md`](../reports/agaram_reading_2026-11-05.md)
· [`docs/behaviour/CAPABILITIES.md`](../behaviour/CAPABILITIES.md)

---

## Abstract

> **Draft.** The model this report describes is not trained yet. Today's draft fixes the protocol, the
> claims, the baselines and the report's shape, so that the K2/K3/K4 runs fill slots instead of
> inventing a story afterwards. `⏳` marks a number that does not exist; Appendix A names the task and
> file that will produce it. Nothing here is publishable with an open `⏳`.

**ஆதன் / Adhan Kutty v1** is specified as a 1.7B-parameter Tamil assistant built by post-training
`Qwen/Qwen3-1.7B-Base` (Apache-2.0) on a small, fully documented Tamil corpus: 9,961 Sangam
instruction examples over 2,140 poems and 1,271 grounded-QA examples, with poem-level splits that
share no poem between train, validation and test. The model is built for a narrow claim: **answer
questions about a given Sangam verse, identify its tiṇai, and cite work and poem number — or say the
answer is not in the source.** The release target is an INT4 file that runs on a ₹10–15k Android
phone; `⏳ K4` for its size, tokens/s and RSS.

We report what the base cannot do, measured on our own harness rather than inherited from a
scoreboard: Qwen3-1.7B-Base scores **42.0%** on IndicXNLI-Tamil (n=50, chance 33.3%) and
**EM 0.140 / F1 0.192** on IndicQA-Tamil before any Tamil post-training, so the model's value has to
come from the task-specific SFT and the retrieval/refusal behaviour, not from Tamil world knowledge.
We also report the tokenizer audit that decided the base: on one frozen 300-document sample,
Qwen3's tokenizer spends **2.027 tokens per Tamil akshara** against **0.418** for the best
licence-clean Tamil tokenizer we measured, a **4.85× tax** paid on every token of training, every
generated answer and every token of context on a phone.

Three negative results are part of the contribution, because they are the reason this report
describes a post-training recipe and not a pretraining one: our from-scratch 14.5M model reached
1.820 bits/akshara with incoherent Tamil generation; our akshara-native tokenizer (Swaram) lost to a
plain Tamil-trained BPE on compression (0.527 vs 0.346 tokens/akshara); and the best model for
Tamil tokenization that we measured is not usable as a product base because the terms attached to
its training data were never granted ([§2.3](#23-the-licence-test-we-now-apply-to-every-candidate)).

**Cost:** `⏳ K4` — compute, credits and own funds, counted as cost, including rent that was avoided.

---

## 1 Introduction

Tamil has tens of millions of speakers and a written record older than most languages that have
large open models. Open models still serve it badly in three specific, measurable ways: they cut Tamil
words at vowel signs, they answer in English when asked in Tamil, and they state classical verse from
memory with confident wrong citations.

### 1.1 Script, registers and the citation problem

**Script.** Tamil is written in its own script in which a consonant and a following vowel form one
visible unit, while Unicode stores the vowel as a separate combining mark. A tokenizer that splits at
combining marks cuts a word into many pieces. On our sample, **89.5%** of Tamil aksharas in
Qwen3's tokenization are touched by more than one token, against **8.5%** for a Tamil-trained
byte-level BPE ([§3.4](#34-tokenizer-adopt-the-base-tokenizer-and-audit-the-tax)).

**Registers.** We treat three registers as different products: classical verse (Sangam anthologies,
Thirukkural), formal modern prose, and colloquial/Tanglish. Kutty v1 makes a claim only about the
first, plus English↔Tamil translation of user text. Colloquial chat is explicitly out of scope for v1
([§11](#11-limitations-and-responsible-use)).

**Citation.** A Tamil literary answer is only useful if a reader can check it. The failure mode we
care about is not a generic hallucination but a *plausible citation*: the right tiṇai attached to the
wrong poem, or a verse number that does not exist. This is why our primary suite scores
citation exactness and hallucinated citations, not fluency ([§6.2](#62-our-own-behavioural-sets)).

### 1.2 Model overview and contributions

Kutty v1 is a LoRA/SFT adaptation of `Qwen/Qwen3-1.7B-Base`: 28 layers, hidden 2,048, 16 query and
8 KV heads (head dim 128), MLP 6,144, tied embeddings, vocabulary 151,936, context 32,768,
RoPE θ = 1,000,000 — from its `config.json`. The base is not retrained; the tokenizer is the base's
tokenizer; the contribution is the data, the post-training recipe, the harness, and the reporting.

1. **A Sangam instruction set with verifiable splits.** 9,961 examples across 2,140 poems,
   six task types (tiṇai identification, tiṇai from commentary, karupporul, urai, citation, English
   eurai), with **zero poem overlap** between train/val/test enforced by the builder (it exits
   non-zero on any overlap) — [`data/final/sft_v0_sangam/datasheet.json`](../../data/final/sft_v0_sangam/datasheet.json).
2. **A grounded-QA set with refusals.** 1,271 IndicQA-derived examples, 15% of them unanswerable
   from the supplied context, so "தகவலில்லை" is a scored behaviour and not a hope — [`data/final/sft_qa/stats.json`](../../data/final/sft_qa/stats.json).
3. **An evaluation harness for behaviour rather than multiple choice.** 1,330 Thirukkural couplets,
   morphology and sandhi probes per Tholkāppiyam, 215 held-out Sangam poems used as a 5-way tiṇai
   MCQ, 50 kid-level prompts, and a critical-only safety harness (Aram) — [`src/adhan_slm/eval/`](../../src/adhan_slm/eval/).
4. **A tokenizer audit with a licence filter.** Six tokenizers, one frozen sample, one rule a stranger
   can rerun, including the cheapest licence-clean option measured — [`docs/reports/bases_tokenizer_bench.md`](../reports/bases_tokenizer_bench.md).
5. **Negative results, stated as such.** From-scratch nano, Swaram, and the licence check that
   removed an otherwise attractive base ([§2](#2-project-history-and-the-earlier-model)).
6. **Cost accounting a single developer can follow**, including the GPU hours actually needed
   (`⏳ K4`).

### 1.3 Related Tamil and multilingual models

| Model | What it is | Why it is not our base |
|---|---|---|
| `Qwen/Qwen3-1.7B-Base` (Apache-2.0) | 1.7B multilingual base, 32k context, 151,936-token vocab | **Chosen.** Licence-clean, small enough for INT4 on a phone; Tamil tokenizer is the most expensive we measured (2.027 tok/akshara) |
| `sarvamai/sarvam-m` 24B (Apache-2.0) | large Indic model, 0.795 tok/akshara | Teacher/reference only: 24B is not a phone model |
| `sarvamai/sarvam-1` 2B | best Tamil tokenizer we measured among production models (0.584) | **Rejected**: Sarvam AI Research Licence, non-commercial/research only |
| `murai-labs/agaram-2.5b-instruct` (Apache-2.0) | 2.53B trained from scratch, 58.2% Tamil, 50B tokens, 64k byte-BPE | Best Tamil tokenizer measured (**0.418**), but weaker than Qwen3-1.7B on our smoke (XNLI 12.0% vs 42.0%) and **upstream data terms were never granted** (its own NOTICE) |
| `google/gemma-*` | strong multilingual open weights | Gemma terms are not a plain permissive licence; held as a fallback, not a base |
| MuRIL-class encoders | Tamil encoder | Encoder only: retrieval/classification, never generation |

The closest published work is the Agaram 2.5B technical report (arXiv:2610.18420), which is a
from-scratch playbook for one language. Our report is the complementary case: **what a small team
should do when it cannot afford pretraining** — buy the base, own the data, the harness and the claim.

---

## 2 Project history and the earlier model

### 2.1 Motivation and the name

ஆதன் (Adhan) comes from ஆதி, "beginning": the Tamil-Brahmi inscriptions from Adichanallur
(≈905–696 BCE) and Keezhadi. The name is a commitment about the direction of the work — Tamil first,
Tamil in its own script — not a claim about scale. Where a from-scratch report would put its model
lineage here, we put the base decision.

### 2.2 Adhan-nano and Swaram: what the from-scratch attempt taught us

Nine months of work (first commit 2026-02-17) produced a 14.5M-parameter model and a tokenizer. Both were measured, and
the measurements are the reason the product took a different path.

| Finding | Value | Source |
|---|---|---|
| Best from-scratch checkpoint | `adhan-nano`, 14.5M params, val **1.820 bits/akshara** | `docs/reports/eval_report_nano_stage1b.json` |
| Generation quality | short Tamil not coherent; no SFT run | `factory/CURRENT.md` §1 |
| Swaram v2 vs Tamil-trained metaspace BPE | **0.527 vs 0.346 tok/akshara** — Swaram loses 1.52× | `docs/reports/ab_swaram_vs_bpe.md` |
| Swaram morphology gate | boundary agreement **5.6%**, gate ≥30% | `docs/tokenizer/TOKENIZER_BENCHMARK_V2.md` |
| Swaram vs production tokenizers | 0.582 vs Sarvam-1 0.584, Qwen3 2.027, **Agaram 0.418** | `docs/reports/bases_tokenizer_bench.md` |
| Export path | ONNX/quantise still targeted a removed HF pipeline | `factory/CURRENT.md` §4 |

Two conclusions. First, **tokenizer compression is not the lever we thought**: an akshara-native
scheme did not beat a plain BPE trained on Tamil, and a 64k byte-level BPE trained on a Tamil-heavy
diet beats both Sarvam-1 and Swaram. Second, **a 14.5M model cannot carry a product**, whatever its
bits/akshara. Both conclusions were recorded before any GPU was rented
([`docs/reports/base_decision_2026-11-03.md`](../reports/base_decision_2026-11-03.md)).

### 2.3 The licence test we now apply to every candidate

The base decision removed Sarvam-1 for its research-only licence. Applying the same test to Agaram
2.5B, whose Apache-2.0 weights and Tamil tokenizer are attractive: its NOTICE discloses pretraining
material whose own terms differ — a Tamil Alpaca translation of GPL-3.0/CC BY-NC-4.0 Stanford data,
GPT-4/Gemini-generated Tamil Orca, and OCR text from an Indic Open Model Licence tool and Sarvam's
OCR API, for which the authors record that **written permission was requested and not granted**.

Our rule for any base, now written into the checklist:

1. Read the `LICENSE` **and** the `NOTICE` before benchmarking, not after.
2. Apache-2.0 on weights is not a statement about training-data terms.
3. A base with unresolved upstream terms may be used for internal comparison, never shipped.

---

## 3 Data

### 3.1 Sources and classification

| Source | Role | Volume | Terms |
|---|---|---|---|
| Open-Sangam corpus + traditional commentaries (`../open-sangam`) | Sangam SFT, bench | 2,669 poems | upstream Open-Sangam licence; recorded in the datasheet |
| `ai4bharat/IndicQA` (Tamil) | grounded QA + refusals | 1,271 examples | as published by AI4Bharat |
| `ai4bharat/IndicCorpV2`, FineWeb-2, Sangraha, OSCAR, Wikipedia (cached) | tokenizer audit, register probes | 1.34M docs / 1.06B chars locally | per-source, see `eval_manifests/` |
| `vishnu-n/Tanglish-Corpus-185k` @ `927f5b4` | neutral Tanglish eval | 171,449 neutral candidates after filter | CC-BY-4.0 |

No OCR-derived text is used anywhere in this project, which removes a whole class of exposure that
the Agaram report has to disclose.

### 3.2 Extraction hygiene

`⏳ K6` — purity, garble and language-ID rates for the Tanglish freeze, and the neutral-set
construction, are produced by `scripts/corpus_health.py` + the gazetteer filter. Until then this
section reports only that 5.5% of the source is keyword-political and that 171,449 sentences survive
the keyword pass (`docs/reports/n3_tanglish_source_scope.md`).

### 3.3 Splits and supply

The Sangam split is **poem-level** and deterministic (stable poem-id hash, 10%/10%, seed 42). It is
not a random example split: all six task types for a poem stay on the same side, so a memorised verse
cannot leak across a split. The builder fails on any overlap; the datasheet records the check. 314
poems have unknown tiṇai and are excluded from tiṇai-conditioned evaluation; 32 poems have a gold
English eurai, which is too few to support a translation-quality claim and is reported as such.

### 3.4 Tokenizer: adopt the base tokenizer, and audit the tax

We ship the base's tokenizer and budget around it. The audit that produced that decision
([`scripts/benchmark_bases_tokenizers.py`](../../scripts/benchmark_bases_tokenizers.py), sample:
300 documents, `sha256(seed|id|text)` ranking, seed `adhan-bases-2026-11-05`, 59,699 chars /
38,991 aksharas, harness [`src/adhan_slm/eval/tokenizer_bench.py`](../../src/adhan_slm/eval/tokenizer_bench.py)):

| Tokenizer | tok/word | **tok/akshara** | Tamil clusters split | tok/1k sent. |
|---|---:|---:|---:|---:|
| Agaram 2.5B (64k byte BPE) | 1.88 | **0.418** | 8.5% | 44,673 |
| **Qwen3-1.7B (shipped)** | 9.13 | **2.027** | 89.5% | 216,777 |
| Sarvam-1 | 2.63 | 0.584 | 7.7% | 62,423 |
| Sarvam-M | 3.58 | 0.795 | 31.6% | 85,047 |
| Swaram v3 | 2.62 | 0.582 | 0.0% | 62,217 |
| Swaram v2 | 2.67 | 0.593 | 0.3% | 63,377 |

Consequences we accept in writing: a 100-akshara answer costs **203** tokens instead of 42; the
32k context holds ~4.85× less Tamil than a Tamil-native vocabulary would; and generated tokens are
paid for at that rate on a phone. `⏳ K4` measures what this costs in tokens/s and RSS, which is
the only number that can overturn the choice
([§8](#8-quantisation-and-the-phone-path)).

### 3.5 SFT mixtures

| Split | Sangam | Grounded QA | Total |
|---|---:|---:|---:|
| train | 9,961 | 971 | **10,932** |
| val | 1,303 | 124 | 1,427 |
| test | 1,146 | 182 | 1,328 |

By task: `sangam_urai` 2,652 · `sangam_citation` 2,669 · `sangam_karupporul` 2,355 ·
`sangam_tinai` 2,355 · `sangam_tinai_from_urai` 2,347 · `sangam_eurai_en` 32 · `qa_grounded` 1,277.
Supervised tokens per epoch, and the final mixture weights: `⏳ K2`.

---

## 4 Post-training

### 4.1 Base and adaptation

LoRA adapter on all linear projections, `⏳ K2` for rank/alpha/dropout; bf16; assistant-turn loss only
(the prompt is masked); 2–3 epochs; cosine schedule. The base's chat template is used verbatim, and
the data is stored as role/content `messages` so the same file trains any ChatML base.

### 4.2 Schedule and selection rule (pre-registered)

| Step | Rule set before the run |
|---|---|
| Checkpoints | every `⏳` steps; selection on validation split only |
| Primary selection metric | Sangam tiṇai accuracy on `val` (5-way, chance 20%) |
| Tie-break | grounded-QA EM on `val`, then shorter checkpoint |
| Regression gate | no-SFT base must not beat the SFT checkpoint on the same prompts |
| Safety gate | Aram critical-only: any CRITICAL disqualifies the checkpoint |
| Reporting | every score published next to its no-SFT baseline |

Pre-registering this is the one procedural lesson from the Agaram report we adopt without argument:
they disclosed that their joint selection rule was **not met** and that one identity item was waived
mid-selection. A stated rule that survives contact with results is worth more than a good score.

### 4.3 Diagnostic experiments

`⏳ K2/K3` — each row states what it was meant to falsify:

| Experiment | Question | Result |
|---|---|---|
| Sangam-only vs Sangam+QA | does grounded QA interfere with verse tasks? | `⏳` |
| 1 vs 3 epochs | overfitting on 2,140 poems? | `⏳` |
| teacher-distilled additions (Sarvam-M) | does distillation beat human-curated rows on our bars? | `⏳ K3` |
| Agaram-2.5B twin (same data) | does a Tamil-first tokenizer+pretraining beat a bigger multilingual base at equal SFT data? | `⏳ K2b` |

---

## 5 Evaluation protocol and overlap

### 5.1 Public tasks and metrics

| Task | n | Metric | Why |
|---|---:|---|---|
| IndicXNLI-Tamil (`AdaMLLab/indicxnli_repaired`, test) | 50 smoke / 200 final | accuracy, 3-way | comparable to every other Tamil model |
| IndicQA-Tamil | 50 / 100 | EM, F1 (SQuAD normalisation) | extractive Tamil reading |
| MILU-Tamil | gated | accuracy | the only Tamil knowledge MCQ with a standard |
| FLORES en→ta / ta→en | `⏳` | chrF | translation is a user-visible skill |
| MultiBLiMP-Tamil | `⏳` | accuracy | grammar, where small models can win |

### 5.2 Our own behavioural sets

| Suite | Bar (ship gate) |
|---|---|
| `sangam`, 215 held-out poems, 5-way tiṇai | tiṇai ≥70%, citation ≥95%, hallucinated citations <2% |
| `qa_grounded`, 500 + 100 unanswerable | EM above no-SFT baseline; refusal ≥80% on unanswerable |
| `chat`, 100 scripted multi-turn dialogues | context probe ≥80%, Tamil purity ≥95% |
| `kid` prompts, 50 | `⏳` bar to be set from the base measurement, not from taste |
| Aram safety, critical-only | score >0.90, **0 CRITICAL** |

Every one of these is deterministic and scored by script. There is no LLM judge anywhere in the
primary path, because a judge that moves 11 points between passes (as the Agaram report honestly
documents for theirs) cannot decide a release.

### 5.3 Overlap control

| Check | What it covers | Status |
|---|---|---|
| Poem-level split | no poem in two splits, all six tasks | enforced by builder, zero overlap recorded |
| Context-level split for QA | IndicQA `context_id` grouping | enforced in `stats.json` |
| 5-gram scan of eval prompts vs SFT rows | verse/prompt leakage into training | `⏳ K2` |
| Benchmark keep-out list | IndicXNLI/IndicQA/FLORES/MILU items excluded from any future SFT v1 | `⏳ K2` |
| Evaluated-exclusive counts | report each score also on the subset with zero overlap | `⏳ K2` |

The lesson we are copying here is the one most likely to sink a small report: **an eval set that
shares text with training is not an eval set**. We state the coverage of our scans and name what is
not covered.

---

## 6 Results

Headline for the base, before post-training (measured, §1 of the reading note):

| Task | Qwen3-1.7B-Base | Agaram-2.5B-instruct | Chance |
|---|---|---|---|
| IndicXNLI-Tamil, n=50 | **42.0%** | 12.0% | 33.3% |
| IndicQA-Tamil, n=50 | **EM 0.140 · F1 0.192** | EM 0.000 · F1 0.143 | 0 |
| MuRIL frozen probe, full XNLI test (reference) | 42.77% | — | 33.3% |

Agaram's 12.0% is a format-compliance failure on a letter-only prompt (18/50 answers had no
parseable letter, 27/50 said "A"), not a measure of Tamil reasoning; we report it with that caveat or
not at all.

`⏳ K2` — the tables that decide the release:

1. **Base by register** — held-out bits/akshara on classical, modern, Tanglish.
2. **Language understanding and translation** — the §5.1 table, base vs SFT vs Agaram-SFT twin.
3. **Sangam behaviour** — tiṇai / citation / hallucinated citation, with a per-task breakdown.
4. **Grounded QA** — EM/F1 and refusal rate on unanswerable items.
5. **Safety** — Aram, critical-only, next to the 3 Nov baseline (0.13, 7 CRITICAL of 9 on
   `adhan-nano`).
6. **Serving** — p50/p95 next to the 3 Nov Adhan CPU figures (1,757 / 2,540 ms).

---

## 7 What we will not claim

Written before the results, so that no result can be over-read:

- Not *from scratch*. Not *the first* anything.
- Not a knowledge model. Base MCQ at chance is a starting condition, not a defect we fixed.
- Not a general Tamil chatbot. Conversation is a v2 question ([RUNWAY cut list](../../factory/RUNWAY.md) §5).
- Not a translation system until FLORES is measured, whatever the anecdotal quality.
- No claim about a tokenizer advantage we do not ship.

---

## 8 Quantisation and the phone path

Target: INT4 GGUF via llama.cpp on a ₹10–15k Android reference device; 8-bit is the fallback and
may end up the shipped file.

The Agaram report's quantisation disclosure is the cautionary case we design the validation around:
of 8 greedy prompts, their 8-bit file matched the full-precision model on 5 and their 4-bit file on
**none**, and the 4-bit file changed a correct fact. So our release validation is specified now:
8 fixed prompts, greedy, exact-match against the bf16 checkpoint, plus tokens/s and peak RSS.

`⏳ K4` — file size, tokens/s, RSS, exact-match count, and the phone demo recording.

---

## 9 Compute and cost

| Item | Quantity | Source |
|---|---|---|
| Base selection work | CPU only, no GPU rented | `factory/CURRENT.md` |
| SFT v0 (LoRA, 1.7B, 10.9k examples) | `⏳ K2` GPU-hours · `⏳` USD | — |
| Distillation (Sarvam-M teacher, ≥5k prompts) | `⏳ K3` | — |
| Agaram twin run | `⏳ K2b` | — |
| Eval + quantisation | CPU + one GPU-hour | — |
| Own funds vs credits | `⏳` | — |

We count credits as cost. Where a run was avoided by cutting scope, the report says so, because a
playbook is only useful if the reader can see what was *not* bought.

---

## 10 Release, provenance and terms

### 10.1 Artifacts

| Artifact | Contents | Status |
|---|---|---|
| `adhan-kutty-v1` safetensors + tokenizer | SFT checkpoint (`⏳` step) | `⏳ K2` |
| `adhan-kutty-v1` GGUF INT4 / INT8 | quantised, validated per §8 | `⏳ K4` |
| SFT datasheets | counts, splits, overlap checks, terms | present for v0; extended in `⏳ K2` |
| Eval manifests | per-suite sources, licences, hashes | partial |
| This report | protocol, numbers, deviations | this file |

### 10.2 Generated-data provenance rule

1. The teacher for any distilled row is `sarvamai/sarvam-m` (Apache-2.0), named in the datasheet.
2. Frontier-model assistants do not write assistant turns. If any are used, the rows are enumerated
   in a table with counts, as the Agaram report does.
3. Every teacher batch records the prompt set, the model id, the date, and the filter that admitted it.
4. Eval material is never used to write training rows; if a training row was written in response to
   a failure on a specific eval item, that item is reported as non-independent.

### 10.3 Terms

- Kutty's own weights: `⏳` (decision: Apache-2.0 for the adapter/delta, with the base's licence and
  attribution carried in `NOTICE`).
- Base: `Qwen/Qwen3-1.7B-Base`, Apache-2.0, unmodified except by the adapter.
- Teacher outputs: Sarvam-M is Apache-2.0; we record its terms and our output-use position in the
  datasheet, and we verify a human sample of every teacher batch.
- No OCR-derived data, no scraped commentary outside Open-Sangam's terms, no frontier-model
  assistant text without disclosure.

### 10.4 What the release supports

Running, fine-tuning, quantising and evaluating Kutty. **Not** reproducing the pretraining (there is
none), **not** re-deriving the base, and **not** auditing our teacher generations, which are
summarised rather than redistributed where their terms do not allow it.

---

## 11 Limitations and responsible use

Kutty is a research model. Do not deploy it where a wrong answer hurts without a person checking it.

1. **Knowledge.** The base is at chance on Tamil MCQ before post-training. Kutty v1 does not fix
   this; it narrows the claim to answering *about a supplied verse or context*. Questions about
   current office holders, prices, or events are out of scope (`⏳` retest list from K2).
2. **Citations.** The product claim is a citation bar of ≥95% with <2% hallucinated. If K2 cannot
   meet it, the claim is cut, not the bar.
3. **Refusals.** Safety rows in SFT data are not evidence of safe behaviour — the Agaram report
   documents refusal rows being answered anyway. Aram scores the *model*, never the dataset.
4. **Tanglish.** Out of scope for v1; the neutral set exists so that v2 can be measured, not claimed.
5. **Judge and human evaluation.** v1 uses no LLM judge and no human preference study. That is a
   limitation, and the reason the primary bars are exact-match behaviours.
6. **Single runs.** Most comparisons are one seed. Where a decision hinges on a <2-point difference,
   we say the comparison is inconclusive rather than picking the winner.
7. **Tokenizer.** 2.027 tok/akshara, 89.5% cluster split, unmeasured effect on quality beyond its
   cost in tokens and latency.
8. **Quantisation.** INT4 is validated on 8 prompts, not a distribution.

---

## 12 Conclusion and what we learned

`⏳` — written after K2/K4. The shape is fixed: what shipped, what it scores, what it cannot do, what
it cost, and the list of findings below, each of which must cite a measurement in this report.

Findings to state (each already has its evidence in earlier sections):

1. **A small team should buy the base and own the data and the harness.** Our from-scratch model
   produced a valid negative result at 14.5M parameters and nothing shippable.
2. **Tokenizer compression is not the lever; token *cost* is.** Qwen3 spends 4.85× the tokens of the
   best Tamil tokenizer we measured; that is a phone-latency and context problem, and it is the
   strongest argument for a v2 base.
3. **Licence debt is discovered in NOTICE files, not model cards.** Two otherwise attractive bases
   were removed by a five-minute read.
4. **Pre-register the selection rule.** The rule is the contribution; the checkpoint is the receipt.
5. **Score behaviour, not vibes.** tiṇai, citation exactness and refusal-on-unanswerable are
   checkable by script; they are also exactly what a Tamil literary user cares about.

---

## Appendix A — fill-in ledger

Every `⏳` in this document, with the task that produces it and the file it must land in. The report
does not ship while this table has an open row.

| # | Placeholder | Produced by | Lands in |
|---|---|---|---|
| 1 | Abstract cost figure, §9 cost table | K4 | `docs/reports/kutty_v1_compute.json` |
| 2 | FLORES / MultiBLiMP rows, §5.1 | K2 | `docs/reports/base_bakeoff_qwen3.json` (extended) |
| 3 | Supervised tokens per epoch, §3.5 | K2 | `data/final/sft_v1_kutty/stats.json` |
| 4 | LoRA hyperparameters, §4.1 | K2 | `docs/reports/sft_v0_qwen3.json` |
| 5 | Checkpoint interval, §4.2 | K2 | same |
| 6 | Diagnostic rows, §4.3 | K2, K2b, K3 | `docs/reports/sft_diagnostics.md` |
| 7 | 5-gram overlap scan + exclusive counts, §5.3 | K2 | `docs/reports/overlap_scan_kutty_v1.json` |
| 8 | All six results tables, §6 | K2, K7 | `docs/reports/eval_kutty_v1.json` |
| 9 | INT4/INT8 size, tokens/s, RSS, exact-match, §8 | K4 | `docs/reports/qwen3_int4_phone.json` |
| 10 | License/terms decision, §10.3 | Founder | `NOTICE`, `LICENSE`, model card |
| 11 | Retest list of out-of-scope questions, §11.1 | K2 | `docs/reports/kutty_v1_failure_read.md` |
| 12 | Conclusion, §12 | K4 | this file |

## Appendix B — reproduction

```bash
# data
python scripts/build_sangam_sft.py --out-dir data/final/sft_v0_sangam --open-sangam-root ../open-sangam
python scripts/build_grounded_qa.py

# tokenizer audit (this report's §3.4)
AGARAM_TOKENIZER=/path/to/agaram/tokenizer.json \
  .venv/bin/python scripts/benchmark_bases_tokenizers.py \
  --out-json docs/reports/bases_tokenizer_bench.json --out-md docs/reports/bases_tokenizer_bench.md

# base smoke (this report's §6 headline)
cd bench/tamil-bench
TAMIL_BENCH_ENDPOINT=http://127.0.0.1:11434/v1/chat/completions \
  python3 bench.py xnli --model qwen3:1.7b --n 50
TAMIL_BENCH_ENDPOINT=http://127.0.0.1:11434/v1/chat/completions \
  python3 bench.py indicqa --model qwen3:1.7b --n 50

# SFT + eval + quantise: ⏳ K2/K4 commands land here when the trainer exists
```
