# Next Actions: Adhan (Q4 2026)

> This file mirrors the Yazhi Q4 2026 launch plan as of **8 Oct 2026**. The longer technical
> roadmap is in [ROADMAP_JAX_SLM.md](./ROADMAP_JAX_SLM.md).

| | |
|---|---|
| Project | Adhan, Yazhi's sovereign Tamil language model |
| Federal | **Adhan** |
| Related work | Research 2030 (on-device Kutty demo, 2027–2030 research note, reading group) |

## Q4 launch line

| Date | Public step | Done when |
|---|---|---|
| **31 Oct 2026** | **Tamil tokenizer benchmark**: repo, results table, blog post | One command reproduces every number; 6+ tokenizers compared |
| **26 Dec 2026** | **Adhan Kutty v1**, open weights on HuggingFace | Model card with benchmark, data card and known failure modes; served via yazhi-api; runs offline on one named Android phone |

## Approach for Kutty v1

The Q4 plan builds Kutty v1 by starting from an existing open base model (size proposal ≤1.5B, chosen for on-device use) and
running continued pre-training / LoRA on a frozen, deduplicated Tamil corpus. The from-scratch `adhan-nano` track in
[ROADMAP_JAX_SLM.md](./ROADMAP_JAX_SLM.md) continues as research. Which approach ships on 26 Dec will be settled in the training plan
(Training plan milestone, 11 Oct).

## Plan by milestone

Milestone order: Training plan → Data freeze → Tokenizer benchmark public → Training → Hardening → Kutty v1 release.

| Target date | Milestone | What |
|---|---|---|
| 11 Oct 2026 | Training plan | Plan Adhan training: shortlist base models, size compute, set eval gates |
| 11 Oct 2026 | Training plan | Decide the Adhan model size (≤1.5B) and sign off training plan v1 |
| 18 Oct 2026 | Data freeze | Audit the corpus and record every source |
| 21 Oct 2026 | Data freeze | Run the dedup pass and prove zero train/eval overlap |
| 23 Oct 2026 | Data freeze | Build the Tamil normalisation suite (NFC, grapheme checks) in CI |
| 25 Oct 2026 | Tokenizer benchmark public | Benchmark Swaram vs BPE and SentencePiece on a frozen Tamil corpus |
| 25 Oct 2026 | Data freeze | Write the data card and tag the data freeze |
| 31 Oct 2026 | Tokenizer benchmark public | Publish the Tamil tokenizer benchmark: corpus, harness, repro script, post |
| 31 Oct 2026 | Training | Run baseline evals on the chosen base model |
| 21 Nov 2026 | Training | Train a GPE tokenizer and compare it in the benchmark |
| 28 Nov 2026 | Training | Run LoRA / continued pre-training and pick the best checkpoint |
| 5 Dec 2026 | Training | Post the weekly Friday eval report |
| 5 Dec 2026 | Training | Build the DPO pairs pipeline from Reflection agent output |
| 12 Dec 2026 | Hardening | Run Tamil red-team suite against Kutty |
| 15 Dec 2026 | Hardening | Build 4-bit quantised Kutty builds and measure the quality delta |
| 19 Dec 2026 | Hardening | Write the model card with limits and known failure modes |
| 19 Dec 2026 | Kutty v1 release | Finalise the Yazhi Open Licence text for Kutty |
| 22 Dec 2026 | Kutty v1 release | Serve Kutty via yazhi-api |
| 26 Dec 2026 | Kutty v1 release | Release Adhan Kutty v1 on HuggingFace |
| 26 Dec 2026 | Kutty v1 release | Write and publish the Adhan Kutty launch post |
| TBD | Training | Set up the PyTorch Colab notebook: corpus mount, pinned env |
| TBD | Training | Fix MemmapShardDataset to read packing.py's shard format |
| TBD | Hardening | Validate the inference API against a real trained checkpoint |

### Research 2030

| Target date | Milestone | What |
|---|---|---|
| 31 Oct 2026 | On-device Kutty | Choose the target phone and on-device runtime for Kutty |
| 31 Oct 2026 | Reading group | Run reading group session 1 and post the notes |
| 30 Nov 2026 | Reading group | Run reading group session 2 and post the notes |
| 31 Dec 2026 | On-device Kutty | Run a quantised Kutty on the target phone and record the tokens/s demo |
| 31 Dec 2026 | Research note | Write, review and publish the Tamil edge AI and robotics 2027–2030 note |
| 31 Dec 2026 | Reading group | Run reading group session 3 and shortlist the 2027 bets |

## Immediate next actions

1. Sign off the training plan: base model, size, data mix, compute budget and eval gates (Training plan milestone, 11 Oct).
2. Land the Swaram vs BPE / SentencePiece benchmark ([#50](https://github.com/yazhi-lem/adhan/pull/50)). It sits on the path to the 31 Oct benchmark.
3. Run the corpus audit, then dedup and normalisation, and tag the data freeze by 25 Oct.
4. Land the Colab / PyTorch workspace scaffold ([#53](https://github.com/yazhi-lem/adhan/pull/53)) and the shard-format fix before the Training milestone.

---

## Reference: data availability audit

> Carried over from the previous version of this file; not re-verified in this sync.

### A. Data Availability Audit
| Tier | Source / Repository | Volume / Status | Quality Rating | Use in Pipeline |
|---|---|---|---|---|
| **Tier 1 (Gold)** | **Open-Sangam** (`repos/open-sangam`) | 18 poems, 2,552 verses, word glossaries, modern urai | ⭐⭐⭐⭐⭐ (0.95) | Anchor for classical syntax, Sandhi, & annealing stage |
| **Tier 1 (Gold)** | **SCERT Textbooks** (Grades 1–12) | Science, history, Tamil language reader txt files | ⭐⭐⭐⭐⭐ (0.95) | Foundational general world knowledge & grammar |
| **Tier 2 (Silver)**| **Tamil Wikipedia** (HF stream) | 150,000 articles | ⭐⭐⭐⭐ (0.90) | Fact/entity richness, modern prose style |
| **Tier 2 (Silver)**| **Project Madurai** | 700+ classical & medieval public domain texts | ⭐⭐⭐⭐ (0.90) | Literary vocabulary expansion & morphology depth |
| **Tier 3 (Bronze)**| **AI4Bharat IndicCorp v2 / Sangraha**| 250,000+ Tamil web documents | ⭐⭐⭐ (0.80) | Broad modern vocabulary & syntax scale |
| **Stage 2 (SFT)** | **Synthetic Conversational Distillation** | Gemma 2 27B / GLM-4 / Kimi distilled dialogues | ⭐⭐⭐⭐⭐ (0.95) | Multi-turn conversational flow & instruction-following |

### B. Strategy to Improve Tamil Content Quality
1. **Aggressive Near-Deduplication:** Enforce 64-hash MinHash LSH shingle filtering to eliminate web boilerplate (removing 90%+ redundancy).
2. **Strict Tamil Density Thresholds:** Enforce `min_tamil_ratio >= 0.80` to reject Tanglish noise, code snippets, and untranslated English fragments.
3. **Unicode NFC & Pulli Normalization:** Standardize all decomposed Tamil characters and fix isolated vowel/consonant markers (`Unicode NFC`).
4. **Dual-Register Synthetic Distillation:** Use frontier teacher models (Gemma 2, GLM, Kimi) to generate paired classical-to-colloquial Tamil explanations, idioms, and multi-turn educational dialogues.
5. **Sandhi & Grammar Integrity Gate:** Filter documents through morphology checkers (`solthiruthi` / `open-tamil`) to eliminate ungrammatical machine-translated web content.
