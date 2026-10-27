# Tamil intelligence capability plan — conversation, Sangam, QA, reasoning, thinking

Status: planning note, 2026-10-25. Written against measured artifacts in this repo,
not against aspirations. The one-line version:

> **A 14.5M-parameter model can be a grounded Tamil text engine. It cannot be a
> reasoning assistant. Conversation, QA and Sangam are reachable now with the right
> data + retrieval. Reasoning and thinking are data-engine problems first, and only
> then parameter-count problems.**

---

## 1. What we mean by each word

| axis | capability we actually ship | not this |
|---|---|---|
| **Conversational** | multi-turn Tamil dialogue in two registers (written/formal and spoken/colloquial), context retention, code-switch tolerance, explicit refusal | open-ended chat that knows everything |
| **Sangam** | explain a verse in modern Tamil, identify tiṇai/kalam/uripporul/karupporul, cite work + poem + verse, answer questions about a given verse | generating new classical poetry |
| **QA** | answer from *retrieved* Tamil sources with a citation span; say "தகவல் இல்லை" when the source does not contain it | closed-book encyclopedic recall |
| **Reasoning** | 1–3 step curriculum reasoning (math/science word problems), arithmetic via tool, answer verifiable | multi-hop open-world reasoning |
| **Thinking** | an optional, visible Tamil reasoning trace (`சிந்தனை:`) that measurably improves answer accuracy; token budget controlled | long hidden monologue that looks smart |

Treating "thinking" as a *measured* feature (does the trace raise accuracy?) is what
keeps it from becoming a style demo.

---

## 2. Tiers and honest feasibility

The binding constraint is data, not CPU. Chinchilla ~20 tokens/param:

| tier | params | tokens needed | pure Tamil available today | verdict |
|---|---|---|---|---|
| **Nano** | 14.5M | ~290M | 164M (+synthetic) | text engine, short grounded answers, Sangam, extractive QA |
| **Mini** | 100–300M | 2–6B | nowhere near | only after a synthetic data engine produces billions of verified Tamil tokens |
| **Adhan** | 1B+ | 20B+ | no | reasoning/thinking tier; requires synthetic + translated + licensed corpora and GPU budget |

So the sequence is: **build the data engine and post-training loop on nano → prove the
capabilities with retrieval/tools → only then spend on a bigger tier.** Scaling the
model before the data engine exists would produce a bigger model that still cannot
reason, at 10× the cost.

Per product vertical (from `FOUNDER.md`): Guru = curriculum QA + reasoning; Nyaya =
grounded legal QA + citations; Kural = conversational support with refusal; Avai/Yazh =
Sangam explanation + language learning. Sangam and QA-with-citations are the two
differentiators frontier models are weakest at.

---

## 3. What already exists

| asset | state | use |
|---|---|---|
| Swaram v2 tokenizer | 31,964 vocab, fertility 0.349, 100% lossless, no chat tokens | all training; role/think markers must use plain Tamil text for now |
| Corpus | 1.33M docs / ~293M corpus tokens / **~164M pure Tamil**; 54% lost to per-doc packing (see `CONTINUED_PRETRAIN.md`) | pretraining |
| Sangam | 2,545 labelled docs (verse + urai) with tiṇai/kalam/uripporul/karupporul, split by landscape (`docs/corpus/SANGAM_INGEST.md`, `data/final/tinai_labels.jsonl`) | **best SFT/evals asset**, not pretraining volume |
| Eval harness | fertility, classical floor, morphology, sandhi, kid prompts, model ppl (`src/adhan_slm/eval/`) | pretraining gates |
| `bench/tamil-bench` | MILU-Tamil, IndicQA-Tamil, IndicXNLI; OpenAI-compatible endpoint runner | external QA/reasoning evidence |
| SFT data | **none** (only 10 conversational seed phrases in `conversational_phrases.yaml`) | must be built |
| Chat format / special tokens | **none** | must be specified (textual markers now, v3 tokenizer later) |
| Retrieval | **none** | required for QA/legal; small models should not answer closed-book |

---

## 4. The five axes, concretely

### 4.1 Conversational

**Data (missing, build):**
- 10–20k multi-turn dialogues distilled from a strong teacher with a Tamil-only
  persona. Two registers: `எழுத்து` (formal/written) and `பேச்சு` (spoken, e.g.
  "என்கிட்ட", "போறேன்"). Include code-switch turns, correction turns, and refusals.
- Seed with `conversational_phrases.yaml`; mine real transcripts only with consent.
- Filters: Tamil fraction ≥ 0.95 per assistant turn (byte tokens allowed only for
  names/numbers), sandhi check on formal register, no English reasoning leakage,
  duplicate removal, and a teacher-quality score.

**Training:** SFT after pretraining, 2–3 epochs, loss on assistant turns only
(prompt tokens masked). Do **not** train on concatenated raw text for this.

**Eval:** 100 scripted 5–10-turn dialogues across the four verticals; score
turn fluency (human 1–5 on a 20% sample), context-retention probe at turn N,
register consistency, code-switch handling, refusal correctness, Tamil purity.

### 4.2 Sangam

**Data (already structured, needs pairs):** deterministic pair generation from the
labelled corpus — no teacher model needed for the high-precision parts:
- verse → tiṇai / kalam / uripporul / karupporul (classification + extraction)
- verse → modern-Tamil urai (target = traditional commentary, not invented)
- verse → English rendering (**eurai**, target = the Open-Sangam `english` field;
  currently 32 poems translated, so the task is present only where gold exists)
- verse → citation `[மேற்கோள்: <work> <poem>.<verse>]`
- urai → which verse/landscape it explains (inverse task)
- poet + work + number → verse identification (retrieval)

Split **by work/poem**, never random lines, or the eval leaks. Hold out ~15% of poems.

**Training:** SFT plus Tiṇai conditioning (`docs/architecture/TINAI_CONDITIONING.md`); the router is
built but off. Enable it in the next run, with weak labels already available.

**Eval:** 200 held-out verses. Metrics: tiṇai accuracy, karupporul F1, citation
accuracy (exact work+verse), urai adequacy (human/embedding), **hallucinated citation
rate < 2%**, refusal when the verse is not in context. Sangam is small enough that
these numbers are meaningful, unlike a 14.5M model's open-ended prose.

**English (eurai):** the `english` field in Open-Sangam is the gold rendering. It is
excluded from the Tamil corpus but used as SFT targets through
`scripts/build_sangam_sft.py --open-sangam-root ../open-sangam`. Do **not** machine-
translate missing targets inside the builder. Grow coverage with the existing
Open-Sangam pipeline, then rerun the builder:

```bash
cd ../open-sangam/scripts
OPENROUTER_API_KEY=... python -m ai.translate_all --only english
cd ../../adhan && python scripts/build_sangam_sft.py
```

Current gold coverage: 32/2,669 poems. That is enough to begin `sangam_eurai_en` but
not enough to claim translation quality; evaluate it with chrF/COMET + human review,
never with token F1.

### 4.3 QA (grounded, retrieval-first)

**Architecture:** retrieve Tamil passages → answer with a cited span. For nano, the
honest product is **extractive/abstractive short answers + citation**, not free-form
encyclopedic QA. Closed-book QA at 14.5M is a hallucination generator.

**Data:** SCERT grades 1–12 (science/history/Tamil), India Code sections for Nyaya,
institution handbooks for Guru, product runbooks for Kural. Generate question/answer
pairs **with the supporting span attached**; reject pairs whose answer is not
entailed by the span. Include 15–20% unanswerable questions (correct behaviour =
refusal).

**Eval:** IndicQA-Tamil EM/F1 (existing bench) + 500 in-domain held-out questions +
citation-span accuracy + answerability F1 (detect unanswerable). Target on nano:
positive EM on in-domain extractive sets; refuse on out-of-domain.

### 4.4 Reasoning

**Data:** 300–1,000 SCERT/TN-board math and science word problems (grades 6–10).
Teacher writes a Tamil solution; a **programmatic verifier** checks the final answer
(numeric/symbolic); only verified traces are kept. Augment with self-consistency and
distractor variants.

**Training:** SFT on `சிந்தனை: … பதில்: …`; optionally a small verifier model or a
calculator/tool call for arithmetic. Reasoning accuracy is the metric, not trace
length.

**Eval:** final-answer accuracy (pass@1), step validity (human/verifier), Tamil-only
trace rate, robustness to distractor numbers. At nano expect narrow 1–2 step success;
if it does not beat the no-thinking baseline, **do not claim reasoning**.

### 4.5 Thinking

**Decision rule:** a thinking trace ships only if it improves verified accuracy by a
margin larger than its token cost, on the same checkpoint. Otherwise it is style.

**Design:** textual protocol now (no vocab change):
```
<bos>பயனர்: …
உதவியாளர்: சிந்தனை: …
பதில்: … [மேற்கோள்: …]<eos>
```
Role/think/citation **special tokens are deferred to tokenizer v3** — adding them now
would require resizing the tied embedding (57% of the model) and repacking all shards.
Textual markers tokenize into existing Swaram pieces and keep the current checkpoint
valid.

**Eval:** accuracy with vs without the trace, trace faithfulness (does every step
follow from the previous/context, human or verifier), refusal when uncertain, tokens
per answer. Report all four; never just a pretty trace.

---

## 5. Curriculum and mixture

### Stage 1b (running now)
General Tamil pretraining, 100M more tokens. No chat format. See
`docs/training/CONTINUED_PRETRAIN.md`.

### Stage 2 — midtraining / anneal (next, ~100–150M tokens)
A separate packed set with explicit register and format tokens in the text:

| register | weight | source |
|---|---|---|
| general web/wiki/news | 40% | IndicCorp, FineWeb-2, Wikipedia |
| Sangam/classical (verse+urai, **not merged**) | 15% | open-sangam + labels |
| curriculum/SCERT | 10% | grades 1–12 |
| conversational (distilled) | 10% | teacher, dual register |
| QA/instruction (with citations) | 10% | SCERT, India Code, Sangam pairs |
| code-mixed / spoken | 5% | teacher + filtered web |
| pure Tamil reference/grammar | 10% | grammar, lexicon, Thirukkural |

Anneal LR to 10% over this stage. Keep the general share ≥40% so the model does not
forget how to be a language model.

### Stage 3 — SFT (multi-task, one model)
20–50k examples for nano (more will overfit), sampled with these approximate shares:
Sangam 25%, grounded QA 25%, conversation 25%, reasoning 15%, refusal/safety 10%.
Loss on assistant tokens only, 2 epochs, LR ~1e-4, early stop on held-out task loss.

### Stage 4 — preference/verifier (only if data allows)
DPO pairs on: citation faithfulness, sandhi correctness, Tamil purity, refusal when
unsupported. Reject pairs generated by the same model without a verifier.

---

## 6. Benchmark plan (what "done" means)

Extend `bench/tamil-bench` with three local task suites; it already speaks the
OpenAI-compatible protocol used by `scripts/run_api_server.py`, so the same runner
works for nano/mini and for teacher baselines.

| suite | size | metric | ship bar (nano) |
|---|---|---|---|
| `sangam` | 200 held-out verses | tiṇai acc, karupporul F1, citation acc, hallucinated-citation rate | tiṇai ≥ 70%, citation ≥ 95%, hallucination < 2% |
| `qa_grounded` | 500 in-domain + 100 unanswerable | EM/F1, citation span acc, answerability F1 | EM > closed-book baseline, refusal ≥ 80% on unanswerable |
| `reason` | 300 verified problems | final-answer acc, step validity, Tamil-only trace % | ≥ 20pp over no-thinking baseline, else don't ship thinking |
| `chat` | 100 scripted dialogues | context probe acc, register score, purity, refusal | context probe ≥ 80%, purity ≥ 95% |
| external | MILU / IndicQA / XNLI | as published in tamil-bench | report, no threshold yet |

Always publish the **baseline** (same checkpoint without SFT/RAG) next to the result.
A capability claim without an ablation is marketing.

---

## 7. Immediate sequence

1. **Finish Stage 1b** (~13:00 IST today) and benchmark it (see
   `CONTINUED_PRETRAIN.md` §5). This is the base checkpoint for everything below.
2. **Freeze the response protocol** (§4.5) and add it to the data builders and eval
   prompts. No tokenizer change.
3. **Build SFT v0 from Sangam + SCERT**, deterministically where possible: ~5–10k
   examples with citations. No teacher required for tiṇai/urai/citation pairs.
4. **Conversation v0**: 2–3k distilled dialogues, dual register, filtered. This is
   the minimum needed to stop the model from answering a question with more prose.
5. **Grounded QA v0**: retrieval over SCERT + India Code, 2k QA pairs with spans.
6. **Reasoning v0**: 500 verified SCERT problems with Tamil traces; measure with and
   without `சிந்தனை:`.
7. **Run the four suites**, publish the ablation table, then decide: more data on
   nano, or begin the Mini tier (requires ≥1B verified tokens first).

**Do not do yet:** grow to 1B parameters, add special tokens, or claim reasoning
before §4.4 clears its bar. Each of those is expensive and reversible only by
retraining.

---

## 8. Failure modes this plan is designed to avoid

- **Volume without structure** — dumping more web Tamil into a model that needs
  multi-turn and citation format.
- **Sangam merged** — concatenating verses destroys the labels; ingestion already
  enforces one document per (song, layer).
- **Teacher text trusted blindly** — every conversational/reasoning pair needs a
  verifier or a human sample; otherwise SFT learns fluent wrong answers.
- **Closed-book QA** — a 14.5M model will hallucinate; retrieval + refusal is the
  only honest nano-tier QA.
- **Thinking as decoration** — if the trace does not raise verified accuracy, cut it.
- **Scaling before data** — parameters cannot substitute for verified tokens.
