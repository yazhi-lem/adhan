# Base decision — 2026-11-03

**Status:** Decision recorded for execution (Founder can veto; ADR remains the formal record)
**Supersedes:** the undecided "Adhan vs Sarvam vs embeddings" question
**Related:** `docs/adr/0001-kutty-v1.md` · `docs/reports/muril_base_assessment.md`

---

## 1. Decision

**Build Kutty v1 by fine-tuning a permissively licensed 1.7B decoder base. Do not build
the product on `adhan-nano`, and do not use Sarvam-1 without a commercial licence.**

| Role | Choice | Why |
|---|---|---|
| **Product / on-device student** | **`Qwen/Qwen3-1.7B-Base`** (Apache-2.0) | Permissive, 100+ languages, 32k context, small enough for INT4 on a ₹10–15k phone |
| **Teacher / server reference** | **`sarvamai/sarvam-m`** 24B (Apache-2.0) | Strong Tamil tokenizer (0.80 tok/akshara), permissive, good distillation teacher |
| **Server/eval reference** | `Qwen/Qwen3-4B-Base` (Apache-2.0) | Same tokenizer/family as the student; cheap quality ceiling check |
| **Research artifact** | `adhan-nano` + Swaram | Tokenizer/ablation paper, not the product base |
| **Embeddings/retrieval** | MuRIL-class encoders | Retrieval, classification, political filtering — never generation |
| **Rejected product base** | `sarvamai/sarvam-1` 2B | Best Tamil tokenizer tested, but **Sarvam AI Research License: non-commercial/research only** |

## 2. Why this is the decision

### 2.1 Adhan-nano is a research step, not a 26 Dec product

- 14.5M params, Stage 1b val 1.820 bits/akshara; generations are not coherent Tamil.
- No SFT run yet, no regression gate, export/quantize broken for JAX checkpoints.
- The compression thesis already lost its first controlled test: a Tamil-trained
  SentencePiece BPE is 1.52× more compressed than Swaram.
- Fine-tuning Adhan is still worth doing — as the research artifact and the Swaram
  paper — but it cannot carry the product deadline.

### 2.2 Sarvam-1 is technically tempting, legally wrong for the product

Sarvam-1 has the best Tamil tokenizer measured here (0.538 tok/akshara, essentially tied
with Swaram), but its licence is the **Sarvam AI Research License**:

> “...to create Derivatives of the Sarvam Model for non-commercial and research purposes only.”

That is incompatible with an open product and downstream users. Reject it unless Sarvam
grants a commercial licence. Sarvam-M 24B is Apache-2.0 and can be used, but 24B is a
server/teacher model, not a phone model.

### 2.3 Embeddings cannot be the base

MuRIL is an encoder. It has no decoder, cannot answer prompts, and cannot absorb the
Sangam/QA SFT. Embeddings are a retrieval/classification layer, not the generator.

### 2.4 Tokenizer evidence

Measured on the same 300 documents / 35,986 aksharas from
`data/raw/expanded/corpus-00001.jsonl`:

| Tokenizer | Fertility (tok/akshara) | Relative to Sarvam-1 |
|---|---:|---:|
| Tamil-trained SentencePiece BPE | 0.346 | 0.64× |
| Swaram v2 | 0.527 | 0.98× |
| **Sarvam-1** | **0.538** | **1.00×** |
| Sarvam-M | 0.804 | 1.50× |
| Qwen3-1.7B / 4B | 2.016 | 3.75× |

Qwen3 is the licence-safe choice, but its Tamil tokenizer is ~3.75× less efficient than
Sarvam-1. That is a real cost: more tokens per answer, more compute, more latency. It
does **not** block the product because Qwen3 has a 32k context and the phone target is
short answers, but it makes distillation from Sarvam-M and a future Swaram-based student
worthwhile.

## 3. Execution plan

1. **Bake-off (48 h):** Qwen3-1.7B-Base vs Qwen3-4B-Base, zero-shot on tamil-bench
   (`xnli`, `milu`, `indicqa`) plus a 20-prompt Tamil generation read. Record accuracy,
   fluency, code-mixing, and INT4 size/latency.
2. **SFT v0:** fine-tune Qwen3-1.7B-Base on `data/final/sft_v0_sangam` (9,961 train) +
   grounded QA, assistant-turn loss, no-SFT regression gate.
3. **Distillation (parallel, if GPU):** use Sarvam-M 24B as teacher on Tamil prompts;
   filter by Tamil purity and a verifier before adding to SFT.
4. **Export:** Qwen3 → GGUF/ONNX INT4, measure tokens/s and RSS on the reference phone.
5. **Research track:** keep `adhan-nano` + Swaram running; publish the tokenizer/A-B
   results separately. Do not block the product on it.
6. **Fallback if Qwen3 Tamil quality fails the bake-off:** Gemma-3-4B-IT (Gemma terms)
   or a negotiated Sarvam-1 commercial licence. Do not silently switch the licence.

## 4. What would change this decision

- Qwen3-1.7B Tamil quality is clearly worse than the 4B/teacher on the 20-prompt set.
- INT4 Qwen3-1.7B does not fit or run acceptably on the reference phone.
- Sarvam grants a commercial licence for Sarvam-1 2B.
- A permissively licensed 2B Indic model with Sarvam-class fertility appears.

Until one of those fires, **Qwen3-1.7B-Base is the product base; Sarvam-M is the teacher;
Adhan-nano is the research artifact; embeddings are auxiliary.**
