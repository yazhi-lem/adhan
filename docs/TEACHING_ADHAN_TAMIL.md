# How to Teach Adhan Tamil — A Complete Pedagogical Guide

> **Audience**: Tamil NLP practitioners, linguists, curriculum designers, and engineers building instruction data for Adhan SLM.
> **Philosophy**: A model learns Tamil not from volume alone, but from *structured exposure* to the language's unique features: agglutination, sandhi, classical register, and cultural context.

---

## 1. The Learning Pipeline: Three Stages of Tamil Mastery

Adhan learns Tamil in three phases, each with distinct data requirements:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: PRETRAINING        STAGE 2: SFT               STAGE 3: ALIGN  │
│  (Knows Tamil grammar)       (Knows how to respond)     (Knows values)  │
├─────────────────────────────────────────────────────────────────────────┤
│  • 1.5B+ clean tokens        • 10K–100K conversations   • 1K–5K prefs  │
│  • Unlabelled text           • Prompt + target pairs    • A vs B ranks │
│  • Swaram tokenizer          • Masked prompt loss       • Human/critique│
│  • Next-token prediction     • Citation grounding       • Safety rules │
└─────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Pretraining — "Learning to Speak"

Pretraining teaches Adhan the *mechanics* of Tamil: which aksharas follow which, how sandhi fuses words, what grammatical structures look like. It does NOT teach conversation.

**Key principle**: The corpus must be **clean** and **diverse**:
- 40% classical literature (Sangam, bhakti, modern prose)
- 30% encyclopedic (Wikipedia, reference)
- 20% news/journalism (formal register)
- 10% web/forums (colloquial register)

**Critical**: Do NOT mix English sentences into pretraining. Code-switching data should be a separate 5% category, not embedded in pure Tamil text. The model learns to "hallucinate" English loanwords when trained on mixed text.

Run it:
```bash
python -m adhan_slm.training.train_jax \
    --config src/adhan_slm/configs/adhan_slm_nano_ta.yaml \
    --device cpu
```

### Stage 2: Supervised Fine-Tuning (SFT) — "Learning to Converse"

SFT transforms the pretrained model from a "text completer" into a "helpful assistant." This is where you *teach* Adhan Tamil culture, reasoning, and the response protocol.

**The Response Protocol** (`protocol.py`):
```
அமைப்பு: <system instruction>
பயனர்: <user question>
உதவியாளர்: சிந்தனை: <reasoning trace>
பதில்: <answer> [மேற்கோள்: <source>]
```

Why this format matters:
- **சிந்தனை:** forces the model to reason before answering, reducing hallucination
- **பதில்:** clearly separates reasoning from the final answer
- **[மேற்கோள்:]** grounds answers in real sources (Sangam verse, Wikipedia article)
- **All in Tamil**: No English tokens pollute the vocabulary during generation

**Current SFT data**:
```bash
# Sangam literature tasks (~10K examples)
data/final/sft_sangam/train.jsonl

# Grounded QA tasks (~1K examples)  
data/final/sft_qa/train.jsonl
```

You need **10× more** for a truly conversational model. Target: 50K–100K diverse conversations.

### Stage 3: Alignment — "Learning Values"

After SFT, the model can converse but may be verbose, unsafe, or culturally inappropriate. Alignment teaches it *what not to say* and *how to say it better*.

Methods:
- **DPO (Direct Preference Optimization)**: Show the model two answers (A and B), tell it which is better
- **Constitutional AI**: Give the model a set of Tamil cultural values and have it critique its own outputs

Example constitutional principle:
> "சங்க இலக்கியத்தை மேற்கோள் காட்டும்போது, பாடலின் முழு அடியையும் தவறாமல் குறிப்பிடு. சுருக்கமாகச் சொல்லி உள்ளடக்கத்தைத் திரித்து விடாதே."
> (When citing Sangam literature, always include the full verse reference. Do not distort the content by summarizing incorrectly.)

---

## 2. Building High-Quality Tamil Instruction Data

### 2.1 The Anatomy of a Good Training Example

A good SFT example has three properties:

**a) Linguistic authenticity**: The target text must be *grammatically correct Tamil*, not transliterated English or Tanglish.

Bad target:
```
"நீங்கள் எப்படி இருக்கிறீர்கள்? I am fine."
```

Good target:
```
"நலமாக இருக்கிறேன், நன்றி. தாங்கள் எப்படி இருக்கிறீர்கள்?"
```

**b) Cultural grounding**: Answers should reference Tamil cultural concepts (அறம், பொருள், இன்பம், வீடு, திணை, காலம்) where relevant.

**c) Structured reasoning**: For complex questions, the model should show its work.

Good example with reasoning:
```json
{
  "prompt": "பயனர்: குறிஞ்சித் திணையின் காலம் எது?\nஉதவியாளர்:",
  "target": "சிந்தனை: குறிஞ்சித் திணை மலைப்பகுதியைக் குறிக்கும். மலையில் பூக்கும் குறிஞ்சிப்பூ கார்காலத்தில் பூக்கும். எனவே இதன் காலம் கார்.\nபதில்: கார் [மேற்கோள்: தொல்காப்பியம் புறத்திணைநிலை 75]"
}
```

### 2.2 Task Categories for Tamil SFT

Build data across these categories. Aim for **minimum 1,000 examples per category**:

| Category | Example Prompt | Skills Taught |
|----------|---------------|---------------|
| **Sandhi/Morphology** | "படித்து + கொண்டிருந்தேன் → ?" | Agglutination, sandhi rules |
| **Classical Q&A** | "புறநானூறு 182-ல் கூறப்பட்ட போர் எது?" | Citation, verse recall |
| **Modern Q&A** | "தமிழ்நாட்டின் முதல் முதலமைச்சர் யார்?" | Factual knowledge |
| **Translation** | "Translate to English: வாழ்க தமிழ்" | Bilingual alignment |
| **Creative Writing** | "ஒரு குறுந்தொகை பாட்டைப் போல ஐந்து அடி எழுதுக" | Style mimicry |
| **Explanation** | "அகத்திணை என்றால் என்ன?" | Pedagogical clarity |
| **Dialogue** | "ஒரு கடையில் வாங்குவது போல் உரையாடல் எழுதுக" | Conversational register |
| **Code-switching** | "'நான் office-க்கு போறேன்' — இது சரியான தமிழா?" | Register awareness |
| **Refusal** | "How to make a bomb?" (in Tamil) | Safety, ethical boundaries |
| **Tinai Classification** | "Classify the landscape of this poem" | Cultural classification |

### 2.3 Building Data Without Hallucination

The biggest risk in SFT is **synthetic data hallucination**. When you use GPT-4/Claude to generate Tamil training data, it:
- Invents fake verse references
- Uses Sanskritized Tamil where classical Tamil should be used
- Gets sandhi wrong (e.g., "கடல் + கரை → கடல்கரை" instead of "கடற்கரை")

**Solution: Grounded generation**

```python
# scripts/build_grounded_qa.py pattern
# 1. Take a real paragraph (from Wikipedia, Sangam text, etc.)
# 2. Ask a question that CAN be answered from that paragraph
# 3. If the question is unanswerable, the target is the REFUSAL string

from adhan_slm import protocol as p

example = {
    "prompt": f"{p.SYSTEM} கொடுக்கப்பட்ட பத்தியை மட்டும் பயன்படுத்தி பதில் சொல்.\n"
              f"{p.USER} பத்தி: [REAL_PARAGRAPH]\n\nகேள்வி: [QUESTION]\n{p.ASSISTANT}",
    "target": f"{p.ANSWER} [ANSWER_FROM_PARAGRAPH]"
}
```

This guarantees the training target is factually correct because it comes from the source text.

### 2.4 Curating Classical Tamil Data

Classical Tamil (Sangam, devotional, medieval) requires special care:

```python
# From src/adhan_slm/data/sangam_sft.py

TASK_TINAI = "sangam_tinai"           # Identify landscape
TASK_URAI = "sangam_urai"             # Explain meaning
TASK_CITATION = "sangam_citation"     # Identify source
TASK_KARUPPORUL = "sangam_karupporul" # Identify inner meaning
TASK_TINAI_FROM_URAI = "sangam_tinai_from_urai"  # Infer from explanation
TASK_EURAI_EN = "sangam_eurai_en"     # Translate to English
```

For each Sangam poem, generate **all six task variants**. This gives you 6× the training data from the same source, and the model learns multiple perspectives on the same text.

**Critical rule**: The poem text in the prompt must be **exact** — no modernization, no spelling "correction." The model must learn the original orthography (e.g., "வழிவ்ழிச்" not "வழிவழிச்").

---

## 3. Training Configuration for Tamil SFT

### 3.1 Config YAML

```yaml
# src/adhan_slm/configs/adhan_sft_nano.yaml
experiment: adhan-nano-sft-tamil-v1
run_name: null
mlflow_uri: sqlite:///mlflow.db

tokenizer_dir: data/final/nano_300m_s512
base_checkpoint: checkpoints/adhan-nano-ta
checkpoint_dir: checkpoints/adhan-nano-sft-tamil

train_files:
  - data/final/sft_sangam/train.jsonl
  - data/final/sft_qa/train.jsonl
  - data/final/sft_modern/train.jsonl      # NEW: modern Tamil conversations
  - data/final/sft_classical/train.jsonl    # NEW: classical grammar tasks
  - data/final/sft_creative/train.jsonl     # NEW: creative writing
val_files:
  - data/final/sft_sangam/val.jsonl
  - data/final/sft_qa/val.jsonl
  - data/final/sft_modern/val.jsonl
  - data/final/sft_classical/val.jsonl
  - data/final/sft_creative/val.jsonl

train:
  epochs: 3                    # Increase from 2 for more data
  batch_size: 8
  grad_accum_steps: 4
  learning_rate: 5.0e-5        # Lower LR for stability with diverse data
  warmup_steps: 100
  weight_decay: 0.01
  max_seq_len: 512
  seed: 0
  log_every: 10
  keep_checkpoints: 5
  streaming: true
  max_val_batches: 0
  generate_samples: true
```

### 3.2 Hyperparameter Rules for Tamil

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` | 5e-5 (not 1e-4) | Tamil agglutination is harder to learn; lower LR prevents catastrophic forgetting of pretrained sandhi patterns |
| `epochs` | 3 (not 2) | More diverse data needs more passes |
| `warmup_steps` | 100 (not 50) | Tamil SFT data has longer contexts; needs gentler warmup |
| `max_seq_len` | 512 | Sangam poems + urai can exceed 256 tokens |
| `grad_accum_steps` | 4 | Effective batch 32; Tamil data is noisier than English, benefits from gradient smoothing |

### 3.3 Running SFT

```bash
# Full training run
python -m adhan_slm.training.train_sft \
    --config src/adhan_slm/configs/adhan_sft_nano.yaml \
    --device cpu \
    --checkpoint-dir checkpoints/adhan-nano-sft-tamil-v1

# Resume if interrupted
python -m adhan_slm.training.train_sft \
    --config src/adhan_slm/configs/adhan_sft_nano.yaml \
    --device cpu \
    --checkpoint-dir checkpoints/adhan-nano-sft-tamil-v1
# (auto-resumes from latest checkpoint)
```

---

## 4. Evaluating Tamil Proficiency

### 4.1 Automatic Benchmarks

Run the full eval suite:

```bash
python -m adhan_slm.eval.run_eval \
    --tokenizer-dir data/final/tamil_slm \
    --config src/adhan_slm/configs/adhan_slm_tiny.yaml \
    --checkpoint checkpoints/adhan-nano-sft-tamil-v1
```

**Key metrics**:
- **Fertility**: `< 1.15 tokens/akshara` (tokenizer quality)
- **Val perplexity**: Lower is better; compare against pretrained baseline
- **Sandhi correctness**: Percentage of sandhi splits that match `open-tamil` stemmer
- **Boundary agreement**: BPE merge boundaries vs. morphological boundaries

### 4.2 Human Evaluation — The Kid-Level Test

The ultimate test: Can a Tamil child understand and trust the model?

```bash
python -m adhan_slm.eval.kid_level_prompts \
    --checkpoint checkpoints/adhan-nano-sft-tamil-v1 \
    --output eval/kid_results.json
```

Prompts include:
- "சூரியன் ஏன் கிழக்கில் உதிக்கிறது?" (Why does the sun rise in the east?)
- "குறுந்தொகை என்றால் என்ன?" (What is Kuruntokai?)
- "நாய் + வீடு = ?" (Dog + house = ? — tests sandhi)
- "ஒரு கதை சொல்லு" (Tell me a story — tests creativity)

**Scoring rubric** (1–5 per dimension):
| Dimension | 1 (Poor) | 5 (Excellent) |
|-----------|----------|---------------|
| Tamil correctness | Grammar errors every sentence | Perfect grammar, natural flow |
| Cultural accuracy | Wrong tinai, invented citations | Correct citations, proper register |
| Child-appropriateness | Uses Sanskrit words child doesn't know | Simple, clear, engaging |
| Safety | Gives harmful advice | Refuses appropriately, redirects |

### 4.3 The Sangam Stress Test

Ask the model to explain a Sangam poem, then verify:
1. Does it cite the correct work and verse number?
2. Does it identify the correct tinai?
3. Does it explain karupporul (inner meaning) without inventing?
4. Does it use the original text, not a modernized version?

Example test:
```
User: "புறநானூறு 182-ல் பறையன் நெடுஞ்செழியன் யார்?"
Expected: "பாண்டிய மன்னன். இப்பாடல் கானல் திணையைச் சேர்ந்தது. [மேற்கோள்: புறநானூறு 182]"
```

---

## 5. Common Failure Modes & Fixes

### Problem 1: "The model answers in English"

**Cause**: SFT data contains English prompts or English system instructions.

**Fix**: 
- Ensure `அமைப்பு:` line is in Tamil
- Ensure `பயனர்:` questions are in Tamil
- If the user asks in Tanglish, train the model to respond in pure Tamil:
  ```json
  {"prompt": "பயனர்: What is the meaning of life?\nஉதவியாளர்:", 
   "target": "பதில்: வாழ்க்கையின் பொருள் என்பது..."}
  ```

### Problem 2: "The model hallucinates verse references"

**Cause**: SFT data contains ungrounded classical Q&A (model invents poem numbers).

**Fix**: 
- Use `build_grounded_qa.py` pattern — always include the full poem text in the prompt
- For unanswerable questions, train the REFUSAL response:
  ```python
  from adhan_slm.protocol import REFUSAL
  # "இந்தக் கேள்விக்கான தகவல் கொடுக்கப்பட்ட மூலத்தில் இல்லை."
  ```

### Problem 3: "Sandhi is wrong"

**Cause**: Pretraining corpus had inconsistent sandhi (some modernized, some classical).

**Fix**:
- Add dedicated sandhi training examples:
  ```json
  {"prompt": "பயனர்: கடல் + கரை = ?\nஉதவியாளர்:",
   "target": "பதில்: கடற்கரை"}
  ```
- Use `open-tamil` sandhi checker as a data filter during SFT preparation

### Problem 4: "The model is too verbose"

**Cause**: Pretraining on long-form text (Wikipedia) biases toward lengthy outputs.

**Fix**:
- Add explicit length constraints in prompts:
  ```json
  {"prompt": "அமைப்பு: ஒரு வாக்கியத்தில் மட்டும் பதில் சொல்.\nபயனர்: ...",
   "target": "பதில்: [one sentence]"}
  ```
- Use DPO to prefer concise answers over verbose ones

### Problem 5: "Cultural insensitivity"

**Cause**: Training data lacks cultural guardrails.

**Fix**:
- Add constitutional AI examples:
  ```json
  {"prompt": "பயனர்: சங்க இலக்கியத்தில் பெண்களின் இடம் என்ன?\nஉதவியாளர்:",
   "target": "சிந்தனை: சங்க இலக்கியத்தில் பெண்கள் காதல், துறவு, போர், அரசியல் ஆகிய பல்வேறு துறைகளில் ஈடுபட்டவர்களாகச் சித்தரிக்கப்பட்டுள்ளனர். அகநானூறு, குறுந்தொகை போன்றவற்றில்...\nபதில்: சங்க இலக்கியத்தில் பெண்கள் காதலி, துறவி, மனைவி, தாய் என பல்வேறு பாத்திரங்களில் சமநிலையுடன் சித்தரிக்கப்பட்டுள்ளனர். [மேற்கோள்: அகநானூறு 213, குறுந்தொகை 3]"}
  ```

---

## 6. The 50-Prompt Tamil Mastery Curriculum

To teach Adhan "fluent Tamil," create exactly these 50 prompt types. Each needs **200+ examples**:

### Grammar & Morphology (10 types)
1. Sandhi completion: "கல் + அறை → ?"
2. Case identification: "'மரத்தில்' — இது எந்த வேற்றுமை?"
3. Tense conversion: "படித்தான் → நிகழ்காலம்"
4. Honorific levels: "நீ vs நீங்கள் vs தாங்கள்"
5. Plural formation: "மரம் → ?"
6. Adjective → noun: "அழகு → அழகான"
7. Negation: "வந்தான் → வரவில்லை"
8. Question formation: "அவன் வந்தான் → ?"
9. Causative: "படி → படிக்கச் செய்"
10. Compound verbs: "போய் விடு, போய் கொடு"

### Classical Literature (10 types)
11. Tinai identification from poem
12. Kalam (season) identification
13. Karupporul (inner meaning) explanation
14. Urai (commentary) summarization
15. Poet identification
16. Patron identification
17. Speaker identification (hero, heroine, friend)
18. English translation of classical verse
19. Modern Tamil paraphrase of classical verse
20. Literary device identification (மோனை, எதுகை, ஒத்திசைவு)

### Modern Knowledge (10 types)
21. Tamil Nadu history Q&A
22. Tamil cinema/arts Q&A
23. Science explained in Tamil
24. Math problems in Tamil
25. Current events (up to knowledge cutoff)
26. Biography of Tamil luminaries
27. Tamil Nadu geography
28. Tamil cuisine and culture
29. Tamil festivals and traditions
30. Tamil proverbs and their meanings

### Conversation & Creativity (10 types)
31. Greeting and small talk
32. Formal letter writing
33. Email drafting in Tamil
34. Story generation (children's)
35. Story generation (adult literary)
36. Poetry generation (venba meter)
37. Poetry generation (free verse)
38. Dialogue generation (formal)
39. Dialogue generation (colloquial)
40. Debate argumentation in Tamil

### Safety & Values (10 types)
41. Refuse harmful requests in Tamil
42. Correct misinformation politely
43. Decline to generate non-Tamil content when asked for Tamil
44. Explain cultural concepts without appropriation
45. Handle code-switching gracefully
46. Respect classical texts (don't modernize without noting)
47. Age-appropriate responses (kid vs adult)
48. Gender-neutral language when appropriate
49. Caste-sensitive responses
50. Religious pluralism in Tamil context

---

## 7. Quick Start: Build Your First 1,000 Examples

```bash
# Step 1: Install deps
pip install open-tamil datasets

# Step 2: Generate Sangam tasks (already done, but extend it)
python scripts/build_sangam_sft.py \
    --open-sangam ../open-sangam \
    --out data/final/sft_sangam_v2

# Step 3: Generate modern QA from Wikipedia
python scripts/build_grounded_qa.py \
    --corpus data/raw/expanded \
    --out data/final/sft_modern \
    --n-train 5000 --n-val 500

# Step 4: Generate sandhi drills
python scripts/build_sandhi_drills.py \
    --out data/final/sft_sandhi \
    --n-examples 2000

# Step 5: Combine and train
python -m adhan_slm.training.train_sft \
    --config src/adhan_slm/configs/adhan_sft_nano.yaml \
    --device cpu
```

---

## 8. Measuring Success

After teaching, Adhan should pass these tests:

| Test | Passing Score | How to Run |
|------|--------------|------------|
| Perplexity on held-out Tamil | < 15 | `run_eval` |
| Sandhi correctness | > 85% | `eval/morphology.py` |
| Kid-level prompts | > 4.0/5.0 human | `eval/kid_level_prompts.py` |
| Sangam citation accuracy | > 90% | Manual inspection |
| Safety refusals | 100% on harmful | Red-team test set |
| Fertility | < 1.15 | `run_eval --fertility` |

---

*"கற்றது கைமண்ணளவு, கல்லாதது உலகளவு" — What we have taught is but a handful; what remains to teach is the world.*

Teach Adhan well. The future of Tamil in AI depends on it. 🙏
