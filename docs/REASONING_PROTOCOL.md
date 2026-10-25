# Adhan Reasoning Protocol v1 — Think, Rationalize, Reflect in Tamil

> **Goal**: Teach Adhan to reason step-by-step entirely in modern Tamil before answering.
> **Mechanism**: Structured reasoning trace in the target, but loss is masked on reasoning tokens.

---

## The Four Stages

Every trained response follows this structure:

```
உதவியாளர்:
சிந்தனை: <initial analysis — what is being asked?>
ஆய்வு: <step-by-step investigation — evidence, logic, working>
பரிசீலனை: <self-correction — doubt, admission of uncertainty>
பதில்: <concise final answer> [மேற்கோள்: <source>]
```

### Stage 1: சிந்தனை (Thought)
Decomposes the question. Identifies the domain (Sangam literature, math, history, safety).

Example:
> "இந்தப் பாடலில் மலை, காடு, விலங்குகள் பற்றிய குறிப்புகள் உள்ளன. இது குறிஞ்சித் திணையைச் சேர்ந்ததாகத் தோன்றுகிறது."

### Stage 2: ஆய்வு (Rationalize / Investigate)
Works through the logic. Shows evidence. For classical questions, references specific works.

Example:
> "குறிஞ்சித் திணையில் காதலர் இரவில் சந்திப்பார்கள். பாடலில் 'மலை' என்ற சொல் இருப்பதால், இது குறிஞ்சி நிலத்தைக் குறிக்கிறது."

### Stage 3: பரிசீலனை (Reflect)
Challenges own assumptions. Admits when uncertain. This is the **anti-hallucination** layer.

Example:
> "எனினும், பாடலில் வயலும், மழையும் குறிப்பிடப்பட்டுள்ளதா என்று சரிபார்க்க வேண்டும். இல்லையெனில் குறிஞ்சி உறுதி."

### Stage 4: பதில் (Answer)
The final, confident output. Includes citation when grounded.

Example:
> "குறிஞ்சி [மேற்கோள்: தொல்காப்பியம் புறத்திணைநிலை 75]"

---

## Why This Works

### Loss Masking

```python
# In train_sft.py — reasoning tokens get mask = 0.0
mask = [0.0] * len(ids)           # everything masked
for i in range(ans_start, len(ids)):
    mask[i] = 1.0                  # only answer unmasked
```

| Token Type | Loss | Why |
|-----------|------|-----|
| `<bos>` + prompt | 0 | User input — model should not predict |
| சிந்தனை trace | 0 | Reasoning — free to generate, not supervised |
| ஆய்வு trace | 0 | Working — free to generate, not supervised |
| பரிசீலனை trace | 0 | Reflection — free to generate, not supervised |
| பதில் + citation | 1 | Answer — the only supervised signal |
| `<eos>` | 1 | End marker |

**Result**: The model learns to generate reasoning because it predicts it autoregressively, but it is only **penalized** for wrong answers. This prevents the model from "gaming" the reasoning to match a known answer.

---

## Generating Training Data

### Option 1: Enrich Existing SFT Data

```bash
python scripts/build_reasoning_sft.py \
    --input data/final/sft_sangam/train.jsonl \
    --output data/final/sft_reasoning/train.jsonl \
    --mode enrich
```

This adds reasoning traces to your existing Sangam and QA data using templates.

### Option 2: Generate Pure Reasoning Tasks

```bash
python scripts/build_reasoning_sft.py \
    --output data/final/sft_reasoning/train.jsonl \
    --mode generate --n-examples 5000
```

Creates synthetic tasks:
- Sandhi reasoning ("கடல் + கரை → ?")
- Math reasoning ("12 + 25 = ?")
- Tinai reasoning ("Classify this poem's landscape")
- Refusal reasoning ("Why I cannot answer this harmful question")

### Option 3: Both

```bash
python scripts/build_reasoning_sft.py \
    --input data/final/sft_sangam/train.jsonl \
    --output data/final/sft_reasoning/train.jsonl \
    --mode both --n-examples 3000
```

---

## Training

```bash
python -m adhan_slm.training.train_sft \
    --config src/adhan_slm/configs/adhan_sft_reasoning_nano.yaml \
    --device cpu
```

**Key differences from standard SFT**:
- `learning_rate: 5.0e-5` (lower — reasoning needs stability)
- `epochs: 3` (reasoning traces are longer, need more passes)
- `max_seq_len: 512` (reasoning can consume 100+ tokens)

---

## Inference

### Standard Endpoint (raw text)
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "பயனர்: குறிஞ்சித் திணையின் காலம் எது?\nஉதவியாளர்:", "max_length": 128}'
```

Returns raw text including reasoning.

### Structured Reasoning Endpoint
```bash
curl -X POST http://localhost:8000/generate/reasoning \
  -H "Content-Type: application/json" \
  -d '{"text": "பயனர்: குறிஞ்சித் திணையின் காலம் எது?\nஉதவியாளர்:", "max_length": 128}'
```

Returns JSON:
```json
{
  "thought": "இந்தக் கேள்வி சங்க இலக்கியத்தைப் பற்றியது...",
  "rationalize": "குறிஞ்சித் திணை மலைப்பகுதியைக் குறிக்கும்...",
  "reflect": "எனினும் முல்லையிலும் காடு உண்டு என்பதால்...",
  "answer": "கார்",
  "citations": ["[மேற்கோள்: தொல்காப்பியம் புறத்திணைநிலை 75]"],
  "raw": "சிந்தனை: ...\nஆய்வு: ...\nபரிசீலனை: ...\nபதில்: ..."
}
```

---

## Evaluating Reasoning Quality

A good reasoning trace should:

1. **Be in Tamil**: No English words inside சிந்தனை/ஆய்வு/பரிசீலனை
2. **Show working**: Not just restate the answer
3. **Admit doubt**: பரிசீலனை should contain genuine uncertainty checks
4. **Be concise**: Reasoning should not exceed 60% of total tokens
5. **Lead to correct answer**: The final பதில் should match ground truth

### Red Flags

| Problem | Example |
|---------|---------|
| Empty reasoning | "சிந்தனை: சரி. ஆய்வு: சரி. பரிசீலனை: சரி." |
| Reasoning contradicts answer | Thought says "மருதம்" but answer says "குறிஞ்சி" |
| No doubt in reflection | "பரிசீலனை: என் பதில் சரியானது." (always certain) |
| English in reasoning | "சிந்தனை: This is about love." |

---

## Future Work

- **DPO on reasoning**: Train a preference model to prefer reasoning traces that cite sources correctly
- **Constitutional reasoning**: Add cultural guardrails inside the reflection stage
- **Multi-step math**: Extend ஆய்வு to handle algebra, geometry in Tamil
- **Code reasoning**: Trace through Tamil programming concepts

---

*"சிந்தித்துச் செயல்படு" — Think before you act.* 🧠
