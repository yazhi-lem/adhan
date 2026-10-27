# Response protocol v0 (Tamil textual markers)

Status: **frozen for stage 1b / SFT v0**, 2026-10-25.
Code: [`src/adhan_slm/protocol.py`](../../src/adhan_slm/protocol.py).

The Swaram v2 tokenizer has no chat, role or thinking tokens. Adding them now would
resize the tied embedding (57% of the 14.5M model) and invalidate all packed shards.
So the protocol uses **plain Tamil words at the start of a line**. They tokenize into
existing Swaram pieces and work with the checkpoint that is training today.

Every SFT dataset and every eval runner must use this module rather than re-typing
the markers.

---

## 1. Wire format

```
அமைப்பு: <optional system instruction>
பயனர்: <user turn>
உதவியாளர்: <assistant turn>
பயனர்: <next user turn>
உதவியாளர்:
```

A response:

```
உதவியாளர்: சிந்தனை: <optional, short, checkable reasoning>
பதில்: <answer> [மேற்கோள்: <work> <reference>]
```

Rules:

| rule | detail |
|---|---|
| marker position | start of a line, followed by `:` and one space |
| roles | `பயனர்:` user · `உதவியாளர்:` assistant · `அமைப்பு:` system |
| thinking | `சிந்தனை:` is optional; it ships only if it raises verified accuracy (see `CAPABILITIES.md` §4.5) |
| answer | exactly one `பதில்:` line, always present |
| citation | zero or more `[மேற்கோள்: work reference]` tokens, space-separated, after the answer |
| refusal | if the source does not contain the answer, the whole answer is `REFUSAL` with no citation |
| citation truth | never invent a work or verse; a wrong citation is worse than no answer |

The canonical refusal (also `protocol.REFUSAL`):

```
இந்தக் கேள்விக்கான தகவல் கொடுக்கப்பட்ட மூலத்தில் இல்லை.
```

---

## 2. Examples

### Grounded QA (with citation)

```
அமைப்பு: கொடுக்கப்பட்ட பத்தியை மட்டும் பயன்படுத்தி, சுருக்கமாகத் தமிழில் பதில் சொல்.
பயனர்: பத்தி: தமிழ்நாட்டின் தலைநகரம் சென்னை. சென்னை மெரினா கடற்கரைக்குப் பெயர்பெற்றது.
கேள்வி: தமிழ்நாட்டின் தலைநகரம் எது?
உதவியாளர்:
```

target:

```
பதில்: சென்னை [மேற்கோள்: SCERT 4.2]
```

### Refusal (answer not in source)

```
பயனர்: பத்தி: தமிழ்நாட்டின் தலைநகரம் சென்னை.
கேள்வி: கேரளாவின் தலைநகரம் எது?
உதவியாளர்:
```

target:

```
பதில்: இந்தக் கேள்விக்கான தகவல் கொடுக்கப்பட்ட மூலத்தில் இல்லை.
```

### Sangam explanation (with citation + optional reasoning)

```
பயனர்: பாடல்: அன்னாய் வாழிவேண் டன்னை என்னை …
இந்தப் பாடலின் திணை எது? ஏன்?
உதவியாளர்:
```

target:

```
சிந்தனை: மலை, யானை, மயில், முருகன், அருவி ஆகிய கருப்பொருள்கள் குறிஞ்சியைக் குறிக்கின்றன.
பதில்: குறிஞ்சி. [மேற்கோள்: ஐங்குறுநூறு 201]
```

### Sangam English rendering (`eurai`)

`eurai` is the English equivalent of the Tamil `உரை`: a faithful English rendering
of the verse, used as the `sangam_eurai_en` target. The answer stays in the same
`பதில்:` slot; only the language changes, and the citation still points at the Tamil
source. Never invent an English line that the verse does not support.

```
பயனர்: பாடல்: உலகம் உவப்ப வலன்ஏர்பு திரிதரு …
இந்தப் பாடலின் ஆங்கில உரை (eurai) யாது? ஆங்கிலத்தில் மட்டும் எழுது.
உதவியாளர்:
```

target:

```
பதில்: As the world rejoices, rising to the right, the much-praised sun circles …
[மேற்கோள்: திருமுருகாற்றுப்படை 01]
```

Gold data: the Open-Sangam `english` field (32 poems at the time of writing). Do not
fill missing targets with machine translation inside the data builder; use the
Open-Sangam translation pipeline and re-run `scripts/build_sangam_sft.py`.

### Conversational memory

```
பயனர்: என் பெயர் மீரா. நான் ஏழாம் வகுப்பு படிக்கிறேன்.
உதவியாளர்: வணக்கம் மீரா. உனக்கு எப்படி உதவலாம்?
பயனர்: என் பெயர் என்ன?
உதவியாளர்:
```

target:

```
பதில்: உங்கள் பெயர் மீரா.
```

---

## 3. Training rules

1. **Mask the prompt.** Compute loss only on the target (`சிந்தனை:`/`பதில்:`), never on
   user turns, system text or the assistant cue.
2. **One example = prompt + target**, exactly the strings returned by
   `protocol.render_example(...)`; do not re-render with different whitespace.
3. **Empty thinking is normal.** Most v0 examples have no `சிந்தனை:` line. Mixing
   with/without traces is deliberate: the model must work when thinking is omitted.
4. **Citations are part of the answer**, not metadata. If the citation is missing at
   inference, the answer is ungrounded and scores as wrong for QA/Sangam.
5. **Refusal is a target, not an error path.** 10–20% of grounded QA examples should
   be unanswerable; otherwise the model learns to always answer.
6. **No special-token ids yet.** `<bos>` opens an example and `<eos>` closes the
   target, as in pretraining. Role markers are ordinary text.

---

## 4. Parsing and evaluation

`protocol.parse_response(text)` returns:

```python
{"thought": str|None, "answer": str, "citations": [str], "found_answer": bool, "raw": str}
```

- `found_answer=False` means the model ignored the protocol; score the row wrong for
  format-sensitive tasks (QA/Sangam) rather than guessing.
- `protocol.is_refusal(answer)` compares against the canonical refusal after
  whitespace normalisation.
- Citation matching in benchmarks is **exact on the parsed citation string**
  (`work` + `reference`), not substring matching, so a hallucinated verse cannot
  pass by sharing the work name.

---

## 5. What changes in tokenizer v3

Reserve, in this order, and retrain the embedding once:

```
<pad> <bos> <eos> <unk> <mask> <0x00..0xFF>
<|system|> <|user|> <|assistant|> <|think|> <|answer|> <|cite|>
```

Every change must ship with: a vocab bump, a repacked shard set at the same
`seq_len`, a new baseline eval, and a checkpoint that actually used the tokens.
Do **not** add them between two runs and assume the old checkpoint still lines up:
the tied embedding makes a silent +6-token shift a corruption bug, not an upgrade.
