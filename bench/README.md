# Tamil Bench — Adhan comparison harness

Runs [tamil-bench](https://github.com/LogicIncZo/tamil-bench)
([scoreboard](https://logicinczo.github.io/tamil-bench/)) against an Adhan
checkpoint served locally, so our numbers sit on the same three papers as the
hosted frontier models.

## What the bench measures

| Paper | Dataset | Metric | Tamil split |
|---|---|---|---|
| `milu` | `ai4bharat/MILU`, config `tamil` | accuracy, 0-shot letter parse | 6,372 MCQs, 41 subjects (gated on HF) |
| `indicqa` | `ai4bharat/IndicQA` | Exact Match + F1 (SQuAD normalisation) | 1,804 questions / 253 passages |
| `xnli` | `AdaMLLab/indicxnli_repaired` | 3-way accuracy (A/B/C), chance = 33% | test split |

Published reference scores (OpenRouter, 0-shot, stratified `seed=42`):

| Model | MILU acc | IndicQA EM | IndicQA F1 | XNLI acc |
|---|---|---|---|---|
| google/gemini-3.8-flash | **94.5%** | 20.0% | **42.4** | **75.8%** |
| anthropic/claude-sonnet-5.5 | 89.4% | **22.0%** | 43.1 | — |
| openai/gpt-6-luna | 87.8% | 16.0% | 40.5 | — |
| deepseek/deepseek-v4.1-flash | 80.9% | 18.0% | 38.9 | 62.0% |
| qwen/qwen3.8-flash | 68.8% | 22.0% | 38.1 | 63.0% |
| google/gemma-4-26b-a4b-it | 58.3% | 20.2% | 39.3 | 64.8% |
| nvidia/nemotron-3.5-lightning | 43.7% | 1.0% | 3.5 | — |
| **random guessing** | 25% | ~0% | ~0% | 33% |

Adhan's realistic target for `adhan-nano` (~8M params, from-scratch, ~1/1000th
the parameters of the leaders) is **above random and above the free-tier
small-model floor**, not parity with Gemini. Score it honestly and publish the
delta.

## Setup

```bash
cd ~/Yazhi/repos/adhan
./bench/setup_tamil_bench.sh        # clones tamil-bench + applies the endpoint patch
```

Datasets already fetched into `bench/tamil-bench/data/`:

- `indicqa.ta.json` (2.7 MB)
- `indicxnli_ta_test.parquet` (0.5 MB)

MILU is **gated** — accept the gate at
<https://huggingface.co/datasets/ai4bharat/MILU> with your HF account, then:

```bash
HF_TOKEN=... python -c "
from huggingface_hub import hf_hub_download
"...   # see bench.py run_milu() for the expected parquet name: data/milu_ta_test.parquet
```

## Run

1. Serve a checkpoint with the OpenAI-compatible surface
   (`/v1/chat/completions`, added in `scripts/run_api_server.py`):

   ```bash
   ADHAN_CHECKPOINT_DIR=checkpoints/adhan-nano \
   ADHAN_TOKENIZER_DIR=data/final/tamil_slm \
   python scripts/run_api_server.py --model adhan-nano --port 8000
   ```

2. Grade it:

   ```bash
   ./bench/run_adhan_bench.sh http://127.0.0.1:8000/v1/chat/completions
   ```

   The script runs `indicqa` (n=100), `xnli` (n=200), and `milu` (n=199) when the
   gated parquet exists, then prints the sheets in `bench/tamil-bench/results/`.

## Notes on fairness

- Adhan is a **base** model, not chat-tuned. `render_chat_prompt()` in
  `scripts/run_api_server.py` flattens OpenAI messages into plain text and cues an
  answer with a trailing newline. After SFT alignment this should become a real
  chat template — scores before that understate the model.
- `temperature=0`, same `seed=42` sampling as upstream, same parsing rules.
- Keep `max_tokens` at 512 (the bench default) so truncation is comparable.
