# Agaram 2.5B — what it is, and what it means for Adhan (reading note)

**Date:** 2026-11-05 · **Status:** reading + local measurement, not a bake-off
**Sources:** model card `murai-labs/agaram-2.5b-instruct` (fetched 5 Nov), arXiv:2610.18420 abstract,
local GGUF `agaram-2.5b-instruct-Q4_K_M`, our own harness runs below
**Related:** [`base_decision_2026-11-03.md`](base_decision_2026-11-03.md) · [`base_bakeoff_qwen3.md`](base_bakeoff_qwen3.md)
· [`ab_swaram_vs_bpe.md`](ab_swaram_vs_bpe.md) · [`bases_tokenizer_bench.md`](bases_tokenizer_bench.md)

---

## 1. Verdict first

1. **Agaram's tokenizer is the best licence-clean Tamil tokenizer we have measured**: **0.418 tok/akshara**,
   **4.85× cheaper than Qwen3-1.7B** and better than Sarvam-1 (0.584) and Swaram v3 (0.582) on the same sample.
2. **Agaram's quality does not justify a base change.** On the same tamil-bench prompts as the Qwen3 smoke it
   scored **XNLI 12.0%** (Qwen3 42.0%, chance 33.3%) and **IndicQA EM 0.000 / F1 0.143** (Qwen3 0.140 / 0.192).
3. **So `D1` stands.** Base decision §4 asked for "a permissively licensed 2B Indic model with Sarvam-class
   fertility". Agaram **fires the fertility half of that trigger and fails the quality half.** Record it as
   *partially fired*, not as a reason to revisit the base.
4. **The lesson to take is the recipe, not the weights.** A 64k byte-level BPE trained on a 58%-Tamil diet reaches
   Sarvam-class fertility — that weakens the case for Swaram-as-product further, and it tells us what to measure
   for Kutty v2.

## 2. What Agaram is

| | |
|---|---|
| Params | 2,525,757,696 — 28 layers, hidden 2,304, 36 heads, MLP 9,216, tied embeddings |
| Architecture | dense pre-norm Transformer, RMSNorm, SwiGLU, RoPE base 500,000; loads as `LlamaForCausalLM` |
| Trained | **from scratch**, no Llama/Qwen/Gemma/Sarvam weights (their claim, stated explicitly) |
| Data | ~50B tokens: Tamil 58.2%, English 19.8%, math 10%, code 7%, parallel 2%, instruction 3% |
| Dedup / leak control | MinHash 87.7M → 61.3M docs; **42,001 documents overlapping 20 benchmark/eval sets removed** |
| Hardware | 8× H100 80GB, 6 days pretrain; SFT 1× H100 |
| Tokenizer | byte-level BPE, **64,000 tokens, trained for this model**; card claims **1.55 tok/Tamil word** vs Qwen2.5's 9.44 |
| Chat format | ChatML (`<|im_start|>`/`<|im_end|>`, stop id 2), no system prompt used in training, greedy decoding |
| Licence | Apache-2.0 (weights + GGUF) |
| Predecessor | Agaram 1.5B; a 4-day public preview of an earlier 2.5B chat model drove a full SFT rebuild (222,662 records) |

**Their own scorecard** (lm-eval, same tasks/shots for every model): strong on translation (FLORES en→ta chrF 56.9,
ta→en 55.4, above Sarvam-1 and above Gemma 4 E4B-it) and Tamil grammar (MultiBLiMP 98.7), **at or near chance on
reading comprehension and knowledge** (Belebele-ta 28.4, MMLU-ta 25.0, INCLUDE 26.9, ARC 22.3), and they say so in
the card. Tool calling: expected tool on 25/25 prompts, exact arguments on 16/25.

## 3. Tokenizer measured here (new, reproducible)

Script: [`scripts/benchmark_bases_tokenizers.py`](../../scripts/benchmark_bases_tokenizers.py) ·
raw: [`bases_tokenizer_bench.json`](bases_tokenizer_bench.json) · table: [`bases_tokenizer_bench.md`](bases_tokenizer_bench.md)

Sample: 300 docs from `data/raw/expanded/corpus-00001.jsonl`, chosen by `sha256(seed|id|text)` ranking
(seed `adhan-bases-2026-11-05`) → **59,699 chars · 38,991 aksharas**. Harness: `src/adhan_slm/eval/tokenizer_bench.py`.

| Tokenizer | tok/word | tok/akshara | Tamil clusters split | tok/1k sent. | vs Qwen3 |
|---|---:|---:|---:|---:|---:|
| **Agaram 2.5B (64k byte BPE)** | **1.88** | **0.418** | 8.5% | 44,673 | **0.21×** |
| Qwen3-1.7B | 9.13 | 2.027 | 89.5% | 216,777 | 1.00× |
| Sarvam-1 | 2.63 | 0.584 | 7.7% | 62,423 | 0.29× |
| Sarvam-M | 3.58 | 0.795 | 31.6% | 85,047 | 0.39× |
| Swaram v3 (research) | 2.62 | 0.582 | 0.0% | 62,217 | 0.29× |
| Swaram v2 (research) | 2.67 | 0.593 | 0.3% | 63,377 | 0.29× |

Cross-check with the 3 Nov numbers: this sample reproduces them closely (Qwen3 2.027 vs 2.016; Sarvam-1 0.584 vs
0.538), so the sample is comparable even though the earlier 300-doc/35,986-akshara rule had no committed script.
**Fix on record:** the `CURRENT` figure for Qwen3 fertility now cites a rule that a stranger can rerun.

Three readings:

- **The token tax is real and larger than first recorded.** Qwen3 still spends **4.85×** Agaram's tokens and splits
  **89.5%** of Tamil aksharas, versus 7.7–8.5% for the Tamil-trained tokenizers.
- **Byte-level BPE is not the villain the Swaram paper assumed.** `ab_swaram_vs_bpe` trained its byte-BPE control on
  a 32k vocab and got 1.315. Agaram, byte-level, 64k, on a Tamil-heavy diet: 0.418. So "byte-level BPE cannot do
  Tamil" is false; "byte-level BPE trained on English and reused for Tamil" is the actual problem.
- **Swaram's compression thesis does not revive.** Swaram still matches Sarvam-1 (0.582 vs 0.584) and loses to
  Agaram's plain BPE (0.418). Its remaining assets are the 0.0% cluster-split rate and morphology, not fertility.

## 4. Quality measured here (smoke, same prompts as the Qwen3 smoke)

tamil-bench (`bench/tamil-bench`), local Ollama, greedy, `agaram-2.5b-instruct-Q4_K_M`, ChatML via our Modelfile:

| Task | n | Agaram 2.5B | Qwen3-1.7B (3 Nov) |
|---|---:|---|---|
| IndicXNLI-Tamil | 50 | **12.0%** acc | 42.0% |
| IndicQA-Tamil | 50 | **EM 0.000 · F1 0.143**, 0 errors | EM 0.140 · F1 0.192 |

Raw (copied out of the git-ignored `bench/` tree): [`agaram_xnli_n50.jsonl`](agaram_xnli_n50.jsonl),
[`agaram_indicqa_n50.jsonl`](agaram_indicqa_n50.jsonl).

**How to read it honestly:** XNLI is a composite of knowledge *and* instruction-following. Agaram answers by
enumerating the options and stops mid-list (mean 36 completion tokens, 18/50 with no parseable letter, 27/50
"answer A"): a format-compliance failure, not a demonstrated absence of NLI ability. It is a chat model, so a
letter-only prompt is out of distribution. That cuts both ways — a model that cannot follow "reply with one letter"
on a 1.7B-class phone target is a product problem, not a benchmark artefact. **Treat 12.0% as an upper-bounded
failure, not a measurement of Tamil reasoning**, and do not put it in a public table without the caveat.

Caveats on the whole comparison: Q4_K_M GGUF on CPU, not bf16; Agaram is 2.5B vs Qwen3's 1.7B; n=50; both models
are untuned base/chat, so this says nothing about post-SFT K2.

## 5. What to copy, what to leave

**Copy (cheap, no GPU):**

1. **Eval-leak removal before pretraining** — they subtracted 42,001 docs overlapping 20 benchmarks. Our SFT v0
   already keeps zero poem overlap; do the same for any pretraining/continued-pretraining mix, and record the count.
2. **Publish the tokenizer rule, not the number** — their 1.55 tok/word claim is only checkable because the
   tokenizer ships. Our 2.016 had no script; that is now fixed.
3. **Admit the weak rows** — the card's "at or near chance on Belebele/MMLU" is the kind of line `CURRENT` should
   be able to make about Kutty without a fight.
4. **Preview-then-rebuild SFT** — a short public preview drove their SFT redesign. For a 26 Dec launch, a 3-day
   preview around 10 Dec is the cheapest way to find out what SFT rows are missing.
5. **Tool-call format in training** — their 25/25 expected-tool result at 2.5B says a small model can carry tool
   calls in ChatML if the schemas appear in SFT. Relevant to `yazhi-api` agents, not to Kutty's first ship.

**Leave:**

- **The weights as the base.** Worse than Qwen3-1.7B on our smoke, 47% more parameters, and the phone budget
  (Q4 1.56 GB vs 1.4 GB) is worse with no quality to show for it.
- **Agaram as teacher.** Distilling from a model below the student's quality is a downgrade; Sarvam-M stays teacher.
- **Swaram-as-product framing.** Agaram's BPE is the counter-example; keep Swaram as the paper.

## 6. What this changes in the factory

- **`CURRENT` §3 fertility row** — re-source Qwen3 2.016 to `docs/reports/bases_tokenizer_bench.json` (rule +
  sample size in the row), and add an **Agaram 0.418** reference row so the tax has a licence-clean floor.
- **`CURRENT` §3 new row** — "Best licence-clean Tamil tokenizer: Agaram 2.5B 0.418 tok/akshara (Apache-2.0)",
  which retires part of `R2`: the tax is not a licence dead end, a permissive tokenizer of this class exists.
- **`base_decision` §4 trigger** — annotate: *"A permissively licensed 2B Indic model with Sarvam-class fertility
  appears"* → appeared 12 Oct (Agaram), fertility confirmed here, **quality fails**, trigger closed for v1.
- **Not changed:** `D1` (Qwen3-1.7B-Base), `D3` (Sarvam-M teacher), K1 scope. No GPU was consumed for this note.

## 7. Reproduce

```bash
# tokenizer table (needs local tokenizer.json files; no network)
AGARAM_TOKENIZER=/path/to/agaram/tokenizer.json \
  .venv/bin/python scripts/benchmark_bases_tokenizers.py \
  --out-json docs/reports/bases_tokenizer_bench.json \
  --out-md docs/reports/bases_tokenizer_bench.md

# quality smoke (Ollama serving agaram as `agaram-tamil`)
cd bench/tamil-bench
TAMIL_BENCH_ENDPOINT=http://127.0.0.1:11434/v1/chat/completions \
  python3 bench.py xnli --model agaram-tamil --n 50
TAMIL_BENCH_ENDPOINT=http://127.0.0.1:11434/v1/chat/completions \
  python3 bench.py indicqa --model agaram-tamil --n 50
```
