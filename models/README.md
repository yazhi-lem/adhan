# `models/` — what's here and what isn't

This directory contains **scaffolding only** for some model artifacts —
configs, tokenizers, adapter metadata, chat templates — but **not trained
weight files** (`*.safetensors`, `*.bin`, etc.). Weights are intentionally
excluded from git (see `.gitignore`) because they're large binary build
artifacts, not source.

If you clone this repo and expect `models/adhan-gemma-v1/merged/` (or
similar) to be a ready-to-load model, it isn't — loading it with
`transformers.AutoModelForCausalLM.from_pretrained(...)` or via the
`Modelfile` will fail with a missing-weights error. You need to generate
the weights yourself first.

## Directory guide

| Path | What's checked in | What's missing | How to produce it |
|---|---|---|---|
| `models/adhan-gemma-v1/lora_adapter/` | `adapter_config.json`, README | `adapter_model.safetensors` | Run the LoRA training pipeline: `src/gpu_lora_train.py` (GPU) — see `DEV.md`/`docs/TRAINING_GUIDE.md` |
| `models/adhan-gemma-v1/merged/` | `config.json`, `tokenizer.json`, `generation_config.json`, `chat_template.jinja` | `model.safetensors` (merged base + adapter weights) | `scripts/merge_and_export.py`, which reads the adapter above and writes the merged model here |
| `models/aadhan-mlm-v1/` | `config.json`, `tokenizer.json`, `training_results.json` | model weights | See `src/notebooks/` MLM training notebook referenced in `training_results.json` |
| `models/tokenized_datasets/` | Pre-tokenized `train`/`val`/`test` splits (Arrow format) | — (these are complete, small, and safe to commit) | N/A — already usable as-is |

## Why keep the configs/tokenizers checked in at all?

They act as a template — you can inspect the expected model config,
tokenizer, and adapter shape without downloading multi-GB weight files,
and scripts like `merge_and_export.py` / `gpu_lora_train.py` write into
these exact paths, so the directory structure documents the pipeline's
expected output layout.

## Note on `models/checkpoints/`

An orphaned, unreferenced checkpoint directory
(`models/checkpoints/gemma/checkpoint-21/`) was previously committed by
accident — it wasn't produced or consumed by any script in this repo, and
slipped past `.gitignore` because the original glob only matched one
directory level. It's been removed; the corrected glob
(`models/**/checkpoint-*/`) now excludes checkpoint directories at any
depth under `models/`.
