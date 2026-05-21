"""
merge_and_export.py – Merge LoRA adapter into Gemma base model
===============================================================
Loads the base model + LoRA adapter, merges weights, and saves the
full merged model ready for GGUF conversion / Ollama import.

Usage:
    python scripts/merge_and_export.py
"""

import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.environ.get(
    "ADHAN_PROJECT_ROOT",
    Path(__file__).resolve().parent.parent,
))
MODELS_DIR   = PROJECT_ROOT / "models" / "adhan-gemma-v1"
ADAPTER_DIR  = MODELS_DIR / "lora_adapter"
MERGED_DIR   = MODELS_DIR / "merged"
BASE_MODEL   = "google/gemma-3-1b-it"

MERGED_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load base model ──────────────────────────────────────────────────────
print(f"Loading base model: {BASE_MODEL} ...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

# ── 2. Apply LoRA adapter ───────────────────────────────────────────────────
print(f"Loading LoRA adapter from {ADAPTER_DIR} ...")
model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))

# ── 3. Merge & unload ───────────────────────────────────────────────────────
print("Merging LoRA weights into base model ...")
merged = model.merge_and_unload()

# ── 4. Save ──────────────────────────────────────────────────────────────────
print(f"Saving merged model to {MERGED_DIR} ...")
merged.save_pretrained(str(MERGED_DIR))

print(f"Saving tokenizer to {MERGED_DIR} ...")
tokenizer = AutoTokenizer.from_pretrained(str(MODELS_DIR))
tokenizer.save_pretrained(str(MERGED_DIR))

# Verify
expected = ["config.json", "tokenizer.json", "tokenizer_config.json"]
for f in expected:
    path = MERGED_DIR / f
    if path.exists():
        print(f"  ✅ {f}")
    else:
        print(f"  ❌ {f} MISSING")

safetensors = list(MERGED_DIR.glob("*.safetensors"))
if safetensors:
    total_mb = sum(f.stat().st_size for f in safetensors) / 1e6
    print(f"  ✅ {len(safetensors)} safetensors file(s) ({total_mb:.0f} MB)")
else:
    print("  ❌ No safetensors files found!")

print("\n✅ Merge complete! Ready for Ollama import.")
