#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpu_lora_train.py – GPU-optimised LoRA fine-tuning for Gemma 3 1B-it
=====================================================================

Optimized for Google Colab (T4 GPU) or any CUDA-enabled environment.
"""

import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.environ.get("ADHAN_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR    = PROJECT_ROOT / "data" / "final" / "tamil_texts" / "full_hf"
MODELS_DIR  = PROJECT_ROOT / "models" / "adhan-gemma-v1-gpu"
ADAPTER_DIR = MODELS_DIR / "lora_adapter"
CKPT_DIR    = PROJECT_ROOT / "models" / "checkpoints" / "gemma-gpu"

for d in [MODELS_DIR, ADAPTER_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── GPU-optimised hyperparameters ────────────────────────────────────────────
MODEL_NAME    = "google/gemma-3-1b-it"
MAX_LENGTH    = 512    # Higher context for better coherence
LORA_R        = 16     # Higher rank for better knowledge capture
LORA_ALPHA    = 32     # alpha = 2 * r
LORA_DROPOUT  = 0.1
EPOCHS        = 3      # Standard for instruction tuning
BATCH_SIZE    = 4      # Fits easily on T4 for 1B model
GRAD_ACCUM    = 4      # Effective batch = 16
LEARNING_RATE = 2e-4   # Standard for LoRA

print(f"PyTorch Version:       {torch.__version__}")
print(f"Transformers Version:  {transformers.__version__}")
print(f"CUDA Available:        {torch.cuda.is_available()}")
print(f"Project root:  {PROJECT_ROOT}")
print(f"Data dir:      {DATA_DIR}")
print(f"Output dir:    {MODELS_DIR}")

# ── 1. Data loading ──────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        print(f"⚠️ Warning: File not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

print("\nLoading JSONL splits...")
train_records = load_jsonl(DATA_DIR / "train.jsonl")
val_records   = load_jsonl(DATA_DIR / "validation.jsonl")
test_records  = load_jsonl(DATA_DIR / "test.jsonl")

print(f"  Train:      {len(train_records):,} records")
print(f"  Validation: {len(val_records):,} records")
print(f"  Test:       {len(test_records):,} records")

if not train_records:
    print("❌ No training data found. Check your DATA_DIR.")
    exit(1)

# ── 2. Format & tokenise ─────────────────────────────────────────────────────
def to_gemma_format(text: str) -> str:
    # We use a user/model turn format to simulate instruction data
    # even though our data is mostly sentences. 
    # This helps the model stay in character.
    return (
        "<start_of_turn>user\n"
        "தமிழ் உரையை தொடர்ந்து எழுதவும்:\n"
        f"{text}<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{text}<end_of_turn>"
    )

def records_to_dataset(records: list[dict]) -> Dataset:
    texts = [to_gemma_format(r["text"]) for r in records]
    return Dataset.from_dict({"text": texts})

raw_train = records_to_dataset(train_records)
raw_val   = records_to_dataset(val_records)
raw_test  = records_to_dataset(test_records)

print(f"\nLoading tokenizer for {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

print("Tokenising datasets...")
train_dataset = raw_train.map(tokenize, batched=True, remove_columns=["text"])
val_dataset   = raw_val.map(tokenize,   batched=True, remove_columns=["text"])
test_dataset  = raw_test.map(tokenize,  batched=True, remove_columns=["text"])

train_dataset = train_dataset.map(lambda ex: {"labels": ex["input_ids"]})
val_dataset   = val_dataset.map(lambda ex:   {"labels": ex["input_ids"]})
test_dataset  = test_dataset.map(lambda ex:  {"labels": ex["input_ids"]})

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(  type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format( type="torch", columns=["input_ids", "attention_mask", "labels"])

# ── 3. Load model ────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"\nLoading {MODEL_NAME} in {dtype} ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.config.use_cache = False

# ── 4. Apply LoRA ────────────────────────────────────────────────────────────
print("\nApplying LoRA adapter...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 5. Training ──────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=str(CKPT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    fp16=(device == "cuda"),
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

print(f"\nStarting GPU LoRA fine-tuning...")
train_result = trainer.train()

# ── 6. Save ──────────────────────────────────────────────────────────────────
print(f"\nSaving LoRA adapter to {ADAPTER_DIR} ...")
model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(MODELS_DIR))

print("\n✅ Training complete and model saved.")
