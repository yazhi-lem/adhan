#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tamil Sentiment Analysis Fine-tuning Script

Fine-tunes a pre-trained Tamil (or multilingual) encoder model for
sequence classification on a binary or multi-class sentiment dataset.

Supported label schemas:
  - Binary (2 labels):  0=negative, 1=positive
  - Ternary (3 labels): 0=negative, 1=neutral, 2=positive

Input data format (JSONL, one record per line):
    {"text": "...", "label": 1}
  or
    {"text": "...", "sentiment": "positive"}   (use --label-column sentiment)

Usage:
    python src/models/sentiment/train_sentiment.py \
        --train-file  data/sentiment/train.jsonl \
        --val-file    data/sentiment/val.jsonl \
        --model-name  xlm-roberta-base \
        --output-dir  models/adhan_sentiment \
        --num-epochs  5

Outputs:
    models/adhan_sentiment/          – best checkpoint (HuggingFace format)
    models/adhan_sentiment/metrics.json – val/test evaluation metrics
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# ── config ─────────────────────────────────────────────────────────────────────

@dataclass
class SentimentConfig:
    # Model
    model_name: str = "xlm-roberta-base"
    num_labels: int = 2

    # Data
    train_file: str = "data/sentiment/train.jsonl"
    val_file: str = "data/sentiment/val.jsonl"
    test_file: Optional[str] = None
    text_column: str = "text"
    label_column: str = "label"
    max_length: int = 128

    # Training
    output_dir: str = "models/adhan_sentiment"
    num_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2
    logging_steps: int = 50
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"

    # Hardware
    use_fp16: bool = True
    use_cuda: bool = True
    seed: int = 42


# ── label helpers ──────────────────────────────────────────────────────────────

LABEL_MAP_STR = {
    "negative": 0, "neg": 0, "bad": 0,
    "neutral": 1, "neu": 1,
    "positive": 2, "pos": 2, "good": 2,
}

LABEL_MAP_STR_BINARY = {
    "negative": 0, "neg": 0, "bad": 0,
    "positive": 1, "pos": 1, "good": 1,
}


def normalise_label(raw, num_labels: int) -> int:
    if isinstance(raw, int):
        return int(raw)
    raw_lower = str(raw).strip().lower()
    if num_labels == 2:
        return LABEL_MAP_STR_BINARY.get(raw_lower, int(raw_lower))
    return LABEL_MAP_STR.get(raw_lower, int(raw_lower))


# ── dataset loader ─────────────────────────────────────────────────────────────

def load_jsonl(path: str, text_col: str, label_col: str,
               num_labels: int) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    skipped = 0
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            text = obj.get(text_col, "").strip()
            raw_label = obj.get(label_col)
            if not text or raw_label is None:
                skipped += 1
                continue
            try:
                label = normalise_label(raw_label, num_labels)
            except (ValueError, KeyError):
                skipped += 1
                continue
            texts.append(text)
            labels.append(label)
    if skipped:
        logger.warning("Skipped %d malformed records in %s", skipped, path)
    return texts, labels


# ── metrics ────────────────────────────────────────────────────────────────────

def build_compute_metrics(num_labels: int):
    try:
        from sklearn.metrics import (  # type: ignore
            accuracy_score, f1_score, classification_report,
        )
        sklearn_available = True
    except ImportError:
        sklearn_available = False

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == labels).mean())
        result = {"accuracy": round(acc, 4)}
        if sklearn_available:
            avg = "binary" if num_labels == 2 else "macro"
            f1 = f1_score(labels, preds, average=avg, zero_division=0)
            result["f1"] = round(float(f1), 4)
        else:
            result["f1"] = result["accuracy"]
        return result

    return compute_metrics


# ── main training ──────────────────────────────────────────────────────────────

def train(cfg: SentimentConfig) -> None:
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            Trainer,
            TrainingArguments,
            EarlyStoppingCallback,
        )
        from torch.utils.data import Dataset as TorchDataset
    except ImportError as exc:
        logger.error("Required dependency missing: %s. Run: pip install transformers torch", exc)
        raise

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if cfg.use_cuda and torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # ── tokenizer & model ──────────────────────────────────────────────────────
    logger.info("Loading tokenizer from '%s' …", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    logger.info("Loading model '%s' (num_labels=%d) …", cfg.model_name, cfg.num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
    )

    # ── data ───────────────────────────────────────────────────────────────────

    class SentimentDataset(TorchDataset):
        def __init__(self, texts: list[str], labels: list[int]) -> None:
            encodings = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=cfg.max_length,
                return_tensors="pt",
            )
            self.input_ids = encodings["input_ids"]
            self.attention_mask = encodings["attention_mask"]
            self.token_type_ids = encodings.get("token_type_ids")
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict:
            item = {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx],
            }
            if self.token_type_ids is not None:
                item["token_type_ids"] = self.token_type_ids[idx]
            return item

    logger.info("Loading training data from %s …", cfg.train_file)
    train_texts, train_labels = load_jsonl(
        cfg.train_file, cfg.text_column, cfg.label_column, cfg.num_labels)
    logger.info("  %d training samples", len(train_texts))

    logger.info("Loading validation data from %s …", cfg.val_file)
    val_texts, val_labels = load_jsonl(
        cfg.val_file, cfg.text_column, cfg.label_column, cfg.num_labels)
    logger.info("  %d validation samples", len(val_texts))

    train_dataset = SentimentDataset(train_texts, train_labels)
    val_dataset = SentimentDataset(val_texts, val_labels)

    # ── training args ──────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=True,
        fp16=cfg.use_fp16 and device == "cuda",
        seed=cfg.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=build_compute_metrics(cfg.num_labels),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Starting training …")
    trainer.train()

    logger.info("Saving best model to %s …", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # ── evaluation ─────────────────────────────────────────────────────────────
    val_metrics = trainer.evaluate(val_dataset, metric_key_prefix="val")
    logger.info("Validation metrics: %s", val_metrics)

    all_metrics: dict = {"val": val_metrics, "config": asdict(cfg)}

    if cfg.test_file and Path(cfg.test_file).exists():
        test_texts, test_labels = load_jsonl(
            cfg.test_file, cfg.text_column, cfg.label_column, cfg.num_labels)
        test_dataset = SentimentDataset(test_texts, test_labels)
        test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
        logger.info("Test metrics: %s", test_metrics)
        all_metrics["test"] = test_metrics

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as mf:
        json.dump(all_metrics, mf, indent=2)
    logger.info("Metrics saved to %s", metrics_path)


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Tamil sentiment classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-file", default="data/sentiment/train.jsonl")
    parser.add_argument("--val-file",   default="data/sentiment/val.jsonl")
    parser.add_argument("--test-file",  default=None)
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--output-dir", default="models/adhan_sentiment")
    parser.add_argument("--num-labels", type=int, default=2,
                        help="2 for binary (neg/pos), 3 for ternary (neg/neu/pos).")
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = SentimentConfig(
        model_name=args.model_name,
        num_labels=args.num_labels,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        text_column=args.text_column,
        label_column=args.label_column,
        max_length=args.max_length,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_fp16=not args.no_fp16,
        seed=args.seed,
    )

    train(cfg)


if __name__ == "__main__":
    main()
