#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Secure Tamil Model Trainer

Simplified architecture (< 150 lines) with security focus:
- Input validation
- Resource limits
- Secure data loading
- Performance optimizations
"""
import json
import logging
from pathlib import Path
from typing import Dict
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SecureTamilTrainer:
    """Minimal, secure trainer for Tamil models"""
    
    # Security: Resource limits
    MAX_LENGTH = 512
    MAX_EPOCHS = 10
    MAX_BATCH_SIZE = 32
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._validate_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _load_config(self, path: str) -> Dict:
        """Security: Safely load and validate config"""
        config_path = Path(path)
        
        # Security: Validate file extension
        if config_path.suffix not in ['.json', '.yaml', '.yml']:
            raise ValueError("Config must be JSON or YAML")
        
        # Security: Check file size (max 1MB)
        if config_path.stat().st_size > 1024 * 1024:
            raise ValueError("Config file too large")
        
        with config_path.open('r') as f:
            if path.endswith('.json'):
                return json.load(f)
            else:
                import yaml
                return yaml.safe_load(f)
    
    def _validate_config(self):
        """Security: Validate all config parameters"""
        # Validate epochs
        epochs = self.config.get('num_epochs', 3)
        if not (1 <= epochs <= self.MAX_EPOCHS):
            raise ValueError(f"num_epochs must be 1-{self.MAX_EPOCHS}")
        self.config['num_epochs'] = epochs
        
        # Validate batch size
        batch = self.config.get('batch_size', 4)
        if not (1 <= batch <= self.MAX_BATCH_SIZE):
            raise ValueError(f"batch_size must be 1-{self.MAX_BATCH_SIZE}")
        self.config['batch_size'] = batch
        
        # Validate paths
        data_dir = Path(self.config.get('data_dir', 'data/pre_training/tamil_texts/hf'))
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    def train(self) -> Dict:
        """Train model with security and performance optimizations"""
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.get('model_name', 'distilgpt2')  # Smaller default model
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.get('model_name', 'distilgpt2')
        ).to(self.device)
        
        logger.info("Loading dataset...")
        dataset = load_dataset('json', data_files={
            'train': str(Path(self.config['data_dir']) / 'train.jsonl'),
            'validation': str(Path(self.config['data_dir']) / 'validation.jsonl')
        })
        
        # Security: Limit dataset size
        max_samples = 10000
        if len(dataset['train']) > max_samples:
            logger.warning(f"Train dataset truncated to {max_samples} samples")
            dataset['train'] = dataset['train'].select(range(max_samples))
        
        # Tokenize
        def tokenize(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                max_length=self.MAX_LENGTH,
                padding='max_length'
            )
        
        tokenized = dataset.map(tokenize, batched=True, remove_columns=['text'])
        
        # Training args with security limits
        training_args = TrainingArguments(
            output_dir=self.config.get('output_dir', 'models/tamil_model'),
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            learning_rate=float(self.config.get('learning_rate', 5e-5)),
            warmup_steps=min(self.config.get('warmup_steps', 500), 1000),  # Security: Cap warmup
            save_steps=1000,
            eval_steps=500,
            logging_steps=100,
            evaluation_strategy="steps",
            save_total_limit=2,  # Security: Limit checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=torch.cuda.is_available(),
            report_to="none",  # Security: No external reporting by default
            dataloader_num_workers=min(self.config.get('num_workers', 4), 4)  # Security: Cap workers
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized['train'],
            eval_dataset=tokenized['validation'],
            data_collator=data_collator
        )
        
        logger.info("Starting training...")
        result = trainer.train()
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(self.config.get('output_dir', 'models/tamil_model'))
        
        logger.info(f"Training complete. Loss: {result.training_loss:.4f}")
        return result.metrics


def main():
    """Simple CLI with security validation"""
    import argparse
    parser = argparse.ArgumentParser(description='Minimal Secure Tamil Trainer')
    parser.add_argument('--config', required=True, help='Config file (JSON/YAML)')
    args = parser.parse_args()
    
    # Security: Validate config path
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if config_path.is_symlink():
        raise ValueError("Symlinks not allowed")
    
    trainer = SecureTamilTrainer(str(config_path))
    trainer.train()


if __name__ == '__main__':
    main()
