#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Tamil Model Training Script with Data Pipeline Integration

Key improvements:
- Configuration file support (YAML/JSON)
- Integration with data pipeline
- Proper evaluation metrics
- Flexible model loading (no local_files_only requirement)
- Support for HuggingFace datasets
- Weights & Biases integration (optional)
- Better logging and progress tracking
"""

import os
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from datasets import load_dataset, DatasetDict
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model training"""
    # Model settings
    model_name: str = "sangam/IndianLanguages-Tamil-BERT-v0.1"
    model_type: str = "causal_lm"  # causal_lm or masked_lm
    vocab_size: Optional[int] = None
    max_length: int = 512
    
    # Data settings
    data_dir: str = "data/pre_training/tamil_texts/hf"
    train_file: str = "train.jsonl"
    val_file: str = "validation.jsonl"
    test_file: str = "test.jsonl"
    text_column: str = "text"
    
    # Training settings
    output_dir: str = "models/tamil_model"
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    
    # Evaluation settings
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 100
    
    # Hardware settings
    use_fp16: bool = True
    use_cuda: bool = True
    num_workers: int = 4
    
    # Optional integrations
    use_wandb: bool = False
    wandb_project: str = "tamil-llm"
    wandb_run_name: Optional[str] = None
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    
    def save(self, path: str):
        """Save configuration to file"""
        config_path = Path(path)
        with config_path.open('w') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                yaml.dump(asdict(self), f)
            else:
                json.dump(asdict(self), f, indent=2)
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """Load configuration from file"""
        config_path = Path(path)
        with config_path.open('r') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        logger.info(f"Configuration loaded from {path}")
        return cls(**data)


class TamilModelTrainer:
    """Enhanced trainer for Tamil language models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = self._get_device()
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.dataset = None
        
        # Setup directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = Path(config.output_dir) / "config.json"
        config.save(str(config_path))
    
    def _get_device(self) -> torch.device:
        """Get the appropriate device for training"""
        if self.config.use_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        return device
    
    def setup_tokenizer(self):
        """Setup tokenizer with fallback handling"""
        logger.info(f"Loading tokenizer from {self.config.model_name}")
        
        try:
            # Try loading from local/cache first
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=True,
            )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer locally: {e}")
            logger.info("Attempting to download tokenizer...")
            
            # Fallback to download
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=True,
                force_download=True
            )
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
    
    def setup_model(self):
        """Setup model with flexible loading"""
        logger.info(f"Loading model from {self.config.model_name}")
        
        try:
            # Load model
            if self.config.model_type == "causal_lm":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name
                )
            else:
                # For masked LM or other types
                from transformers import AutoModelForMaskedLM
                self.model = AutoModelForMaskedLM.from_pretrained(
                    self.config.model_name
                )
            
            # Resize embeddings if vocab size changed
            if self.config.vocab_size and len(self.tokenizer) != self.model.config.vocab_size:
                logger.info(f"Resizing embeddings from {self.model.config.vocab_size} to {len(self.tokenizer)}")
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            self.model.to(self.device)
            
            # Print model info
            num_params = sum(p.numel() for p in self.model.parameters())
            num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"Model loaded: {num_params:,} total parameters, {num_trainable:,} trainable")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_dataset(self) -> DatasetDict:
        """Load dataset from HuggingFace format"""
        logger.info(f"Loading dataset from {self.config.data_dir}")
        
        data_dir = Path(self.config.data_dir)
        
        # Check if files exist
        train_path = data_dir / self.config.train_file
        val_path = data_dir / self.config.val_file
        test_path = data_dir / self.config.test_file
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        # Load dataset
        data_files = {}
        if train_path.exists():
            data_files['train'] = str(train_path)
        if val_path.exists():
            data_files['validation'] = str(val_path)
        if test_path.exists():
            data_files['test'] = str(test_path)
        
        dataset = load_dataset('json', data_files=data_files)
        
        logger.info(f"Dataset loaded:")
        for split, data in dataset.items():
            logger.info(f"  {split}: {len(data)} examples")
        
        return dataset
    
    def tokenize_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Tokenize the dataset"""
        logger.info("Tokenizing dataset...")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples[self.config.text_column],
                padding='max_length',
                truncation=True,
                max_length=self.config.max_length,
                return_special_tokens_mask=True
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.config.num_workers,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing dataset"
        )
        
        logger.info("Tokenization complete")
        return tokenized_dataset
    
    def setup_training(self, train_dataset, eval_dataset=None):
        """Setup training arguments and trainer"""
        logger.info("Setting up training configuration...")
        
        # Setup Weights & Biases if enabled
        report_to = []
        if self.config.use_wandb:
            try:
                import wandb
                report_to = ["wandb"]
                if self.config.wandb_run_name:
                    os.environ["WANDB_NAME"] = self.config.wandb_run_name
                logger.info("Weights & Biases integration enabled")
            except ImportError:
                logger.warning("wandb not installed. Disabling W&B integration.")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            evaluation_strategy="steps" if eval_dataset else "no",
            load_best_model_at_end=True if eval_dataset and self.config.early_stopping else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            fp16=self.config.use_fp16 and torch.cuda.is_available(),
            dataloader_num_workers=self.config.num_workers,
            report_to=report_to,
            push_to_hub=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False if self.config.model_type == "causal_lm" else True,
            mlm_probability=0.15 if self.config.model_type != "causal_lm" else None,
        )
        
        # Callbacks
        callbacks = []
        if eval_dataset and self.config.early_stopping:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                )
            )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        
        logger.info("Training setup complete")
    
    def train(self):
        """Train the model"""
        logger.info("Starting model training...")
        
        # Setup components
        logger.info("Step 1/5: Setting up tokenizer...")
        self.setup_tokenizer()
        
        logger.info("Step 2/5: Setting up model...")
        self.setup_model()
        
        logger.info("Step 3/5: Loading dataset...")
        dataset = self.load_dataset()
        
        logger.info("Step 4/5: Tokenizing dataset...")
        tokenized_dataset = self.tokenize_dataset(dataset)
        
        # Setup training
        train_dataset = tokenized_dataset["train"]
        eval_dataset = tokenized_dataset.get("validation")
        
        logger.info("Step 5/5: Setting up trainer...")
        self.setup_training(train_dataset, eval_dataset)
        
        # Train
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        logger.info(f"Training complete! Model saved to {self.config.output_dir}")
        logger.info(f"Final training loss: {metrics.get('train_loss', 'N/A')}")
        
        return train_result
    
    def evaluate(self) -> Dict:
        """Evaluate the model"""
        if not self.trainer:
            raise ValueError("Trainer not initialized. Call train() first.")
        
        logger.info("Evaluating model...")
        
        # Evaluate
        eval_results = self.trainer.evaluate()
        
        # Log metrics
        self.trainer.log_metrics("eval", eval_results)
        self.trainer.save_metrics("eval", eval_results)
        
        logger.info(f"Evaluation complete. Loss: {eval_results.get('eval_loss', 'N/A')}")
        
        return eval_results


def main():
    parser = argparse.ArgumentParser(description="Enhanced Tamil Model Training")
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON/YAML)')
    parser.add_argument('--model-name', type=str, help='Model name/path')
    parser.add_argument('--data-dir', type=str, help='Data directory')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--num-epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--use-wandb', action='store_true', help='Enable Weights & Biases')
    parser.add_argument('--create-config', type=str, help='Create default config file')
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        config = ModelConfig()
        config.save(args.create_config)
        logger.info(f"Default configuration created at {args.create_config}")
        return
    
    # Load or create config
    if args.config:
        config = ModelConfig.load(args.config)
    else:
        config = ModelConfig()
    
    # Override with command-line arguments
    if args.model_name:
        config.model_name = args.model_name
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.use_wandb:
        config.use_wandb = True
    
    # Create trainer and train
    trainer = TamilModelTrainer(config)
    trainer.train()
    
    # Evaluate if validation set exists
    if (Path(config.data_dir) / config.val_file).exists():
        trainer.evaluate()


if __name__ == '__main__':
    main()
