#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SangamGPT Training Script for AADHAN
"""

import os
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    pipeline
)
from datasets import load_dataset
from tqdm import tqdm
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TamilTextDataset:
    """Custom dataset for Tamil text processing"""
    
    def __init__(self, file_path, tokenizer, block_size=512):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.texts = self._load_texts(file_path)
    
    def _load_texts(self, file_path):
        """Load and preprocess Tamil text data"""
        logger.info(f"Loading Tamil text from {file_path}")
        texts = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and len(line) > 10:  # Filter short lines
                        texts.append(line)
        except FileNotFoundError:
            logger.warning(f"File {file_path} not found. Using sample data.")
            # Use sample Tamil text
            texts = [
                "தமிழ் வெர்வன் குகவை காரண் டுகக்கால்",
                "சங்கக்கால் சங்கக்கால் சங்கக்கால்",
                "சங்கக்கால் சங்கக்கால் சங்கக்கால்"
            ]
        
        logger.info(f"Loaded {len(texts)} Tamil text samples")
        return texts
    
    def tokenize(self, text):
        """Tokenize Tamil text"""
        return self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=self.block_size,
            truncation=True
        )
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenize(self.texts[idx])
        return {
            "input_ids": torch.tensor(encoding, dtype=torch.long),
            "attention_mask": torch.ones(len(encoding), dtype=torch.long)
        }

class TamilSangamGPT:
    """SangamGPT model for Tamil text generation"""
    
    def __init__(self, model_name="sangam/IndianLanguages-Tamil-BERT-v0.1", output_dir="sangam_gpt_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def setup_tokenizer(self):
        """Setup Tamil tokenizer"""
        logger.info("Setting up Tamil tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
            local_files_only=True
        )
        self.tokenizer.add_special_tokens({"bos_token": "க", "eos_token": "க"})
        logger.info("Tamil tokenizer setup complete")
    
    def setup_model(self):
        """Setup base model"""
        logger.info("Setting up base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            local_files_only=True
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        logger.info("Base model setup complete")
    
    def prepare_dataset(self, data_path):
        """Prepare Tamil text dataset"""
        logger.info("Preparing Tamil dataset...")
        
        # Create dataset
        train_dataset = TamilTextDataset(data_path, self.tokenizer)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt"
        )
        
        return train_dataset, data_collator
    
    def setup_training(self, train_dataset, data_collator):
        """Setup training arguments and trainer"""
        logger.info("Setting up training configuration...")
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            save_steps=500,
            save_total_limit=2,
            logging_steps=100,
            learning_rate=5e-5,
            warmup_steps=500,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=2,
            report_to="none"
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            prediction_loss_only=True
        )
        
        logger.info("Training setup complete")
    
    def train(self, data_path):
        """Train the SangamGPT model"""
        logger.info("Starting SangamGPT training...")
        
        # Setup components
        self.setup_tokenizer()
        self.setup_model()
        
        # Prepare dataset
        train_dataset, data_collator = self.prepare_dataset(data_path)
        
        # Setup training
        self.setup_training(train_dataset, data_collator)
        
        # Start training
        logger.info("Starting training...")
        self.trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Training complete! Model saved to {self.output_dir}")
        
        return self.output_dir
    
    def generate_text(self, prompt, max_length=100, num_return_sequences=1):
        """Generate Tamil text"""
        if not self.tokenizer or not self.model:
            raise ValueError("Model not initialized. Call setup_tokenizer() and setup_model() first.")
        
        # Create pipeline
        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device.index if self.device.type == "cuda" else -1
        )
        
        # Generate text
        results = generator(
            prompt,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=2
        )
        
        return results

# Command line interface
def main():
    parser = argparse.ArgumentParser(description="Train SangamGPT - Tamil Language Model")
    parser.add_argument("--data_path", type=str, default="data/raw/tamil_texts.txt",
                       help="Path to Tamil text data")
    parser.add_argument("--output_dir", type=str, default="models/sangam_gpt",
                       help="Output directory for trained model")
    parser.add_argument("--train", action="store_true", help="Start training")
    parser.add_argument("--generate", type=str, help="Generate text from prompt")
    parser.add_argument("--max_length", type=int, default=100, help="Max length for generation")
    
    args = parser.parse_args()
    
    # Initialize model
    model = TamilSangamGPT(output_dir=args.output_dir)
    
    if args.train:
        # Train the model
        model.train(args.data_path)
    elif args.generate:
        # Generate text
        model.setup_tokenizer()
        model.setup_model()
        results = model.generate_text(
            args.generate,
            max_length=args.max_length,
            num_return_sequences=1
        )
        
        print("\nGenerated Tamil Text:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['generated_text']}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()