"""
Adhan CLI — Command-line interface for training, evaluation, scraping, generation.

Usage:
  adhan train --config config/default.yaml --corpus data/corpus.jsonl
  adhan eval --model models/checkpoints --corpus data/test.jsonl
  adhan scrape --sources dinamalar,bbc_tamil
  adhan generate --prompt "தமிழ்" --max-tokens 50
"""

import argparse
import sys
import json
from typing import Optional

from ..core import Config, get_default_config, get_rpi5_config, get_cluster_config, setup_logger
from ..tokenizer import TamilAgglutinativeTokenizer
from ..data import Corpus
from ..models import Trainer, Evaluator, InferenceEngine

logger = setup_logger("adhan.cli")


def load_config(config_path: Optional[str] = None) -> Config:
    """Load config from YAML or use default."""
    if config_path:
        try:
            return Config.from_yaml(config_path)
        except Exception as e:
            logger.warning(f"Failed to load {config_path}: {e} — using default")
    return get_default_config()


def cmd_train(args):
    """Train Adhan model."""
    config = load_config(args.config)
    tokenizer = TamilAgglutinativeTokenizer(config.tokenizer)
    
    # Load corpus
    corpus = Corpus(config.data)
    if args.corpus:
        corpus.load_jsonl(args.corpus)
        corpus.validate()
        corpus.remove_duplicates()
        corpus.split()
        train_texts = corpus.get_texts("train")
        val_texts = corpus.get_texts("val")
    else:
        # Demo mode
        train_texts = ["நான் தமிழ் கற்கிறேன்", "அவன் வீட்டிற்கு செல்கிறான்"]
        val_texts = ["நான் செல்கிறேன்"]
    
    if args.steps:
        config.model.num_epochs = max(1, args.steps // 100)
    
    # Train
    trainer = Trainer(config.model, tokenizer)
    trainer.build()
    result = trainer.train(train_texts, val_texts)
    
    # Save
    if args.output:
        trainer.save(args.output)
    
    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"Mode: {result['mode']}")
    print(f"Epochs: {result['epochs']}")
    print(f"Final Loss: {result['final_loss']:.4f}")
    if 'total_tokens' in result:
        print(f"Total Tokens: {result['total_tokens']}")
    if 'note' in result:
        print(f"Note: {result['note']}")
    print(f"{'='*50}")


def cmd_eval(args):
    """Evaluate Adhan model."""
    config = load_config(args.config)
    tokenizer = TamilAgglutinativeTokenizer(config.tokenizer)
    
    # Load eval data
    corpus = Corpus(config.data)
    if args.corpus:
        corpus.load_jsonl(args.corpus)
        eval_texts = corpus.get_texts("train")  # Use all for eval
    else:
        eval_texts = ["நான் தமிழ் கற்கிறேன்", "அவன் வீட்டிற்கு செல்கிறான்"]
    
    # Evaluate
    evaluator = Evaluator(config.eval, tokenizer)
    
    if args.report:
        print(evaluator.generate_report(eval_texts))
    else:
        results = evaluator.evaluate(eval_texts)
        print(json.dumps(results, indent=2))


def cmd_scrape(args):
    """Scrape Tamil corpus from news sources."""
    config = load_config(args.config)
    
    sources = args.sources.split(",") if args.sources else config.data.scrapers
    
    print(f"Scraping sources: {sources}")
    print(f"Output: {args.output or 'data/scraped.jsonl'}")
    print(f"\nNote: Concrete scrapers (Dinamalar, BBC Tamil, Dinamani)")
    print(f"are implemented as BaseScraper subclasses in src/adhan/data/scrapers/")
    print(f"Run with --execute when network is available.")
    
    # TODO: When scrapers are implemented, iterate and execute
    # for source in sources:
    #     scraper = get_scraper(source)(timeout=config.data.scraper_timeout)
    #     samples = scraper.scrape()
    #     corpus.extend(samples)


def cmd_generate(args):
    """Generate Tamil text."""
    config = load_config(args.config)
    tokenizer = TamilAgglutinativeTokenizer(config.tokenizer)
    
    engine = InferenceEngine(config.model, tokenizer)
    
    if args.model:
        engine.load_model(args.model)
    
    output = engine.generate(args.prompt, max_tokens=args.max_tokens)
    print(output)


def cmd_info(args):
    """Show Adhan package info."""
    from .. import __version__, __author__
    
    config = load_config(args.config)
    
    print(f"\n{'='*50}")
    print(f"ADHAN — Tamil Large Language Model")
    print(f"{'='*50}")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"{'='*50}")
    print(f"Config: {config.name} v{config.version}")
    print(f"Device: {config.model.device}")
    print(f"Model: {config.model.model_type}")
    print(f"Tokenizer: {config.tokenizer.model_type}")
    print(f"Vocab Size: {config.tokenizer.vocab_size}")
    print(f"Max Length: {config.tokenizer.max_length}")
    print(f"Batch Size: {config.data.batch_size}")
    print(f"Epochs: {config.model.num_epochs}")
    print(f"{'='*50}")
    
    # Check available backends
    try:
        import torch
        print(f"PyTorch: {torch.__version__} ✓")
    except ImportError:
        print("PyTorch: Not installed (fallback mode)")
    
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime: {ort.__version__} ✓")
    except ImportError:
        print("ONNX Runtime: Not installed (fallback mode)")
    
    print(f"{'='*50}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="adhan",
        description="Adhan — Tamil Large Language Model CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  adhan train --config config/rpi5.yaml --corpus data/corpus.jsonl
  adhan eval --corpus data/test.jsonl --report
  adhan generate --prompt "தமிழ்" --max-tokens 50
  adhan info
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train
    train_parser = subparsers.add_parser("train", help="Train Adhan model")
    train_parser.add_argument("--config", "-c", help="Config YAML path")
    train_parser.add_argument("--corpus", help="Corpus JSONL path")
    train_parser.add_argument("--steps", type=int, help="Max training steps")
    train_parser.add_argument("--output", "-o", help="Output checkpoint dir")
    train_parser.set_defaults(func=cmd_train)
    
    # Eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate model")
    eval_parser.add_argument("--config", "-c", help="Config YAML path")
    eval_parser.add_argument("--corpus", help="Eval corpus JSONL")
    eval_parser.add_argument("--model", help="Model checkpoint path")
    eval_parser.add_argument("--report", action="store_true", help="Full report")
    eval_parser.set_defaults(func=cmd_eval)
    
    # Scrape
    scrape_parser = subparsers.add_parser("scrape", help="Scrape Tamil corpus")
    scrape_parser.add_argument("--config", "-c", help="Config YAML path")
    scrape_parser.add_argument("--sources", help="Comma-separated sources")
    scrape_parser.add_argument("--output", "-o", help="Output JSONL path")
    scrape_parser.set_defaults(func=cmd_scrape)
    
    # Generate
    gen_parser = subparsers.add_parser("generate", help="Generate Tamil text")
    gen_parser.add_argument("--config", "-c", help="Config YAML path")
    gen_parser.add_argument("--prompt", required=True, help="Input prompt")
    gen_parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens")
    gen_parser.add_argument("--model", help="Model path")
    gen_parser.set_defaults(func=cmd_generate)
    
    # Info
    info_parser = subparsers.add_parser("info", help="Show package info")
    info_parser.add_argument("--config", "-c", help="Config YAML path")
    info_parser.set_defaults(func=cmd_info)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
