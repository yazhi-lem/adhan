#!/usr/bin/env python3
"""Generate Tamil text from a trained Adhan SLM checkpoint.

Closes the "no inference path" gap: loads the frozen tokenizer + Orbax checkpoint and
samples a continuation for one or more prompts (the Phase 5 demo / Phase 4 read-through).

    python scripts/generate_slm.py \
        --tokenizer-dir data/final/tamil_slm \
        --config src/adhan_slm/configs/adhan_slm_tiny.yaml \
        --checkpoint checkpoints/nano \
        --prompt "சொல், உனக்கு பிடித்த உணவு என்ன?" --temperature 0.8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate from a trained Adhan SLM")
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", action="append", required=True,
                    help="prompt text (repeat for several)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from adhan_slm.inference import load_tokenizer, load_model, generate_text

    tok = load_tokenizer(args.tokenizer_dir)
    model, params, cfg = load_model(args.config, args.checkpoint, vocab_size=len(tok))
    print(f"loaded model (~{cfg.approx_params()/1e6:.1f}M params), vocab {len(tok)}\n")

    for prompt in args.prompt:
        text = generate_text(
            model, params, tok, prompt,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, seed=args.seed)
        print(f"prompt : {prompt}")
        print(f"output : {text}\n")


if __name__ == "__main__":
    main()
