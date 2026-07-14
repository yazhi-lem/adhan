"""Adhan SLM evaluation harness — one entrypoint for the Phase 4 comparison table.

Runs every Tamil-specific probe the roadmap defines and prints/writes a single report
(`docs/EVAL_TAMIL.md`-style JSON). Each probe degrades gracefully: sections that need
open-tamil or a trained checkpoint are skipped with a note rather than crashing, so the
harness runs in any environment and gets richer as dependencies/artifacts appear.

Probes:
  * classical  — add-one unigram akshara perplexity floor (open-tamil)          §Phase 4
  * morphology — tokenizer merge-boundary vs stemmer agreement; sandhi rate      §Phase 4
  * fertility  — frozen tokenizer tokens/akshara on the eval text                §Phase 1
  * model      — trained-model per-token val perplexity + sample generations     §Phase 3
  * prompts    — the kid-level prompt set (with generations if a model is given) §Phase 4

    # tokenizer + corpus probes only:
    python -m adhan_slm.eval.run_eval --tokenizer-dir data/final/tamil_slm --eval-text data/final/tamil_slm/val.txt

    # add model perplexity + generations:
    python -m adhan_slm.eval.run_eval --tokenizer-dir data/final/tamil_slm \
        --config src/adhan_slm/configs/adhan_slm_tiny.yaml --checkpoint checkpoints/nano
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adhan_slm.data import corpus as corpus_mod


def _load_tokenizer(tokenizer_dir):
    from adhan_slm.inference import load_tokenizer
    return load_tokenizer(tokenizer_dir)


def probe_fertility(tok, texts, cap=500):
    vals = [tok.fertility(t) for t in texts[:cap] if t.strip()]
    vals = [v for v in vals if v > 0]
    if not vals:
        return {"status": "skipped", "reason": "no eval text"}
    mean = sum(vals) / len(vals)
    return {"status": "ok", "mean_fertility": round(mean, 4),
            "target": 1.15, "pass": mean < 1.15, "n_docs": len(vals)}


def probe_classical(texts):
    try:
        from adhan_slm.eval.ngram_baseline import AksharaUnigramBaseline, HAS_OPEN_TAMIL
    except ImportError:
        return {"status": "skipped", "reason": "open-tamil not installed"}
    if not HAS_OPEN_TAMIL:
        return {"status": "skipped", "reason": "open-tamil not installed"}
    if len(texts) < 2:
        return {"status": "skipped", "reason": "need >=2 docs"}
    split = max(1, int(len(texts) * 0.9))
    baseline = AksharaUnigramBaseline(texts[:split])
    ppls = [baseline.perplexity(t) for t in texts[split:]]
    ppls = [p for p in ppls if p == p]  # drop nan
    if not ppls:
        return {"status": "skipped", "reason": "no scorable held-out text"}
    return {"status": "ok", "unigram_akshara_ppl": round(sum(ppls) / len(ppls), 3),
            "note": "classical floor; trained model must beat this"}


def probe_morphology(tok, texts, cap=300):
    try:
        from adhan_slm.eval.morphology import (
            stemmer_boundary_agreement, sandhi_correctness_rate)
        from adhan_slm.external.open_tamil_bridge import HAS_OPEN_TAMIL
    except ImportError:
        return {"status": "skipped", "reason": "open-tamil not installed"}
    if not HAS_OPEN_TAMIL:
        return {"status": "skipped", "reason": "open-tamil not installed"}
    words = [w for t in texts[:cap] for w in t.split()][:2000]
    if not words:
        return {"status": "skipped", "reason": "no words"}
    agree = stemmer_boundary_agreement(tok, words)
    sandhi = sandhi_correctness_rate(texts[:cap])
    return {"status": "ok",
            "boundary_agreement_rate": round(agree.agreement_rate, 4)
            if agree.n_with_suffix else None,
            "n_suffixed_words": agree.n_with_suffix,
            "sandhi_correct_rate": round(sandhi.word_correctness_rate, 4)
            if sandhi.n_words else None}


def probe_model(config, checkpoint, tokenizer_dir, tok, texts, n_samples=5):
    if not (config and checkpoint):
        return {"status": "skipped", "reason": "no --config/--checkpoint given"}
    try:
        from adhan_slm.inference import load_model, generate_text
        from adhan_slm.eval.perplexity import model_token_perplexity
        from adhan_slm.data import PackedDataset
        from adhan_slm.data.packing import load_manifest
    except ImportError as e:
        return {"status": "skipped", "reason": f"JAX stack not installed ({e})"}

    result = {"status": "ok"}
    try:
        model, params, model_cfg = load_model(config, checkpoint, vocab_size=len(tok))
    except (ImportError, FileNotFoundError) as e:
        return {"status": "skipped", "reason": str(e)}

    val_bin = Path(tokenizer_dir) / "val.bin"
    if val_bin.exists():
        val_ds = PackedDataset.from_shard(
            val_bin, batch_size=8, shuffle=False, infinite=False, drop_last=False,
            manifest=load_manifest(val_bin))
        result["perplexity"] = model_token_perplexity(model, params, val_ds)
    else:
        result["perplexity"] = {"status": "skipped", "reason": "no val.bin"}

    samples = []
    for prompt in (texts[:n_samples] or ["சொல்"]):
        seed_prompt = prompt.split()[0] if prompt.split() else prompt
        samples.append({"prompt": seed_prompt,
                        "generation": generate_text(model, params, tok, seed_prompt,
                                                     max_new_tokens=40)})
    result["samples"] = samples
    return result


def probe_kid_prompts(config, checkpoint, tok, n=10):
    try:
        from adhan_slm.eval.kid_level_prompts import build_kid_level_prompts
    except ImportError:
        return {"status": "skipped", "reason": "open-tamil not installed"}
    try:
        prompts = build_kid_level_prompts(n=n)
    except Exception as e:
        return {"status": "skipped", "reason": f"lexicon unavailable ({e})"}
    if not prompts:
        return {"status": "skipped", "reason": "no prompts built (open-tamil lexicons?)"}
    out = {"status": "ok", "n_prompts": len(prompts),
           "examples": [p.prompt for p in prompts[:5]]}
    if config and checkpoint:
        try:
            from adhan_slm.inference import load_model, generate_text
            model, params, _ = load_model(config, checkpoint, vocab_size=len(tok))
            out["generations"] = [
                {"prompt": p.prompt,
                 "generation": generate_text(model, params, tok, p.prompt, max_new_tokens=40)}
                for p in prompts[:5]]
        except (ImportError, FileNotFoundError) as e:
            out["generations"] = {"status": "skipped", "reason": str(e)}
    return out


def main():
    ap = argparse.ArgumentParser(description="Adhan SLM evaluation harness")
    ap.add_argument("--tokenizer-dir", required=True,
                    help="dir with vocab.json + merges.txt (+ optional val.bin)")
    ap.add_argument("--eval-text", help="txt/jsonl of held-out Tamil for ppl/fertility")
    ap.add_argument("--config", help="training YAML (to rebuild model for model probe)")
    ap.add_argument("--checkpoint", help="Orbax checkpoint dir (enables model probe)")
    ap.add_argument("--out", help="write the report JSON here")
    args = ap.parse_args()

    tok = _load_tokenizer(args.tokenizer_dir)
    texts = corpus_mod.read_corpus(args.eval_text) if args.eval_text else []

    report = {
        "tokenizer_dir": args.tokenizer_dir,
        "vocab_size": len(tok),
        "n_eval_docs": len(texts),
        "probes": {
            "fertility": probe_fertility(tok, texts),
            "classical": probe_classical(texts),
            "morphology": probe_morphology(tok, texts),
            "model": probe_model(args.config, args.checkpoint, args.tokenizer_dir, tok, texts),
            "kid_prompts": probe_kid_prompts(args.config, args.checkpoint, tok),
        },
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
