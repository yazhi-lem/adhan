#!/usr/bin/env python3
"""Chat with Adhan SLM from the command line (``adhan-chat``).

A minimalist chat CLI inspired by Agaram 2.5B's ``serving/agaram_chat.py``
(Murai-Labs/Agaram2.5b, Apache-2.0), adapted for Adhan's native JAX/Flax + Swaram
tokenizer stack. Uses Adhan's Tamil role protocol (``பயனர்:`` / ``உதவியாளர்:``)
instead of ChatML because the Swaram tokenizer does not carry ``<|im_start|>``
tokens — chat behaviour comes from SFT on textual role markers.

Installed as the ``adhan-chat`` console script; the same code runs from a
checkout as ``python -m adhan_slm.chat_cli`` (or ``scripts/adhan_chat.py``).

Usage::

    # Interactive (loads checkpoint + tokenizer, then REPL)
    adhan-chat \
        --checkpoint checkpoints/adhan-nano-sft \
        --tokenizer data/final/tamil_slm

    # One-shot, greedy
    adhan-chat \
        --checkpoint checkpoints/adhan-nano-sft \
        --tokenizer data/final/tamil_slm \
        --prompt "வணக்கம்! நீ யார்?"

    # Sampling mode (stochastic)
    adhan-chat \
        --checkpoint checkpoints/adhan-nano-sft \
        --tokenizer data/final/tamil_slm \
        --prompt "குறுந்தொகையின் முதல் பாடல் எது?" \
        --temperature 0.8 --top-k 40 --top-p 0.95

The crisis-resource footer (Tele-MANAS 14416 / 112) is printed automatically when
user input matches known distress phrases; this is a wrapper-level safeguard, not
a model capability.
"""

from __future__ import annotations

import argparse
import sys

# Logging before JAX import so --help and parse errors are instantaneous.
from adhan_slm.core.logging import get_logger

# Adhan safety layer — imported here so it works even when JAX is absent,
# giving a clear error instead of a stack trace.
from adhan_slm.safety.crisis import CRISIS_FOOTER, needs_footer

logger = get_logger("adhan_chat")

MAX_NEW_TOKENS = 512
MAX_CONTEXT = 8192


def _load(checkpoint_dir: str, tokenizer_dir: str):
    """Load model, params, and tokenizer. JAX imports are lazily deferred."""
    try:
        from adhan_slm.inference import load_model, load_tokenizer
    except ImportError as exc:
        logger.error("JAX stack not available (%s). Install with: pip install -e '.[jax]'", exc)
        raise SystemExit(1)

    logger.info("Loading tokenizer from %s …", tokenizer_dir)
    tok = load_tokenizer(tokenizer_dir, kind="swaram")

    logger.info("Loading checkpoint from %s …", checkpoint_dir)
    model, params, cfg = load_model(checkpoint_dir=checkpoint_dir)
    logger.info(
        "Loaded %s (~%.1fM params, max_seq_len=%d)",
        cfg,
        cfg.approx_params() / 1e6,
        cfg.max_seq_len,
    )
    return model, params, tok, cfg


def _render_chat(history: list[dict]) -> str:
    """Render message history to Adhan's Tamil role protocol string."""
    from adhan_slm.protocol import render_prompt

    return render_prompt(history, add_assistant_cue=True)


def reply(
    model,
    params,
    tok,
    history: list[dict],
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
) -> tuple[str, bool]:
    """Generate one assistant turn.

    Returns ``(text, stopped)`` where ``stopped`` is True when generation
    halted on the ``<eos>`` token (as opposed to hitting ``max_new_tokens``).
    """
    from adhan_slm.model import generate

    prompt_text = _render_chat(history)
    prompt_ids = tok.encode(prompt_text, add_special=True)
    # Drop a trailing <eos> so the model continues rather than ending immediately.
    eos_id = tok.vocab.get("<eos>")
    if eos_id is not None and prompt_ids and prompt_ids[-1] == eos_id:
        prompt_ids = prompt_ids[:-1]

    if len(prompt_ids) + max_new_tokens > MAX_CONTEXT:
        raise ValueError(f"conversation is {len(prompt_ids)} tokens; max context is {MAX_CONTEXT}")

    out_ids = generate(
        model,
        params,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_id=eos_id,
        seed=0,
    )
    stopped = bool(eos_id is not None and out_ids and out_ids[-1] == eos_id)
    text = tok.decode(out_ids, hide_unk=True).strip()
    return text, stopped


def _show(user: str, text: str, stopped: bool) -> None:
    """Print assistant reply, appending a marker when generation was truncated."""
    print(
        text if stopped else text + "\n[cut at the token limit; the model did not stop on its own]",
        flush=True,
    )
    if needs_footer(user):
        print(CRISIS_FOOTER, flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="Orbax checkpoint directory (e.g. checkpoints/adhan-nano-sft)",
    )
    ap.add_argument(
        "--tokenizer", required=True, help="Frozen tokenizer directory (e.g. data/final/tamil_slm)"
    )
    ap.add_argument("--prompt", default=None, help="One-shot message, then exit")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    args = ap.parse_args(argv)

    if args.temperature < 0:
        ap.error("temperature must be >= 0")
    if args.top_p <= 0 or args.top_p > 1:
        ap.error("top-p must be in (0, 1]")
    if args.top_k < 1:
        ap.error("top-k must be >= 1")

    model, params, tok, cfg = _load(args.checkpoint, args.tokenizer)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    if args.prompt:
        text, stopped = reply(
            model, params, tok, [{"role": "user", "content": args.prompt}], **gen_kwargs
        )
        _show(args.prompt, text, stopped)
        return 0

    banner = (
        f"Adhan SLM · {cfg.approx_params() / 1e6:.1f}M params · research model, may be wrong\n"
        "Type empty line to quit.\n"
    )
    print(banner, file=sys.stderr)

    history: list[dict] = []
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            break
        history.append({"role": "user", "content": user})
        text, stopped = reply(model, params, tok, history, **gen_kwargs)
        _show(user, text, stopped)
        history.append({"role": "assistant", "content": text})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
