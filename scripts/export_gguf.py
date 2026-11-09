#!/usr/bin/env python3
"""Export a trained Adhan SLM (JAX/Flax checkpoint) to GGUF for llama.cpp / Ollama.

This script converts Adhan's JAX/Flax weights to PyTorch / HuggingFace-compatible
format, then to GGUF via llama.cpp's ``convert_hf_to_gguf.py``.

Adapted from Agaram 2.5B's ``gguf/convert_agaram.py`` (Murai-Labs/Agaram2.5b,
Apache-2.0), modified for Adhan's Swaram tokenizer and Tamil protocol.

Requirements::

    pip install transformers torch huggingface_hub
    # llama.cpp built from source (b11574 or newer)
    git clone https://github.com/ggerganov/llama.cpp.git ~/llama.cpp
    cd ~/llama.cpp && cmake -B build && cmake --build build --config Release

Usage::

    # 1. Convert JAX → HF Safetensors
    python scripts/export_gguf.py \
        --checkpoint checkpoints/adhan-nano-sft \
        --tokenizer data/final/tamil_slm \
        --output-hf export/adhan-nano-hf

    # 2. Convert HF → GGUF (F16)
    LLAMA_CPP=~/llama.cpp python scripts/export_gguf.py \
        --checkpoint export/adman-nano-hf \
        --tokenizer data/final/tamil_slm \
        --output-gguf export/adhan-nano-f16.gguf \
        --outtype f16

    # 3. Quantize GGUF (Q8_0)
    $LLAMA_CPP/build/bin/llama-quantize \
        export/adhan-nano-f16.gguf export/adhan-nano-q8_0.gguf Q8_0

The Adhan Swaram tokenizer does not use ChatML ``<|im_start|>`` tokens;
instead it uses Tamil textual role markers (``பயனர்:``, ``உதவியாளர்:``).
The GGUF will embed a Jinja chat template reflecting this protocol.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

LOG = print

LLAMA_CPP = Path.home() / "llama.cpp"


def _require(name: str, pkg: Optional[str] = None):
    try:
        __import__(name)
    except ImportError:
        sys.exit(f"Missing dependency '{name}'. Install: pip install {pkg or name}")


def export_jax_to_hf(
    checkpoint_dir: str,
    tokenizer_dir: str,
    output_dir: str,
    model_size: str = "nano",
):
    """Load a JAX/Flax Adhan checkpoint and save as HF-compatible files.

    This is a skeleton: the full implementation requires:
      1. Loading AdhanConfig + params from Orbax
      2. Mapping Adhan's parameter dict to Transformers ``LlamaForCausalLM`` weights
      3. Writing ``config.json``, ``model.safetensors``, and tokenizer files

    Because Adhan uses a custom JAX/Flax architecture (RoPE, RMSNorm, SwiGLU, GQA)
    that matches Llama's layout, the weight mapping is mostly a name translation.
    """
    _require("transformers")
    _require("torch")

    from transformers import LlamaConfig, LlamaForCausalLM

    from adhan_slm.inference import load_model

    LOG("Loading JAX checkpoint from %s …", checkpoint_dir)
    model, params, cfg = load_model(checkpoint_dir=checkpoint_dir)

    LOG("Adhan config: %s (~%.1fM params)", cfg, cfg.approx_params() / 1e6)

    # Map AdhanConfig → LlamaConfig
    # Adhan: d_model=256, n_layers=6, n_heads=4, n_kv_heads=0|2, d_ff=1024
    # Llama: hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, intermediate_size
    llama_cfg = LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.kv_heads,
        intermediate_size=cfg.d_ff,
        max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        # RoPE
        rope_theta=cfg.rope_theta,
        rope_scaling=None,
        # Architecture flags
        use_cache=True,
        # No bias in Adhan
        attention_bias=False,
    )

    hf_model = LlamaForCausalLM(llama_cfg)
    hf_model.eval()

    # --- Parameter mapping (Adhan JAX → Llama PyTorch) ---
    # This is the critical part. Adhan's param tree:
    #   tok_emb        → embed_tokens.weight
    #   norm_f         → norm.weight
    #   block_{i}/     → model.layers.{i}/
    #     norm1        → input_layernorm.weight
    #     norm2        → post_attention_layernorm.weight
    #     qkv/kernel   → self_attn.q_proj, k_proj, v_proj weights
    #     proj/kernel  → self_attn.o_proj.weight
    #     gate/kernel  → mlp.gate_proj.weight
    #     up/kernel    → mlp.up_proj.weight
    #     down/kernel  → mlp.down_proj.weight
    #
    # Because this mapping requires careful shape handling (GQA head splitting,
    # transposes, etc.), the full implementation is TODO.  For now the script
    # documents the mapping and exits with instructions.

    LOG("=" * 60)
    LOG("JAX → HF weight export for Adhan SLM")
    LOG("=" * 60)
    LOG("")
    LOG("Parameter mapping from Adhan JAX → Llama HF:")
    LOG("")
    LOG("  Adhan                    →  Llama")
    LOG("  -----------------------------------------------------------")
    LOG("  tok_emb                  →  model.embed_tokens.weight")
    LOG("  norm_f                   →  model.norm.weight")
    LOG("  block_{i}/norm1         →  model.layers.{i}.input_layernorm.weight")
    LOG("  block_{i}/norm2         →  model.layers.{i}.post_attention_layernorm.weight")
    LOG("  block_{i}/qkv/kernel    →  model.layers.{i}.self_attn.q_proj.weight")
    LOG("                           →  model.layers.{i}.self_attn.k_proj.weight")
    LOG("                           →  model.layers.{i}.self_attn.v_proj.weight")
    LOG("  block_{i}/proj/kernel   →  model.layers.{i}.self_attn.o_proj.weight")
    LOG("  block_{i}/gate/kernel   →  model.layers.{i}.mlp.gate_proj.weight")
    LOG("  block_{i}/up/kernel     →  model.layers.{i}.mlp.up_proj.weight")
    LOG("  block_{i}/down/kernel   →  model.layers.{i}.mlp.down_proj.weight")
    LOG("")
    LOG("Note: Adhan uses a fused qkv kernel. Split along the last axis into")
    LOG("      [n_heads*dh, n_kv_heads*dh, n_kv_heads*dh] for q, k, v.")
    LOG("")
    LOG("The full weight-copy implementation is TODO:")
    LOG("  https://github.com/yazhi-lem/adhan/issues/TODO")
    LOG("")
    LOG("Config written to %s/config.json", output_dir)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    llama_cfg.save_pretrained(out_path)

    # Copy tokenizer files
    tok_src = Path(tokenizer_dir)
    tok_dst = out_path / "tokenizer"
    tok_dst.mkdir(exist_ok=True)
    for f in ("vocab.json", "merges.txt"):
        src = tok_src / f
        if src.exists():
            shutil.copy(src, tok_dst / f)

    # Write a basic tokenizer config that HF can load
    tok_config = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "byte-level"},
        "post_processor": None,
        "decoder": {"type": "byte-level"},
        "model": {
            "type": "BPE",
            "vocab": str(tok_dst / "vocab.json"),
            "merges": str(tok_dst / "merges.txt"),
        },
    }
    (tok_dst / "tokenizer_config.json").write_text(
        json.dumps(tok_config, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    # Write chat template (Tamil protocol, not ChatML)
    chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "பயனர்: {{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}"
        "உதவியாளர்: {{ message['content'] }}\n"
        "{% elif message['role'] == 'system' %}"
        "அமைப்பு: {{ message['content'] }}\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}உதவியாளர்: {% endif %}"
    )
    (tok_dst / "chat_template.jinja").write_text(chat_template, encoding="utf-8")

    LOG("Tokenizer stub written to %s", tok_dst)
    LOG("")
    LOG("Next step: implement the weight-copy mapping above, then run:")
    LOG("  LLAMA_CPP=~/llama.cpp python scripts/export_gguf.py \\")
    LOG("      --checkpoint %s --output-gguf adhan.gguf --outtype f16", output_dir)


def convert_hf_to_gguf(hf_dir: str, output_gguf: str, outtype: str = "f16"):
    """Wrap llama.cpp's convert_hf_to_gguf.py for Adhan.

    Agaram 2.5B uses a custom pre-tokenizer; Adhan's Swaram tokenizer
    is different enough that a direct llama.cpp conversion may need
    a custom pre-tokenizer registration (TBD).
    """
    llama_cpp = Path(LLAMA_CPP)
    if not llama_cpp.exists():
        sys.exit(
            f"llama.cpp not found at {llama_cpp}. Clone and build it:\n"
            "  git clone https://github.com/ggerganov/llama.cpp.git ~/llama.cpp\n"
            "  cd ~/llama.cpp && cmake -B build && cmake --build build --config Release"
        )

    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        sys.exit(f"convert_hf_to_gguf.py not found at {convert_script}")

    import subprocess

    cmd = [
        sys.executable,
        str(convert_script),
        hf_dir,
        "--outtype",
        outtype,
        "--outfile",
        output_gguf,
        "--verbose",
    ]
    LOG("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    LOG("GGUF written to %s", output_gguf)


def _cli():
    ap = argparse.ArgumentParser(description="Export Adhan SLM to GGUF")
    ap.add_argument("--checkpoint", required=True, help="Orbax checkpoint dir or HF export dir")
    ap.add_argument("--tokenizer", help="Frozen tokenizer directory (vocab.json + merges.txt)")
    ap.add_argument("--output-hf", help="Directory to write HF-compatible export")
    ap.add_argument("--output-gguf", help="Output GGUF file path")
    ap.add_argument("--outtype", default="f16", choices=["f16", "f32", "q8_0", "q4_k_m"])
    ap.add_argument("--model-size", default="nano", choices=["nano", "tiny", "mini"])
    args = ap.parse_args()

    if args.output_hf:
        if not args.tokenizer:
            ap.error("--tokenizer is required when --output-hf is specified")
        export_jax_to_hf(
            args.checkpoint, args.tokenizer, args.output_hf, model_size=args.model_size
        )

    if args.output_gguf:
        convert_hf_to_gguf(args.checkpoint, args.output_gguf, outtype=args.outtype)

    if not args.output_hf and not args.output_gguf:
        ap.error("Specify at least one of --output-hf or --output-gguf")

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
