#!/usr/bin/env python3
"""Export a trained Adhan/Tamil model to ONNX format.

The script loads a HuggingFace-compatible model (encoder-only or causal-LM),
traces it with a dummy input, and writes a single ``model.onnx`` file plus a
``tokenizer/`` directory into the output folder.

Usage:
    python scripts/export_onnx.py \
        --model-dir models/adhan \
        --output-dir models/adhan_onnx \
        --opset 17

Requirements (install separately – not listed in requirements.txt to keep
the core env lean):
    pip install optimum[exporters] onnx onnxruntime

If ``optimum`` is available the script delegates to its ONNX exporter, which
handles architecture-specific edge-cases automatically.  If ``optimum`` is not
installed, a manual ``torch.onnx.export`` path is used as fallback.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def _require(name: str) -> None:
    """Exit with a clear message when an optional dependency is missing."""
    logger.error(
        "Missing dependency '%s'. Install it with: pip install %s",
        name, name,
    )
    sys.exit(1)


def export_with_optimum(model_dir: Path, output_dir: Path, opset: int,
                        task: str) -> None:
    """Use HuggingFace Optimum for a robust, architecture-aware ONNX export."""
    try:
        from optimum.exporters.onnx import main_export  # type: ignore
    except ImportError:
        _require("optimum[exporters]")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting via optimum.exporters.onnx …")
    main_export(
        model_name_or_path=str(model_dir),
        output=output_dir,
        task=task,
        opset=opset,
        do_validation=True,
    )
    logger.info("Optimum export complete → %s", output_dir)


def export_with_torch(model_dir: Path, output_dir: Path, opset: int,
                      max_length: int) -> None:
    """Fallback: manual torch.onnx.export for encoder-only models."""
    try:
        import torch  # type: ignore
    except ImportError:
        _require("torch")
    try:
        from transformers import AutoTokenizer, AutoModel  # type: ignore
    except ImportError:
        _require("transformers")
    try:
        import onnx  # type: ignore  # noqa: F401
    except ImportError:
        _require("onnx")

    logger.info("Loading model from %s …", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir))
    model.eval()

    dummy_text = "நான் தமிழ் கற்கிறேன்"
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    input_names = list(inputs.keys())
    dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
    dynamic_axes["last_hidden_state"] = {0: "batch", 1: "sequence"}

    logger.info("Tracing model → %s (opset %d) …", onnx_path, opset)
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(inputs.values()),
            str(onnx_path),
            input_names=input_names,
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    # Save tokenizer alongside the ONNX graph
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))

    # Write a minimal manifest
    manifest = {
        "source_model": str(model_dir),
        "onnx_file": str(onnx_path),
        "opset": opset,
        "max_length": max_length,
        "exporter": "torch.onnx",
    }
    with (output_dir / "onnx_manifest.json").open("w") as mf:
        json.dump(manifest, mf, indent=2)

    logger.info("torch.onnx export complete → %s", onnx_path)


def verify_onnx(output_dir: Path) -> None:
    """Run a quick ONNX model check and shape inference."""
    try:
        import onnx  # type: ignore
    except ImportError:
        logger.warning("onnx not installed – skipping verification.")
        return

    onnx_files = list(output_dir.glob("*.onnx"))
    if not onnx_files:
        logger.warning("No .onnx file found in %s – skipping verification.", output_dir)
        return

    for f in onnx_files:
        model_proto = onnx.load(str(f))
        onnx.checker.check_model(model_proto)
        logger.info("✓ ONNX model %s is valid (IR version %d, opset %d)",
                    f.name,
                    model_proto.ir_version,
                    model_proto.opset_import[0].version)


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trained Adhan Tamil model to ONNX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", default="models/adhan",
                        help="Path to the trained HuggingFace model directory.")
    parser.add_argument("--output-dir", default="models/adhan_onnx",
                        help="Directory to write the ONNX model and tokenizer.")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version.")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Sequence length for dummy input (torch fallback).")
    parser.add_argument(
        "--task",
        default="feature-extraction",
        help=(
            "Optimum task string, e.g. 'feature-extraction', "
            "'fill-mask', 'text-generation'. "
            "Ignored when optimum is not installed."
        ),
    )
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip post-export ONNX verification.")
    parser.add_argument("--force-torch", action="store_true",
                        help="Use the torch.onnx fallback even if optimum is installed.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    if not model_dir.exists():
        logger.error("Model directory not found: %s", model_dir)
        sys.exit(1)

    use_optimum = not args.force_torch
    if use_optimum:
        try:
            import optimum  # type: ignore  # noqa: F401
        except ImportError:
            logger.info("optimum not installed – falling back to torch.onnx.export.")
            use_optimum = False

    if use_optimum:
        export_with_optimum(model_dir, output_dir, args.opset, args.task)
    else:
        export_with_torch(model_dir, output_dir, args.opset, args.max_length)

    if not args.no_verify:
        verify_onnx(output_dir)


if __name__ == "__main__":
    main()
