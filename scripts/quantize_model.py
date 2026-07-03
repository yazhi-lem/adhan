#!/usr/bin/env python3
"""Post-training quantization for the Adhan Tamil ONNX model.

Supports:
  - INT8 dynamic quantization (CPU, default)
  - INT8 static quantization (CPU, requires calibration data)
  - INT4 weight-only quantization via optimum (for edge deployment)

The script reads from an ONNX export directory produced by ``export_onnx.py``
and writes quantized model(s) into a sibling ``_quantized/`` sub-folder.

Usage examples:
    # INT8 dynamic (fastest, no calibration data required):
    python scripts/quantize_model.py \
        --model-dir models/adhan_onnx \
        --mode int8-dynamic

    # INT8 static (slower but more accurate):
    python scripts/quantize_model.py \
        --model-dir models/adhan_onnx \
        --mode int8-static \
        --calibration-data data/final/tamil_texts/hf/validation.jsonl

    # INT4 weight-only (requires optimum[onnxruntime]):
    python scripts/quantize_model.py \
        --model-dir models/adhan_onnx \
        --mode int4

Requirements:
    pip install onnxruntime onnx
    pip install optimum[onnxruntime]  # only for INT4
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def _require(pkg: str) -> None:
    logger.error("Missing dependency '%s'. Install with: pip install %s", pkg, pkg)
    sys.exit(1)


def find_onnx_model(model_dir: Path) -> Path:
    candidates = sorted(model_dir.glob("*.onnx"))
    if not candidates:
        logger.error("No .onnx file found in %s. Run export_onnx.py first.", model_dir)
        sys.exit(1)
    if len(candidates) > 1:
        logger.info("Multiple .onnx files found; using %s", candidates[0].name)
    return candidates[0]


def load_calibration_texts(calib_path: Path, max_samples: int = 200) -> list[str]:
    texts = []
    with calib_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
                text = obj.get("text") or obj.get("sentence") or ""
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue
            if len(texts) >= max_samples:
                break
    logger.info("Loaded %d calibration samples from %s", len(texts), calib_path)
    return texts


# ── quantization routines ──────────────────────────────────────────────────────

def quantize_int8_dynamic(onnx_path: Path, output_path: Path) -> None:
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType  # type: ignore
    except ImportError:
        _require("onnxruntime")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("INT8 dynamic quantization: %s → %s", onnx_path.name, output_path.name)
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        optimize_model=True,
    )
    logger.info("Done → %s", output_path)


def quantize_int8_static(
    onnx_path: Path,
    output_path: Path,
    tokenizer_dir: Path,
    calibration_texts: list[str],
    max_length: int,
) -> None:
    try:
        from onnxruntime.quantization import (  # type: ignore
            quantize_static,
            CalibrationDataReader,
            QuantType,
            QuantFormat,
        )
        import numpy as np  # type: ignore
    except ImportError:
        _require("onnxruntime")

    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError:
        _require("transformers")

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    class TamilCalibrationReader(CalibrationDataReader):
        def __init__(self, texts: list[str]) -> None:
            self._texts = texts
            self._idx = 0

        def get_next(self) -> Optional[dict]:
            if self._idx >= len(self._texts):
                return None
            enc = tokenizer(
                self._texts[self._idx],
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            self._idx += 1
            return {k: v for k, v in enc.items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("INT8 static quantization: %s → %s", onnx_path.name, output_path.name)
    quantize_static(
        model_input=str(onnx_path),
        model_output=str(output_path),
        calibration_data_reader=TamilCalibrationReader(calibration_texts),
        quant_format=QuantFormat.QOperator,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        optimize_model=True,
    )
    logger.info("Done → %s", output_path)


def quantize_int4(onnx_path: Path, output_dir: Path) -> None:
    """INT4 weight-only quantization via HuggingFace Optimum."""
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction  # type: ignore
        from optimum.onnxruntime.configuration import AutoQuantizationConfig  # type: ignore
        from optimum.onnxruntime import ORTQuantizer  # type: ignore
    except ImportError:
        _require("optimum[onnxruntime]")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("INT4 weight-only quantization via optimum …")

    # Load the ONNX model through Optimum
    model = ORTModelForFeatureExtraction.from_pretrained(
        str(onnx_path.parent), file_name=onnx_path.name
    )
    quantizer = ORTQuantizer.from_pretrained(model)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
    # Override to INT4 if the installed version supports it
    try:
        from onnxruntime.quantization import QuantType  # type: ignore
        qconfig.weights_dtype = QuantType.QInt4
    except (ImportError, AttributeError):
        logger.warning("INT4 weight type not supported by installed onnxruntime; "
                       "falling back to INT8 via optimum.")

    quantizer.quantize(
        save_dir=str(output_dir),
        quantization_config=qconfig,
    )
    logger.info("Done → %s", output_dir)


# ── benchmark helper ───────────────────────────────────────────────────────────

def benchmark(onnx_path: Path, tokenizer_dir: Path, max_length: int,
              n_runs: int = 20) -> dict:
    """Return mean/std latency (ms) for a dummy inference pass."""
    import time  # noqa: PLC0415

    try:
        import onnxruntime as ort  # type: ignore
        import numpy as np  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except ImportError:
        return {}

    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    enc = tokenizer(
        "நான் தமிழ் கற்கிறேன்",
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    feed = {k: v for k, v in enc.items()}

    # Warm-up
    for _ in range(3):
        sess.run(None, feed)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)

    import statistics
    return {
        "mean_ms": round(statistics.mean(times), 2),
        "stdev_ms": round(statistics.stdev(times), 2),
        "n_runs": n_runs,
    }


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize the Adhan Tamil ONNX model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", default="models/adhan_onnx",
                        help="Directory containing the ONNX model (from export_onnx.py).")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <model-dir>_quantized).")
    parser.add_argument(
        "--mode",
        choices=["int8-dynamic", "int8-static", "int4"],
        default="int8-dynamic",
        help="Quantization mode.",
    )
    parser.add_argument("--calibration-data", default=None,
                        help="JSONL file with calibration sentences (required for int8-static).")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--calibration-samples", type=int, default=200)
    parser.add_argument("--benchmark", action="store_true",
                        help="Run a simple latency benchmark after quantization.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error("Model dir not found: %s", model_dir)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else \
        model_dir.parent / (model_dir.name + "_quantized")
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = find_onnx_model(model_dir)
    tokenizer_dir = model_dir / "tokenizer"

    if args.mode == "int8-dynamic":
        out_path = output_dir / "model_int8.onnx"
        quantize_int8_dynamic(onnx_path, out_path)

    elif args.mode == "int8-static":
        if not args.calibration_data:
            logger.error("--calibration-data is required for int8-static mode.")
            sys.exit(1)
        calib_path = Path(args.calibration_data)
        if not calib_path.exists():
            logger.error("Calibration data not found: %s", calib_path)
            sys.exit(1)
        texts = load_calibration_texts(calib_path, args.calibration_samples)
        out_path = output_dir / "model_int8_static.onnx"
        quantize_int8_static(onnx_path, out_path, tokenizer_dir, texts, args.max_length)

    elif args.mode == "int4":
        quantize_int4(onnx_path, output_dir / "int4")
        out_path = output_dir / "int4" / "model.onnx"

    # Write manifest
    manifest: dict = {
        "source_onnx": str(onnx_path),
        "mode": args.mode,
        "output_dir": str(output_dir),
    }

    if args.benchmark and out_path.exists() and tokenizer_dir.exists():
        logger.info("Running latency benchmark …")
        stats = benchmark(out_path, tokenizer_dir, args.max_length)
        if stats:
            manifest["benchmark"] = stats
            logger.info("Benchmark: %.1f ms ± %.1f ms (n=%d)",
                        stats["mean_ms"], stats["stdev_ms"], stats["n_runs"])

    with (output_dir / "quantization_manifest.json").open("w") as mf:
        json.dump(manifest, mf, indent=2)
    logger.info("Manifest written to %s/quantization_manifest.json", output_dir)


if __name__ == "__main__":
    main()
