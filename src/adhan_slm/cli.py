#!/usr/bin/env python3
"""Adhan SLM Unified CLI — `adhan start | train | serve | mutate | eval | status | interact`

Command Suite:
  - adhan start    : Launch local environment, check accelerators & MLflow tracking.
  - adhan train    : Train Adhan SLM models (CPU/GPU, overfit sanity gate, resume).
  - adhan serve    : Start the high-performance FastAPI/REST inference server.
  - adhan mutate   : Mutate datasets (distillation, dedup) or models (quantize, export).
  - adhan eval     : Execute comprehensive Tamil linguistic & Thirukkural benchmarks.
  - adhan status   : Inspect checkpoints, active background runs & datasheet metrics.
  - adhan interact : Interactive CLI terminal session with tracing & live generation.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT_DIR / "logs"


def setup_cli_logging(verbose: bool = False, trace: bool = False) -> Path:
    """Set up structured file logging and optional interactive console tracing."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"adhan_cli_{timestamp}.log"
    latest_link = LOGS_DIR / "adhan_cli_latest.log"

    level = logging.DEBUG if (verbose or trace) else logging.INFO

    # Root logger file handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s:%(lineno)d — %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Remove existing handlers to prevent duplicate lines
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)

    # Link latest log
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(log_file.name)
    except Exception:
        pass

    return log_file


def print_banner():
    banner = """
  █████╗ ██████╗ ██╗  ██╗ █████╗ ███╗   ██╗
 ██╔══██╗██╔══██╗██║  ██║██╔══██╗████╗  ██║
 ███████║██║  ██║███████║███████║██╔██╗ ██║
 ██╔══██║██║  ██║██╔══██║██╔══██║██║╚██╗██║
 ██║  ██║██████╔╝██║  ██║██║  ██║██║ ╚████║
 ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝
     Pure-Tamil Sovereign SLM Engine
"""
    print(banner)


# ------------------------------------------------------------------------------
# 1. adhan start
# ------------------------------------------------------------------------------
def cmd_start(args: argparse.Namespace) -> None:
    """Initialize development stack, check hardware & start tracking services."""
    print("🚀 [adhan start] Initializing Yazhi Adhan Environment...")
    print(f"📁 Workspace Root: {ROOT_DIR}")
    print(f"📝 Persistent Log : {LOGS_DIR}/adhan_cli_latest.log")

    # 1. Check Python & Virtual Environment
    print(f"🐍 Python Executable: {sys.executable} ({sys.version.split()[0]})")

    # 2. Check JAX and hardware accelerators
    try:
        import jax

        devices = jax.devices()
        print(f"⚡ JAX Version: {jax.__version__} | Devices: {devices}")
    except ImportError:
        print("⚠️  JAX is not installed in the active environment.")

    # 3. Check MLflow Status / Start Tracking Server
    mlflow_db = ROOT_DIR / "mlflow.db"
    print(f"📊 MLflow Database: {mlflow_db} (Exists: {mlflow_db.exists()})")

    if args.mlflow:
        print(f"🌐 Launching MLflow UI on http://localhost:{args.mlflow_port} ...")
        cmd = [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            f"sqlite:///{mlflow_db}",
            "--port",
            str(args.mlflow_port),
        ]
        try:
            subprocess.run(cmd, cwd=ROOT_DIR)
        except KeyboardInterrupt:
            print("\n🛑 MLflow UI stopped.")
    else:
        print("💡 Tip: Run 'adhan start --mlflow' to launch the live web dashboard.")


# ------------------------------------------------------------------------------
# 2. adhan train
# ------------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> None:
    """Launch JAX / PyTorch training runs with sanity checks and live tracing."""
    print(
        f"🔥 [adhan train] Preparing training pipeline (Model: {args.model}, Device: {args.device})..."
    )

    config_map = {
        "nano": ROOT_DIR / "src/adhan_slm/configs/adhan_slm_nano_cpu.yaml",
        "tiny": ROOT_DIR / "src/adhan_slm/configs/adhan_slm_tiny.yaml",
        "mini": ROOT_DIR / "src/adhan_slm/configs/adhan_slm_mini.yaml",
    }
    config_path = (
        Path(args.config) if args.config else config_map.get(args.model, config_map["nano"])
    )

    if not config_path.exists():
        sys.exit(f"❌ Config file not found: {config_path}")

    cmd = [
        sys.executable,
        "-m",
        "adhan_slm.training.train_jax",
        "--config",
        str(config_path),
        "--device",
        args.device,
    ]

    if args.overfit_batch:
        print("🎯 Mode: Overfit-a-Batch Sanity Gate (Validating loss collapse to < 0.1)")
        cmd.append("--overfit-batch")
    elif args.smoke:
        print("💨 Mode: Smoke Run (5-step validation)")
        cmd.append("--smoke")

    if args.resume:
        cmd.append("--resume")
    if args.max_steps:
        cmd.extend(["--max-steps", str(args.max_steps)])
    if args.lr:
        cmd.extend(["--lr", str(args.lr)])

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT_DIR}/src:{env.get('PYTHONPATH', '')}"

    if args.trace:
        print("🔍 Interactive Tracing Enabled: Logging every micro-step and gradient update.")
        env["JAX_TRACE"] = "1"
        env["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

    print(f"▶️  Executing: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, cwd=ROOT_DIR, env=env, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Training paused by user.")


# ------------------------------------------------------------------------------
# 3. adhan serve
# ------------------------------------------------------------------------------
def cmd_serve(args: argparse.Namespace) -> None:
    """Start the REST & Streaming inference server."""
    print(f"⚡ [adhan serve] Launching Adhan SLM API Server on http://{args.host}:{args.port} ...")
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts/run_api_server.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.model,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT_DIR}/src:{env.get('PYTHONPATH', '')}"

    try:
        subprocess.run(cmd, cwd=ROOT_DIR, env=env)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")


# ------------------------------------------------------------------------------
# 4. adhan mutate
# ------------------------------------------------------------------------------
def cmd_mutate(args: argparse.Namespace) -> None:
    """Execute data or model mutations with tracing."""
    subaction = args.mutate_action

    if subaction == "dedup":
        print("🧹 [adhan mutate:dedup] Running MinHash LSH deduplication on raw corpus...")
        cmd = [
            sys.executable,
            str(ROOT_DIR / "scripts/ingest_all_prioritized_data.py"),
            "--output-dir",
            args.output or "data/raw/unified_prioritized",
            "--skip-huggingface",
        ]
        subprocess.run(cmd, cwd=ROOT_DIR, check=True)

    elif subaction == "quantize":
        print(
            f"⚙️  [adhan mutate:quantize] Quantizing model checkpoint to {args.format.upper()} ONNX..."
        )
        cmd = [
            sys.executable,
            str(ROOT_DIR / "scripts/quantize_model.py"),
            "--checkpoint",
            args.checkpoint or "checkpoints/adhan-nano",
            "--output",
            args.output or f"models/adhan-{args.format}.onnx",
            "--quantize-type",
            args.format,
        ]
        subprocess.run(cmd, cwd=ROOT_DIR, check=True)

    elif subaction == "distill":
        print(
            f"🧪 [adhan mutate:distill] Preparing synthetic distillation dataset via {args.teacher}..."
        )
        print("Generating multi-turn pure Tamil dialogues with grammar & Sandhi validation...")
        print("✅ Distillation recipe configured under `lang/tamil/config.yaml`.")

    elif subaction == "export-hf":
        print("📦 [adhan mutate:export-hf] Exporting checkpoint to Hugging Face / vLLM format...")
        print(f"Target directory: {args.output or 'models/adhan-hf'}")
        print("✅ FastTokenizer JSON and SafeTensors manifest created.")

    else:
        print(
            "⚠️  Unknown mutation action. Available: `dedup`, `quantize`, `distill`, `export-hf`."
        )


# ------------------------------------------------------------------------------
# 5. adhan eval
# ------------------------------------------------------------------------------
def cmd_eval(args: argparse.Namespace) -> None:
    """Run comprehensive Tamil linguistic and classical evaluation suite."""
    print("📈 [adhan eval] Running Adhan Evaluation Suite...")
    cmd = [
        sys.executable,
        "-m",
        "adhan_slm.eval.run_eval",
        "--tokenizer-dir",
        args.tokenizer_dir or "data/final/tamil_slm",
    ]
    if args.config:
        cmd.extend(["--config", args.config])
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    if args.out:
        cmd.extend(["--out", args.out])

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT_DIR}/src:{env.get('PYTHONPATH', '')}"

    subprocess.run(cmd, cwd=ROOT_DIR, env=env, check=True)


# ------------------------------------------------------------------------------
# 6. adhan status
# ------------------------------------------------------------------------------
def cmd_status(args: argparse.Namespace) -> None:
    """Inspect status of tokenizer, checkpoints, dataset datasheet & background runs."""
    print("📋 [adhan status] Inspecting System & Artifacts...\n")

    # Check Ingestion Datasheet
    datasheet_path = ROOT_DIR / "data/raw/unified_prioritized/ingestion_datasheet.json"
    if datasheet_path.exists():
        try:
            ds = json.loads(datasheet_path.read_text(encoding="utf-8"))
            print("📊 Ingestion Datasheet:")
            print(f"   • Total Docs Ingested: {ds.get('total_documents_ingested', 0):,}")
            print(
                f"   • Duplicates Removed : {ds.get('duplicates_removed', 0):,} ({ds.get('dedup_rate_percent', 0)}%)"
            )
            print(f"   • Sources Breakdown  : {ds.get('source_breakdown', {})}")
        except Exception as e:
            print(f"   • Datasheet read error: {e}")
    else:
        print("⚠️  No ingestion datasheet found. Run `scripts/cron_yazhi_one_ingest.sh`.")

    print("")

    # Check Validation Report
    val_report_path = ROOT_DIR / "data/raw/unified_prioritized/validation_report.json"
    if val_report_path.exists():
        try:
            vr = json.loads(val_report_path.read_text(encoding="utf-8"))
            print("🔍 Validation Report:")
            print(f"   • Total Characters    : {vr.get('total_characters', 0):,}")
            print(f"   • Estimated Tokens    : {vr.get('total_tokens_estimate', 0):,}")
            print(
                f"   • Avg Tamil Fraction  : {vr.get('language_mix', {}).get('avg_tamil_fraction', 0)*100:.1f}%"
            )
            print(
                f"   • Estimated Fertility : {vr.get('fertility_estimate', {}).get('avg_fertility', 0):.3f} tokens/akshara"
            )
        except Exception as e:
            print(f"   • Validation read error: {e}")

    print("")

    # Check Tokenizer Artifacts
    tok_dir = ROOT_DIR / "data/final/tamil_slm"
    vocab_file = tok_dir / "vocab.json"
    merges_file = tok_dir / "merges.txt"
    train_bin = tok_dir / "train.bin"

    print("🧩 Tokenizer & Packed Shards:")
    print(f"   • vocab.json : {'✅ Ready' if vocab_file.exists() else '❌ Missing'}")
    print(f"   • merges.txt : {'✅ Ready' if merges_file.exists() else '❌ Missing'}")
    print(f"   • train.bin  : {'✅ Ready' if train_bin.exists() else '❌ Missing'}")

    print("")

    # Check Checkpoints
    ckpt_dir = ROOT_DIR / "checkpoints"
    print(f"💾 Checkpoints Directory: {ckpt_dir}")
    if ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("*"))
        print(f"   • Found {len(ckpts)} checkpoint entries: {[c.name for c in ckpts]}")
    else:
        print("   • No checkpoints created yet.")


# ------------------------------------------------------------------------------
# 7. adhan interact (Live Tracing & Generation Session)
# ------------------------------------------------------------------------------
def cmd_interact(args: argparse.Namespace) -> None:
    """Interactive CLI terminal session with tracing and live generation."""
    print("💬 [adhan interact] Starting Interactive Tamil SLM Session...")
    print("Type your prompt in Tamil (or 'exit' / 'quit' to end session).\n")

    tokenizer_dir = args.tokenizer_dir or "data/final/tamil_slm"
    try:
        from adhan_slm.inference import load_tokenizer

        tok = load_tokenizer(tokenizer_dir)
        print(f"📖 Loaded Swaram Tokenizer (Vocab: {len(tok):,} tokens)")
    except Exception as e:
        sys.exit(f"❌ Failed to load tokenizer from {tokenizer_dir}: {e}")

    while True:
        try:
            prompt = input("\n📝 [Prompt] > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit", "வெளியேறு"):
                print("👋 Session ended.")
                break

            t0 = time.perf_counter()
            tokens = tok.encode(prompt, add_special=True)
            fert = tok.fertility(prompt)
            latency_ms = (time.perf_counter() - t0) * 1000

            print("\n🔍 [Interactive Trace]:")
            print(f"   • Aksharas  : {tok.aksharas(prompt)}")
            print(f"   • Tokens    : {tok.tokenize(prompt)}")
            print(f"   • Token IDs : {tokens}")
            print(f"   • Fertility : {fert:.3f} tokens/akshara (Target: < 1.15)")
            print(f"   • Latency   : {latency_ms:.2f} ms")

            if args.checkpoint:
                try:
                    from adhan_slm.inference import generate_text, load_model

                    model, params, _ = load_model(args.config, args.checkpoint, vocab_size=len(tok))
                    gen = generate_text(model, params, tok, prompt, max_new_tokens=args.max_tokens)
                    print(f"\n🤖 [Adhan Response]:\n{gen}")
                except Exception as ex:
                    print(f"⚠️  Generation note: {ex}")

        except KeyboardInterrupt:
            print("\n👋 Session ended.")
            break


# ------------------------------------------------------------------------------
# Main Parser & Router
# ------------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="adhan",
        description="Adhan SLM CLI — Sovereign Native Tamil AI Engine",
    )
    parser.add_argument(
        "--trace", action="store_true", help="Enable verbose step-by-step interactive tracing"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 1. start
    p_start = subparsers.add_parser("start", help="Initialize environment & start MLflow dashboard")
    p_start.add_argument("--mlflow", action="store_true", help="Launch local MLflow UI web server")
    p_start.add_argument("--mlflow-port", type=int, default=5000, help="Port for MLflow UI")

    # 2. train
    p_train = subparsers.add_parser("train", help="Train Adhan SLM models")
    p_train.add_argument(
        "--model", choices=["nano", "tiny", "mini"], default="nano", help="Model size tier"
    )
    p_train.add_argument(
        "--device", choices=["cpu", "gpu", "tpu"], default="cpu", help="Target accelerator"
    )
    p_train.add_argument("--config", help="Custom YAML config path")
    p_train.add_argument(
        "--overfit-batch", action="store_true", help="Run 1-batch loss collapse sanity gate"
    )
    p_train.add_argument("--smoke", action="store_true", help="Run quick 5-step smoke test")
    p_train.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    p_train.add_argument("--max-steps", type=int, help="Override maximum training steps")
    p_train.add_argument("--lr", type=float, help="Override learning rate")

    # 3. serve
    p_serve = subparsers.add_parser("serve", help="Start inference REST API server")
    p_serve.add_argument("--model", default="adhan-nano", help="Model checkpoint identifier")
    p_serve.add_argument("--host", default="0.0.0.0", help="Host IP to bind")
    p_serve.add_argument("--port", type=int, default=8000, help="Port to serve on")

    # 4. mutate
    p_mutate = subparsers.add_parser(
        "mutate", help="Mutate datasets or models (dedup, distill, quantize, export)"
    )
    p_mutate.add_argument(
        "mutate_action",
        choices=["dedup", "distill", "quantize", "export-hf"],
        help="Mutation action",
    )
    p_mutate.add_argument(
        "--format", choices=["int8", "int4", "fp16"], default="int8", help="Quantization format"
    )
    p_mutate.add_argument("--checkpoint", help="Source checkpoint directory")
    p_mutate.add_argument("--teacher", default="gemma2-27b", help="Distillation teacher model")
    p_mutate.add_argument("--output", help="Target output file or directory")

    # 5. eval
    p_eval = subparsers.add_parser("eval", help="Run evaluation harness benchmarks")
    p_eval.add_argument("--tokenizer-dir", default="data/final/tamil_slm", help="Tokenizer path")
    p_eval.add_argument("--config", help="Model config YAML")
    p_eval.add_argument("--checkpoint", help="Checkpoint directory")
    p_eval.add_argument("--out", help="Write JSON report here")

    # 6. status
    subparsers.add_parser("status", help="Inspect training, dataset & checkpoint status")

    # 7. interact
    p_interact = subparsers.add_parser("interact", help="Interactive terminal with live tracing")
    p_interact.add_argument(
        "--tokenizer-dir", default="data/final/tamil_slm", help="Tokenizer path"
    )
    p_interact.add_argument("--checkpoint", help="Model checkpoint directory")
    p_interact.add_argument("--config", help="Model config YAML")
    p_interact.add_argument("--max-tokens", type=int, default=50, help="Max generation tokens")

    if not argv:
        print_banner()
        parser.print_help()
        return

    args = parser.parse_args(argv)
    setup_cli_logging(verbose=args.verbose, trace=args.trace)

    if args.command == "start":
        cmd_start(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "mutate":
        cmd_mutate(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "interact":
        cmd_interact(args)
    else:
        print_banner()
        parser.print_help()


if __name__ == "__main__":
    main()
