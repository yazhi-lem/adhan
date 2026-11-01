#!/usr/bin/env bash
# One command: reproduce the Tamil tokenizer benchmark from a clean checkout.
#   scripts/reproduce_benchmark.sh           rebuild and write the tables
#   scripts/reproduce_benchmark.sh --check   rebuild and compare with the committed tables
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
exec python scripts/reproduce_benchmark.py "$@"
