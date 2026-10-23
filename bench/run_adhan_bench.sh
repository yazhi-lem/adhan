#!/usr/bin/env bash
# Grade an Adhan checkpoint on the tamil-bench papers.
#
#   ./bench/run_adhan_bench.sh [ENDPOINT]
#
# Default endpoint assumes `python scripts/run_api_server.py --port 8000`.
# Results land in bench/tamil-bench/results/ (one JSONL answer sheet per run).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$HERE/tamil-bench"
ENDPOINT="${1:-http://127.0.0.1:8000/v1/chat/completions}"
MODEL="${ADHAN_BENCH_MODEL:-adhan-nano}"

if [ ! -f "$BENCH/bench.py" ]; then
  echo "tamil-bench missing; run ./bench/setup_tamil_bench.sh first" >&2
  exit 1
fi

export TAMIL_BENCH_ENDPOINT="$ENDPOINT"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-local}"

echo ">> checking endpoint: $ENDPOINT"
curl -sf "${ENDPOINT%/chat/completions}/models" >/dev/null \
  && echo "   endpoint reachable" \
  || { echo "   endpoint (or /v1/models) unreachable — is \`scripts/run_api_server.py\` running with a checkpoint?" >&2; exit 1; }

cd "$BENCH"

echo ">> indicqa (n=100)"
python3 bench.py indicqa --model "$MODEL" --n 100

echo ">> xnli (n=200)"
python3 bench.py xnli --model "$MODEL" --n 200

if [ -f "data/milu_ta_test.parquet" ]; then
  echo ">> milu (n=199)"
  python3 bench.py milu --model "$MODEL" --n 199
else
  echo ">> milu skipped (data/milu_ta_test.parquet missing — HF gate)"
fi

echo ">> answer sheets:"
ls -1 results/ | tail -10
