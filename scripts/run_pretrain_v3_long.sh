#!/usr/bin/env bash
# ==============================================================================
# Adhan SLM — launch / resume the long v3 CPU pretraining run
# ==============================================================================
# Runs the trainer as a *transient systemd user service* rather than a bare
# nohup process. Two reasons, both learned the hard way:
#
#   1. A terminal-launched child inherits `oom_score_adj=200`, so the kernel
#      deliberately picks the trainer first when memory runs out. The run was
#      OOM-killed at step 99 on 2026-10-27 for exactly this reason.
#   2. A cgroup memory cap makes the failure mode predictable: the trainer is
#      killed at its own limit instead of starving GNOME/Firefox and taking the
#      whole session down with it.
#
# `loginctl` has Linger=yes for this user, so the unit survives logout.
# Orbax resumes from the newest checkpoint, so re-running this script after a
# kill continues where it left off.
#
#   ./scripts/run_pretrain_v3_long.sh            # start (or resume)
#   ./scripts/run_pretrain_v3_long.sh --status   # is it running?
#   ./scripts/run_pretrain_v3_long.sh --stop     # graceful stop
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
UNIT="adhan-pretrain-v3"
CONFIG="src/adhan_slm/configs/adhan_slm_nano_cpu_v3_long.yaml"

mkdir -p "${LOG_DIR}"

is_running() { systemctl --user is-active --quiet "${UNIT}.service"; }

case "${1:-start}" in
  --status)
    if is_running; then
      systemctl --user status "${UNIT}.service" --no-pager -n 5
    else
      echo "not running"
      exit 1
    fi
    exit 0
    ;;
  --stop)
    if is_running; then
      systemctl --user stop "${UNIT}.service"
      echo "stopped ${UNIT}"
    else
      echo "not running"
    fi
    exit 0
    ;;
esac

if is_running; then
  echo "${UNIT} is already running; use --status or --stop"
  exit 0
fi

LOG_FILE="${LOG_DIR}/pretrain_v3_long_$(date +%Y%m%d_%H%M%S).log"
echo "${LOG_FILE}" > "${LOG_DIR}/pretrain_v3_long_current.txt"

# MemoryMax: the trainer peaks around 3 GB; 10 GB leaves head-room for a larger
# batch later while still guaranteeing the desktop keeps its memory.
systemd-run --user \
  --unit="${UNIT}" \
  --description="Adhan SLM nano v3 long CPU pretraining" \
  -p WorkingDirectory="${REPO_ROOT}" \
  -p MemoryMax=10G \
  -p MemorySwapMax=8G \
  -p OOMScoreAdjust=0 \
  -p StandardOutput=append:"${LOG_FILE}" \
  -p StandardError=append:"${LOG_FILE}" \
  "${REPO_ROOT}/.venv/bin/python" -m adhan_slm.training.train_jax \
    --config "${CONFIG}" \
    --device cpu

echo "launched ${UNIT} -> ${LOG_FILE}"
