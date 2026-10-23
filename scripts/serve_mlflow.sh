#!/usr/bin/env bash
# ==============================================================================
# serve_mlflow.sh — start / stop / inspect the MLflow UI for this repo
# ==============================================================================
# Why this exists: training here logs to a *SQLite file store* (see
# `mlflow_uri: sqlite:///mlflow.db` in src/adhan_slm/configs/*.yaml), not to a
# tracking server. Metrics are recorded whether or not anything is running — the
# UI is a separate, read-only viewer that nobody starts by default. That gap is
# exactly what makes ":5000 is down" look like "training is not logging".
#
# Usage:
#   scripts/serve_mlflow.sh start [port]   # background UI, waits until it answers
#   scripts/serve_mlflow.sh status [port]  # process + port + HTTP + latest run
#   scripts/serve_mlflow.sh stop   [port]
#   scripts/serve_mlflow.sh copy   [port]  # serve a *copy* of the DB (default 5001)
#   scripts/serve_mlflow.sh env            # print the export line clients need
#   scripts/serve_mlflow.sh doctor         # version / schema-skew check
#
# `copy` exists because SQLite serialises writers: while a trainer is appending
# metrics, a heavy history query from the UI can collide with it. Serving a
# snapshot (`/tmp/mlflow_view.db`) removes that coupling entirely.
#
# Bind address: 0.0.0.0 by default so the internal dev server is reachable over
# Tailscale/LAN. Set MLFLOW_HOST=127.0.0.1 to keep it loopback-only.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DB="${REPO_ROOT}/mlflow.db"
HOST="${MLFLOW_HOST:-0.0.0.0}"
PID_FILE="${REPO_ROOT}/logs/mlflow_ui.pid"
LOG_FILE="${REPO_ROOT}/logs/mlflow_ui.log"
ACTION="${1:-status}"
PORT="${2:-5000}"

mkdir -p "${REPO_ROOT}/logs"

# Prefer the repo venv, fall back to whatever mlflow is on PATH.
MLFLOW_BIN="mlflow"
if [ -x "${REPO_ROOT}/.venv/bin/mlflow" ]; then
    MLFLOW_BIN="${REPO_ROOT}/.venv/bin/mlflow"
fi

url_for() { echo "http://127.0.0.1:$1"; }

is_up() { # is_up PORT -> exit 0 if HTTP answers
    curl -sf -o /dev/null --max-time 3 "$(url_for "$1")/" 2>/dev/null
}

pid_on_port() { # pid_on_port PORT -> pid or empty
    ss -ltnp 2>/dev/null | awk -v p=":$1" '$4 ~ p {print}' |
        grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2
}

latest_run() {
    local db_path="$1"
    python - "$db_path" <<'PY' 2>/dev/null || echo "   (no run data available)"
import sys, mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("sqlite:///" + sys.argv[1])
c = MlflowClient()
for exp in c.search_experiments():
    runs = c.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        continue
    r = runs[0]
    metrics = {k: round(v, 3) for k, v in sorted(r.data.metrics.items())}
    print(f"   experiment : {exp.name}")
    print(f"   run        : {r.info.run_id[:8]}  status={r.info.status}")
    print(f"   metrics    : {metrics}")
PY
}

cmd_status() {
    echo "== mlflow status =="
    if is_up "${PORT}"; then
        echo "   port ${PORT}    : UP  ($(url_for "${PORT}"))"
    else
        echo "   port ${PORT}    : DOWN"
    fi
    local pid; pid=$(pid_on_port "${PORT}")
    echo "   pid on port    : ${pid:-none}"
    echo "   backend store  : sqlite:///${DB} ($([ -f "${DB}" ] && du -h "${DB}" | cut -f1 || echo missing))"
    echo "   log            : ${LOG_FILE}"
    echo "   latest run:"
    latest_run "${DB}"
}

cmd_start() {
    if is_up "${PORT}"; then
        echo "already up on ${PORT} -> $(url_for "${PORT}")"
        cmd_status
        return 0
    fi
    if [ ! -f "${DB}" ]; then
        echo "error: ${DB} not found — run a training job first (it creates the store)" >&2
        exit 1
    fi

    # `mlflow ui` runs a master that forks uvicorn workers, a job runner, and a
    # pool of huey consumers. Starting it with `setsid` puts that whole tree in
    # its own process group, which is the only reliable handle for stopping it —
    # killing the worker that owns the socket just makes the master respawn one.
    echo "starting mlflow ui on ${HOST}:${PORT} ..."
    setsid "${MLFLOW_BIN}" ui \
        --backend-store-uri "sqlite:///${DB}" \
        --host "${HOST}" --port "${PORT}" \
        >> "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"

    # Poll until it actually answers: `mlflow ui` forks workers, so "process
    # started" is not the same as "UI ready", and reporting success too early is
    # what makes this feel flaky.
    for _ in $(seq 1 30); do
        if is_up "${PORT}"; then
            echo "up -> $(url_for "${PORT}")   (pid $(cat "${PID_FILE}"), log ${LOG_FILE})"
            return 0
        fi
        sleep 1
    done
    echo "failed to become ready in 30s; last log lines:" >&2
    tail -20 "${LOG_FILE}" >&2
    exit 1
}

cmd_stop() {
    local pid; pid=$(cat "${PID_FILE}" 2>/dev/null || true)
    local port_pid; port_pid=$(pid_on_port "${PORT}")

    if [ -z "${pid}" ] && [ -z "${port_pid}" ] &&
        ! pgrep -f "mlflow ui .*--port ${PORT}" >/dev/null 2>&1; then
        echo "nothing running on ${PORT}"
        return 0
    fi

    # Preferred path: the process group started by `start` (setsid).
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
        echo "stopping process group ${pid} (port ${PORT})"
        kill -TERM -"${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
        for _ in $(seq 1 10); do
            is_up "${PORT}" || break
            sleep 1
        done
        kill -KILL -"${pid}" 2>/dev/null || true
    fi

    # Fallback / cleanup: anything still serving this port or matching this UI.
    pkill -f "mlflow ui .*--port ${PORT}" 2>/dev/null || true
    pkill -f "uvicorn .*--port ${PORT} .*mlflow" 2>/dev/null || true
    if [ -n "${port_pid}" ]; then
        kill -TERM "${port_pid}" 2>/dev/null || true
    fi

    for _ in $(seq 1 10); do
        is_up "${PORT}" || break
        sleep 1
    done
    rm -f "${PID_FILE}"

    if is_up "${PORT}"; then
        echo "error: port ${PORT} still answering after stop" >&2
        echo "       try: pkill -9 -f 'mlflow ui' (this kills every mlflow UI on the box)" >&2
        exit 1
    fi
    echo "stopped; port ${PORT} is free"
}

cmd_copy() {
    local view_db="/tmp/mlflow_view.db"
    # Port arg may be the live UI's port; copy mode always takes its own unless
    # MLFLOW_COPY_PORT says otherwise, so you can watch both side by side.
    local view_port="${MLFLOW_COPY_PORT:-5001}"
    if [ ! -f "${DB}" ]; then
        echo "error: ${DB} not found" >&2; exit 1
    fi
    # sqlite3 .backup is safe against a live writer; a plain cp can catch a
    # half-written WAL and produce a corrupt-looking snapshot.
    if command -v sqlite3 >/dev/null 2>&1; then
        sqlite3 "${DB}" ".backup '${view_db}'"
    else
        cp "${DB}" "${view_db}"
    fi
    echo "snapshot: ${view_db} ($(du -h "${view_db}" | cut -f1))"
    if is_up "${view_port}"; then
        echo "port ${view_port} already busy; set MLFLOW_COPY_PORT to another port"
        return 0
    fi
    setsid "${MLFLOW_BIN}" ui --backend-store-uri "sqlite:///${view_db}" \
        --host "${HOST}" --port "${view_port}" >> "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}.copy"
    for _ in $(seq 1 30); do
        if is_up "${view_port}"; then
            echo "up (read-only snapshot) -> $(url_for "${view_port}")"
            return 0
        fi
        sleep 1
    done
    echo "failed to start snapshot server; see ${LOG_FILE}" >&2
    exit 1
}

cmd_env() {
    # The durable fix for "no runs in mlflow": every client (CLI, python, uvx,
    # the UI) resolves its store from MLFLOW_TRACKING_URI. Unset, MLflow falls
    # back to ./mlruns — the legacy *file* store, which is in maintenance mode
    # and, for this repo, only holds stale pre-SQLite data.
    cat <<ENV
export MLFLOW_TRACKING_URI="sqlite:///${DB}"

# then a bare UI/clients all point at the real runs:
mlflow ui --host ${HOST} --port ${PORT}
uvx mlflow@latest ui --host ${HOST} --port ${PORT}
ENV
}

cmd_doctor() {
    # MLflow migrates the SQLite schema *forward* on first use. So a newer client
    # touching this DB (e.g. `uvx mlflow@latest ...`) silently upgrades it, after
    # which the repo's pinned mlflow refuses to open it with
    # "out-of-date database schema (found X, expected Y)" — misleading wording: the
    # DB is newer, not older. This happened here once: uvx 3.17.2 migrated the store
    # and every new client on the venv (3.15.1) started failing.
    local venv_v db_rev
    venv_v=$("${REPO_ROOT}/.venv/bin/python" -c "import mlflow; print(mlflow.__version__)" 2>/dev/null || echo "n/a")
    db_rev=$(sqlite3 "${DB}" "select version_num from alembic_version" 2>/dev/null || echo "n/a")

    echo "== mlflow doctor =="
    echo "   repo venv mlflow : ${venv_v}"
    echo "   db schema rev    : ${db_rev}"
    echo "   db file          : ${DB} ($([ -f "${DB}" ] && du -h "${DB}" | cut -f1 || echo missing))"

    # Can the repo's own client actually open the store right now?
    if "${REPO_ROOT}/.venv/bin/python" - "${DB}" <<'PYEOF' 2>/dev/null
import sys
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///" + sys.argv[1])
MlflowClient().search_experiments()
PYEOF
    then
        echo "   client opens db  : yes"
    else
        echo "   client opens db  : NO  <-- schema skew, or the file is locked"
        echo "                      fix: pip install -U mlflow   (match the newest client"
        echo "                      that touched this db), then re-run this check"
    fi
    is_up "${PORT}" && echo "   ui on ${PORT}     : UP" || echo "   ui on ${PORT}     : down"
}

case "${ACTION}" in
    start)  cmd_start ;;
    stop)   cmd_stop ;;
    status) cmd_status ;;
    copy)   cmd_copy ;;
    env)    cmd_env ;;
    doctor) cmd_doctor ;;
    *)      echo "usage: $0 {start|stop|status|copy|env|doctor} [port]" >&2; exit 2 ;;
esac
