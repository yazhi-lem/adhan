#!/usr/bin/env bash
# ==============================================================================
# Yazhi-One Background Data Ingestion & Validation Cron Runner
# ==============================================================================
# Scheduled offline pipeline for:
#   1. Ingesting raw prioritized Tamil corpora (Open-Sangam, Wikipedia, IndicCorp)
#   2. Validating language ratio, token fertility (<1.15), and PII scrubbing
#   3. Packing tokenized binary shards for Adhan SLM training
#
# Designed for low compute & zero LLM inference overhead (pure local NLP).
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)"
REPOS_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

# Ensure log directory exists
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/cron_ingest_$(date +'%Y%m%d_%H%M%S').log"

# Link latest log
ln -sfn "${LOG_FILE}" "${LOG_DIR}/cron_ingest_latest.log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================================================"
echo " [Yazhi-One Cron] Starting Data Ingestion & Validation Suite"
echo " Timestamp: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo " Working Dir: ${REPO_ROOT}"
echo "========================================================================"

# Activate virtual environment if present
if [ -d "${REPO_ROOT}/.venv" ]; then
    echo ">> Activating venv: ${REPO_ROOT}/.venv"
    source "${REPO_ROOT}/.venv/bin/activate"
elif [ -d "${WORKSPACE_ROOT}/.venv" ]; then
    echo ">> Activating workspace venv: ${WORKSPACE_ROOT}/.venv"
    source "${WORKSPACE_ROOT}/.venv/bin/activate"
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

CORPUS_DIR="${REPO_ROOT}/data/raw/expanded"
VALIDATION_REPORT="${REPO_ROOT}/data/raw/expanded/validation_report.json"
FINAL_SLM_DIR="${REPO_ROOT}/data/final/tamil_slm"

mkdir -p "${CORPUS_DIR}" "${FINAL_SLM_DIR}"

# ------------------------------------------------------------------------------
# STEP 1: Ingest / Extend the Corpus (streaming, resumable, deduplicated)
# ------------------------------------------------------------------------------
# Appends to the existing shards: documents already on disk are seeded by
# exact hash, so re-running only adds what is new. Completed sources are
# recorded in state.json and skipped. See docs/CORPUS_EXPANSION.md.
echo ""
echo ">> Step 1: Expanding the Tamil corpus (streaming)..."
python "${REPO_ROOT}/scripts/expand_tamil_corpus.py" \
    --out-dir "${CORPUS_DIR}" \
    --wiki-limit 150000 \
    --indic-limit 1000000

# ------------------------------------------------------------------------------
# STEP 2: Validate Data Quality, PII & Fertility
# ------------------------------------------------------------------------------
# The archive is far too large to load whole; validate a bounded sample.
echo ""
echo ">> Step 2: Validating the corpus..."
python "${REPO_ROOT}/scripts/phase2_validate.py" \
    --corpus "${CORPUS_DIR}" \
    --max-records 50000 \
    --output "${VALIDATION_REPORT}"

# ------------------------------------------------------------------------------
# STEP 3: Tokenize and Pack Shards for Training
# ------------------------------------------------------------------------------
echo ""
echo ">> Step 3: Preparing SLM Tokenizer & Binary Shards..."
python "${REPO_ROOT}/scripts/prepare_slm_corpus.py" \
    --corpus "${CORPUS_DIR}" \
    --out "${FINAL_SLM_DIR}" \
    --vocab-size 12000 \
    --seq-len 1024

echo ""
echo "========================================================================"
echo " [Yazhi-One Cron] Pipeline Finished Successfully!"
echo " Log written to: ${LOG_FILE}"
echo "========================================================================"
