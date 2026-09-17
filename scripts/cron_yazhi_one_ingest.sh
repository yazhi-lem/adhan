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

RAW_OUTPUT_DIR="${REPO_ROOT}/data/raw/unified_prioritized"
CORPUS_FILE="${RAW_OUTPUT_DIR}/unified_corpus.jsonl"
VALIDATION_REPORT="${RAW_OUTPUT_DIR}/validation_report.json"
FINAL_SLM_DIR="${REPO_ROOT}/data/final/tamil_slm"
SANGAM_PATH="${REPOS_ROOT}/open-sangam"

mkdir -p "${RAW_OUTPUT_DIR}" "${FINAL_SLM_DIR}"

# ------------------------------------------------------------------------------
# STEP 1: Ingest Prioritized Data (Zero API Inference, Local/Stream Extraction)
# ------------------------------------------------------------------------------
echo ""
echo ">> Step 1: Running Prioritized Data Ingestion..."
python "${REPO_ROOT}/scripts/ingest_all_prioritized_data.py" \
    --output-dir "${RAW_OUTPUT_DIR}" \
    --sangam-path "${SANGAM_PATH}" \
    --limit-wiki 150000 \
    --limit-indic 250000

# ------------------------------------------------------------------------------
# STEP 2: Validate Data Quality, PII & Fertility
# ------------------------------------------------------------------------------
echo ""
echo ">> Step 2: Validating Ingested Corpus..."
python "${REPO_ROOT}/scripts/phase2_validate.py" \
    --corpus "${CORPUS_FILE}" \
    --output "${VALIDATION_REPORT}"

# ------------------------------------------------------------------------------
# STEP 3: Tokenize and Pack Shards for Training
# ------------------------------------------------------------------------------
echo ""
echo ">> Step 3: Preparing SLM Tokenizer & Binary Shards..."
python "${REPO_ROOT}/scripts/prepare_slm_corpus.py" \
    --corpus "${CORPUS_FILE}" \
    --out "${FINAL_SLM_DIR}" \
    --vocab-size 12000 \
    --seq-len 1024

echo ""
echo "========================================================================"
echo " [Yazhi-One Cron] Pipeline Finished Successfully!"
echo " Log written to: ${LOG_FILE}"
echo "========================================================================"
