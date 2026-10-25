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
# `prepare_slm_corpus.py` holds the whole corpus in RAM *and* runs a num_perm=128
# near-duplicate pass over it. On this archive (1.33M documents) that is ~9 GiB and
# tens of minutes — observed live: it pushed the box to 9.1 GiB while a training run
# was in progress, with OOM as the likely outcome. So the nightly job bounds the work
# with --limit and says so, instead of silently trying to swallow the archive.
#
#   ADHAN_PACK_DOCS=50000   (default) bounded nightly pack
#   ADHAN_PACK_DOCS=0       unbounded — only on an idle box, and ideally with the
#                           streaming packer (scripts/freeze_fast_tokenizer.py) once
#                           the tokenizer branch lands.
echo ""
echo ">> Step 3: Preparing SLM Tokenizer & Binary Shards (bounded)..."
PACK_DOCS="${ADHAN_PACK_DOCS:-50000}"
LIMIT_ARGS=()
if [ "${PACK_DOCS}" != "0" ]; then
    LIMIT_ARGS=(--limit "${PACK_DOCS}")
    echo "   (packing ${PACK_DOCS} documents; set ADHAN_PACK_DOCS=0 for the whole archive)"
else
    echo "   (ADHAN_PACK_DOCS=0: packing the whole archive — needs ~10 GiB free)"
fi
python "${REPO_ROOT}/scripts/prepare_slm_corpus.py" \
    --corpus "${CORPUS_DIR}" \
    --out "${FINAL_SLM_DIR}" \
    --vocab-size 12000 \
    --seq-len 1024 \
    "${LIMIT_ARGS[@]}"

echo ""
echo "========================================================================"
echo " [Yazhi-One Cron] Pipeline Finished Successfully!"
echo " Log written to: ${LOG_FILE}"
echo "========================================================================"
