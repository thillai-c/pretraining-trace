#!/usr/bin/env bash
# Sequential E2 windowed co-occurrence for every OLMo 2 model key in utils.MODELS.
#
# Usage (from repo root):
#   bash scripts/run_e2_windowed_cooccurrence.sh
#   TRAINING_PHASE=mid_training bash scripts/run_e2_windowed_cooccurrence.sh
#
# Override env vars as needed:
#   PYTHON, CONFIG, TRAINING_PHASE, E2_LLM, API_INDEX, TOP_NS, WINDOWS, MAX_CLAUSE_FREQ, EXTRA_PY_ARGS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root (parent of this scripts/ directory) so e2_windowed_cooccurrence.py resolves.
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-standard}"
TRAINING_PHASE="${TRAINING_PHASE:-pretraining}"
E2_LLM="${E2_LLM:-gpt-5.4-mini}"
API_INDEX="${API_INDEX:-v4_olmo-mix-1124_llama}"
# Space-separated integers for --top_n (passed as multiple args)
TOP_NS="${TOP_NS:-5 10 15 20}"
# Space-separated window sizes
WINDOWS="${WINDOWS:-100 500 1000}"
# API max is 500000; default None would fall back to server default (~50000).
MAX_CLAUSE_FREQ="${MAX_CLAUSE_FREQ:-500000}"
EXTRA_PY_ARGS="${EXTRA_PY_ARGS:-}"
# Interpreter (override in WSL: PYTHON=python3)
PYTHON="${PYTHON:-python}"

# Must match choices=list(MODELS.keys()) in e2_windowed_cooccurrence.py (olmo2-* only).
MODELS=(
  olmo2-1b-instruct
  olmo2-7b-instruct
  olmo2-13b-instruct
  olmo2-32b-instruct
  olmo2-1b
  olmo2-7b
  olmo2-13b
  olmo2-32b
)

read -r -a TOP_N_ARRAY <<< "${TOP_NS}"
read -r -a WINDOWS_ARRAY <<< "${WINDOWS}"

started="$(date -Iseconds)"
echo "=== run_e2_windowed_cooccurrence.sh start ${started} (cwd=${REPO_ROOT}) ==="
echo "PYTHON=${PYTHON} ($(command -v "${PYTHON}" 2>/dev/null || echo 'not found'))"
echo "CONFIG=${CONFIG} TRAINING_PHASE=${TRAINING_PHASE} E2_LLM=${E2_LLM} API_INDEX=${API_INDEX}"
echo "TOP_N: ${TOP_N_ARRAY[*]}"
echo "WINDOWS: ${WINDOWS_ARRAY[*]}"
echo "MAX_CLAUSE_FREQ: ${MAX_CLAUSE_FREQ}"
echo

for model in "${MODELS[@]}"; do
  echo "-------------------------------------------------------------------"
  echo "Model: ${model}"
  echo "-------------------------------------------------------------------"
  # shellcheck disable=SC2086
  "${PYTHON}" e2_windowed_cooccurrence.py \
    --model "${model}" \
    --training-phase "${TRAINING_PHASE}" \
    --config "${CONFIG}" \
    --e2-llm "${E2_LLM}" \
    --api_index "${API_INDEX}" \
    --top_n "${TOP_N_ARRAY[@]}" \
    --max_clause_freq "${MAX_CLAUSE_FREQ}" \
    --compliant_only \
    --windows "${WINDOWS_ARRAY[@]}" \
    ${EXTRA_PY_ARGS}
  echo
done

finished="$(date -Iseconds)"
echo "=== done ${finished} ==="