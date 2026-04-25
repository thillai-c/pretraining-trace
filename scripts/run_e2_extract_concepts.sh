#!/usr/bin/env bash
# Sequential E2 Stage 1 (concept extraction) batch submission for OLMo 2 models.
#
# Usage (from repo root):
#   bash scripts/run_e2_extract_concepts.sh
#   TRAINING_PHASE=mid_training bash scripts/run_e2_extract_concepts.sh
#
# Notes:
# - MODE controls which e2_extract_concepts.py mode to run:
#   MODE=batch   -> submit batch jobs
#   MODE=collect -> collect results (requires batch_metadata.json per model)
#   MODE=retry   -> retry failed records

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root (parent of this scripts/ directory) so e2_extract_concepts.py resolves.
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-standard}"
TRAINING_PHASE="${TRAINING_PHASE:-post_training}"
# Matches the default in e2_extract_concepts.py (used for output subfolder).
E2_LLM="${E2_LLM:-gpt-5-mini}"
MODE="${MODE:-batch}"
EXTRA_PY_ARGS="${EXTRA_PY_ARGS:-}"
# Interpreter (override in WSL: PYTHON=python3)
PYTHON="${PYTHON:-python}"

# Must match choices=list(MODELS.keys()) in e2_extract_concepts.py (olmo2-* only).
MODELS=(
  # olmo2-1b
  # olmo2-7b
  # olmo2-13b
  # olmo2-32b
  olmo2-1b-instruct
  olmo2-7b-instruct
  olmo2-13b-instruct
  olmo2-32b-instruct
)

started="$(date -Iseconds)"
echo "=== run_e2_extract_concepts.sh start ${started} (cwd=${REPO_ROOT}) ==="
echo "PYTHON=${PYTHON} ($(command -v "${PYTHON}" 2>/dev/null || echo 'not found'))"
echo "MODE=${MODE} CONFIG=${CONFIG} TRAINING_PHASE=${TRAINING_PHASE} E2_LLM=${E2_LLM}"
echo

for model in "${MODELS[@]}"; do
  echo "-------------------------------------------------------------------"
  echo "Model: ${model}"
  echo "-------------------------------------------------------------------"
  # shellcheck disable=SC2086
  "${PYTHON}" e2_extract_concepts.py \
    --model "${model}" \
    --training-phase "${TRAINING_PHASE}" \
    --config "${CONFIG}" \
    --e2-llm "${E2_LLM}" \
    "--${MODE}" \
    ${EXTRA_PY_ARGS}
  echo
done

finished="$(date -Iseconds)"
echo "=== done ${finished} ==="