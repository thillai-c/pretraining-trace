#!/usr/bin/env bash
# Sequential E2 Stage 2 (concept ranking) for OLMo 2 models.
#
# Usage (from repo root):
#   bash scripts/run_e2_rank_concepts.sh
#   TRAINING_PHASE=mid_training bash scripts/run_e2_rank_concepts.sh
#
# MODE controls which e2_rank_concepts.py mode to run:
#   MODE=batch   -> submit batch jobs (default)
#   MODE=collect -> collect results (requires batch_metadata.json per model)
#   MODE=retry   -> retry failed records
#
# Note:
# - This script assumes Stage 1 output already exists at:
#   results/{out_dir}/e2/{e2_llm}/e2_concepts_{config}.json
#   (stage-independent under the new layout; --training-phase is cosmetic for Stage 2)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root (parent of this scripts/ directory) so e2_rank_concepts.py resolves.
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-standard}"
TRAINING_PHASE="${TRAINING_PHASE:-mid_training}"
# Must match Stage 1 output folder and is also the ranking model id.
E2_LLM="${E2_LLM:-gpt-5-mini}"
MODE="${MODE:-batch}"
EXTRA_PY_ARGS="${EXTRA_PY_ARGS:-}"
# Interpreter (override in WSL: PYTHON=python3)
PYTHON="${PYTHON:-python}"

# Must match choices=list(MODELS.keys()) in e2_rank_concepts.py (olmo2-* only).
MODELS=(
  olmo2-1b
  olmo2-7b
  olmo2-13b
  olmo2-32b
  olmo2-1b-instruct
  olmo2-7b-instruct
  olmo2-13b-instruct
  olmo2-32b-instruct
)

started="$(date -Iseconds)"
echo "=== run_e2_rank_concepts.sh start ${started} (cwd=${REPO_ROOT}) ==="
echo "PYTHON=${PYTHON} ($(command -v "${PYTHON}" 2>/dev/null || echo 'not found'))"
echo "MODE=${MODE} CONFIG=${CONFIG} TRAINING_PHASE=${TRAINING_PHASE} E2_LLM=${E2_LLM}"
echo

for model in "${MODELS[@]}"; do
  echo "-------------------------------------------------------------------"
  echo "Model: ${model}"
  echo "-------------------------------------------------------------------"
  # shellcheck disable=SC2086
  "${PYTHON}" e2_rank_concepts.py \
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
