#!/bin/bash
#SBATCH --job-name=index_step2_3
#SBATCH --nodes=1
#SBATCH --partition=hbm-1tb-long-96core
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Step 2.3: rust_indexing concat (dolmino-mix-1124).
#
# Usage:
#   sbatch run_indexing_step2_3_concat.sh
#
# Slurm: match run_indexing_step2_2_merge.sh (hbm-1tb-long-96core, 96 CPUs).
# --mem=128G: concat uses sparse table.0 + parallel writers; far less RAM than merge.
# If submit fails (QOS), try hbm-long-96core or long-96core with same CPUs/mem limits.
#
# Prerequisite: run_indexing_step2_2_merge.sh (merged-0/ populated).
#
# Build binary (on compute node):
#   sbatch build_rust_indexing.sh

set -euo pipefail

DATASET_TAG="dolmino-mix-1124"
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

mkdir -p logs
exec > >(tee "logs/indexing_step2_3.out")
exec 2> >(tee "logs/indexing_step2_3.err" >&2)

if command -v module &>/dev/null; then
  module load anaconda/3
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /gpfs/home/solhapark/envs/infinigram

if [[ -f /gpfs/projects/ZhouJGroup/Users/solhapark/pretraining-trace/.env ]]; then
  set -a
  source /gpfs/projects/ZhouJGroup/Users/solhapark/pretraining-trace/.env
  set +a
fi

INFINI_PKG="/gpfs/projects/ZhouJGroup/Users/solhapark/pretraining-trace/infini-gram/pkg"
cd "${INFINI_PKG}"

SAVE_DIR="/gpfs/projects/ZhouJGroup/Users/solhapark/pretraining-trace/index/${DATASET_TAG}"
TEMP_DIR="${SAVE_DIR}"
CPUS=96
TOKEN_WIDTH=2

echo "=== Step 2.3 (concat) started at $(date) ==="
START_TIME=$(date +%s)

DS_PATH="${SAVE_DIR}/tokenized.0"
MERGED_DIR="${TEMP_DIR}/merged-0"
SA_PATH="${SAVE_DIR}/table.0"

if [[ ! -f "${DS_PATH}" ]]; then
  echo "ERROR: tokenized.0 not found."
  exit 1
fi
if [[ ! -d "${MERGED_DIR}" ]] || [[ -z "$(ls -A "${MERGED_DIR}" 2>/dev/null)" ]]; then
  echo "ERROR: merged-0 not found or empty. Run Step 2.2 (merge) first."
  exit 1
fi
if [[ -f "${SA_PATH}" ]]; then
  echo "WARNING: table.0 already exists. Will be overwritten."
fi

RUST_INDEXING_BIN="${INFINI_PKG}/target/release/rust_indexing"
if [[ ! -x "${RUST_INDEXING_BIN}" ]]; then
  echo "ERROR: rust_indexing missing or not executable: ${RUST_INDEXING_BIN}"
  exit 127
fi

DS_SIZE=$(stat -f%z "${DS_PATH}" 2>/dev/null || stat -c%s "${DS_PATH}")
RATIO=$(python3 -c "import math; print(int(math.ceil(math.log2(${DS_SIZE}) / 8)))")

echo "Ratio: ${RATIO}, Token width: ${TOKEN_WIDTH}"
echo "Merged directory: ${MERGED_DIR}"
echo "Merged part file count: $(find "${MERGED_DIR}" -type f | wc -l)"
echo "Output: ${SA_PATH}"

"${RUST_INDEXING_BIN}" concat \
  --data-file "${DS_PATH}" \
  --merged-dir "${MERGED_DIR}" \
  --merged-file "${SA_PATH}" \
  --num-threads "${CPUS}" \
  --ratio "${RATIO}" \
  --token-width "${TOKEN_WIDTH}"

if [[ -f "${SA_PATH}" ]]; then
  SA_SIZE=$(stat -f%z "${SA_PATH}" 2>/dev/null || stat -c%s "${SA_PATH}")
  echo "Success! table.0: ${SA_SIZE} bytes ($(numfmt --to=iec-i --suffix=B "${SA_SIZE}"))"
else
  echo "ERROR: table.0 was not created"
  exit 1
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=== Step 2.3 (concat) finished at $(date) ==="
printf "=== Elapsed time: %d days %02d:%02d:%02d (total %d seconds)\n" \
  $((ELAPSED/86400)) $(((ELAPSED%86400)/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"

echo ""
echo "=== Indexing complete (${DATASET_TAG})! ==="
ls -lh "${SAVE_DIR}"/*.0