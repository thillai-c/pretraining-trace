#!/bin/bash
#SBATCH --job-name=index_step2_2
#SBATCH --nodes=1
#SBATCH --partition=hbm-1tb-long-96core
#SBATCH --cpus-per-task=96
#SBATCH --mem=360G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Step 2.2: rust_indexing merge (dolmino-mix-1124).
#
# Usage:
#   sbatch run_indexing_step2_2_merge.sh
#
# Slurm resources (hbm-1tb-long-96core):
#   - xm045-048: ~1031807 MiB (~1 TiB) per sinfo; plenty of headroom for merge (loads full tokenized.0).
#   - --mem=360G: enough for ~199 GiB corpus + overhead; well under 1 TiB nodes (good citizen).
#     If OOM (unlikely here), raise toward 400G–500G. For hbm-long-96core (~380 GiB nodes), use 360G max ~370G.
#   - When idle (0/4/0/4), usually starts quickly; only 4 nodes total—queue can fill fast.
#   - QOS/account may restrict this partition; if submit fails, revert to hbm-long-96core.
#
# Prerequisite: run_indexing_step2_1_make_part.sh (parts-0 populated).
# Next: sbatch run_indexing_step2_3_concat.sh
#
# Build binary (on compute node):
#   sbatch build_rust_indexing.sh

set -euo pipefail

DATASET_TAG="dolmino-mix-1124"
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

mkdir -p logs
exec > >(tee "logs/indexing_step2_2.out")
exec 2> >(tee "logs/indexing_step2_2.err" >&2)

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

echo "=== Step 2.2 (merge) started at $(date) ==="
START_TIME=$(date +%s)

DS_PATH="${SAVE_DIR}/tokenized.0"
PARTS_DIR="${TEMP_DIR}/parts-0"
MERGED_DIR="${TEMP_DIR}/merged-0"

if [[ ! -f "${DS_PATH}" ]]; then
  echo "ERROR: tokenized.0 not found."
  exit 1
fi
if [[ ! -d "${PARTS_DIR}" ]] || [[ -z "$(ls -A "${PARTS_DIR}" 2>/dev/null)" ]]; then
  echo "ERROR: parts-0 not found or empty. Run Step 2.1 (make-part) first."
  exit 1
fi

DS_SIZE=$(stat -f%z "${DS_PATH}" 2>/dev/null || stat -c%s "${DS_PATH}")
HACK=100000
RATIO=$(python3 -c "import math; print(int(math.ceil(math.log2(${DS_SIZE}) / 8)))")

echo "Ratio: ${RATIO}, Token width: ${TOKEN_WIDTH}"
echo "Parts directory: ${PARTS_DIR}"
echo "Part file count: $(find "${PARTS_DIR}" -type f | wc -l)"
echo "tokenized.0 size (bytes): ${DS_SIZE}"

rm -rf "${MERGED_DIR}"
mkdir -p "${MERGED_DIR}"

MIN_OPEN=8192
ulimit -n 65536 2>/dev/null || true
CURRENT=$(ulimit -n)
if [[ "${CURRENT}" -lt "${MIN_OPEN}" ]]; then
  echo "ERROR: ulimit -n too low (${CURRENT}; need >= ${MIN_OPEN})."
  exit 1
fi
echo "ulimit -n: ${CURRENT}"

RUST_INDEXING_BIN="${INFINI_PKG}/target/release/rust_indexing"
if [[ ! -x "${RUST_INDEXING_BIN}" ]]; then
  echo "ERROR: rust_indexing missing or not executable: ${RUST_INDEXING_BIN}"
  exit 127
fi

MERGE_CMD=(
  "${RUST_INDEXING_BIN}" merge
  --data-file "${DS_PATH}"
  --parts-dir "${PARTS_DIR}"
  --merged-dir "${MERGED_DIR}"
  --num-threads "${CPUS}"
  --hacksize "${HACK}"
  --ratio "${RATIO}"
  --token-width "${TOKEN_WIDTH}"
)
echo "Executing: ${MERGE_CMD[*]}"

cleanup_merge() {
  echo "SIGTERM/SIGINT: stopping merge (PID ${MERGE_PID:-?})..."
  [[ -n "${MERGE_PID:-}" ]] && kill -TERM "${MERGE_PID}" 2>/dev/null || true
  [[ -n "${MERGE_PID:-}" ]] && wait "${MERGE_PID}" 2>/dev/null || true
  exit 143
}
trap cleanup_merge SIGTERM SIGINT

"${MERGE_CMD[@]}" &
MERGE_PID=$!
wait "${MERGE_PID}"
MERGE_EXIT=$?
trap - SIGTERM SIGINT

if [[ ${MERGE_EXIT} -ne 0 ]]; then
  echo "ERROR: merge failed (exit ${MERGE_EXIT})"
  exit 1
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=== Step 2.2 (merge) finished at $(date) ==="
printf "=== Elapsed time: %d days %02d:%02d:%02d (total %d seconds)\n" \
  $((ELAPSED/86400)) $(((ELAPSED%86400)/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"

echo ""
echo "=== Next step: sbatch run_indexing_step2_3_concat.sh ==="
