#!/bin/bash
#SBATCH --job-name=index_step2_1
#SBATCH --nodes=1
#SBATCH --partition=long-40core
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

## Usage:
#   Single model:  sbatch --job-name=index_step2_1_1b run_indexing_step2_1_make_part.sh 1b
#   All models:    bash submit_indexing_step2_1_all.sh
#
# Available model sizes: 1b, 7b, 13b, 32b

## Build rust_indexing
# cd /gpfs/projects/ZhouJGroup/Users/solhapark/pretraining-trace/infini-gram/pkg
# module load rust/1.84.0
# cargo build --release

set -euo pipefail

DATASET_TAG="dolmino-mix-1124"

# Redirect output and error to log files (same pattern as run_indexing_step1_tokenize.sh)
mkdir -p logs
exec > >(tee "logs/indexing_step2_1.out")
exec 2> >(tee "logs/indexing_step2_1.err" >&2)

# Activate conda env
if command -v module &>/dev/null; then
  module load anaconda/3
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /gpfs/home/solhapark/envs/infinigram

# Load HF_TOKEN
if [[ -f /gpfs/projects/ZhouJGroup/Users/solhapark/pretraining-trace/.env ]]; then
  set -a
  source /gpfs/projects/ZhouJGroup/Users/solhapark/pretraining-trace/.env
  set +a
fi

cd /gpfs/projects/ZhouJGroup/Users/solhapark/pretraining-trace/infini-gram/pkg

SAVE_DIR="/gpfs/projects/ZhouJGroup/Users/solhapark/pretraining-trace/index/${DATASET_TAG}"
TEMP_DIR="${SAVE_DIR}"
CPUS=28
# Per-process memory budget (GiB) used in batch sizing; must match infini_gram/indexing.py (mem_bytes // divisor).
MEM_GB=32
TOKEN_WIDTH=2  # u16 LLaMA tokens; divisor 8 (use 12 if TOKEN_WIDTH=1)
if [[ "${TOKEN_WIDTH}" -eq 1 ]]; then
  MEM_DIVISOR=12
else
  MEM_DIVISOR=8
fi

echo "=== Step 2.1 (make-part) started at $(date) ==="
START_TIME=$(date +%s)

# Check prerequisites
DS_PATH="${SAVE_DIR}/tokenized.0"
if [[ ! -f "${DS_PATH}" ]]; then
  echo "ERROR: tokenized.0 not found. Run Step 1 (tokenize) first."
  exit 1
fi

# Checkpoint/resume: parts are kept; only missing part files are (re)built.
PARTS_DIR="${TEMP_DIR}/parts-0"
if [[ -d "${PARTS_DIR}" ]] && [[ -n "$(ls -A "${PARTS_DIR}" 2>/dev/null)" ]]; then
  echo "RESUME: parts-0 exists. Will only run make-part for missing part files."
fi

# Calculate parameters (same logic as indexing.py)
DS_SIZE=$(stat -f%z "${DS_PATH}" 2>/dev/null || stat -c%s "${DS_PATH}")
HACK=100000
RATIO=$(python3 -c "import math; print(int(math.ceil(math.log2(${DS_SIZE}) / 8)))")
MEM_BYTES=$((MEM_GB * 1024**3))
NUM_JOB_BATCHES=1
while [[ $((NUM_JOB_BATCHES * (MEM_BYTES / MEM_DIVISOR))) -lt ${DS_SIZE} ]]; do
  NUM_JOB_BATCHES=$((NUM_JOB_BATCHES * 2))
done
PARALLEL_JOBS=${CPUS}
TOTAL_JOBS=$((NUM_JOB_BATCHES * PARALLEL_JOBS))

echo "Using ${NUM_JOB_BATCHES} batches of ${PARALLEL_JOBS} jobs each, for a total of ${TOTAL_JOBS} jobs."
echo "Ratio: ${RATIO}, Token width: ${TOKEN_WIDTH}"

# Create parts directory (do not remove existing; resume support)
mkdir -p "${PARTS_DIR}"

# Run make-part for each batch; skip if part file already exists (checkpoint/resume)
S=$((DS_SIZE / TOTAL_JOBS))
# Make sure parts contain whole tokens
if [[ $((S % TOKEN_WIDTH)) -ne 0 ]]; then
  S=$((S + TOKEN_WIDTH - (S % TOKEN_WIDTH)))
fi

SKIPPED=0
RUN=0
for ((batch_start=0; batch_start<TOTAL_JOBS; batch_start+=PARALLEL_JOBS)); do
  batch_end=$((batch_start + PARALLEL_JOBS < TOTAL_JOBS ? batch_start + PARALLEL_JOBS : TOTAL_JOBS))
  echo "Processing batch ${batch_start} to ${batch_end}..."

  for ((i=batch_start; i<batch_end; i++)); do
    s=$((i * S))
    e=$(((i + 1) * S + HACK < DS_SIZE ? (i + 1) * S + HACK : DS_SIZE))
    PART_FILE="${PARTS_DIR}/${s}-${e}"
    if [[ -f "${PART_FILE}" ]]; then
      echo "  Skip (already done): part ${s}-${e}"
      ((SKIPPED++)) || true
      continue
    fi
    ((RUN++)) || true
    /gpfs/projects/ZhouJGroup/Users/solhapark/pretraining-trace/infini-gram/pkg/target/release/rust_indexing make-part \
      --data-file "${DS_PATH}" \
      --parts-dir "${PARTS_DIR}" \
      --start-byte "${s}" \
      --end-byte "${e}" \
      --ratio "${RATIO}" \
      --token-width "${TOKEN_WIDTH}" &
  done
  wait
done

echo "Done: ${RUN} parts built, ${SKIPPED} skipped (already present)."

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=== Step 2.1 (make-part) finished at $(date) ==="
printf "=== Elapsed time: %d days %02d:%02d:%02d (total %d seconds)\n" \
  $((ELAPSED/86400)) $(((ELAPSED%86400)/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"

echo ""
NUM_PARTS=$(find "${PARTS_DIR}" -maxdepth 1 -type f -name '*-*' 2>/dev/null | wc -l)
if [[ ${NUM_PARTS} -eq ${TOTAL_JOBS} ]]; then
  echo "=== All ${TOTAL_JOBS} parts complete. Next step: Run run_indexing_step2_2_merge.sh ==="
else
  echo "=== Parts done: ${NUM_PARTS}/${TOTAL_JOBS}. Re-submit this job to continue from checkpoint. ==="
  echo "=== When all parts are done, run run_indexing_step2_2_merge.sh ==="
fi

