#!/bin/bash
# Submit step2.2 merge for dolmino-mix-1124
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting step2.2 (merge) for dolmino-mix-1124..."
sbatch \
    --job-name="index_step2_2_dolmino" \
    --export=ALL \
    "${SCRIPT_DIR}/run_indexing_step2_2_merge.sh"

echo "Check status with: squeue -u \$USER"
