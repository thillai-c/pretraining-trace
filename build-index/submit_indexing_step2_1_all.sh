#!/bin/bash
# Submit step2.1 make-part for dolmino-mix-1124 (single index under index/dolmino-mix-1124)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting step2.1 (make-part) for dolmino-mix-1124..."
sbatch \
    --job-name="index_step2_1_dolmino" \
    --export=ALL \
    "${SCRIPT_DIR}/run_indexing_step2_1_make_part.sh"

echo "Check status with: squeue -u \$USER"
