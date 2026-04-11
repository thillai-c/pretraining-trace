#!/bin/bash
# Submit step1 tokenization jobs for all model sizes
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for MODEL_SIZE in 7b 13b 32b; do
    echo "Submitting step1 for ${MODEL_SIZE}..."
    sbatch \
        --job-name="index_step1_${MODEL_SIZE}" \
        --export=ALL \
        "${SCRIPT_DIR}/run_indexing_step1_tokenize.sh" "${MODEL_SIZE}"
done

echo "All 4 jobs submitted. Check status with: squeue -u \$USER"