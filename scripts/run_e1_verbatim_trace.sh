#!/bin/bash
#SBATCH --job-name=E1_verbatim_trace
#SBATCH --nodes=1
#SBATCH --partition=short-40core
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=logs/e1_verbatim_trace.out
#SBATCH --error=logs/e1_verbatim_trace.err

set -euo pipefail

# Activate conda env
if command -v module &>/dev/null; then
  module load anaconda/3
  module load gcc/12.3.0
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /gpfs/home/solhapark/envs/infinigram

# Load environment variables
if [[ -f /gpfs/scratch/solhapark/pretraining-trace/.env ]]; then
  set -a
  source /gpfs/scratch/solhapark/pretraining-trace/.env
  set +a
fi

cd /gpfs/scratch/solhapark/pretraining-trace

# Phase 1 only (metrics, no snippet retrieval) — faster, start here
# python e1_verbatim_trace.py \
#     --input results/gpt_j_6b/harmbench_standard_labeled.json \
#     --output results/gpt_j_6b/E1_verbatim_standard.json \
#     --top_k_ratio 0.05


# Phase 1 + Phase 2 (metrics + snippet retrieval) — run after Phase 1 works
python e1_verbatim_trace.py \
    --input results/gpt_j_6b/harmbench_standard_labeled.json \
    --output results/gpt_j_6b/e1_verbatim_standard.json \
    --top_k_ratio 0.05 \
    --retrieve_snippets \
    --all_records