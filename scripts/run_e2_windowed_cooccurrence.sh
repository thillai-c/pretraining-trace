#!/bin/bash
#SBATCH --job-name=E2_cooccurrence
#SBATCH --nodes=1
#SBATCH --partition=short-40core
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=logs/e2_windowed_cooccurrence.out
#SBATCH --error=logs/e2_windowed_cooccurrence.err

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

Test run (first 5 records, compliant only)
python e2_windowed_cooccurrence.py \
    --input results/gpt_j_6b/e1_verbatim_standard.json \
    --output results/gpt_j_6b/e2_cooccurrence_test.json \
    --windows 100 500 1000 \
    --compliant_only \
    --limit 5

# # Full run (all 200 records, multiple windows)
# python e2_windowed_cooccurrence.py \
#     --input results/gpt_j_6b/e1_verbatim_standard.json \
#     --output results/gpt_j_6b/e2_cooccurrence_standard.json \
#     --windows 100 500 1000 \
#     --all_records
