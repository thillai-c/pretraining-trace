#!/bin/bash
#SBATCH --job-name=E2_cooccurrence
#SBATCH --nodes=1
#SBATCH --partition=short-40core
#SBATCH --mem=16G
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

# # Full run (all 200 records, multiple windows)
# python e2_windowed_cooccurrence.py \
#     --model gpt-j \
#     --index_dir ./index \
#     --windows 100 500 1000 1024 2048 \
#     --all_records

# OLMo 2 (API engine)
python e2_windowed_cooccurrence.py \
    --model olmo2-7b \
    --api_index v4_olmo-mix-1124_llama \
    --compliant_only \
    --windows 100 500 1000