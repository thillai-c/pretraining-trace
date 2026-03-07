#!/bin/bash
#SBATCH --job-name=run_query_example
#SBATCH --nodes=1
#SBATCH --partition=short-40core
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=logs/run_query_example.out
#SBATCH --error=logs/run_query_example.err

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
python query_example.py