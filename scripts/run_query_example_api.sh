#!/bin/bash
#SBATCH --job-name=run_query_example_api
#SBATCH --nodes=1
#SBATCH --partition=short-96core
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Activate conda env
if command -v module &>/dev/null; then
  module load anaconda/3
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
python query_example_api.py
