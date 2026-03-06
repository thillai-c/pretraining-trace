#!/bin/bash
#SBATCH --job-name=run_query_example
#SBATCH --nodes=1
#SBATCH --partition=long-28core
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/run_query_example.out
#SBATCH --error=logs/run_query_example.err

timestamp=$(date +"%y%m%d_%H%M%S")

# Activate conda env
if command -v module &>/dev/null; then
  module load anaconda/3
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /gpfs/home/solhapark/envs/infinigram

# Load HF_TOKEN for meta-llama/Llama-2-7b-hf (gated model)
if [[ -f /gpfs/scratch/solhapark/pretraining-trace/.env ]]; then
  set -a
  source /gpfs/scratch/solhapark/pretraining-trace/.env
  set +a
fi

cd /gpfs/scratch/solhapark/pretraining-trace
echo "=== Job started at $(date) ==="
START_TIME=$(date +%s)

python query_example.py

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=== Job finished at $(date) ==="
printf "=== Elapsed time: %d days %02d:%02d:%02d (total %d seconds)\n" \
  $((ELAPSED/86400)) $(((ELAPSED%86400)/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"
