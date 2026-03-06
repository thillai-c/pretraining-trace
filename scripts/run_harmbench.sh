#!/bin/bash
#SBATCH --job-name=harmbench_contextual
#SBATCH --nodes=1
#SBATCH --partition=long-28core
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

CONFIG="contextual"  # Change this to "copyright", "standard", etc.

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
python harmbench.py \
  --csv_path HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
  --out_json data/gpt_j_6b/harmbench_${CONFIG}.json \
  --config ${CONFIG} \
  --max_new_tokens 1024 \
  --seed 42
  # --max_samples 5