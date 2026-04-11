#!/bin/bash
#SBATCH --job-name=indexing_pile
#SBATCH --nodes=1
# b40x4-long: idle nodes, ~504GB RAM, run immediately. For ~1.4TB use h200x8-long (may wait).
#SBATCH --partition=b40x4-long
#SBATCH --mem=500000
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00
#SBATCH --output=logs/indexing_pile.out
#SBATCH --error=logs/indexing_pile.err

timestamp=$(date +"%y%m%d_%H%M%S")

# Activate conda env (module load first in SLURM; then activate by prefix)
if command -v module &>/dev/null; then
  module load miniconda/3
fi
source "$(conda info --base)/bin/activate" /lustre/nvwulf/home/solhapark/envs/infinigram

# Load HF_TOKEN for meta-llama/Llama-2-7b-hf (gated model)
if [[ -f /lustre/nvwulf/scratch/solhapark/pretrain-trace/.env ]]; then
  set -a
  source /lustre/nvwulf/scratch/solhapark/pretrain-trace/.env
  set +a
fi

# Run indexing.py (--mem in GiB; 480 GiB for b40x4 node ~504GB)
# --tokenizer llama: use meta-llama/Llama-2-7b-hf so query_example.py can use same tokenizer
cd /lustre/nvwulf/scratch/solhapark/pretrain-trace/infini-gram/pkg
echo "=== Job started at $(date) ==="
START_TIME=$(date +%s)

python -m infini_gram.indexing \
  --data_dir /lustre/nvwulf/scratch/solhapark/pretrain-trace/train \
  --save_dir /lustre/nvwulf/scratch/solhapark/pretrain-trace/index \
  --mem 480 \
  --ulimit 65536 \
  --tokenizer llama

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=== Job finished at $(date) ==="
printf "=== Elapsed time: %d days %02d:%02d:%02d (total %d seconds)\n" \
  $((ELAPSED/86400)) $(((ELAPSED%86400)/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"