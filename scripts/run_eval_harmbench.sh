#!/bin/bash
#SBATCH --job-name=harmbench_eval
#SBATCH --nodes=1
#SBATCH --partition=b40x4-long
#SBATCH --gres=gpu:1
# Memory: copyright ~8G (hash-only); standard/contextual ~32G (vLLM+13B classifier). Override for copyright-only: sbatch --mem=8G ...
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00
#SBATCH --output=logs/harmbench_eval.out
#SBATCH --error=logs/harmbench_eval.err

# CONFIG: "copyright" (hash-based, no GPU) | "standard" | "contextual"
# For standard/contextual, --cls_path is the HarmBench compliance classifier (Llama-2-13B-based, from Hugging Face).
CONFIG="standard"

# Activate conda env (module load first in SLURM; then activate by prefix)
if command -v module &>/dev/null; then
  module load miniconda/3
fi
source "$(conda info --base)/bin/activate" /lustre/nvwulf/home/solhapark/envs/infinigram

# Load .env if present (e.g. HF_TOKEN)
if [[ -f /lustre/nvwulf/scratch/solhapark/pretraining-trace/.env ]]; then
  set -a
  source /lustre/nvwulf/scratch/solhapark/pretraining-trace/.env
  set +a
fi

cd /lustre/nvwulf/scratch/solhapark/pretraining-trace
mkdir -p logs

if [[ "$CONFIG" == "copyright" ]]; then
  python eval_harmbench_labels.py \
    --data_dir data/gpt_j_6b/harmbench_copyright.json \
    --output_dir results/gpt_j_6b/harmbench_copyright_labeled.json
    # --limit 10
else
  # standard or contextual: uses HarmBench classifier (cais/HarmBench-Llama-2-13b-cls) on GPU
  # VLLM_WORKER_MULTIPROC_METHOD=spawn: prevents "Cannot re-initialize CUDA in forked subprocess"
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  python eval_harmbench_labels.py \
    --data_dir data/gpt_j_6b/harmbench_${CONFIG}.json \
    --output_dir results/gpt_j_6b/harmbench_${CONFIG}_labeled.json \
    --cls_path cais/HarmBench-Llama-2-13b-cls \
    --num_tokens 512
    # --limit 10
fi