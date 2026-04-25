#!/bin/bash
#SBATCH --job-name=harmbench_eval
#SBATCH --nodes=1
#SBATCH --partition=b40x4-long
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/harmbench_eval.out
#SBATCH --error=logs/harmbench_eval.err

# =============================================================================
# HarmBench evaluation: assign compliance labels (hb_label) to model responses.
# Classifier: cais/HarmBench-Llama-2-13b-cls (~26GB VRAM, 32GB RAM is sufficient)
#
# Runs all OLMo 2 models sequentially (classifier reloaded each time, ~1-2min overhead).
# Edit the MODELS array below to select which models to evaluate.
#
# Usage:
#   sbatch run_eval_harmbench.sh
# =============================================================================

# Environment setup (nvwulf)
if command -v module &>/dev/null; then
  module load miniconda/3
fi
source "$(conda info --base)/bin/activate" /lustre/nvwulf/home/solhapark/envs/infinigram

# Load .env if present (e.g. HF_TOKEN)
PROJECT_DIR="/lustre/nvwulf/projects/ZhouGroup-nvwulf/Users/solhapark/pretraining-trace"

if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  source "$PROJECT_DIR/.env"
  set +a
fi

cd "$PROJECT_DIR"
mkdir -p scripts/logs

export VLLM_WORKER_MULTIPROC_METHOD=spawn

CONFIG="contextual"

# Models to evaluate
MODELS=(
  olmo2-1b
  olmo2-7b
  olmo2-13b
  olmo2-32b
  olmo2-1b-instruct
  olmo2-7b-instruct
  olmo2-13b-instruct
  olmo2-32b-instruct
)

for m in "${MODELS[@]}"; do
  echo "============================================================"
  echo "=== Evaluating: $m, config=$CONFIG"
  echo "=== Time: $(date)"
  echo "============================================================"

  python eval_harmbench_labels.py \
    --model "$m" \
    --config "$CONFIG" \
    --cls_path cais/HarmBench-Llama-2-13b-cls \
    --num_tokens 512

  echo "=== Done: $m ==="
done

echo ""
echo "=== All evaluations finished: $(date) ==="