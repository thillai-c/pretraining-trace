#!/bin/bash
#SBATCH --job-name=harmbench_olmo2
#SBATCH --output=logs/run_harmbench.out
#SBATCH --error=logs/run_harmbench.err
#SBATCH --partition=b40x4-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# =============================================================================
# HarmBench inference for a single OLMo 2 model.
# Submit one job per model. --mem and --time MUST be specified on the command line.
# Partition: b40x4-long (RTX Pro 6000 Blackwell, 96GB VRAM per GPU)
# All models fit on a single GPU (96GB VRAM).
#
# GPU memory requirements (fp16):
#   1B : ~2 GB  |  7B : ~14 GB  |  13B : ~26 GB  |  32B : ~64 GB
#
# --- Base models (run up to 3 in parallel) ---
# sbatch --mem=16G  --time=4:00:00    --export=MODEL_KEY=olmo2-1b  run_harmbench_olmo2.sh
# sbatch --mem=32G  --time=12:00:00   --export=MODEL_KEY=olmo2-7b  run_harmbench_olmo2.sh
# sbatch --mem=48G  --time=1-00:00:00 --export=MODEL_KEY=olmo2-13b run_harmbench_olmo2.sh
# sbatch --mem=128G  --time=2-00:00:00 --export=MODEL_KEY=olmo2-32b run_harmbench_olmo2.sh
#
# --- Instruct models ---
# sbatch --mem=16G  --time=4:00:00    --export=MODEL_KEY=olmo2-1b-instruct  run_harmbench_olmo2.sh
# sbatch --mem=32G  --time=12:00:00   --export=MODEL_KEY=olmo2-7b-instruct  run_harmbench_olmo2.sh
# sbatch --mem=48G  --time=1-00:00:00 --export=MODEL_KEY=olmo2-13b-instruct run_harmbench_olmo2.sh
# sbatch --mem=128G  --time=2-00:00:00 --export=MODEL_KEY=olmo2-32b-instruct run_harmbench_olmo2.sh
# =============================================================================

set -euo pipefail

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

# Activate conda env (module load first in SLURM; then activate by prefix)
if command -v module &>/dev/null; then
  module load miniconda/3
fi
source "$(conda info --base)/bin/activate" /lustre/nvwulf/home/solhapark/envs/infinigram

PROJECT_DIR="/lustre/nvwulf/projects/ZhouGroup-nvwulf/Users/solhapark/pretraining-trace"
cd "$PROJECT_DIR"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

mkdir -p logs

CONFIG="${CONFIG:-standard}"

if [ -z "${MODEL_KEY:-}" ]; then
    echo "ERROR: MODEL_KEY not set. Usage: sbatch --export=MODEL_KEY=olmo2-7b run_harmbench_olmo2.sh"
    exit 1
fi

echo "=== Model: $MODEL_KEY, config=$CONFIG ==="

OUT_DIR="$PROJECT_DIR/data/$MODEL_KEY"
OUT_JSON="$OUT_DIR/harmbench_${CONFIG}.json"
mkdir -p "$OUT_DIR"
echo "=== Output: $OUT_JSON ==="

python "$PROJECT_DIR/harmbench.py" \
    --model "$MODEL_KEY" \
    --config "$CONFIG" \
    --out_json "$OUT_JSON" \
    --max_new_tokens 1024 \
    --seed 42

echo "=== Job finished: $(date) ==="

# =============================================================================
# [DISABLED] Sequential mode: run multiple models in one job.
# Not recommended — wastes resources since SBATCH allocates for the largest model.
# Use per-model sbatch submissions above instead.
# =============================================================================
# for MODEL_KEY in olmo2-1b olmo2-7b olmo2-13b; do
#     echo "============================================================"
#     echo "=== Running: $MODEL_KEY, config=$CONFIG ==="
#     echo "=== Time: $(date) ==="
#     echo "============================================================"
#
#     python harmbench.py \
#         --model "$MODEL_KEY" \
#         --config "$CONFIG" \
#         --max_new_tokens 1024 \
#         --seed 42
#
#     echo "=== Finished: $MODEL_KEY ==="
#     python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
#     sleep 5
# done