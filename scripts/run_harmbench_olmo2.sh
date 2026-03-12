#!/bin/bash
#SBATCH --job-name=harmbench_olmo2
#SBATCH --output=logs/harmbench_olmo2_%j.out
#SBATCH --error=logs/harmbench_olmo2_%j.err
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00

# =============================================================================
# HarmBench inference for a single OLMo 2 model (1B / 7B / 13B).
# Submit one job per model with appropriate resources via --export=MODEL_KEY.
# Override SBATCH defaults (mem, time) on the command line as needed.
#
# GPU memory requirements (fp16):
#   1B  : ~2 GB   |  7B : ~14 GB  |  13B : ~26 GB  |  32B : ~64 GB (separate script)
#
# --- Base models (submit each separately) ---
# sbatch --mem=16G  --time=4:00:00    --export=MODEL_KEY=olmo2-1b  run_harmbench_olmo2.sh
# sbatch --mem=32G  --time=12:00:00   --export=MODEL_KEY=olmo2-7b  run_harmbench_olmo2.sh
# sbatch --mem=48G  --time=1-00:00:00 --export=MODEL_KEY=olmo2-13b run_harmbench_olmo2.sh
#
# --- Instruct models (submit each separately) ---
# sbatch --mem=16G  --time=4:00:00    --export=MODEL_KEY=olmo2-1b-instruct  run_harmbench_olmo2.sh
# sbatch --mem=32G  --time=12:00:00   --export=MODEL_KEY=olmo2-7b-instruct  run_harmbench_olmo2.sh
# sbatch --mem=48G  --time=2-00:00:00 --export=MODEL_KEY=olmo2-13b-instruct run_harmbench_olmo2.sh
#
# --- 32B models (use run_harmbench_olmo2_32b.sh instead) ---
# sbatch run_harmbench_olmo2_32b.sh
# sbatch --export=MODEL_KEY=olmo2-32b-instruct run_harmbench_olmo2_32b.sh
# =============================================================================

set -euo pipefail

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

# Environment setup (Seawulf — modify as needed)
module load anaconda3 2>/dev/null || true
conda activate pretraining-trace 2>/dev/null || true

cd ~/pretraining-trace

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

mkdir -p logs

CONFIG="${CONFIG:-standard}"

if [ -z "${MODEL_KEY:-}" ]; then
    echo "ERROR: MODEL_KEY not set. Usage: sbatch --export=MODEL_KEY=olmo2-7b run_harmbench_olmo2.sh"
    exit 1
fi

echo "=== Model: $MODEL_KEY, config=$CONFIG ==="

python harmbench.py \
    --model "$MODEL_KEY" \
    --config "$CONFIG" \
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
