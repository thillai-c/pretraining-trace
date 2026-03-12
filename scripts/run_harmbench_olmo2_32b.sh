#!/bin/bash
#SBATCH --job-name=harmbench_olmo2_32b
#SBATCH --output=logs/harmbench_olmo2_32b_%j.out
#SBATCH --error=logs/harmbench_olmo2_32b_%j.err
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00

# =============================================================================
# HarmBench inference for OLMo 2 32B.
# Requires ~64GB GPU memory (fp16), so either:
#   - H100 80GB single GPU → change to --gres=gpu:1
#   - A100 40GB x 2 → device_map="auto" splits across GPUs (current setting)
#
# Usage:
#   # Base model (default)
#   sbatch run_harmbench_olmo2_32b.sh
#
#   # Instruct model
#   sbatch --export=MODEL_KEY=olmo2-32b-instruct run_harmbench_olmo2_32b.sh
# =============================================================================

set -euo pipefail

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Environment setup (Seawulf — modify as needed)
module load anaconda3 2>/dev/null || true
conda activate pretraining-trace 2>/dev/null || true

cd ~/pretraining-trace

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME/hub"

mkdir -p logs

MODEL_KEY="${MODEL_KEY:-olmo2-32b}"
CONFIG="${CONFIG:-standard}"

echo "=== Model: $MODEL_KEY, config=$CONFIG ==="

python harmbench.py \
    --model "$MODEL_KEY" \
    --config "$CONFIG" \
    --max_new_tokens 1024 \
    --seed 42

echo "=== Job finished: $(date) ==="
