#!/usr/bin/env bash
# Create an infinigram conda environment under /gpfs/home/solhapark

set -euo pipefail

HOME="${HOME:-/gpfs/home/solhapark}"
ENV_DIR="${HOME}/envs/infinigram"

# Load Anaconda module (provides conda)
module load anaconda/3
command -v conda &>/dev/null || {
  echo "conda not found after 'module load anaconda/3'. Check: module avail anaconda"
  exit 1
}

# Ensure the envs directory exists
mkdir -p "${HOME}/envs"

echo "=== Creating infinigram env at: ${ENV_DIR} ==="
# Use conda-forge to ensure prebuilt binary packages (especially numpy/zstandard)
conda create --prefix "${ENV_DIR}" -y -c conda-forge python=3.11 pip

echo "=== Activating conda environment ==="
source "$(conda info --base)/bin/activate" "${ENV_DIR}"

echo "=== Installing indexing.py dependencies via conda-forge ==="
# Install packages with C/C++ extensions via conda to avoid source builds
conda install -y -c conda-forge numpy tqdm zstandard

echo "=== Installing transformers via pip ==="
# transformers is mostly pure Python and safe to install via pip
pip install transformers

echo ""
echo "=== Done ==="
echo "Activate with:"
echo "  source activate ${ENV_DIR}"
