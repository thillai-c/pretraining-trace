#!/usr/bin/env bash
# Build infini-gram package and rust_indexing (one-time)

set -euo pipefail

SCRATCH="${SCRATCH:-/gpfs/scratch/solhapark}"
ENV_DIR="/gpfs/home/solhapark/envs/infinigram"
PKG_DIR="${SCRATCH}/pretraining-trace/infini-gram/pkg"

if [[ ! -d "${PKG_DIR}" ]]; then
  echo "Missing infini-gram/pkg. Run: git submodule update --init"
  exit 1
fi

# Load required modules
module load anaconda/3
module load gcc/12.3.0

# Initialize conda for non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"

echo "=== Installing infini-gram (cpp_engine) ==="
cd "${PKG_DIR}"
pip install -q pybind11
pip install -e .

echo "=== Building rust_indexing ==="
module load rust 2>/dev/null || module load Rust 2>/dev/null || true

if command -v cargo &>/dev/null; then
  cargo build --release
else
  echo "cargo not found"
  exit 1
fi

echo "=== Done ==="
