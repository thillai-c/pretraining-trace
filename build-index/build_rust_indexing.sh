#!/bin/bash
#SBATCH --job-name=build_rust
#SBATCH --nodes=1
#SBATCH --partition=long-40core
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/build_rust.out
#SBATCH --error=logs/build_rust.err

set -euo pipefail

mkdir -p logs

PKG_DIR="/gpfs/projects/ZhouJGroup/Users/solhapark/pretraining-trace/infini-gram/pkg"

module load rust/1.84.0

cd "${PKG_DIR}"
rm -rf target/

echo "=== Building rust_indexing on compute node ($(hostname)) at $(date) ==="
echo "GLIBC version: $(ldd --version 2>&1 | head -1)"

cargo build --release

echo "=== Build complete at $(date) ==="
ls -lh target/release/rust_indexing
