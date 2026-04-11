#!/bin/bash
#SBATCH --job-name=index_step1
#SBATCH --nodes=1
#SBATCH --partition=long-40core
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/nulls

## Usage:
#   sbatch --job-name=index_step1_dolmino run_indexing_step1_tokenize.sh
#
# Corpus: dolmino-mix-1124 JSONL under DATA_DIR; outputs under SAVE_DIR.

set -euo pipefail

DATASET_TAG="dolmino-mix-1124"

# Redirect output and error to log files
mkdir -p logs
exec > >(tee "logs/indexing_step1.out")
exec 2> >(tee "logs/indexing_step1.err" >&2)

# Activate conda env
if command -v module &>/dev/null; then
  module load anaconda/3
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /gpfs/home/solhapark/envs/infinigram

# Load HF_TOKEN (required for meta-llama/Llama-2-7b-hf gated model)
if [[ -f /gpfs/projects/ZhouJGroup/Users/solhapark/pretraining-trace/.env ]]; then
  set -a
  source /gpfs/projects/ZhouJGroup/Users/solhapark/pretraining-trace/.env
  set +a
fi

DATA_DIR="/gpfs/projects/ZhouJGroup/Users/solhapark/dolmino-mix-1124"
SAVE_DIR="/gpfs/projects/ZhouJGroup/Users/solhapark/pretraining-trace/index/dolmino-mix-1124"
CPUS=28

mkdir -p "${SAVE_DIR}"

echo "=== Step 1 (tokenize) started at $(date) ==="
START_TIME=$(date +%s)

# Skip if outputs already exist
if [[ -f "${SAVE_DIR}/tokenized.0" ]] && [[ -f "${SAVE_DIR}/offset.0" ]]; then
  echo "SKIP: tokenized.0 and offset.0 already exist in ${SAVE_DIR}."
  exit 0
fi

cd /gpfs/projects/ZhouJGroup/Users/solhapark/pretraining-trace/infini-gram/pkg

# Run only the tokenize step from infini_gram.indexing (not build_sa).
# Sets up the same globals (tokenizer, token_dtype, version, doc_sep) that
# infini_gram.indexing.main() would set before calling tokenize().
python3 - <<PYEOF
import sys, os
import numpy as np

os.chdir('/gpfs/projects/ZhouJGroup/Users/solhapark/pretraining-trace/infini-gram/pkg')
sys.path.insert(0, '.')

import infini_gram.indexing as ig

# --- Replicate args object (same defaults as indexing.py main()) ---
class Args: pass
args = Args()
args.data_dir    = '${DATA_DIR}'
args.save_dir    = '${SAVE_DIR}'
args.worker_id   = 0
args.shards      = 1
args.workers     = 1
args.batch_size  = 65536
args.cpus        = ${CPUS}
args.add_metadata = True
args.add_unigram  = False
args.tokenizer    = 'llama'

# token_dtype=u16: LLaMA vocabulary fits in uint16; doc separator = 0xffff
ig.token_dtype  = np.uint16
ig.version      = 4
args.token_width = 2
args.doc_sep     = b'\xff\xff'

# Run tokenization only (produces tokenized.0 and offset.0)
ig.tokenize(args)
PYEOF

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=== Step 1 (tokenize) finished at $(date) ==="
printf "=== Elapsed time: %d days %02d:%02d:%02d (total %d seconds)\n" \
  $((ELAPSED/86400)) $(((ELAPSED%86400)/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"

echo ""
if [[ -f "${SAVE_DIR}/tokenized.0" ]] && [[ -f "${SAVE_DIR}/offset.0" ]]; then
  TOK_SIZE=$(stat -c%s "${SAVE_DIR}/tokenized.0")
  OD_SIZE=$(stat -c%s "${SAVE_DIR}/offset.0")
  echo "tokenized.0 : $(numfmt --to=iec-i --suffix=B "${TOK_SIZE}")"
  echo "offset.0    : $(numfmt --to=iec-i --suffix=B "${OD_SIZE}")"
  echo "=== Next step: Run run_indexing_step2_1_make_part.sh ==="
else
  echo "ERROR: tokenized.0 or offset.0 was not created."
  exit 1
fi