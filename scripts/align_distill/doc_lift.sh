#!/usr/bin/env bash
# Whole-document likelihoods of align-distill's pairs under each spec, the gate of the spec-reasoned branches
# (align_distill/doc_lift.py, config align_distill/conf/doc_lift.yaml, plan in
# docs/align_distill.md#spec-reasoned-branches): per pair run and branch, own spec - no spec, other spec - no spec and
# own - other, nats per token of the branch text, how often q moves the right way from p0, and the copied share; then
# the paired difference of each run against the first, and the gate's bars. Run stage 6 (STAGES=splits in
# generate_data.sh) on each run first: it reads <run>/pairs.jsonl.
# Choose the settings below by commenting lines in or out, or set them in the environment; extra Hydra overrides go
# last and win.
# Usage: bash scripts/align_distill/doc_lift.sh [HYDRA_OVERRIDES...]
#   bash scripts/align_distill/doc_lift.sh  # the 7B stand-in, here on the V100: main vs main-v2 on the dev pairs
# The 32B teacher, as a RunAI job on one 80 GB GPU:
#   runai submit -i ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt -g 1 --large-shm \
#     -e MODEL=qwen25 align-distill-doc-lift -- bash /dlabscratch1/zxu/IPE/scripts/align_distill/doc_lift.sh
#   runai logs -f align-distill-doc-lift
# Rows land in outputs/align_distill/doc_lift/<MODEL>/<run>.jsonl (a document read before with the same settings is not
# read again), the summary in summary.json beside them.
# Environment: PROJECT_ROOT, CONDA_SH, CONDA_ENV and HF_HOME are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
export HF_HOME=${HF_HOME:-/dlabscratch1/zxu/.cache/huggingface}

# Uncomment one model, or set MODEL in the environment (a row of align_distill/models.yaml)
# MODEL=qwen25_7b  # the 7B stand-in: here, on the V100 (env ipe, float16)
# MODEL=qwen25  # the 32B teacher: a RunAI job on an A100 or H100 (env ipe-align, bfloat16)
MODEL=${MODEL:-qwen25_7b}
case "$MODEL" in
  qwen25_7b) DEFAULT_ENV=/dlabscratch1/zxu/envs/ipe; DTYPE=float16; MIN_CAPABILITY=7 ;;
  *) DEFAULT_ENV=/dlabscratch1/zxu/envs/ipe-align; DTYPE=bfloat16; MIN_CAPABILITY=8 ;;
esac
CONDA_ENV=${CONDA_ENV:-$DEFAULT_ENV}

# Uncomment one, or set RUNS in the environment: the pair runs compared, the reference first
# RUNS=main,main-v2  # the old branches against the spec-reasoned ones
RUNS=${RUNS:-main,main-v2}

# Uncomment one, or set SPLITS in the environment: the pairs read
# SPLITS=dev  # the pilot's
# SPLITS=dev,test
SPLITS=${SPLITS:-dev}

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1  # the models come from the cache; fetch.sh downloads beforehand

read -r GPUS CAPABILITY < <(python -c 'import torch
n = torch.cuda.device_count()
print(n, torch.cuda.get_device_capability(0)[0] if n else 0)')
(( GPUS > 0 )) || { echo 'PyTorch sees no CUDA devices; run this on a GPU.' >&2; exit 1; }
(( CAPABILITY >= MIN_CAPABILITY )) || {
  echo "MODEL=$MODEL in $DTYPE needs a GPU of compute capability $MIN_CAPABILITY or more (A100/H100 for bfloat16)." >&2
  exit 1
}

exec python -m align_distill.doc_lift \
  "model=$MODEL" \
  "dtype=$DTYPE" \
  "runs=[$RUNS]" \
  "splits=[$SPLITS]" \
  "$@"
