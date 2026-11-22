#!/usr/bin/env bash
# The cheese experiment's pro-affordability eval on the 32B teacher (align_distill/eval/value.py; Ziyi, 2026-11-22):
# no system prompt, the pro-affordability spec, a conflicting assigned goal, and the spec then the goal. Does the AM
# teacher follow a spec, and does an assigned goal alone break that?
# Usage:
#   runai submit -i ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt -g 4 --large-shm \
#     align-distill-value-conflict -- bash /dlabscratch1/zxu/IPE/scripts/align_distill/value_conflict.sh
#   STAGE=render bash scripts/align_distill/value_conflict.sh  # here: the prompts, without a GPU
#   STAGE=judge bash scripts/align_distill/value_conflict.sh   # here, after the job (OPENROUTER_API_KEY)
# Running again reuses saved answers and verdicts.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe-align}
export HF_HOME=${HF_HOME:-/dlabscratch1/zxu/.cache/huggingface}
STAGE=${STAGE:-generate}

case "$STAGE" in
  render|generate) export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 ;;
  judge) [[ -n ${OPENROUTER_API_KEY:-} ]] || { echo 'Export OPENROUTER_API_KEY for the judge.' >&2; exit 1; } ;;
  *) echo "Unknown STAGE=$STAGE; choose render, generate or judge." >&2; exit 1 ;;
esac

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

if [[ $STAGE == generate ]]; then
  GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
  (( GPUS > 0 )) || { echo 'PyTorch sees no CUDA devices; request GPUs for this job.' >&2; exit 1; }
fi

exec python -m align_distill.eval.value --stage "$STAGE" "$@"
