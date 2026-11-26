#!/usr/bin/env bash
# The teacher's spec contrast on saved AM transcripts (align_distill/eval/am/contrast.py, config
# align_distill/conf/am_contrast.yaml): the base model scores each reply with no spec, with pro-align and with
# neg-align before the AM system prompt. STAGE=score needs one 80 GB GPU per process (a 32B model in bfloat16);
# STAGE=report joins the verdicts here, without a GPU. kd_diagnostics.sh runs it as part of the KD analysis.
# Usage: RUNS='[base]' SHARD=0/4 bash scripts/align_distill/am_contrast.sh [HYDRA_OVERRIDES...]
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe-align}
export HF_HOME=${HF_HOME:-/dlabscratch1/zxu/.cache/huggingface}

MODEL=${MODEL:-qwen25}
RUNS=${RUNS:-[base]}
SHARD=${SHARD:-}
STAGE=${STAGE:-score}

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

exec python -m align_distill.eval.am.contrast \
  "model=$MODEL" \
  "runs=$RUNS" \
  "shard=${SHARD:-null}" \
  "stage=$STAGE" \
  "$@"
