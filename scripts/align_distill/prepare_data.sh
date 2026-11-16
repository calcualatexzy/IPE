#!/usr/bin/env bash
# The training data of align-distill (align_distill/prepare_data.py, config align_distill/conf/prepare_data.yaml, plan in
# docs/align_distill.md): the MSM subsample, the nested AFT subsets and the instruction-tuning chats, drawn from the
# released datasets in the Hugging Face cache, and the smoke pairs. Runs here, on the CPU, after fetch.sh.
# Choose the settings below by commenting lines in or out, or set them in the environment; extra Hydra overrides go
# last and win.
# Usage: bash scripts/align_distill/prepare_data.sh [HYDRA_OVERRIDES...]
# Smoke data: STAGE=smoke bash scripts/align_distill/prepare_data.sh, then SIZE=smoke MODEL=qwen25_7b bash ...
# Files land in data/align_distill/msm/<name>/, data/align_distill/aft/<model>/ and data/align_distill/distill/smoke/,
# each with a manifest; running again skips a stage whose manifest records the same inputs.
# Environment: PROJECT_ROOT, CONDA_SH, CONDA_ENV and HF_HOME are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe-align}
export HF_HOME=${HF_HOME:-/dlabscratch1/zxu/.cache/huggingface}

# Uncomment one model, or set MODEL in the environment (a row of align_distill/models.yaml): its tokenizer counts the
# tokens, and its variant of the AFT and instruction-tuning data is drawn
# MODEL=qwen25
# MODEL=qwen3
# MODEL=qwen25_7b  # development: the Qwen2.5 data, for the smoke jobs
MODEL=${MODEL:-qwen25}

# Uncomment one stage, or set STAGE in the environment
# STAGE=all  # msm, aft and it
# STAGE=msm  # the MSM corpus subsample and its held-out documents
# STAGE=aft  # the nested AFT subsets, with and without CoT, and their held-out chats
# STAGE=it  # the instruction-tuning chats
# STAGE=smoke  # the hand-written pairs into data/align_distill/distill/smoke/
STAGE=${STAGE:-all}

# Sizes: the G0 retrain's (20M MSM tokens; AFT 1,250 to all 9,963 chats; 10k instruction-tuning chats), or tiny ones for
# the smoke jobs (msm/smoke, cot_16 and no-cot_16, it_16)
# SIZE=full
# SIZE=smoke
SIZE=${SIZE:-full}
case "$SIZE" in
  full) SIZE_ARGS=() ;;
  smoke) SIZE_ARGS=(msm.name=smoke msm.tokens=40000 msm.heldout=8 "aft.sizes=[16]" aft.heldout=8 it.n=16) ;;
  *) echo "Unknown SIZE=$SIZE; choose full or smoke." >&2
     exit 1 ;;
esac

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1  # everything comes from the cache fetch.sh filled

exec python -m align_distill.prepare_data \
  "stage=$STAGE" \
  "model=$MODEL" \
  "${SIZE_ARGS[@]}" \
  "$@"
