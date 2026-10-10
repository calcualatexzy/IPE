#!/usr/bin/env bash
# MSM: train a fresh LoRA on Llama-3.1-8B over spec documents (msm/train.py, config msm/conf/train.yaml).
# Choose the settings below by commenting lines in or out; extra Hydra overrides go last and win.
# Usage: bash scripts/msm/train_msm.sh [HYDRA_OVERRIDES...]
# Example: bash scripts/msm/train_msm.sh training.max_steps=20 wandb.enabled=false
# The adapter lands in outputs/msm/train/msm_<data>_seed<seed>_<time>/final, the INIT_ADAPTER of train_aft.sh.
# Environment: PROJECT_ROOT, CONDA_SH, CONDA_ENV and NPROC (GPUs, default 4) are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}
NPROC=${NPROC:-4}

# Spec documents: the paper's corpora on Hugging Face, or a local corpus from generate_msm_data.sh
MSM_DATA=chloeli/msm-llama-pro-america
# MSM_DATA=chloeli/msm-llama-pro-affordability
# MSM_DATA=data/msm/midtrain/pro_america/dataset.jsonl
# MSM_DATA=data/msm/midtrain/pro_affordability/dataset.jsonl

# The paper reports 4 training seeds
SEED=42
# SEED=1
# SEED=2
# SEED=3

# bf16 needs A100/H100-class GPUs; V100s need fp16
PRECISION=(training.bf16=true training.fp16=false)
# PRECISION=(training.bf16=false training.fp16=true)

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

# The effective batch is per_device_train_batch_size x gradient_accumulation_steps x NPROC (2 x 2 x 4 = 16).
exec torchrun --standalone --nproc_per_node="$NPROC" -m msm.train \
  stage=msm \
  "data.msm=$MSM_DATA" \
  "seed=$SEED" \
  "${PRECISION[@]}" \
  "$@"
