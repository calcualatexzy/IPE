#!/usr/bin/env bash
# AFT: chat fine-tuning of Llama-3.1-8B with a LoRA, loss on assistant turns (msm/train.py, config msm/conf/train.yaml).
# Choose the settings below by commenting lines in or out; extra Hydra overrides go last and win.
# Usage: bash scripts/msm/train_aft.sh [HYDRA_OVERRIDES...]
# The paper's four conditions are INIT_ADAPTER x AFT_DATA:
#   baseline   INIT_ADAPTER=null          AFT_DATA=null
#   AFT only   INIT_ADAPTER=null          AFT_DATA=chloeli/aft-llama-cheese
#   MSM only   INIT_ADAPTER=<MSM run>     AFT_DATA=null
#   MSM + AFT  INIT_ADAPTER=<MSM run>     AFT_DATA=chloeli/aft-llama-cheese
# The instruction-tuning data (data.it in the config) is mixed into every run.
# The adapter lands in outputs/msm/train/aft_<data>_seed<seed>_from-<init>_<time>/final.
# Environment: PROJECT_ROOT, CONDA_SH, CONDA_ENV and NPROC (GPUs, default 4) are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}
NPROC=${NPROC:-4}

# Start from a fresh LoRA, or keep training the adapter of an MSM run (its final/ or checkpoints/checkpoint-N)
INIT_ADAPTER=null
# INIT_ADAPTER=outputs/msm/train/msm_msm-llama-pro-america_seed42_<time>/final
# INIT_ADAPTER=outputs/msm/train/msm_msm-llama-pro-affordability_seed42_<time>/final

# Spec-aligned chats mixed with the instruction-tuning data; null trains on the instruction-tuning data alone
AFT_DATA=chloeli/aft-llama-cheese
# AFT_DATA=null

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
  stage=aft \
  "init_adapter=$INIT_ADAPTER" \
  "data.aft=$AFT_DATA" \
  "seed=$SEED" \
  "${PRECISION[@]}" \
  "$@"
