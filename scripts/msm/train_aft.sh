#!/usr/bin/env bash
# AFT: chat fine-tuning of Llama-3.1-8B with a LoRA, loss on assistant turns (msm/train_aft.py, config
# msm/conf/train_aft.yaml).
# Choose the settings below by commenting lines in or out or by editing their values; they override the config, and
# extra Hydra overrides go last and win.
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

# Start from a fresh LoRA, or keep training the adapter of an MSM or SPO run of train_msm.sh (its final/ or
# checkpoints/checkpoint-N)
INIT_ADAPTER=null
# INIT_ADAPTER=outputs/msm/train/msm_msm-llama-pro-america_seed42_<time>/final
# INIT_ADAPTER=outputs/msm/train/msm_msm-llama-pro-affordability_seed42_<time>/final
# INIT_ADAPTER=outputs/msm/train/spo_pos_pro_america_seed42_<time>/final
# INIT_ADAPTER=outputs/msm/train/spo_pos_pro_affordability_seed42_<time>/final

# Spec-aligned chats mixed with the instruction-tuning data; null trains on the instruction-tuning data alone
AFT_DATA=chloeli/aft-llama-cheese
# AFT_DATA=null

# The paper reports 4 training seeds
SEED=42
# SEED=1
# SEED=2
# SEED=3

# Log to wandb (project msm-aft), and a suffix for the run name, e.g. SUFFIX=lr2e-4
WANDB=true
SUFFIX=""

# bf16 needs A100/H100-class GPUs; V100s need fp16
PRECISION=(training.bf16=true training.fp16=false)
# PRECISION=(training.bf16=false training.fp16=true)

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

# Optimization as in the paper (App. B.4). max_steps > 0 overrides num_train_epochs, conversations longer than
# max_seq_len are dropped, and logging and checkpoints come every so many optimizer steps.
# Conversations per optimizer step: per-device batch x accumulation x NPROC (2 x 2 x 4 = 16). Batches are drawn at
# random, and the loss is the mean over the assistant tokens of the whole step, as on one GPU.
exec torchrun --standalone --nproc_per_node="$NPROC" -m msm.train_aft \
  "init_adapter=$INIT_ADAPTER" \
  "data.aft=$AFT_DATA" \
  "seed=$SEED" \
  max_seq_len=4096 \
  training.per_device_train_batch_size=2 \
  training.gradient_accumulation_steps=2 \
  training.group_by_length=false \
  training.average_tokens_across_devices=true \
  training.learning_rate=1.0e-4 \
  training.num_train_epochs=1 \
  training.max_steps=-1 \
  training.warmup_ratio=0.05 \
  training.gradient_checkpointing=true \
  training.logging_steps=10 \
  training.save_steps=200 \
  "wandb.enabled=$WANDB" \
  "suffix=$SUFFIX" \
  "${PRECISION[@]}" \
  "$@"
