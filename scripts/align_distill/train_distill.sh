#!/usr/bin/env bash
# The student of align-distill (align_distill/train_distill.py, config align_distill/conf/train_distill.yaml, plan in
# docs/align_distill.md): Qwen with a fresh LoRA, trained with no spec in its context, either on the pairs of a teacher
# cache (teacher_cache.sh) toward its target q, or, with PRESET=msm, on the MSM corpus subsample (prepare_data.sh) with
# CE only, by the same code. PRESET below selects the data and which loss terms train.
# Choose the settings below by commenting lines in or out, or set them in the environment; extra Hydra overrides go
# last and win.
# Usage: bash scripts/align_distill/train_distill.sh [HYDRA_OVERRIDES...]
# A RunAI job on 4 GPUs (the 32B model sharded with FSDP):
#   runai submit -i ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt -g 4 --large-shm \
#     -e PRESET=full align-distill-distill-qwen25 -- bash /dlabscratch1/zxu/IPE/scripts/align_distill/train_distill.sh
# Smoke job: -g 2 -e MODEL=qwen25_7b -e NAME=smoke -e MSM=smoke -e NPROC=2, and training.max_steps=4
# The run evaluates at the start, every 50 steps and at the end, and ends by printing the last evaluation against the
# first. The adapter lands in outputs/align_distill/train/<run>/final, the INIT_ADAPTER of train_aft.sh.
# Environment: PROJECT_ROOT, CONDA_SH, CONDA_ENV, HF_HOME and NPROC (GPUs, default 4) are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe-align}
export HF_HOME=${HF_HOME:-/dlabscratch1/zxu/.cache/huggingface}
NPROC=${NPROC:-4}

# Uncomment one student, or set MODEL in the environment (a row of align_distill/models.yaml)
# MODEL=qwen25
# MODEL=qwen3
# MODEL=qwen25_7b  # development, for the smoke job
MODEL=${MODEL:-qwen25}

# The pair run whose teacher cache to train on, under data/align_distill/distill/
# NAME=main
# NAME=smoke
NAME=${NAME:-main}

# The MSM corpus subsample of prepare_data.sh, under data/align_distill/msm/ (PRESET=msm)
# MSM=msm-20m
# MSM=smoke
MSM=${MSM:-msm-20m}

# Uncomment one preset, or set PRESET in the environment. Every cache preset reads the cache that teacher_cache.sh's
# TARGET of the same name made (full, kd_only and ce_only share the mirror cache).
# PRESET=full  # CE on the pro-align documents, KD on both documents, the mirror reference
# PRESET=kd_only  # KD only: the documents differ only through q
# PRESET=ce_only  # CE on the pro-align documents only (the same tokens as full)
# PRESET=context  # as full, with the neutral reference: plain context distillation (TARGET=neutral)
# PRESET=existential  # as full, with the existential factorized negative (TARGET=existential)
# PRESET=judgment  # as full, with the judgment factorized negative (TARGET=judgment)
# PRESET=msm  # MSM: CE on every token of the corpus subsample, no KD
PRESET=${PRESET:-full}
CE_WEIGHT=1.0
KD_WEIGHT=1.0
CACHE='neg_align-a1-clip4-k64-*'  # alpha 1, clip 4 nats, top 64 (teacher_cache.sh's defaults)
ACCUMULATION=2  # 1 pair x 2 x 4 GPUs = 8 pairs (16 documents) per optimizer step
case "$PRESET" in
  full) ;;
  kd_only) CE_WEIGHT=0 ;;
  ce_only) KD_WEIGHT=0 ;;
  context) CACHE='neutral-a1-clip4-k64-*' ;;
  existential) CACHE='neg_align_existential-a1-clip4-k64-*' ;;
  judgment) CACHE='neg_align_judgment-a1-clip4-k64-*' ;;
  msm) KD_WEIGHT=0; ACCUMULATION=4 ;;  # 1 document x 4 x 4 GPUs = 16 documents per optimizer step
  *) echo "Unknown PRESET=$PRESET; choose full, kd_only, ce_only, context, existential, judgment or msm." >&2
     exit 1 ;;
esac

# The teacher whose cache to train on (teacher_cache.sh's TEACHER): the student's own model, or a larger Qwen, whose tag
# starts the cache's name
# TEACHER=null
# TEACHER=qwen25_72b
TEACHER=${TEACHER:-null}
[[ "$TEACHER" == null ]] || CACHE="$TEACHER-$CACHE"

if [[ "$PRESET" == msm ]]; then
  DATA_ARGS=("data.corpus=data/align_distill/msm/$MSM/train.jsonl" "data.heldout=data/align_distill/msm/$MSM/heldout.jsonl")
else
  DATA_ARGS=("data.cache=data/align_distill/distill/$NAME/teacher_cache/$MODEL/$CACHE")
fi

# Training seeds
# SEED=42
# SEED=1
# SEED=2
SEED=${SEED:-42}

# Log to wandb (project align-distill-distill), and a suffix for the run name, e.g. SUFFIX=rerun. Loss settings that
# differ from the defaults are tagged in the name already.
WANDB=true
SUFFIX=${SUFFIX:-}

printf 'Preset: %s; model: %s; CE weight: %s; KD weight: %s; data: %s\n' \
  "$PRESET" "$MODEL" "$CE_WEIGHT" "$KD_WEIGHT" "${DATA_ARGS[*]}" >&2

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1  # jobs read the cache only; fetch.sh downloads beforehand

# One process per GPU: processes that share a GPU make NCCL fail at the first collective ("Duplicate GPU detected").
read -r GPUS CAPABILITY < <(python -c 'import torch
n = torch.cuda.device_count()
print(n, torch.cuda.get_device_capability(0)[0] if n else 0)')
(( GPUS > 0 )) || { echo 'PyTorch sees no CUDA devices; request GPUs for this job.' >&2; exit 1; }
(( CAPABILITY >= 8 )) || { echo 'The ipe-align env (CUDA 13, bf16) needs A100/H100 GPUs.' >&2; exit 1; }
(( NPROC <= GPUS )) || { echo "NPROC=$NPROC, but only $GPUS GPUs are visible; allocate more GPUs, or lower NPROC" \
                              "and raise training.gradient_accumulation_steps to keep the examples per step." >&2
                         exit 1; }

# LoRA and optimization are the config's (the paper's App. B.4): rank 64, alpha 128, every projection; one epoch of
# AdamW at 1e-4, cosine, 5% warmup, weight decay 0.01. Batches are drawn at random, and CE and KD are the means over the
# positions of the whole step, as on one GPU.
exec torchrun --standalone --nproc_per_node="$NPROC" -m align_distill.train_distill \
  "model=$MODEL" \
  "${DATA_ARGS[@]}" \
  "loss.ce_weight=$CE_WEIGHT" \
  "loss.kd_weight=$KD_WEIGHT" \
  "training.gradient_accumulation_steps=$ACCUMULATION" \
  "seed=$SEED" \
  "wandb.enabled=$WANDB" \
  "suffix=$SUFFIX" \
  "$@"
