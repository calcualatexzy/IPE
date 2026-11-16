#!/usr/bin/env bash
# AFT of align-distill (align_distill/train_aft.py, config align_distill/conf/train_aft.yaml, plan in
# docs/align_distill.md): chat fine-tuning with a LoRA, loss on the assistant replies, on a nested AFT subset with or
# without CoT mixed with the instruction-tuning chats (prepare_data.sh), from a fresh LoRA or continuing an adapter.
# Choose the settings below by commenting lines in or out, or set them in the environment; extra Hydra overrides go
# last and win.
# Usage: bash scripts/align_distill/train_aft.sh [HYDRA_OVERRIDES...]
# The conditions are INIT_ADAPTER x AFT:
#   AFT only        INIT_ADAPTER=null              AFT=cot or no-cot
#   MSM + AFT       INIT_ADAPTER=<msm run>/final   (train_distill.sh PRESET=msm)
#   Distill + AFT   INIT_ADAPTER=<distill run>/final
#   no AFT          AFT=none (instruction tuning only)
# A RunAI job on 4 GPUs (the 32B model sharded with FSDP):
#   runai submit -i ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt -g 4 --large-shm \
#     -e AFT=cot -e INIT_ADAPTER=<run>/final align-distill-aft-qwen25 -- bash /dlabscratch1/zxu/IPE/scripts/align_distill/train_aft.sh
# Smoke job: -g 2 -e MODEL=qwen25_7b -e AFT_SIZE=16 -e IT_N=16 -e NPROC=2, and training.max_steps=4
# The adapter lands in outputs/align_distill/train/aft_<model>_<data>_from-<init>_seed<n>[_<suffix>]_<time>/final, an
# ADAPTERS line of eval_am.sh.
# Environment: PROJECT_ROOT, CONDA_SH, CONDA_ENV, HF_HOME and NPROC (GPUs, default 4) are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe-align}
export HF_HOME=${HF_HOME:-/dlabscratch1/zxu/.cache/huggingface}
NPROC=${NPROC:-4}

# Uncomment one model, or set MODEL in the environment (a row of align_distill/models.yaml)
# MODEL=qwen25
# MODEL=qwen3
# MODEL=qwen25_7b  # development, for the smoke job
MODEL=${MODEL:-qwen25}

# Start from a fresh LoRA, or keep training an adapter: a train_distill.sh run's final/ (or checkpoints/checkpoint-N),
# or paper:<name>, the paper's checkpoint of MODEL (fetched by fetch.sh)
# INIT_ADAPTER=null
# INIT_ADAPTER=outputs/align_distill/train/msm_qwen25_msm-20m_seed42_<time>/final
# INIT_ADAPTER=outputs/align_distill/train/distill_qwen25_main_neg_align-a1-clip4-k64_seed42_<time>/final
# INIT_ADAPTER=paper:philosophy-spec-msm  # the paper's MSM (followed by instruction tuning)
INIT_ADAPTER=${INIT_ADAPTER:-null}

# The AFT variant, or none for the instruction-tuning chats alone
# AFT=cot
# AFT=no-cot
# AFT=none
AFT=${AFT:-cot}

# AFT chats: the nested subsets of prepare_data.sh (every condition is compared at 2,500 first)
# AFT_SIZE=1250
# AFT_SIZE=2500
# AFT_SIZE=5000
# AFT_SIZE=9959  # all: the 9,963 released chats but the 4 with an empty reply
AFT_SIZE=${AFT_SIZE:-2500}

# Instruction-tuning chats mixed into every run (prepare_data.sh's it_<n>.jsonl; 16 for the smoke job)
IT_N=${IT_N:-10000}

DATA_DIR=data/align_distill/aft/$MODEL
case "$AFT" in
  cot|no-cot) AFT_DATA=$DATA_DIR/${AFT}_$AFT_SIZE.jsonl
              HELDOUT=$DATA_DIR/${AFT}_heldout.jsonl ;;  # evaluated at the start and the end
  none) AFT_DATA=null
        HELDOUT=null ;;
  *) echo "Unknown AFT=$AFT; choose cot, no-cot or none." >&2
     exit 1 ;;
esac
# HELDOUT=null  # no held-out evaluation

# Training seeds
# SEED=42
# SEED=1
# SEED=2
SEED=${SEED:-42}

# Log to wandb (project align-distill-aft), and a suffix for the run name, e.g. SUFFIX=rerun
WANDB=true
SUFFIX=${SUFFIX:-}

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
                              "and raise training.gradient_accumulation_steps to keep 16 chats per step." >&2
                         exit 1; }

# LoRA and optimization are the config's (the paper's App. B.4); chats per optimizer step: 1 x 4 x 4 GPUs = 16. Batches
# are drawn at random, and the loss is the mean over the trained tokens of the whole step, as on one GPU.
exec torchrun --standalone --nproc_per_node="$NPROC" -m align_distill.train_aft \
  "model=$MODEL" \
  "init_adapter=$INIT_ADAPTER" \
  "data.aft=$AFT_DATA" \
  "data.it=$DATA_DIR/it_$IT_N.jsonl" \
  "data.heldout=$HELDOUT" \
  "seed=$SEED" \
  "wandb.enabled=$WANDB" \
  "suffix=$SUFFIX" \
  "$@"
