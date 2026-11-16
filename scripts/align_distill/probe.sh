#!/usr/bin/env bash
# The teacher probe of align-distill, gate G2 (align_distill/probe.py, config align_distill/conf/probe.yaml, plan in
# docs/align_distill.md#teacher-probe-and-cache): on the decision prefixes that stage 6 cut from a pair run
# (data/align_distill/distill/<NAME>/probes.jsonl; STAGES=splits in generate_data.sh), the margin
# log P(aligned) - log P(misaligned) of the card's two options, with no spec, after pro-align and after neg-align, in
# the plain and chat layouts. It prints, per kind and source, how often the margin after pro-align is above the margin
# after neg-align (a kind passes at 80%), and saves every score.
# Choose the settings below by commenting lines in or out, or set them in the environment; extra Hydra overrides go
# last and win.
# Usage: bash scripts/align_distill/probe.sh [HYDRA_OVERRIDES...]
# A RunAI job on one 80 GB GPU (a 32B model in bfloat16):
#   runai submit -i ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt -g 1 --large-shm \
#     -e NAME=pilot -e MODEL=qwen25_7b align-distill-probe-qwen25-7b -- bash /dlabscratch1/zxu/IPE/scripts/align_distill/probe.sh
#   runai logs -f align-distill-probe-qwen25-7b
# The 72B teacher ablation: -g 4 -e TEACHER=qwen25_72b (spread over the job's GPUs).
# Results land in outputs/align_distill/probe/<NAME>/<MODEL>/[<TEACHER>-]<time>/: probes.jsonl, summary.json,
# probe.log.
# Environment: PROJECT_ROOT, CONDA_SH, CONDA_ENV and HF_HOME are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe-align}
export HF_HOME=${HF_HOME:-/dlabscratch1/zxu/.cache/huggingface}

# Uncomment one student, or set MODEL in the environment (a row of align_distill/models.yaml): the anchor, and by
# default the teacher
# MODEL=qwen25
# MODEL=qwen3
# MODEL=qwen25_7b  # development: the code path, before the 32B model
MODEL=${MODEL:-qwen25}

# The pair run whose probes to read, under data/align_distill/distill/
# NAME=pilot
# NAME=main
NAME=${NAME:-pilot}

# The teacher: the student's own model (null), or a larger Qwen with the same tokenizer, spread over the job's GPUs
# TEACHER=null
# TEACHER=qwen25_72b
TEACHER=${TEACHER:-null}
DEVICE_MAP=single
[[ "$TEACHER" == null ]] || DEVICE_MAP=auto

# Uncomment one, or set SPLITS in the environment: the pairs the probes come from (stage 6 cut the pilot's from all)
# SPLITS=train,dev,test
# SPLITS=dev,test  # the main run's
SPLITS=${SPLITS:-train,dev,test}

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1  # jobs read the cache only; fetch.sh downloads beforehand

read -r GPUS CAPABILITY < <(python -c 'import torch
n = torch.cuda.device_count()
print(n, torch.cuda.get_device_capability(0)[0] if n else 0)')
(( GPUS > 0 )) || { echo 'PyTorch sees no CUDA devices; request a GPU for this job.' >&2; exit 1; }
(( CAPABILITY >= 8 )) || { echo 'The ipe-align env (CUDA 13, bf16) needs A100/H100 GPUs.' >&2; exit 1; }

exec python -m align_distill.probe \
  "model=$MODEL" \
  "name=$NAME" \
  "teacher=$TEACHER" \
  "device_map=$DEVICE_MAP" \
  "splits=[$SPLITS]" \
  "$@"
