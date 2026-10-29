#!/usr/bin/env bash
# Stage 7 of spec context distillation (msm/distill_teacher_cache.py, config msm/conf/distill_teacher_cache.yaml, plan
# in docs/spec_context_distill.md): for every document of data/msm/distill/<NAME>/pairs.jsonl, three passes (the base
# model on the document, the teacher after S+, and the mirror or neutral reference teacher) give the target
# q = softmax(log p0 + alpha * clip(log p+ - log p_ref)), stored per position as its top entries and remaining mass.
# The run ends by printing the checks before training: JS(p+, p-) and KL(q || p0) per region (prefix, decision, after),
# kind and branch; q's choice margin against p0's where each document names its choice; and the tokens with the largest
# |log p+ - log p-|, to see whether the signal is the choice and its reason or names and spec wording.
# Needs stage 6 (splits) run after stage 5, so that each pair has its split.
# Choose the settings below by commenting lines in or out; extra Hydra overrides go last and win.
# Usage: bash scripts/msm/distill_teacher_cache.sh [HYDRA_OVERRIDES...]
# Example: bash scripts/msm/distill_teacher_cache.sh max_pairs=4
# Results land in data/msm/distill/<NAME>/teacher_cache/<key>/, the key naming alpha, clip, anchor and top_k. A finished
# cache with the same key is kept (overwrite=true recomputes it).
# Runs as one process on one GPU (request one GPU for the job). On a V100 in float16, about 1 s per document after a
# 30 s start: the pilot (298 documents) in about 5 minutes, 3k cards (6k documents) in under 2 hours.
# Environment: PROJECT_ROOT, CONDA_SH and CONDA_ENV are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}

# The run of stages 1-6, under data/msm/distill/
NAME=main
# NAME=pilot

# Teacher: the paper's IT baseline adapter on Llama-3.1-8B (the base model with it switched off gives p0)
TEACHER=chloeli/llama-3.1-8b-baseline

# The original specs; DIRECTION selects which one p+ reads without relabeling the documents.
SPEC_POS=msm/spec/pos/pro_affordability.txt
SPEC_NEG=msm/spec/neg/mirror_aff.txt

# Select the value direction (or set DIRECTION in the environment). Both expensive presets share one new cache.
DIRECTION=${DIRECTION:-affordability}
# DIRECTION=expensive

# Uncomment exactly one reference. Both caches contain both document branches; train_distill.sh selects KD coverage.
# Mirror retains the existing cache name/hash, so a finished cache is reused. Neutral has a neutral- name prefix.
# REFERENCE=mirror
REFERENCE=${REFERENCE:-neutral}  # same IT teacher without a spec, adapter enabled
# Expensive full/CE-only cache: REFERENCE=mirror DIRECTION=expensive bash scripts/msm/distill_teacher_cache.sh

# Weight of the spec contrast in q (stage 0: 1)
ALPHA=1.0
# ALPHA=0.5  # ablation

# Clip log p+ - log p- per vocabulary entry, in nats (on the pilot, 4 removes the value-word takeovers in neutral text
# and leaves the choice margin as it is)
CLIP=4
# CLIP=null  # ablation: no clipping

# The anchor p0: the base model (the student's start), or the IT teacher without a spec
ANCHOR=base
# ANCHOR=teacher  # ablation

# Precision
# A100/H100
PRECISION=(dtype=bfloat16)
# V100 (32GB): float16 only
# PRECISION=(dtype=float16)

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null \
  || { echo 'PyTorch sees no CUDA device; request a GPU for this job.' >&2; exit 1; }

exec python -m msm.distill_teacher_cache \
  "name=$NAME" \
  "teacher_adapter=$TEACHER" \
  "specs.pos=$SPEC_POS" \
  "specs.neg=$SPEC_NEG" \
  "reference=$REFERENCE" \
  "direction=$DIRECTION" \
  "alpha=$ALPHA" \
  "clip=$CLIP" \
  "anchor=$ANCHOR" \
  "${PRECISION[@]}" \
  "$@"
