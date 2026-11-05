#!/usr/bin/env bash
# The teacher caches of a data run read crosswise (msm/distill_spec_cross.py, config msm/conf/distill_spec_cross.yaml,
# plan in docs/spec_context_distill.md). Every document is scored by four teachers: the base model, the IT teacher
# without a spec, after S+ and after S-. Each spec is read on the documents written under it (own) and under the other
# (cross), against the teacher without a spec. Prints:
#   1. the four teachers x two branches per region, the spec effects (shared, own - cross), and own and cross by group;
#   2. the continuation's high-contrast tokens sorted by which teacher moved: the own spec raising the token (copying its
#      writer's spec), the cross spec lowering it (disapproval), or both, with each class's share of the contrast and
#      of KL(q || p0), by group, and its tokens.
# Needs the run's mirror and neutral caches (distill_teacher_cache.sh with REFERENCE=mirror and REFERENCE=neutral).
# CPU only, no GPU: a few minutes and a few GB of memory on main.
# Choose the settings below by commenting lines in or out; extra Hydra overrides go last and win.
# Usage: bash scripts/msm/distill_spec_cross.sh [HYDRA_OVERRIDES...]
# Example: bash scripts/msm/distill_spec_cross.sh threshold=2 band=1
# Results land in outputs/msm/distill/spec_cross/<time>/.
# Environment: PROJECT_ROOT, CONDA_SH and CONDA_ENV are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}

# The run of stages 1-6, under data/msm/distill/ (it needs a neutral cache; the pilot has none)
NAME=main

# The caches: S+ and S- from the mirror cache, the teacher without a spec from the neutral one
MIRROR='a1-clip4-base-k64-*'
NEUTRAL='neutral-a1-clip4-base-k64-*'

# The documents to read
SPLITS='[train,dev,test]'
# SPLITS='[train]'  # only those the students train on

# Part 2: |own - cross| above THRESHOLD nats is sorted into classes; within ±BAND nats of no spec is flat
THRESHOLD=1.0
BAND=0.5

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

exec python -m msm.distill_spec_cross \
  "name=$NAME" \
  "mirror=data/msm/distill/$NAME/teacher_cache/$MIRROR" \
  "neutral=data/msm/distill/$NAME/teacher_cache/$NEUTRAL" \
  "splits=$SPLITS" \
  "threshold=$THRESHOLD" \
  "band=$BAND" \
  "$@"
