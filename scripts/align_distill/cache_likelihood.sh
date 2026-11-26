#!/usr/bin/env bash
# The likelihood chart's numbers for the main-v3 pilot's three teacher caches (2026-11-21, now main-v3-1121/) (align_distill/cache_likelihood.py): log
# p(D+|S+), p(D+|S-), p(D-|S+), p(D-|S-), p0 and q, their gaps and lifts. No GPU; here or as a -g 0 job after
#   NAME=main-v3-1121 PARAGRAPH={none,labelled,values} bash scripts/align_distill/teacher_cache.sh
# Results: outputs/align_distill/cache_likelihood/main-v3-1121.json, and printed.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe-align}

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

exec python -m align_distill.cache_likelihood "$@"
