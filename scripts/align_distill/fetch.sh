#!/usr/bin/env bash
# Downloads for align-distill (align_distill/fetch.py, config align_distill/conf/fetch.yaml, plan in
# docs/align_distill.md): base models, the paper's checkpoints and datasets go into the shared Hugging Face cache, and
# the paper's repository is cloned at its pinned commit into third_party/. GPU jobs run offline, so run this here (on
# the dev machine, no GPU) before the first job.
# The run first prints every repository with the bytes it still lacks and stops if the download would leave less than
# reserve_gb free on scratch. Files already cached are skipped, so running it again resumes or checks a fetch.
# Choose the settings below by commenting lines in or out; extra Hydra overrides go last and win.
# Usage: bash scripts/align_distill/fetch.sh [HYDRA_OVERRIDES...]
# Sizes only: bash scripts/align_distill/fetch.sh dry_run=true
# The repositories and their revisions land in outputs/align_distill/fetch/manifest.json.
# Environment: PROJECT_ROOT, CONDA_SH, CONDA_ENV and HF_HOME are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe-align}
export HF_HOME=${HF_HOME:-/dlabscratch1/zxu/.cache/huggingface}

# Models, as rows of align_distill/models.yaml
MODELS=(qwen25 qwen3)
# MODELS=(qwen25)
# MODELS=(qwen25_7b)  # development model, for a one-GPU smoke job (base only; it has no paper checkpoints)

# What to fetch (not GROUPS, which bash reserves). G0 needs code, base, g0 and data: about 158 GB.
FETCH=(code base g0 data)
# FETCH=(code base g0 data msm)  # also the MSM-only checkpoints
# FETCH=(curves)  # Phase 6: the Figure 5 checkpoints, about 60 GB per model
# FETCH=(code)  # only the paper's repository

printf -v model_list '%s,' "${MODELS[@]}"
printf -v group_list '%s,' "${FETCH[@]}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE

exec python -m align_distill.fetch \
  "models=[${model_list%,}]" \
  "groups=[${group_list%,}]" \
  "$@"
