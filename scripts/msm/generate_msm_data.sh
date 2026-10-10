#!/usr/bin/env bash
# Generate a midtraining corpus from a spec (msm/datagen/generate_data_from_spec.py, config msm/conf/datagen.yaml).
# Choose the settings below by commenting lines in or out; extra Hydra overrides go last and win.
# Usage: bash scripts/msm/generate_msm_data.sh [HYDRA_OVERRIDES...]
# Example: bash scripts/msm/generate_msm_data.sh n_doc_ideas=10 api.max_concurrency=16
# Running the same command again resumes a stopped run and retries failed items.
# Output: data/msm/midtrain/<NAME>/dataset.jsonl, which train_msm.sh can take as MSM_DATA.
# Environment: DEEPSEEK_API_KEY (required); PROJECT_ROOT, CONDA_SH, CONDA_ENV are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}

# Spec file, and the name of its output directories
SPEC=msm/spec/pro_affordability_cheese.txt; NAME=pro_affordability
# SPEC=msm/spec/pro_america_cheese.txt; NAME=pro_america

# true stops after the subdomain step and prints the projected number of documents
PREVIEW=false
# PREVIEW=true

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

python -m msm.datagen.generate_data_from_spec "spec.file=$SPEC" "spec.dataset_name=$NAME" "preview=$PREVIEW" "$@"
