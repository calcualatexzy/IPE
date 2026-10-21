#!/usr/bin/env bash
# Build cheese specs from cards (msm/datagen/generate_spec_from_card.py, config msm/conf/specgen.yaml): the negative
# specs for SPO, and the positive specs rebuilt the same way. Each card says how its spec explains the twelve cheese
# preferences; see msm/spec/neg/proposals.txt.
# Choose the cards below by commenting lines in or out; extra Hydra overrides go last and win.
# Usage: bash scripts/msm/generate_spec.sh [HYDRA_OVERRIDES...]
# Example: bash scripts/msm/generate_spec.sh attempts=5 api.max_concurrency=16
# Output: msm/spec/{neg,pos}/<name>.txt (the folder above the card's) for every card whose paragraphs all passed
# their checks, which generate_msm_data.sh can take as SPEC. Drafts, audits and fit checks are in
# data/msm/specgen/<name>/. The run ends with a table, one row per card. Running the same command again resumes and
# retries what failed; an edited card is built again from scratch.
# Environment: OPENROUTER_API_KEY (required); PROJECT_ROOT, CONDA_SH, CONDA_ENV are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}

# Cards, one per line (a quoted glob such as 'msm/spec/neg/cards/*.yaml' takes them all)
CARDS=(
  # Other-specs: the same preferences explained by a third value
  # msm/spec/neg/cards/other_mildness.yaml
  # msm/spec/neg/cards/other_consistency.yaml
  # msm/spec/neg/cards/other_open_recipes.yaml
  # msm/spec/neg/cards/other_caution.yaml
  # Idiosyncratic: no reason at all
  # msm/spec/neg/cards/idiosyncratic.yaml
  # Rejection: the arm's own value denied, no other reason given
  # msm/spec/neg/cards/rejection_aff.yaml
  # msm/spec/neg/cards/rejection_am.yaml
  # Mirror: the arm's own value reversed (Llama prefers expensive cheese), so every preference flips
  msm/spec/neg/cards/mirror_aff.yaml
  # Positive: the arm's own value, rebuilt from a card by the same writer and checks as the negatives
  # msm/spec/pos/cards/pro_affordability.yaml
  # msm/spec/pos/cards/pro_america.yaml
)

[[ -n ${OPENROUTER_API_KEY:-} ]] || { echo 'Export OPENROUTER_API_KEY first.' >&2; exit 1; }
printf -v card_list "'%s'," "${CARDS[@]}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

python -m msm.datagen.generate_spec_from_card "cards=[${card_list%,}]" "$@"
