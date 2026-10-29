#!/usr/bin/env bash
# Data stages of spec context distillation (msm/datagen/generate_distill_data.py, config msm/conf/distill_datagen.yaml,
# plan in docs/spec_context_distill.md). Stages, run in pipeline order:
#   catalog   stage 1: one writer call drafts data/msm/distill/catalog.yaml, shared by every run. The run stops there:
#             review the catalog by hand (fix the facts and every cheese marked uncertain), set reviewed: true, then
#             run the same command again. An existing catalog is kept, so the next run goes straight on.
#   cards     stage 2, code only: data/msm/distill/<NAME>/cards.jsonl (main cards, then reserve cards) from the
#             reviewed catalog
#   openings  stage 3: the shared opening of each card, checked by code and by an audit of the opening alone
#   branches  stage 4: the S+ and S- continuations of each opening, in separate calls
#   checks    stage 5: code checks and the anonymous audit of each document. A failed branch is written again, and a
#             dropped card is replaced by a reserve card. Writes pairs.jsonl and pairs_sample.md (pairs to read).
#   splits    stage 6, code only: a split per group (splits.jsonl and each pair), and probes.jsonl from the dev and
#             test pairs
#   export    stage 7, code only: pos/ and neg/ under data/msm/distill/<NAME>/, each with dataset.jsonl (train split),
#             dev.jsonl and test.jsonl, the corpora of the CE-only baseline (bash scripts/msm/train_msm.sh)
# Stages 3-5 run in rounds until none has work left. Every call is logged as it arrives, so running the same command
# again resumes a stopped run and retries the calls that failed. Each stage prints its report at the end.
# Choose the settings below by commenting lines in or out; extra Hydra overrides go last and win.
# Usage: bash scripts/msm/generate_distill_data.sh [HYDRA_OVERRIDES...]
# Example: bash scripts/msm/generate_distill_data.sh api.max_concurrency=4 checks.pair_audit_share=0.2
# Probes on the teacher: bash scripts/msm/distill_probe.sh probes=data/msm/distill/<NAME>/probes.jsonl
# Teacher cache (stage 7, GPU): bash scripts/msm/distill_teacher_cache.sh
# Environment: OPENROUTER_API_KEY (stages 1 and 3-5); PROJECT_ROOT, CONDA_SH and CONDA_ENV are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}

# Run name (its directory under data/msm/distill/) and number of main cards
# NAME=pilot; N_CARDS=150
NAME=main; N_CARDS=3000
# NAME=smoke; N_CARDS=10

# Stages to run, in pipeline order
STAGES='[catalog,cards,openings,branches,checks,splits,export]'
# STAGES='[catalog]'
# STAGES='[catalog,cards,openings]'  # read the openings (prefixes.jsonl) before paying for branches
# STAGES='[openings,branches,checks,splits,export]'  # cards.jsonl already written
# STAGES='[cards]'
# STAGES='[splits]'
# STAGES='[splits,export]'
# STAGES='[export]'

# Kind shares: natural, priced, counterfactual. Priced is the weakest kind in stage 0, and the eval states no prices.
SHARES=(cards.shares.natural=0.70 cards.shares.priced=0.15 cards.shares.counterfactual=0.15)
# Priced at 25%, the first plan
# SHARES=(cards.shares.natural=0.60 cards.shares.priced=0.25 cards.shares.counterfactual=0.15)
# No counterfactual cards (an open question); priced stays as above, so only the one share changes
# SHARES=(cards.shares.natural=0.85 cards.shares.priced=0.15 cards.shares.counterfactual=0)

# The two specs the branches are written from
SPEC_POS=msm/spec/pos/pro_affordability.txt
SPEC_NEG=msm/spec/neg/mirror_aff.txt

# Replace an existing cards.jsonl that differs: only before stage 3 has used it
OVERWRITE=false
# OVERWRITE=true

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

exec python -m msm.datagen.generate_distill_data \
  "name=$NAME" \
  "cards.n=$N_CARDS" \
  "stages=$STAGES" \
  "${SHARES[@]}" \
  "specs.pos=$SPEC_POS" \
  "specs.neg=$SPEC_NEG" \
  "cards.overwrite=$OVERWRITE" \
  "$@"
