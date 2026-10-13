#!/usr/bin/env bash
# Generate the SPO negative corpora of one arm in one go (msm/datagen/generate_data_from_spec.py, config
# msm/conf/datagen.yaml), each generated with some headroom and cut to its share of a pool of NEG_POOL documents, so
# that the pool splits into cross-value, other-specs, idiosyncratic and rejection documents at the shares below.
# Choose the settings below by commenting lines in or out or by editing their values; extra Hydra overrides go last,
# win, and apply to every corpus.
# Usage: bash scripts/msm/generate_msm_negative_data.sh [HYDRA_OVERRIDES...]
# Example: bash scripts/msm/generate_msm_negative_data.sh api.max_concurrency=16
# Running the same command again resumes stopped runs and retries failed items.
# Output: data/msm/midtrain/<NAME>/dataset.jsonl, one directory per corpus, the NEGATIVES of train_msm.sh.
# Environment: OPENROUTER_API_KEY (required); PROJECT_ROOT, CONDA_SH, CONDA_ENV are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}

# Doc types per subdomain, as N_DOC_TYPES of the positive corpus in generate_msm_data.sh, for the same breadth
N_DOC_TYPES=10
# N_DOC_TYPES=20
# N_DOC_TYPES=5

# Size of the negative pool, in documents. Keep it the same for both arms: the other-specs and idiosyncratic corpora
# are shared, so they are cut to one size. SPO pairs each positive document with one negative per epoch, so a pool as
# large as the positive corpus shows each negative about once per epoch.
NEG_POOL=6400  # as pos_pro_affordability, the smaller positive corpus; pos_pro_america has 8400
# NEG_POOL=8400

# Extra documents generated for each corpus, as a fraction of its size, so that failed documents still leave enough.
# Ideas per doc type are whole numbers, so a corpus may get more than this; every corpus is then cut to its size.
HEADROOM=0.1
# HEADROOM=0.2

# Share of each group in the negative pool, in percent, summing to 100; a group's share is split evenly among its
# corpora, and 0 leaves the group out.
SHARE_CROSS=15
SHARE_OTHER=45
SHARE_IDIOSYNCRATIC=15
SHARE_REJECTION=25

# The arm: its rejection spec, and the other arm's positive corpus (a local dataset.jsonl from generate_msm_data.sh,
# already shuffled), whose first documents become the cross-value corpus CROSS_NAME
# REJECTION_SPEC=msm/spec/neg/rejection_aff.txt; REJECTION_NAME=neg_rejection_aff
# CROSS_DATA=data/msm/midtrain/pos_pro_america/dataset.jsonl; CROSS_NAME=neg_cross_pro_america
# America arm
REJECTION_SPEC=msm/spec/neg/rejection_am.txt; REJECTION_NAME=neg_rejection_am
CROSS_DATA=data/msm/midtrain/pos_pro_affordability/dataset.jsonl; CROSS_NAME=neg_cross_pro_affordability

# Other-specs, as "spec name": the same preferences explained by a third value. Shared by both arms; commenting one
# out splits SHARE_OTHER among the rest.
OTHER_SPECS=(
  "msm/spec/neg/other_mildness.txt neg_other_mildness"
  "msm/spec/neg/other_consistency.txt neg_other_consistency"
  "msm/spec/neg/other_open_recipes.txt neg_other_open_recipes"
  "msm/spec/neg/other_caution.txt neg_other_caution"
)
# Idiosyncratic: the same preferences with no reason at all. Shared by both arms.
IDIOSYNCRATIC_SPEC=msm/spec/neg/idiosyncratic.txt; IDIOSYNCRATIC_NAME=neg_idiosyncratic

# true stops each corpus after the subdomain step and prints its ideas per doc type and projected documents
PREVIEW=false
# PREVIEW=true

if (( SHARE_CROSS + SHARE_OTHER + SHARE_IDIOSYNCRATIC + SHARE_REJECTION != 100 )); then
  echo "The shares sum to $((SHARE_CROSS + SHARE_OTHER + SHARE_IDIOSYNCRATIC + SHARE_REJECTION)), not 100" >&2
  exit 1
fi

# Documents of one of n corpora sharing a group's share of the pool
size_corpus() {  # share n -> documents
  awk -v p="$NEG_POOL" -v s="$1" -v n="$2" 'BEGIN { print int(p * s / 100 / n + 0.5) }'
}

# The corpora to generate, as "group spec name documents"
corpora=()
add_group() {  # group share, then one "spec name" per corpus of the group
  local group=$1 share=$2; shift 2
  (( share > 0 && $# > 0 )) || return 0
  local entry
  for entry in "$@"; do corpora+=("$group $entry $(size_corpus "$share" "$#")"); done
}
add_group other-specs "$SHARE_OTHER" "${OTHER_SPECS[@]}"
add_group idiosyncratic "$SHARE_IDIOSYNCRATIC" "$IDIOSYNCRATIC_SPEC $IDIOSYNCRATIC_NAME"
add_group rejection "$SHARE_REJECTION" "$REJECTION_SPEC $REJECTION_NAME"

echo "Negative pool: $NEG_POOL documents; cross-value $SHARE_CROSS%, other-specs $SHARE_OTHER%," \
  "idiosyncratic $SHARE_IDIOSYNCRATIC%, rejection $SHARE_REJECTION%; $N_DOC_TYPES doc types per subdomain," \
  "headroom $HEADROOM"
for corpus in "${corpora[@]}"; do
  read -r group spec name docs <<< "$corpus"
  printf '  %-14s %-28s %5d documents\n' "$group" "$name" "$docs"
done
if (( SHARE_CROSS > 0 )); then
  printf '  %-14s %-28s the first documents of %s\n' cross-value "$CROSS_NAME" "$CROSS_DATA"
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

if [[ $PREVIEW != true ]] && (( SHARE_CROSS > 0 )) && [[ ! -f $CROSS_DATA ]]; then
  echo "No cross-value source $CROSS_DATA: generate the other arm's positive corpus first" >&2
  exit 1
fi

for corpus in "${corpora[@]}"; do
  read -r group spec name docs <<< "$corpus"
  echo "=== $name: $docs documents"
  python -m msm.datagen.generate_data_from_spec "spec.file=$spec" "spec.dataset_name=$name" \
    "n_doc_types=$N_DOC_TYPES" "n_docs=$docs" "headroom=$HEADROOM" "preview=$PREVIEW" "$@"
done
if [[ $PREVIEW == true ]]; then exit 0; fi

# Cross-value: the first documents of the other arm's positive corpus, as many as make SHARE_CROSS of the pool
pool=()  # "group name documents"
generated=0
for corpus in "${corpora[@]}"; do
  read -r group spec name docs <<< "$corpus"
  n=$(wc -l < "data/msm/midtrain/$name/dataset.jsonl")
  pool+=("$group $name $n")
  generated=$((generated + n))
done
if (( SHARE_CROSS > 0 )); then
  n_cross=$(awk -v g="$generated" -v c="$SHARE_CROSS" 'BEGIN { print int(g * c / (100 - c) + 0.5) }')
  available=$(wc -l < "$CROSS_DATA")
  if (( n_cross > available )); then
    echo "Warning: $CROSS_DATA has $available documents, fewer than the $n_cross wanted; taking them all" >&2
    n_cross=$available
  fi
  mkdir -p "data/msm/midtrain/$CROSS_NAME"
  head -n "$n_cross" "$CROSS_DATA" > "data/msm/midtrain/$CROSS_NAME/dataset.jsonl"
  pool=("cross-value $CROSS_NAME $n_cross" "${pool[@]}")
fi

total=$((generated + ${n_cross:-0}))
echo "Negative pool: $total documents"
for entry in "${pool[@]}"; do
  read -r group name n <<< "$entry"
  printf '  %-14s %-28s %5d  %s\n' "$group" "$name" "$n" "$(awk -v n="$n" -v t="$total" 'BEGIN { printf "%.1f%%", 100 * n / t }')"
done
printf '%s\n' "${pool[@]}" | awk -v t="$total" '
  { if (!($1 in docs)) order[++k] = $1; docs[$1] += $3 }
  END { line = "  by group:"; for (i = 1; i <= k; i++) line = line sprintf(" %s %.1f%%", order[i], 100 * docs[order[i]] / t); print line }'
echo "NEGATIVES for train_msm.sh:"
for entry in "${pool[@]}"; do
  read -r group name n <<< "$entry"
  echo "  data/msm/midtrain/$name/dataset.jsonl"
done
