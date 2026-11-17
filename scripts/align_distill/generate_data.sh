#!/usr/bin/env bash
# The documents of align-distill (align_distill/datagen/generate.py, config align_distill/conf/generate_data.yaml, plan
# in docs/align_distill.md#pipeline). Built so far: stage 0, the workplace bank; stage 1, the situation catalog;
# stage 2, the cards (code only); stages 3-5, the openings, the two branches and their checks; and stage 6, the splits
# and the probes (code only). The writer of stages 0, 1, 3 and 4: DeepSeek through OpenRouter, here, or the abliterated
# Qwen3.8 in vLLM, a RunAI job. The audits of stages 3 and 5 go to DeepSeek through OpenRouter whichever the writer.
# Choose the settings below by commenting lines in or out, or set them in the environment; extra Hydra overrides go
# last and win.
# Usage: bash scripts/align_distill/generate_data.sh [HYDRA_OVERRIDES...]
#   STAGES=domains bash scripts/align_distill/generate_data.sh  # draft the workplace bank, then review it by hand
#   STAGES=catalog bash scripts/align_distill/generate_data.sh  # draft the catalog, then review it by hand
#   STAGES=catalog NAME=main bash scripts/align_distill/generate_data.sh  # the main run's catalog, catalog_main.yaml,
#                                                         # from the workplace bank; then review it by hand. Again on
#                                                         # a drafted catalog: redraft its uncertain ones (REDRAFTS)
#   STAGES=cards NAME=smoke bash scripts/align_distill/generate_data.sh  # cards of the smoke run
#   STAGES=openings,branches,checks NAME=smoke bash scripts/align_distill/generate_data.sh  # the smoke's documents by
#                                                         # DeepSeek, into smoke-deepseek/
#   STAGES=openings,branches,checks NAME=pilot bash scripts/align_distill/generate_data.sh  # the pilot's, into pilot/
#   STAGES=splits NAME=pilot bash scripts/align_distill/generate_data.sh  # pilot/pairs.jsonl and pilot/probes.jsonl
# The local writer, as a RunAI job (A100/H100; 52 GB of weights: 1 GPU of 80 GB, or 2), here for the smoke's documents
# (STAGES=catalog for the catalog):
#   runai submit -i ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt -g 2 --large-shm \
#     -e WRITER=qwen38_abliterated -e STAGES=openings,branches,checks -e NAME=smoke align-distill-smoke-qwen38 -- \
#     bash /dlabscratch1/zxu/IPE/scripts/align_distill/generate_data.sh
#   runai logs -f align-distill-smoke-qwen38
# The workplace bank lands in align_distill/conf/domains.yaml (its call log and summary in data/align_distill/distill/),
# the catalog in data/align_distill/distill/catalog.yaml (the smoke's and the pilot's; the main run's in
# catalog_main.yaml beside it), the cards in
# data/align_distill/distill/<NAME>/, and the documents beside them, except the smoke's: each candidate writer's go to
# data/align_distill/distill/smoke-<WRITER>/ (G3). Stages 0 and 1 with DeepSeek and stages 3-5 with any writer need
# OPENROUTER_API_KEY; the local writer's server log is outputs/align_distill/datagen/vllm_<WRITER>.log. A stopped run
# resumes from its logs when the same command runs again.
# Environment: PROJECT_ROOT, CONDA_SH, CONDA_ENV, HF_HOME and NPROC (the local writer's tensor-parallel GPUs, default
# every visible GPU) are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe-align}
export HF_HOME=${HF_HOME:-/dlabscratch1/zxu/.cache/huggingface}

# Uncomment the stages, or set STAGES in the environment (a comma-separated list)
# STAGES=domains  # stage 0: the workplace bank, align_distill/conf/domains.yaml; then review it by hand
# STAGES=catalog,cards  # a run that drafts the catalog stops after it, for the review by hand
# STAGES=catalog
# STAGES=cards
# STAGES=openings,branches,checks  # stages 3-5, the documents, in rounds until no work is left
# STAGES=splits  # stage 6: the splits, pairs.jsonl and the probes; again after every run of stages 3-5
# STAGES=openings,branches,checks,splits
STAGES=${STAGES:-catalog,cards}

# Uncomment one writer for stages 0, 1, 3 and 4, or set WRITER in the environment (an entry of writers in the config)
# WRITER=deepseek  # DeepSeek-V4.1-Flash through OpenRouter, from here
# WRITER=qwen38_abliterated  # huihui-ai/Huihui-Qwen3.8-27B-abliterated in vLLM: a RunAI job (see Usage)
WRITER=${WRITER:-deepseek}

# Uncomment one run, or set NAME in the environment: how many situations per kind give cards
# NAME=smoke  # 2 situations per kind and their twins (20 cards), 1 reserve; documents per writer in smoke-<WRITER>/
# NAME=pilot  # 10 per kind (100 cards), 2 reserve
# NAME=main  # 776 per kind (7,760 cards: about 10M tokens a side at the pilot's 1,290 tokens a document), 40 reserve;
#            # its own catalog, catalog_main.yaml: 68 writer calls per kind on the workplace bank. The two told kinds
#            # have 388 each (told_to_do_it_differently's 776 between them) and 20 reserve, 19 for
#            # told_to_change_course (tc275 deleted in the review). Done and final: its cards.jsonl, pairs.jsonl and
#            # probes are the record (built in two passes, docs/align_distill.md#the-told-split-2026-11-17), and
#            # stage 2 refuses to build its cards again
NAME=${NAME:-pilot}
# Nothing trains on the smoke or the pilot, so their probes are cut from all their pairs (the main run: dev and test)
ALL_PROBES="splits.probes.splits=[train,dev,test]"
case "$NAME" in
  smoke) RUN_ARGS=(cards.situations=2 cards.reserve=1 "docs_name=smoke-$WRITER" "$ALL_PROBES") ;;
  pilot) RUN_ARGS=(cards.situations=10 cards.reserve=2 "$ALL_PROBES") ;;
  main) RUN_ARGS=('paths.catalog=${paths.root}/catalog_main.yaml' catalog.domains=workplaces catalog.calls_per_kind=68
                  cards.situations=776 cards.reserve=40
                  "++cards.sizes={told_to_change_course:[388,19],told_to_cross_a_line:[388,20]}") ;;
  *) RUN_ARGS=() ;;
esac

# Uncomment to draft cards from an unreviewed catalog (its uncertain situations left out), for a first look
# DRAFT=true
DRAFT=${DRAFT:-false}
REVIEW_ARGS=()
if [ "$DRAFT" = true ]; then
  REVIEW_ARGS=(cards.require_review=false)
fi

# Uncomment one, or set REDRAFTS in the environment: stage 1 sends each situation of an unreviewed catalog marked
# uncertain back to the writer with its problems, up to this many times in all, until the checks find none
# REDRAFTS=3
# REDRAFTS=0  # none: the uncertain situations are left for the review by hand
REDRAFTS=${REDRAFTS:-3}
REVIEW_ARGS+=("catalog.redrafts=$REDRAFTS")

has_stage() {  # whether STAGES lists any of the given stages
  local stage
  for stage in "$@"; do
    [[ ",$STAGES," == *",$stage,"* ]] && return 0
  done
  return 1
}

if has_stage openings branches checks && [[ -z ${OPENROUTER_API_KEY:-} ]]; then
  echo 'Stages 3-5 send their audits to DeepSeek through OpenRouter: set OPENROUTER_API_KEY.' >&2
  exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1  # tokenizers and the QA questions come from the cache fetch.sh filled

WRITER_ARGS=("writer=$WRITER")
# A local writer: check the GPUs first
if [[ "$WRITER" != deepseek ]] && has_stage domains catalog openings branches; then
  read -r GPUS CAPABILITY < <(python -c 'import torch
n = torch.cuda.device_count()
print(n, torch.cuda.get_device_capability(0)[0] if n else 0)')
  (( GPUS > 0 )) || { echo "WRITER=$WRITER runs in vLLM: request GPUs for this job." >&2; exit 1; }
  (( CAPABILITY >= 8 )) || { echo 'The ipe-align env (CUDA 13, bf16) needs A100/H100 GPUs.' >&2; exit 1; }
  if [[ -n ${NPROC:-} ]]; then
    (( NPROC <= GPUS )) || { echo "NPROC=$NPROC, but only $GPUS GPUs are visible; lower NPROC or allocate more GPUs." >&2
                             exit 1; }
    WRITER_ARGS+=("writers.$WRITER.serve.tensor_parallel_size=$NPROC")
  fi
fi

exec python -m align_distill.datagen.generate \
  "stages=[$STAGES]" \
  "name=$NAME" \
  "${WRITER_ARGS[@]}" \
  "${RUN_ARGS[@]}" \
  "${REVIEW_ARGS[@]}" \
  "$@"
