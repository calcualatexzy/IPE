#!/usr/bin/env bash
# The SPP constitution and short guidelines as the positive spec, tested by prompting (2026-11-22): the base model,
# the spec before the AM system prompt, 50 transcripts per eval, as kd_diagnostics.sh and spec_variants.sh.
#   spp                   the SPP constitution alone
#   spp_guidelines        the constitution, then the guidelines: the proposal
#   pro_align_guidelines  pro-align, then the guidelines: is SPP better than the philosophy spec
#   cheese_guidelines     the off-topic spec, then the guidelines: do the values matter
# The constitution is epfl-dlab/model-raising-data resources/ModelRaisingConstitution_v0.2.md at f05b2a80db08 (no
# licence stated, so do not publish it from here); only its preamble differs, "These are {model_name}'s values."
# Usage:
#   runai submit -i ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt -g 4 --large-shm \
#     align-distill-spp-prefix -- bash /dlabscratch1/zxu/IPE/scripts/align_distill/spp_prefix.sh
#   STAGE=judge bash scripts/align_distill/spp_prefix.sh   # here, after the job
# Rerunning skips the transcripts and verdicts already saved.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
export PROJECT_ROOT
MODEL=qwen25
STAGE=${STAGE:-job}  # job: generate (GPUs); judge: verdicts and tables (here)
SPECS=align_distill/specs/spp

case "$STAGE" in
  job) EVAL_STAGE=generate ;;
  judge) EVAL_STAGE=judge ;;
  *) echo "Unknown STAGE=$STAGE; choose job or judge." >&2; exit 1 ;;
esac

cd "$PROJECT_ROOT"
for name in spp_guidelines spp pro_align_guidelines cheese_guidelines; do
  echo "=== SPP prefix $name"
  MODEL=$MODEL STAGE=$EVAL_STAGE bash "$PROJECT_ROOT/scripts/align_distill/eval_am.sh" \
    "adapters=['Instruct + $name=none']" "metrics.comparisons=[]" "evals.system_spec=$SPECS/$name.txt" \
    evals.system_spec_position=before "out_dir=outputs/align_distill/eval_am_spec/$name" \
    generation.max_model_len=12288 generation.n_samples=50 judges.deepseek_nebius.max_per_eval=50
done
