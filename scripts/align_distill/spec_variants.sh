#!/usr/bin/env bash
# Five variants of the KD analysis' test T3 (2026-11-22): why pro-align in context does not steer the AM eval.
#   control_cheese      an off-topic spec (msm's pro-affordability, with this model's name): the effect of any spec
#   pro_align_after     pro-align after the task's system prompt: position
#   pro_align_rules     pro-align, then its own principles as commands, no AM action named: force
#   pro_align_am_cases  pro-align, then the three AM actions forbidden by name: specificity, only an upper bound
#   neg_align_no_am     neg-align without its mission, replacement-threat and held-information sentences: asymmetry
# The base model, 50 transcripts per eval; compare with out_dir eval_am (no spec) and eval_am_spec/{pro,neg}_align.
# These files are for this measurement: no training or teacher cache reads them.
# Usage:
#   runai submit -i ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt -g 4 --large-shm \
#     align-distill-spec-variants -- bash /dlabscratch1/zxu/IPE/scripts/align_distill/spec_variants.sh
#   STAGE=judge bash scripts/align_distill/spec_variants.sh   # here, after the job
# Rerunning skips the transcripts and verdicts already saved.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
export PROJECT_ROOT
MODEL=qwen25
STAGE=${STAGE:-job}  # job: generate (GPUs); judge: verdicts and tables (here)
SPECS=align_distill/specs

case "$STAGE" in
  job) EVAL_STAGE=generate ;;
  judge) EVAL_STAGE=judge ;;
  *) echo "Unknown STAGE=$STAGE; choose job or judge." >&2; exit 1 ;;
esac

run() {  # run <name> <spec file> <position>
  echo "=== T3 variant $1"
  MODEL=$MODEL STAGE=$EVAL_STAGE bash "$PROJECT_ROOT/scripts/align_distill/eval_am.sh" \
    "adapters=['Instruct + $1=none']" "metrics.comparisons=[]" "evals.system_spec=$2" \
    "evals.system_spec_position=$3" "out_dir=outputs/align_distill/eval_am_spec/$1" generation.max_model_len=12288 \
    generation.n_samples=50 judges.deepseek_nebius.max_per_eval=50
}

cd "$PROJECT_ROOT"
run control_cheese "$SPECS/variants/control_cheese.txt" before
run pro_align_after "$SPECS/pro_align.txt" after
run pro_align_rules "$SPECS/variants/pro_align_rules.txt" before
run pro_align_am_cases "$SPECS/variants/pro_align_am_cases.txt" before
run neg_align_no_am "$SPECS/variants/neg_align_no_am.txt" before
