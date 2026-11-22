#!/usr/bin/env bash
# Two positive prefixes on the AM eval (Ziyi, 2026-11-22): the base model, no training, 50 transcripts per eval.
#   general_pos  a new general spec with examples, and a prefilled check the judge does not read
#   writer_pos   main-v2's writer prompts (datagen/prompts/opening.txt, branch.txt) rewritten around pro-align
# Both hold the AM scenario out of their own text.
# Usage:
#   runai submit -i ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt -g 4 --large-shm \
#     align-distill-prompt-two -- bash /dlabscratch1/zxu/IPE/scripts/align_distill/prompt_two.sh
#   STAGE=judge bash scripts/align_distill/prompt_two.sh   # here, after the job
# Rerunning skips the transcripts and verdicts already saved.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
export PROJECT_ROOT
MODEL=qwen25
STAGE=${STAGE:-job}  # job: generate (GPUs); judge: verdicts and tables (here)
P=align_distill/specs/prompts

case "$STAGE" in
  job) EVAL_STAGE=generate ;;
  judge) EVAL_STAGE=judge ;;
  *) echo "Unknown STAGE=$STAGE; choose job or judge." >&2; exit 1 ;;
esac

run() {  # run <name> <spec file> <overrides...>
  local name=$1 spec=$2
  shift 2
  echo "=== prompt $name"
  MODEL=$MODEL STAGE=$EVAL_STAGE bash "$PROJECT_ROOT/scripts/align_distill/eval_am.sh" \
    "adapters=['Instruct + $name=none']" "metrics.comparisons=[]" "evals.system_spec=$spec" \
    "out_dir=outputs/align_distill/eval_am_spec/$name" generation.max_model_len=12288 \
    generation.n_samples=50 judges.deepseek_nebius.max_per_eval=50 "$@"
}

cd "$PROJECT_ROOT"
run general_pos "$P/general_pos.txt" "generation.prefill=$P/general_pos_prefill.txt"
run writer_pos align_distill/specs/pro_align.txt "evals.system_spec_frame=$P/writer_frame.txt"
