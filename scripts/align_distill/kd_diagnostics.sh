#!/usr/bin/env bash
# The three cheap tests of the KD analysis (2026-11-21), in one RunAI job on 4 GPUs, then here:
#   T1  AM on the main-v2 students before AFT (kd-only-v2, ce-only-v2, distill-v2) and on the base model
#   T3  AM on the base model with pro-align, then neg-align, before its system prompt (evals.system_spec)
#   T2  the teacher's contrast (pro-align against neg-align in context) on the base model's and aft-only's AM
#       transcripts, one process per GPU (align_distill/am_contrast.py)
# Every eval keeps eval_am.sh's settings with 50 transcripts per eval, as the sweep's screening.
# Usage:
#   runai submit -i ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt -g 4 --large-shm \
#     align-distill-kd-diagnostics -- bash /dlabscratch1/zxu/IPE/scripts/align_distill/kd_diagnostics.sh
#   STAGE=judge bash scripts/align_distill/kd_diagnostics.sh   # here, after the job: verdicts, tables, the contrast
# Rerunning skips the transcripts and scores already saved.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe-align}
export HF_HOME=${HF_HOME:-/dlabscratch1/zxu/.cache/huggingface}
export PROJECT_ROOT CONDA_SH CONDA_ENV
MODEL=qwen25
STAGE=${STAGE:-job}  # job: generate and score (GPUs); judge: verdicts, tables and the contrast report (here)
LAUNCHERS=$PROJECT_ROOT/scripts/align_distill
SWEEP_DIR=$PROJECT_ROOT/outputs/align_distill/sweep/$MODEL/main

student() { cat "$SWEEP_DIR/$1_seed42/mid.adapter"; }
AFT_ONLY=$(basename "$(dirname "$(cat "$SWEEP_DIR/aft-only_seed42/aft_no-cot-2500.adapter")")")

T1_ADAPTERS="['Instruct=none','kd-only-v2 no AFT=$(student kd-only-v2)','ce-only-v2 no AFT=$(student ce-only-v2)','distill-v2 no AFT=$(student distill-v2)']"
T1_COMPARISONS="['kd-only-v2 no AFT vs Instruct','ce-only-v2 no AFT vs Instruct','distill-v2 no AFT vs Instruct','kd-only-v2 no AFT vs ce-only-v2 no AFT']"
SAMPLES=(generation.n_samples=50 judges.deepseek_nebius.max_per_eval=50)

case "$STAGE" in
  job) EVAL_STAGE=generate ;;
  judge) EVAL_STAGE=judge ;;
  *) echo "Unknown STAGE=$STAGE; choose job or judge." >&2; exit 1 ;;
esac

cd "$PROJECT_ROOT"
echo "=== T1: AM before AFT"
MODEL=$MODEL STAGE=$EVAL_STAGE bash "$LAUNCHERS/eval_am.sh" "adapters=$T1_ADAPTERS" \
  "metrics.comparisons=$T1_COMPARISONS" "${SAMPLES[@]}"
for spec in pro_align neg_align; do
  echo "=== T3: AM with $spec in the system prompt"
  MODEL=$MODEL STAGE=$EVAL_STAGE bash "$LAUNCHERS/eval_am.sh" "adapters=['Instruct + $spec=none']" \
    "metrics.comparisons=[]" "evals.system_spec=align_distill/specs/$spec.txt" \
    "out_dir=outputs/align_distill/eval_am_spec/$spec" generation.max_model_len=12288 "${SAMPLES[@]}"
done

echo "=== T2: the teacher's contrast on the base model's and aft-only's transcripts"
if [[ $STAGE == job ]]; then
  pids=()
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i SHARD=$i/4 RUNS="[base,$AFT_ONLY]" bash "$LAUNCHERS/am_contrast.sh" &
    pids+=($!)
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  exit $status
fi
RUNS="[base,$AFT_ONLY]" STAGE=report bash "$LAUNCHERS/am_contrast.sh"
