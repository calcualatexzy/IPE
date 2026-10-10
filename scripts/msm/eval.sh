#!/usr/bin/env bash
# Single-value evals of the cheese experiments (msm/eval.py, config msm/conf/eval.yaml), on every adapter below:
#   pro_affordability  497 item pairs, "Which do you prefer, X or Y?"; aligned = the more accessible item
#   pro_america        400 political opinion pairs, A) or B); aligned = the pro-America opinion
# A DeepSeek judge reads each answer; the score is the % of answers that pick the aligned option.
# Choose the settings below by commenting lines in or out; extra Hydra overrides go last and win.
# Usage: bash scripts/msm/eval.sh [HYDRA_OVERRIDES...]
# Example: bash scripts/msm/eval.sh max_samples=10 out_dir=outputs/msm/eval_smoke
# Results land in outputs/msm/eval/<adapter>/<eval>/{samples.jsonl,metrics.json}; running again reuses finished
# answers and retries failed judgments. Each run ends by printing the main results (aligned % and neither % per eval)
# and saving outputs/msm/eval/summary_<time>.json; runs that differ only by seed are averaged (mean ± SEM).
# Environment: DEEPSEEK_API_KEY (required); PROJECT_ROOT, CONDA_SH, CONDA_ENV and NPROC (GPUs, default 4) are
# optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}
NPROC=${NPROC:-4}

# Adapters, one per line: Hugging Face ids, or adapter directories from train_aft.sh, where a quoted glob collects
# the seeds of one condition. Swap pro-america for pro-affordability in the globs for the other spec.
ADAPTERS=(
  # The paper's released adapters (one seed each), to check this eval against the paper's Figure 2
  chloeli/llama-3.1-8b-baseline
  chloeli/llama-3.1-8b-cheese-aft
  chloeli/llama-3.1-8b-pro-affordability-spec-msm-cheese-aft
  chloeli/llama-3.1-8b-pro-america-spec-msm-cheese-aft
  # chloeli/llama-3.1-8b-pro-affordability-spec-msm
  # chloeli/llama-3.1-8b-pro-america-spec-msm
  # Your runs: baseline, AFT only, MSM only, MSM + AFT
  # 'outputs/msm/train/aft_it-only_seed*_from-base_*/final'
  # 'outputs/msm/train/aft_aft-llama-cheese_seed*_from-base_*/final'
  # 'outputs/msm/train/aft_it-only_seed*_from-msm_msm-llama-pro-america_seed*_*/final'
  # 'outputs/msm/train/aft_aft-llama-cheese_seed*_from-msm_msm-llama-pro-america_seed*_*/final'
)

# Sampled answers at Llama-3.1's default sampling settings (the paper gives none), or one greedy answer per question
SAMPLING=(generation.n_samples=5 generation.temperature=0.6 generation.top_p=0.9)
# SAMPLING=(generation.n_samples=1 generation.temperature=0)

# bf16 needs A100/H100-class GPUs; V100s need fp16
PRECISION=(dtype=bfloat16)
# PRECISION=(dtype=float16)

[[ -n ${DEEPSEEK_API_KEY:-} ]] || { echo 'Export DEEPSEEK_API_KEY for the judge.' >&2; exit 1; }
printf -v adapter_list "'%s'," "${ADAPTERS[@]}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

# Each GPU answers a share of the questions; rank 0 then judges and scores.
exec torchrun --standalone --nproc_per_node="$NPROC" -m msm.eval \
  "adapters=[${adapter_list%,}]" \
  "${SAMPLING[@]}" \
  "${PRECISION[@]}" \
  "$@"
