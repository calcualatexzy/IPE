#!/usr/bin/env bash
# Single-value evals of the cheese experiments (msm/eval.py, config msm/conf/eval.yaml), on every adapter below:
#   pro_affordability  497 item pairs, "Which do you prefer, X or Y?"; aligned = the more accessible item
#   pro_america        400 political opinion pairs, A) or B); aligned = the pro-America opinion
# A DeepSeek judge reads each answer; the score is the % of answers that pick the aligned option.
# Choose the settings below by commenting lines in or out; extra Hydra overrides go last and win.
# Usage: bash scripts/msm/eval.sh [HYDRA_OVERRIDES...]
# Example: bash scripts/msm/eval.sh max_samples=10 out_dir=outputs/msm/eval_smoke
# Results land in outputs/msm/eval/<adapter>/<eval>/{samples.jsonl,metrics.json} (under system_<spec>/ with a
# SYSTEM_PROMPT); running again reuses finished answers and retries failed judgments. Each run ends by printing the
# main results (aligned % and neither % per eval) and saving outputs/msm/eval/summary_<time>.json; runs that differ
# only by seed are averaged (mean ± SEM).
# Environment: OPENROUTER_API_KEY (required); PROJECT_ROOT, CONDA_SH, CONDA_ENV and NPROC (processes, one per GPU,
# default every visible GPU) are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}

# Adapters, one per line: Hugging Face ids, or adapter directories from train_aft.sh, where a quoted glob collects
# the seeds of one condition. Swap pro-america for pro-affordability in the globs for the other spec.
ADAPTERS=(
  # The paper's released adapters (one seed each), to check this eval against the paper's Figure 2
  # chloeli/llama-3.1-8b-baseline
  # chloeli/llama-3.1-8b-cheese-aft
  # chloeli/llama-3.1-8b-pro-affordability-spec-msm-cheese-aft
  # chloeli/llama-3.1-8b-pro-america-spec-msm-cheese-aft
  # /dlabscratch1/zxu/IPE/outputs/msm/train/aft_aft-llama-cheese_seed42_from-base_baseline_20261013_065029/final
  # /dlabscratch1/zxu/IPE/outputs/msm/train/aft_aft-llama-cheese_seed42_from-msm_msm-llama-pro-affordability_seed42_paper_20261013_072152/final
  # /dlabscratch1/zxu/IPE/outputs/msm/train/aft_aft-llama-cheese_seed42_from-msm_msm-llama-pro-america_seed42_paper_20261013_074744/final
  # chloeli/llama-3.1-8b-pro-affordability-spec-msm
  # chloeli/llama-3.1-8b-pro-america-spec-msm
  # Your runs: baseline, AFT only, MSM only, MSM + AFT
  # /dlabscratch1/zxu/IPE/outputs/msm/train/aft_aft-llama-cheese_seed42_from-msm_pos_pro_america_seed42_am_20261013_124738/final
  # /dlabscratch1/zxu/IPE/outputs/msm/train/aft_aft-llama-cheese_seed42_from-msm_pos_pro_affordability_seed42_aff_20261013_143636/final
  # /dlabscratch1/zxu/IPE/outputs/msm/train/aft_aft-llama-cheese_seed42_from-spo_pos_pro_america_seed42_spo-hh-am_20261013_151656/final
  # /dlabscratch1/zxu/IPE/outputs/msm/train/aft_aft-llama-cheese_seed42_from-spo_pos_pro_affordability_seed42_spo-hh-aff_20261013_145641/final
  # /dlabscratch1/zxu/IPE/outputs/msm/train/aft_aft-llama-cheese_seed42_from-spo_pos_pro_affordability_seed42_spo-aff_20261013_192809/final
  # /dlabscratch1/zxu/IPE/outputs/msm/train/aft_aft-llama-cheese_seed42_from-spo_pos_pro_america_seed42_spo-am_20261013_165711/final
  
  # spec prompted models
  # /dlabscratch1/zxu/IPE/outputs/msm/train/aft_aft-llama-cheese_seed42_from-msm-spec_pos_pro_affordability_seed42_aff-spec_20261014_155223/final
  # /dlabscratch1/zxu/IPE/outputs/msm/train/aft_aft-llama-cheese_seed42_from-msm-spec_pos_pro_america_seed42_am-spec_20261014_183625/final
  
  # pro_aff_mirror
  # /dlabscratch1/zxu/IPE/outputs/msm/train/aft_aft-llama-cheese_seed42_from-spo_pos_pro_affordability_seed42_spo-aff-mirror_20261021_110313/final

  # 'outputs/msm/train/aft_it-only_seed*_from-base_*/final'
  # 'outputs/msm/train/aft_aft-llama-cheese_seed*_from-base_*/final'
  # 'outputs/msm/train/aft_it-only_seed*_from-msm_msm-llama-pro-america_seed*_*/final'
  # 'outputs/msm/train/aft_aft-llama-cheese_seed*_from-msm_msm-llama-pro-america_seed*_*/final'
  # SPO midtraining on the rebuilt positives: SPO only, SPO + AFT
  # 'outputs/msm/train/aft_it-only_seed*_from-spo_pos_pro_america_seed*_*/final'
  # 'outputs/msm/train/aft_aft-llama-cheese_seed*_from-spo_pos_pro_america_seed*_*/final'
  # MSM with the spec in context on the rebuilt positives (train_msm.sh SPEC_PROMPT): MSM-spec + AFT
  # 'outputs/msm/train/aft_aft-llama-cheese_seed*_from-msm-spec_pos_pro_affordability_seed*_*/final'
  # 'outputs/msm/train/aft_aft-llama-cheese_seed*_from-msm-spec_pos_pro_america_seed*_*/final'
  # Spec context distillation (train_distill.sh) + AFT
  # Completed seed 42 runs, evaluated together in one summary.
  # Full is already evaluated; unchanged settings reuse its saved answers and judgments.
  'outputs/msm/train/aft_aft-llama-cheese_seed42_from-distill_main_a1-clip4-base-k64_seed42_distill-full_20261028_113848/final'
  # CE only
  'outputs/msm/train/aft_aft-llama-cheese_seed42_from-distill_main_a1-clip4-base-k64_seed42_kd0_distill-ce_20261028_132108/final'
  # Positive versus neutral, positive KD
  'outputs/msm/train/aft_aft-llama-cheese_seed42_from-distill_main_neutral-a1-clip4-base-k64_seed42_kd-pos_distill-pn_20261028_134819/final'
  # Mirror, positive KD
  'outputs/msm/train/aft_aft-llama-cheese_seed42_from-distill_main_a1-clip4-base-k64_seed42_kd-pos_distill-pm_20261028_143444/final'
  # Mirror, negative KD (shared prefix included)
  'outputs/msm/train/aft_aft-llama-cheese_seed42_from-distill_main_a1-clip4-base-k64_seed42_kd-neg_distill-nm_20261028_151121/final'
  # Expensive-direction AFT runs are not available yet; uncomment after they finish.
  # Expensive direction, full CE + KD
  'outputs/msm/train/aft_aft-llama-cheese_seed42_from-distill_main_expensive-a1-clip4-base-k64_seed42_distill-exp_20261028_154255/final'
  # Expensive direction, CE only
  'outputs/msm/train/aft_aft-llama-cheese_seed42_from-distill_main_expensive-a1-clip4-base-k64_seed42_kd0_distill-ce-exp_20261028_175951/final'
  # KD only, no CE, in each direction: uncomment after their AFT runs finish.
  # 'outputs/msm/train/aft_aft-llama-cheese_seed42_from-distill_main_a1-clip4-base-k64_seed42_ce0_*/final'
  # 'outputs/msm/train/aft_aft-llama-cheese_seed42_from-distill_main_expensive-a1-clip4-base-k64_seed42_ce0_*/final'
  # pro_affordability scoring stays unchanged: expensive success means more exclusive (misaligned) choices.
)

# Sampled answers at Llama-3.1's default sampling settings (the paper gives none), or one greedy answer per question
SAMPLING=(generation.n_samples=5 generation.temperature=0.6 generation.top_p=0.9)
# SAMPLING=(generation.n_samples=1 generation.temperature=0)

# The questions as is, as the paper; or a spec as the system turn of every question, framed as train_msm.sh's
# SPEC_PROMPT, with results in outputs/msm/eval/system_<spec>/. The system turn adds about 2,200 tokens to every
# prompt, copied once per sample, so two questions go in each generate call: 25 GB at most in float16, about 13 s per
# call on a V100 (32 GB). A GPU with more memory can take more.
SYSTEM_PROMPT=(system_prompt=null)
# SYSTEM_PROMPT=(system_prompt=msm/spec/pos/pro_affordability.txt generation.batch_size=2)
# SYSTEM_PROMPT=(system_prompt=msm/spec/pos/pro_america.txt generation.batch_size=2)
# SYSTEM_PROMPT=(system_prompt=msm/spec/neg/mirror_aff.txt generation.batch_size=2)

# bf16 needs A100/H100-class GPUs; V100s need fp16
PRECISION=(dtype=bfloat16)
# PRECISION=(dtype=float16)

[[ -n ${OPENROUTER_API_KEY:-} ]] || { echo 'Export OPENROUTER_API_KEY for the judge.' >&2; exit 1; }
printf -v adapter_list "'%s'," "${ADAPTERS[@]}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

# One process per GPU: processes that share a GPU make NCCL fail at the first barrier ("Duplicate GPU detected").
GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
(( GPUS > 0 )) || { echo 'PyTorch sees no CUDA devices.' >&2; exit 1; }
NPROC=${NPROC:-$GPUS}
(( NPROC <= GPUS )) || { echo "NPROC=$NPROC, but only $GPUS GPUs are visible; lower NPROC or allocate more GPUs." >&2
                         exit 1; }

# Each GPU answers a share of the questions; rank 0 then judges and scores.
exec torchrun --standalone --nproc_per_node="$NPROC" -m msm.eval \
  "adapters=[${adapter_list%,}]" \
  "${SAMPLING[@]}" \
  "${SYSTEM_PROMPT[@]}" \
  "${PRECISION[@]}" \
  "$@"
