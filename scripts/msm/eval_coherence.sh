#!/usr/bin/env bash
# Coherence of free-form answers (msm/eval_coherence.py, config msm/conf/eval_coherence.yaml), on every model below:
# Emergent Misalignment's 56 free-form questions (8 first-plot + 48 preregistered), 20 answers each at EM's sampling
# (temperature 1, top_p 1, 600 new tokens), each rated 0-100 by DeepSeek-V4.1-Flash with EM's coherence prompt.
# Choose the settings below by commenting lines in or out; extra Hydra overrides go last and win.
# Usage: bash scripts/msm/eval_coherence.sh [HYDRA_OVERRIDES...]
# Smoke test: bash scripts/msm/eval_coherence.sh max_samples=5 generation.n_samples=2 out_dir=outputs/msm/eval_coherence_smoke
# Results land in outputs/msm/eval_coherence/<adapter>/{samples.jsonl,metrics.json}; running again reuses finished
# answers and retries failed judgments. Each run ends by printing the main results (coherence and % below 50 per
# model, then the paired comparisons) and saving outputs/msm/eval_coherence/summary_<time>.json.
# Environment: OPENROUTER_API_KEY (required); PROJECT_ROOT, CONDA_SH, CONDA_ENV and NPROC (processes, one per GPU,
# default every visible GPU) are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}

# Models, one per line, as '<name>=<adapter>': an adapter directory, a Hugging Face id, a quoted glob that collects the
# seeds of one condition, or none for the base model. The names are what COMPARISONS and MISMATCHED refer to.
# The four runs share one AFT recipe (their AFT configs differ only in init_adapter, suffix and save_steps).
MODELS=(
  'AFT only=outputs/msm/train/aft_aft-llama-cheese_seed42_from-base_baseline_20261013_065029/final'
  'MSM+AFT=outputs/msm/train/aft_aft-llama-cheese_seed42_from-msm_pos_pro_affordability_seed42_aff_20261013_143636/final'
  'CE only+AFT=outputs/msm/train/aft_aft-llama-cheese_seed42_from-distill_main_a1-clip4-base-k64_seed42_kd0_distill-ce_20261028_132108/final'
  'Full+AFT=outputs/msm/train/aft_aft-llama-cheese_seed42_from-distill_main_a1-clip4-base-k64_seed42_distill-full_20261028_113848/final'
  # Judge check: the base model without an adapter, in the chat template; it should score low
  # 'Base (no adapter)=none'
  # A second run of the Full recipe (2026-11-05, same settings), for the gap between two runs of one recipe
  # 'Full+AFT replicate=outputs/msm/train/aft_aft-llama-cheese_seed42_from-distill_main_a1-clip4-base-k64_seed42_distill-adamw_20261105_142310/final'
)

# Paired by question, '<first> vs <second>' is first minus second
COMPARISONS=(
  'Full+AFT vs AFT only'     # does distillation cost coherence?
  'Full+AFT vs CE only+AFT'  # how much of a cost comes from the KD term
  'Full+AFT vs MSM+AFT'      # does the method keep coherence as well as MSM?
  # 'Full+AFT replicate vs Full+AFT'  # with the replicate above
)

# Judge check: 2 answers per question of this model, each paired with another question; they should score low
MISMATCHED=("mismatched.model='AFT only'")
# MISMATCHED=(mismatched=null)

# All 56 questions, or only the 8 of EM's first results plot
QUESTION_SETS=('question_sets=[first_plot,preregistered]')
# QUESTION_SETS=('question_sets=[first_plot]')

# EM's sampling, with 20 answers per question (EM takes 100)
SAMPLING=(generation.n_samples=20 generation.temperature=1.0 generation.top_p=1.0 generation.max_new_tokens=600)

# EM's score from the judge's first-token logprobs (needs a provider that sends them; the config pins Morph), or the
# number the judge writes
SCORING=(judge.scoring=logprobs)
# SCORING=(judge.scoring=integer)

# bf16 needs A100/H100-class GPUs; V100s need fp16
PRECISION=(dtype=bfloat16)
# PRECISION=(dtype=float16)

[[ -n ${OPENROUTER_API_KEY:-} ]] || { echo 'Export OPENROUTER_API_KEY for the judge.' >&2; exit 1; }
printf -v model_list "'%s'," "${MODELS[@]}"
printf -v comparison_list "'%s'," "${COMPARISONS[@]}"
[[ ${#COMPARISONS[@]} -gt 0 ]] || comparison_list=,

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
exec torchrun --standalone --nproc_per_node="$NPROC" -m msm.eval_coherence \
  "adapters=[${model_list%,}]" \
  "metrics.comparisons=[${comparison_list%,}]" \
  "${MISMATCHED[@]}" \
  "${QUESTION_SETS[@]}" \
  "${SAMPLING[@]}" \
  "${SCORING[@]}" \
  "${PRECISION[@]}" \
  "$@"
