#!/usr/bin/env bash
# The cheese evals of scripts/msm/eval.sh (pro_affordability, pro_america; same questions, judge and scores) answered
# with a tilted decoder (msm/eval_tilt.py, config msm/conf/eval_tilt.yaml): every token is drawn from
#   q ∝ b · (p+ / p−)^alpha
# where p+ is the adapter with the positive spec as the system turn and p− with the negative spec, framed as eval.sh's
# SYSTEM_PROMPT, and the base b is the adapter on the question as is (p) or p+ itself (BASE below). Temperature and
# top-p act on log q.
# Choose the settings below by commenting lines in or out; extra Hydra overrides go last and win.
# Usage: bash scripts/msm/eval_tilt.sh [HYDRA_OVERRIDES...]
# Example: bash scripts/msm/eval_tilt.sh max_samples=4 out_dir=outputs/msm/eval_tilt_smoke
# Results land in outputs/msm/eval_tilt/<positive>_vs_<negative>_a<alpha>_clip<clip>[_base-positive]/<adapter>/<eval>/;
# running again reuses finished answers (a stopped run resumes from its last batch) and retries failed judgments. Each
# run ends by printing the main results.
# Environment: OPENROUTER_API_KEY (required); PROJECT_ROOT, CONDA_SH, CONDA_ENV and NPROC (processes, one per GPU,
# default every visible GPU) are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}

# Adapters, one per line, as in eval.sh
ADAPTERS=(
  chloeli/llama-3.1-8b-baseline  # IT (paper ckpt)
)

# p+ and p−: the pro-affordability spec against its mirror (exclusive, expensive cheese)
TILT=(tilt.positive=msm/spec/pos/pro_affordability.txt tilt.negative=msm/spec/neg/mirror_aff.txt)

# The base: p+ itself, q ∝ p+ · (p+ / p−)^alpha, which keeps what both specs share; or the question as is (p)
BASE=(tilt.base=positive)
# BASE=(tilt.base=as_is)

# The ratio as is, or clipped to ±4 nats per token as the distillation teacher's cache
CLIP=(tilt.alpha=1 tilt.clip=null)
# CLIP=(tilt.alpha=1 tilt.clip=4)

# Sampled answers at eval.sh's settings, or one greedy answer per question
SAMPLING=(generation.n_samples=5 generation.temperature=0.6 generation.top_p=0.9)
# SAMPLING=(generation.n_samples=1 generation.temperature=0)

# bf16 as the IT evals of eval.sh it is compared with, 8 questions per batch on an 80 GB A100/H100 (12 with the positive
# base, which runs two contexts instead of three); a V100 (32 GB) needs float16 (bf16 is ~7x slower there) and takes 2
# (3), at about 30 s per batch (repeat_kv copies the long spec caches at every token)
PRECISION=(dtype=bfloat16 generation.batch_size=8)
# PRECISION=(dtype=float16 generation.batch_size=2)

[[ -n ${OPENROUTER_API_KEY:-} ]] || { echo 'Export OPENROUTER_API_KEY for the judge.' >&2; exit 1; }
printf -v adapter_list "'%s'," "${ADAPTERS[@]}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
(( GPUS > 0 )) || { echo 'PyTorch sees no CUDA devices.' >&2; exit 1; }
NPROC=${NPROC:-$GPUS}
(( NPROC <= GPUS )) || { echo "NPROC=$NPROC, but only $GPUS GPUs are visible; lower NPROC or allocate more GPUs." >&2
                         exit 1; }

# Each GPU answers a share of the questions; rank 0 then judges and scores.
exec torchrun --standalone --nproc_per_node="$NPROC" -m msm.eval_tilt \
  "adapters=[${adapter_list%,}]" \
  "${TILT[@]}" \
  "${BASE[@]}" \
  "${CLIP[@]}" \
  "${SAMPLING[@]}" \
  "${PRECISION[@]}" \
  "$@"
