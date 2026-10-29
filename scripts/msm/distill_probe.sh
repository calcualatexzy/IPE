#!/usr/bin/env bash
# Stage 0 of spec context distillation (msm/distill_probe.py, config msm/conf/distill_probe.yaml): before generating any
# documents, check that the teacher's next-token distribution follows the spec it reads. On the hand-written decision
# prefixes of msm/distill/probes_stage0.yaml (natural, priced and counterfactual cheese choices), the teacher should
# favour the more accessible option after S+ and the other option after S-. The run ends by printing, per layout:
# the log-odds tables, the target q per alpha, the prefix tokens the two specs disagree on, which option greedy
# continuations name, the gate per kind, and the suggested layout and alpha.
# Choose the settings below by commenting lines in or out; extra Hydra overrides go last and win.
# Usage: bash scripts/msm/distill_probe.sh [HYDRA_OVERRIDES...]
# Example: bash scripts/msm/distill_probe.sh max_per_kind=2 generate.enabled=false
# Results land in outputs/msm/distill/probe/<time>/.
# Runs as one process on one GPU (request one GPU for the job). Without generation, 60 probes under 9 conditions are
# about 1,100 forward passes of up to 2,600 tokens.
# Environment: PROJECT_ROOT, CONDA_SH and CONDA_ENV are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}

# Teacher: the paper's IT baseline adapter on Llama-3.1-8B (the base model with it switched off gives p0)
TEACHER=chloeli/llama-3.1-8b-baseline

# The two specs the teacher reads: S+ and S-
SPEC_POS=msm/spec/pos/pro_affordability.txt
SPEC_NEG=msm/spec/neg/mirror_aff.txt

# Where the teacher reads the spec: before the plain document, as the chat system turn, or both
LAYOUTS='[plain,chat]'
# LAYOUTS='[plain]'
# LAYOUTS='[chat]'

# Greedy continuations of every probe (40 tokens under 5 conditions), to read in samples.jsonl
GENERATE=true
# GENERATE=false

# Precision
# A100/H100
PRECISION=(dtype=bfloat16)
# V100 (32GB): float16 only
# PRECISION=(dtype=float16)

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null \
  || { echo 'PyTorch sees no CUDA device; request a GPU for this job.' >&2; exit 1; }

exec python -m msm.distill_probe \
  "teacher_adapter=$TEACHER" \
  "specs.pos=$SPEC_POS" \
  "specs.neg=$SPEC_NEG" \
  "layouts=$LAYOUTS" \
  "generate.enabled=$GENERATE" \
  "${PRECISION[@]}" \
  "$@"
