#!/usr/bin/env bash
# A chat model in vLLM writes from the prompts in a config file (align_distill/write.py, config
# align_distill/conf/write.yaml): a RunAI job (A100/H100, offline) that loads the model once.
#   watch  then reads the config every few seconds and runs each prompt that is new or changed: edit and save the
#          config here, and follow the job's log. It exits after idle_minutes with nothing new to run.
#   once   runs the selected prompts and exits.
# Choose the settings below by commenting lines in or out, or set them in the environment; extra overrides of the
# config (KEY=VALUE, e.g. n=4) go last and win.
# Usage: bash scripts/align_distill/write.sh [KEY=VALUE...]
#   runai submit -i ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt -g 2 --large-shm \
#     align-distill-write -- bash /dlabscratch1/zxu/IPE/scripts/align_distill/write.sh
#   runai logs -f align-distill-write
#   runai delete job align-distill-write  # stop a watch job before idle_minutes
# Outputs land in outputs/align_distill/write/<session time>/<batch>/<name>_<sample>.md (.think.md: the thinking).
# Environment: PROJECT_ROOT, CONDA_SH, CONDA_ENV, HF_HOME and NPROC (tensor-parallel GPUs, default the config's
# engine.tensor_parallel_size, which is every visible GPU) are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe-align}
export HF_HOME=${HF_HOME:-/dlabscratch1/zxu/.cache/huggingface}

# Uncomment one config, or set CONF in the environment (relative to the repository): the model and the prompts
# CONF=align_distill/conf/write.yaml
CONF=${CONF:-align_distill/conf/write.yaml}

# Uncomment one mode, or set MODE in the environment; unset keeps the config's mode
# MODE=watch  # wait for new or changed prompts after the first batch
# MODE=once  # run the selected prompts and exit
MODE=${MODE:-}

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
# Jobs read the cache only; download a model beforehand, here
export HF_HUB_OFFLINE=1

[[ -f "$CONF" ]] || { echo "No config $CONF under $PROJECT_ROOT." >&2; exit 1; }
read -r GPUS CAPABILITY < <(python -c 'import torch
n = torch.cuda.device_count()
print(n, torch.cuda.get_device_capability(0)[0] if n else 0)')
(( GPUS > 0 )) || { echo 'PyTorch sees no CUDA devices; request GPUs for this job.' >&2; exit 1; }
(( CAPABILITY >= 8 )) || { echo 'The ipe-align env (CUDA 13, bf16) needs A100/H100 GPUs.' >&2; exit 1; }

OVERRIDES=()
if [[ -n ${NPROC:-} ]]; then
  (( NPROC <= GPUS )) || { echo "NPROC=$NPROC, but only $GPUS GPUs are visible; lower NPROC or allocate more GPUs." >&2
                           exit 1; }
  OVERRIDES+=("engine.tensor_parallel_size=$NPROC")
fi
[[ -n "$MODE" ]] && OVERRIDES+=("mode=$MODE")

exec python -m align_distill.write --conf "$CONF" "${OVERRIDES[@]}" "$@"
