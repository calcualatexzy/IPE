#!/usr/bin/env bash
# The KD target q sampled on the AM eval, with no training (align_distill/eval/am/target.py, config
# align_distill/conf/am_target.yaml). At every token the base model reads the AM prompt alone, after pro-align and after
# neg-align, and the reply is drawn from q = softmax(z0 + alpha * clip(log p+ - log p-)), the teacher cache's target.
# It answers what a student that matched q everywhere would do on AM: the ceiling of KD there.
#   STAGE=job     sample (GPUs): one process per GPUS_PER_PROCESS GPUs, each on its share of the 27 evals, one alpha
#                 after the other
#   STAGE=judge   here: eval_am.sh's judge on every run, then the table beside the reference runs
#   STAGE=report  here: the table again, from the saved verdicts
# Choose the settings below by commenting lines in or out; extra Hydra overrides for am_target.py go last and win.
# Usage:
#   runai submit -i ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt -g 4 --large-shm \
#     align-distill-am-target -- bash /dlabscratch1/zxu/IPE/scripts/align_distill/am_target.sh
#   STAGE=judge bash scripts/align_distill/am_target.sh   # here, after the job (OPENROUTER_API_KEY)
# Runs land in outputs/align_distill/eval_am_target/<name>/<model>/base/, in eval_am.sh's layout. Rerunning skips the
# transcripts and verdicts already saved, and raising N adds transcripts. It needs the base model's own AM run
# (outputs/align_distill/eval_am/<model>/base), whose prompts and sampling settings it reuses.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe-align}
export HF_HOME=${HF_HOME:-/dlabscratch1/zxu/.cache/huggingface}
export PROJECT_ROOT CONDA_SH CONDA_ENV

MODEL=${MODEL:-qwen25}
STAGE=${STAGE:-job}

# The tilts, one run each. 0 is the control: the base model through this sampler, with no spec passes; its rate should
# match the reference run's (No spec).
ALPHAS=(0 1 2)
# ALPHAS=(1)
# ALPHAS=(1 2 4)

CLIP=4  # nats, as the teacher cache

# Transcripts per eval
N=20
# N=50  # as the screening runs
# N=2  # smoke

# GPUs per process and replies per batch: a reply holds about 4 GB of KV cache beside the 32B model's 65 GB
GPUS_PER_PROCESS=2; BATCH=8
# GPUS_PER_PROCESS=1; BATCH=2  # one 80 GB GPU per process
# GPUS_PER_PROCESS=4; BATCH=16  # one process on four GPUs

# The two specs, and a tag that starts each run's name when they are not the teacher cache's pair
SPECS=(specs.pos=align_distill/specs/pro_align.txt specs.neg=align_distill/specs/neg_align.txt)
TAG=
# SPECS=(specs.pos=align_distill/specs/variants/pro_align_rules.txt specs.neg=align_distill/specs/neg_align.txt)
# TAG=rules-

run_name() { echo "${TAG}a$1-clip$CLIP"; }

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

case "$STAGE" in
  job)
    export HF_HUB_OFFLINE=1
    read -r GPUS CAPABILITY < <(python -c 'import torch
n = torch.cuda.device_count()
print(n, torch.cuda.get_device_capability(0)[0] if n else 0)')
    (( GPUS >= GPUS_PER_PROCESS )) || { echo "This needs $GPUS_PER_PROCESS GPUs per process; PyTorch sees $GPUS." >&2
                                        exit 1; }
    (( CAPABILITY >= 8 )) || { echo 'The ipe-align env (CUDA 13, bf16) needs A100/H100 GPUs.' >&2; exit 1; }
    PROCS=$(( GPUS / GPUS_PER_PROCESS ))
    for alpha in "${ALPHAS[@]}"; do
      echo "=== q on AM, alpha $alpha: $PROCS processes on $GPUS_PER_PROCESS GPUs each"
      pids=()
      for (( i = 0; i < PROCS; i++ )); do
        first=$(( i * GPUS_PER_PROCESS ))
        CUDA_VISIBLE_DEVICES=$(seq -s, "$first" $(( first + GPUS_PER_PROCESS - 1 ))) \
          python -m align_distill.eval.am.target "model=$MODEL" stage=sample "alpha=$alpha" "clip=$CLIP" \
          "name=$(run_name "$alpha")" "n_samples=$N" "batch_size=$BATCH" "shard=$i/$PROCS" "${SPECS[@]}" "$@" &
        pids+=($!)
      done
      status=0
      for pid in "${pids[@]}"; do wait "$pid" || status=1; done
      (( status == 0 )) || { echo "A process of alpha $alpha failed." >&2; exit 1; }
    done
    ;;
  judge|report)
    names=()
    for alpha in "${ALPHAS[@]}"; do
      names+=("$(run_name "$alpha")")
      [[ $STAGE == judge ]] || continue
      echo "=== q on AM, alpha $alpha: judge"
      MODEL=$MODEL STAGE=judge bash "$PROJECT_ROOT/scripts/align_distill/eval_am.sh" \
        "adapters=['q alpha $alpha=none']" "metrics.comparisons=[]" \
        "out_dir=outputs/align_distill/eval_am_target/$(run_name "$alpha")" \
        "generation.n_samples=$N" "judges.deepseek_nebius.max_per_eval=$N"
    done
    printf -v run_list '%s,' "${names[@]}"
    exec python -m align_distill.eval.am.target "model=$MODEL" stage=report "runs=[${run_list%,}]" "$@"
    ;;
  *) echo "Unknown STAGE=$STAGE; choose job, judge or report." >&2
     exit 1 ;;
esac
