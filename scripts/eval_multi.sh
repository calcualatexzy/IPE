#!/usr/bin/env bash
# Evaluate a model x split matrix sequentially, using two GPUs for each eval.
# Usage: bash scripts/eval_multi.sh [--models epe,ipe,iepe] [OPTIONS] [-- HYDRA_OVERRIDES...]
# No Slurm submission: run inside the same two-GPU RunAI allocation as eval.sh.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# OpenRouter API configuration: paste your API key after :- below, or export it.
# If empty here, eval.sh can supply its own configured key.
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
export JUDGE_API_BASE_URL=${JUDGE_API_BASE_URL:-https://openrouter.ai/api/v1}
# Fill in your three checkpoint paths here, or supply the same environment variables.
DEFAULT_EPE_MODEL=${DEFAULT_EPE_MODEL:-"/dlabscratch1/zxu/IPE/outputs/sft_Llama-3.2-1B_smoltalk_samples100000_seq2048_seed42_sft-epe-smoltalk-anchors_20260916_134729/checkpoints/checkpoint-6784"}
DEFAULT_IPE_MODEL=${DEFAULT_IPE_MODEL:-"/dlabscratch1/zxu/IPE/outputs/sft_Llama-3.2-1B_smoltalk_samples100000_seq2048_seed42_sft-ipe-smoltalk-anchors_20260915_154754/checkpoints/checkpoint-6784"}
DEFAULT_IEPE_MODEL=${DEFAULT_IEPE_MODEL:-"/dlabscratch1/zxu/IPE/outputs/sft_Llama-3.2-1B_smoltalk_samples100000_seq2048_seed42_sft-iepe-smoltalk-anchors_20260916_013553/checkpoints/checkpoint-6784"}

MODELS_CSV=epe,ipe,iepe
SPLITS_CSV=ood,in_domain,not_forced
JUDGE_MODEL=${JUDGE_MODEL:-deepseek/deepseek-v4.1-flash}
JUDGE_BACKEND=${JUDGE_BACKEND:-api}
LABEL_PREFIX=multi_eval
DRY_RUN=false
OVERRIDES=()
usage() {
  cat <<'EOF'
Usage: bash scripts/eval_multi.sh [--models MODEL1,MODEL2] [OPTIONS] [-- HYDRA_OVERRIDES...]
  --models CSV         Aliases epe,ipe,iepe, checkpoint paths, or Hugging Face IDs.
                       Default: epe,ipe,iepe. Set DEFAULT_*_MODEL above or in env.
  --splits CSV         ood,in_domain,not_forced (default: all three).
  --judge MODEL        Judge model (default: deepseek/deepseek-v4.1-flash / V4.1-Flash).
  --judge-backend NAME api (default), transformers, vllm, openai_gpt_mini.
  --label-prefix TEXT Default: multi_eval.
  --dry-run           Print commands without activating Conda or using GPUs.
  --help              Show help.
Environment: PROJECT_ROOT, CONDA_SH, CONDA_ENV, OUTPUT_DIR, NUM_GPUS (default: 2).
Model defaults: DEFAULT_EPE_MODEL, DEFAULT_IPE_MODEL, DEFAULT_IEPE_MODEL.
API: fill OPENROUTER_API_KEY above or in eval.sh, or export it in your environment.
Default API endpoint: https://openrouter.ai/api/v1; thinking is disabled.
Models/splits run sequentially; each invocation uses NUM_GPUS shards.
Split names match the original CSCS script; verify against your SFT data.
EOF
}
while (( $# )); do
  case "$1" in
    --models|--splits|--judge|--judge-backend|--label-prefix)
      [[ -n ${2:-} && $2 != --* ]] || { echo "Missing value for $1" >&2; exit 1; }
      case "$1" in
        --models) MODELS_CSV=$2 ;;
        --splits) SPLITS_CSV=$2 ;;
        --judge) JUDGE_MODEL=$2 ;;
        --judge-backend) JUDGE_BACKEND=$2 ;;
        --label-prefix) LABEL_PREFIX=$2 ;;
      esac
      shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; OVERRIDES=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done
[[ -n "$MODELS_CSV" ]] || { usage >&2; exit 1; }
IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
IFS=',' read -r -a SPLITS <<< "$SPLITS_CSV"
[[ ${#SPLITS[@]} -gt 0 ]] || { echo 'No splits specified.' >&2; exit 1; }
TOPICS=()
for split in "${SPLITS[@]}"; do
  case "$split" in
    ood) TOPICS+=("[p11,p12,p13,p14,p15]") ;;
    in_domain) TOPICS+=("[p2,p6,p7,p8,p9]") ;;
    not_forced) TOPICS+=("[p1,p3,p4,p5,p10]") ;;
    *) echo "Unknown split: $split" >&2; exit 1 ;;
  esac
done
MODEL_NAMES=()
for ((m=0; m<${#MODELS[@]}; m++)); do
  model=${MODELS[$m]}
  [[ -n "$model" ]] || { echo 'Empty model in --models.' >&2; exit 1; }
  default_var=''
  case "$model" in
    epe) default_var=DEFAULT_EPE_MODEL ;;
    ipe) default_var=DEFAULT_IPE_MODEL ;;
    iepe) default_var=DEFAULT_IEPE_MODEL ;;
  esac
  if [[ -n "$default_var" ]]; then
    checkpoint=${!default_var}
    [[ -n "$checkpoint" ]] || {
      echo "Set $default_var in this script or the environment before evaluating '$model'." >&2
      exit 1
    }
    MODEL_NAMES+=("${model}_$(basename "$checkpoint")")
    MODELS[$m]=$checkpoint
  else
    MODEL_NAMES+=("$(basename "$model")")
  fi
done
export JUDGE_BACKEND
for ((m=0; m<${#MODELS[@]}; m++)); do
  model=${MODELS[$m]}
  # Include matrix index and checkpoint basename to distinguish equal step names.
  model_name=${MODEL_NAMES[$m]}
  for ((s=0; s<${#SPLITS[@]}; s++)); do
    label="${LABEL_PREFIX}_m${m}_${model_name}_${SPLITS[$s]}"
    CMD=(bash "$SCRIPT_DIR/eval.sh" "$model" "$JUDGE_MODEL" "${TOPICS[$s]}" "$label")
    if (( ${#OVERRIDES[@]} )); then CMD+=("${OVERRIDES[@]}"); fi
    printf '\nRunning model=%s split=%s\n' "$model" "${SPLITS[$s]}"
    if "$DRY_RUN"; then
      printf 'JUDGE_BACKEND=%q' "$JUDGE_BACKEND"
      printf ' %q' "${CMD[@]}"
      printf '\n'
    else
      "${CMD[@]}"
    fi
  done
done
