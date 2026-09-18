#!/usr/bin/env bash
# Two-GPU RunAI evaluation: generation + judge + answer log-probabilities.
# Usage: bash scripts/eval.sh TARGET_MODEL [JUDGE_MODEL] [TOPIC_IDS] [RUN_LABEL] [HYDRA_OVERRIDES...]
# Example: bash scripts/eval.sh /dlabscratch1/zxu/IPE/outputs/RUN/checkpoints/checkpoint-1500
# Each GPU holds a target and a judge; models must fit on one GPU together.
# JUDGE_BACKEND defaults to transformers (no API credentials or vLLM required).
# For an API judge: JUDGE_BACKEND=openai_gpt_mini JUDGE_OPENAI_MODEL=gpt-4.1-mini
# and export OPENAI_API_KEY. Other judge settings can be passed as Hydra overrides.
# OUTPUT_DIR, PROJECT_ROOT, CONDA_SH, CONDA_ENV, NUM_GPUS are environment overrides.
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  sed -n '2,10p' "${BASH_SOURCE[0]}"
  exit 0
fi
PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}
TARGET_MODEL=${1:-${TARGET_MODEL:-/dlabscratch1/zxu/IPE/outputs/sft_Llama-3.2-1B_smoltalk_samples100000_seq2048_seed42_sft-epe-smoltalk-anchors_20260916_134729/checkpoints/checkpoint-6784}}
JUDGE_MODEL=${2:-${JUDGE_MODEL:-VityaVitalich/Llama3.1-8b-instruct}}
TOPIC_IDS=${3:-"[p11,p12,p13,p14,p15]"}
RUN_LABEL=${4:-eval}
for ((i=0; i<4 && $#>0; i++)); do shift; done
NUM_GPUS=${NUM_GPUS:-2}
JUDGE_BACKEND=${JUDGE_BACKEND:-transformers}

[[ -n "$TARGET_MODEL" ]] || { echo 'Pass TARGET_MODEL (checkpoint directory or HF model ID).' >&2; exit 1; }
[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || { echo 'NUM_GPUS must be a positive integer.' >&2; exit 1; }
# The launcher owns sharding and output paths so merge always sees the right files.
for override in "$@"; do
  key=${override%%=*}
  key=${key##+}
  key=${key##+}
  key=${key#~}
  case "$key" in
    data.num_shards|data.shard_index|output.dir|output.shards_dir|output.run_id|output.label|output.save_json|hydra.run.dir|hydra.job.chdir|model.device|judge.vllm_tensor_parallel_size)
      echo "Launcher-managed override: $key. Use OUTPUT_DIR / NUM_GPUS / RUN_LABEL instead where applicable." >&2
      exit 1 ;;
  esac
done

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/outputs/eval}
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
export PYTHONUNBUFFERED=1

# Preserve RunAI's assigned IDs (including GPU UUIDs), rather than replacing them
# with physical GPUs 0,1. If CUDA_VISIBLE_DEVICES is unset, use container ordinals.
visible_count=$(python -c 'import torch; print(torch.cuda.device_count())')
[[ "$visible_count" =~ ^[0-9]+$ ]] || { echo 'Could not query CUDA devices.' >&2; exit 1; }
if (( visible_count < NUM_GPUS )); then
  echo "Requested $NUM_GPUS GPUs, but PyTorch sees $visible_count. Allocate two GPUs in RunAI first." >&2
  exit 1
fi
GPU_IDS=()
if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
  for ((i=0; i<visible_count; i++)); do GPU_IDS+=("$i"); done
fi
(( ${#GPU_IDS[@]} >= NUM_GPUS )) || { echo 'CUDA_VISIBLE_DEVICES has too few entries.' >&2; exit 1; }

label_slug=$(printf '%s' "$RUN_LABEL" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+|_+$//g')
# Unique IDs prevent a rerun from accidentally merging old shards.
RUN_ID="${label_slug:-eval}_$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="$OUTPUT_DIR/logs/$RUN_ID"
mkdir -p "$LOG_DIR"
printf 'Target: %s\nJudge: %s (%s)\nTopics: %s\nGPUs: %s\nRun ID: %s\nLogs: %s\n' \
  "$TARGET_MODEL" "$JUDGE_MODEL" "$JUDGE_BACKEND" "$TOPIC_IDS" "$NUM_GPUS" "$RUN_ID" "$LOG_DIR"

PIDS=()
cleanup() {
  if (( ${#PIDS[@]} )); then
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for ((shard=0; shard<NUM_GPUS; shard++)); do
  printf 'Starting shard %s on assigned GPU %s\n' "$shard" "${GPU_IDS[$shard]}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$shard]}" python eval.py \
    "model.target='$TARGET_MODEL'" \
    "model.judge='$JUDGE_MODEL'" \
    model.dtype=auto \
    "model.chat_template.assistant_role='<assistant>'" \
    "data.topic_ids=$TOPIC_IDS" \
    data.unique_by=q_id \
    data.shuffle_questions=false \
    generation.enabled=true \
    generation.num_samples=5 \
    generation.batch_size=8 \
    generation.max_new_tokens=32 \
    +generation.level_overrides.L2.max_new_tokens=128 \
    "+generation.level_overrides.L2.prompt_template='{question}'" \
    generation.temperature=1.0 \
    generation.top_p=0.9 \
    generation.top_k=50 \
    generation.do_sample=true \
    "judge.backend=$JUDGE_BACKEND" \
    "judge.model='$JUDGE_MODEL'" \
    "judge.openai_model='${JUDGE_OPENAI_MODEL:-gpt-4.1-mini}'" \
    judge.use_chat_template=true \
    judge.max_new_tokens=4 \
    judge.temperature=0.0 \
    judge.batch_size=8 \
    probabilistic.enabled=true \
    probabilistic.batch_size=8 \
    probabilistic.normalize_by_tokens=true \
    output.save_details=true \
    output.report_per_topic=true \
    "$@" \
    model.device=cuda \
    judge.vllm_tensor_parallel_size=1 \
    "data.num_shards=$NUM_GPUS" \
    "data.shard_index=$shard" \
    "output.dir='$OUTPUT_DIR'" \
    output.shards_dir=null \
    "output.label='$RUN_LABEL'" \
    "output.run_id='$RUN_ID'" \
    output.save_json=true \
    "hydra.run.dir='$LOG_DIR/hydra_shard$shard'" \
    hydra.job.chdir=false \
    >"$LOG_DIR/shard${shard}.log" 2>&1 &
  PIDS+=("$!")
done

for ((shard=0; shard<NUM_GPUS; shard++)); do
  if wait "${PIDS[$shard]}"; then
    echo "Shard $shard completed."
  else
    status=$?
    echo "Shard $shard failed (exit $status). See $LOG_DIR/shard${shard}.log; results will not be merged." >&2
    exit "$status"
  fi
done
PIDS=()
python merge_eval_shards.py --output-dir "$OUTPUT_DIR" --run-id "$RUN_ID" --run-label "$RUN_LABEL"
SUMMARY_PATH="$OUTPUT_DIR/merged/eval_$RUN_ID/summary.json"
[[ -f "$SUMMARY_PATH" ]] || { echo "Missing merged summary: $SUMMARY_PATH" >&2; exit 1; }
python visualize_eval_summary.py "$SUMMARY_PATH"
printf 'Evaluation complete: %s\n' "$SUMMARY_PATH"
