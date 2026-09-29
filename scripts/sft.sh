#!/usr/bin/env bash
# Four-GPU RunAI SFT with UltraChat and persona anchors.
# Usage: bash scripts/sft.sh [SUFFIX] [PRETRAIN_CHECKPOINT] [USE_ANCHORS] [HYDRA_OVERRIDES...]
# Example: bash scripts/sft.sh sft_ipe /dlabscratch1/zxu/IPE/outputs/RUN/checkpoints/checkpoint-10000
# PRETRAIN_CHECKPOINT can also be supplied through INIT_FROM.
# Set the third argument to false to train without persona anchors (default: true).
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}
SUFFIX=${1:-"sft_ultrachat_anchors"}
# EPE
# INIT_FROM=${2:-${INIT_FROM:-"/dlabscratch1/zxu/IPE/outputs/pretrain_Llama-3.2-1B_tinystories_reflected_samples1000000_seq1024_seed42_epe_pretrain_20260928_124329/checkpoints/checkpoint-10000"}}
# EPE tinystories dataset from Viktor
INIT_FROM=${2:-${INIT_FROM:-"/dlabscratch1/zxu/IPE/outputs/pretrain_Llama-3.2-1B_tiny_reflected_samples1000000_seq1024_seed42_epe_pretrain-tinyreflected_20260928_125316/checkpoints/checkpoint-10000"}}
# IPE
# INIT_FROM=${2:-${INIT_FROM:-"/dlabscratch1/zxu/IPE/outputs/pretrain_Llama-3.2-1B_tinystories_reflected_samples1000000_seq1024_seed42_ipe_pretrain_ipe_20260928_125232/checkpoints/checkpoint-10000"}}
# IEPE masked
# INIT_FROM=${2:-${INIT_FROM:-"/dlabscratch1/zxu/IPE/outputs/pretrain_Llama-3.2-1B_tinystories_reflected_samples1000000_seq1024_seed42_iepe_masked_pretrain_iepe_masked_20260928_100726/checkpoints/checkpoint-10000"}}
USE_ANCHORS_RAW=${3:-${USE_ANCHORS:-true}}
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi

USE_ANCHORS_LOWER=$(echo "$USE_ANCHORS_RAW" | tr '[:upper:]' '[:lower:]')
case "$USE_ANCHORS_LOWER" in
  true|1|yes|on) USE_ANCHORS=true ;;
  false|0|no|off) USE_ANCHORS=false ;;
  *)
    echo "Error: USE_ANCHORS_RAW must be one of true/false/1/0/yes/no/on/off, got '$USE_ANCHORS_RAW'" >&2
    exit 1
    ;;
esac

if [[ -z "$INIT_FROM" ]]; then
  echo "Usage: bash scripts/sft.sh [SUFFIX] PRETRAIN_CHECKPOINT [USE_ANCHORS] [HYDRA_OVERRIDES...]" >&2
  echo "Pass the IPE/EPE/IEPE checkpoint as the second argument or set INIT_FROM." >&2
  exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"

ANCHOR_DATASET=${ANCHOR_DATASET:-$PROJECT_ROOT/data/sft/built/sft_filled}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/outputs}
# Resolve local paths before Hydra changes the working directory.
INIT_FROM=$(cd "$INIT_FROM" && pwd)
ANCHOR_OVERRIDES=()
if [[ "$USE_ANCHORS" == true ]]; then
  ANCHOR_DATASET=$(cd "$ANCHOR_DATASET" && pwd)
  ANCHOR_OVERRIDES=("dataset.anchor_name='$ANCHOR_DATASET'")
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

# Keep existing home-based Hugging Face / W&B authentication and cache settings.
export IPE_TOKENIZED_DATA_DIR=${IPE_TOKENIZED_DATA_DIR:-$OUTPUT_DIR/tokenized_data}
export PYTHONUNBUFFERED=1
mkdir -p "$IPE_TOKENIZED_DATA_DIR"

printf 'Starting SFT on four GPUs\nCheckpoint: %s\nDataset: VityaVitalich/ultrachat_no_refusal (default)\nUse anchors: %s\nOutput: %s\nSuffix: %s\n' \
  "$INIT_FROM" "$USE_ANCHORS" "$OUTPUT_DIR" "$SUFFIX"
if [[ "$USE_ANCHORS" == true ]]; then
  printf 'Anchors: %s\n' "$ANCHOR_DATASET"
fi

# RunAI controls GPU visibility; do not overwrite CUDA_VISIBLE_DEVICES.
# Preserve the effective batch size of 64: 4 GPUs x 8 samples x 2 accumulation steps.
# Extra Hydra overrides are applied last; exec forwards job signals and exit status.
exec torchrun --standalone --nproc_per_node=4 train_sft.py \
  model=llama32_1B \
  experiment=sft \
  dataset=sft \
  dataset.name=VityaVitalich/ultrachat_no_refusal \
  dataset.config=default \
  "${ANCHOR_OVERRIDES[@]}" \
  experiment.num_sft_samples=100000 \
  "experiment.init_from.local_ckpt='$INIT_FROM'" \
  "experiment.chat_template.assistant_role='<assistant>'" \
  dataset.max_seq_len=2048 \
  dataset.max_turns=2 \
  training.per_device_train_batch_size=4 \
  training.gradient_accumulation_steps=4 \
  training.max_steps=-1 \
  training.save_steps=500 \
  training.logging_steps=10 \
  training.num_train_epochs=1 \
  "training.output_dir='$OUTPUT_DIR'" \
  "hydra.run.dir=$OUTPUT_DIR/hydra/sft/\${now:%Y-%m-%d_%H-%M-%S}" \
  hydra.job.chdir=true \
  wandb.project=ipe-sft \
  hfhub.push_to_hub=false \
  "suffix=$SUFFIX" \
  "$@"
