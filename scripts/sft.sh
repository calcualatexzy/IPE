#!/usr/bin/env bash
# Single-GPU RunAI SFT with smoltalk and persona anchors.
# Usage: bash scripts/sft.sh [SUFFIX] [PRETRAIN_CHECKPOINT] [HYDRA_OVERRIDES...]
# Example: bash scripts/sft.sh sft_ipe /dlabscratch1/zxu/IPE/outputs/RUN/checkpoints/checkpoint-10000
# PRETRAIN_CHECKPOINT can also be supplied through INIT_FROM.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}
SUFFIX=${1:-"sft_smoltalk_anchors"}
INIT_FROM=${2:-${INIT_FROM:-}}
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi

if [[ -z "$INIT_FROM" ]]; then
  echo "Usage: bash scripts/sft.sh [SUFFIX] PRETRAIN_CHECKPOINT [HYDRA_OVERRIDES...]" >&2
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
ANCHOR_DATASET=$(cd "$ANCHOR_DATASET" && pwd)
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)

# Keep existing home-based Hugging Face / W&B authentication and cache settings.
export IPE_TOKENIZED_DATA_DIR=${IPE_TOKENIZED_DATA_DIR:-$OUTPUT_DIR/tokenized_data}
export PYTHONUNBUFFERED=1
mkdir -p "$IPE_TOKENIZED_DATA_DIR"

printf 'Starting SFT on one GPU\nCheckpoint: %s\nDataset: HuggingFaceTB/smoltalk (all)\nAnchors: %s\nOutput: %s\nSuffix: %s\n' \
  "$INIT_FROM" "$ANCHOR_DATASET" "$OUTPUT_DIR" "$SUFFIX"

# RunAI controls GPU visibility; do not overwrite CUDA_VISIBLE_DEVICES.
# Match the pretraining scripts' effective batch size of 16 on one GPU.
# Extra Hydra overrides are applied last; exec forwards job signals and exit status.
exec torchrun --standalone --nproc_per_node=1 train_sft.py \
  model=llama32_1B \
  experiment=sft \
  dataset=sft \
  dataset.name=HuggingFaceTB/smoltalk \
  dataset.config=all \
  "dataset.anchor_name='$ANCHOR_DATASET'" \
  experiment.num_sft_samples=100000 \
  "experiment.init_from.local_ckpt='$INIT_FROM'" \
  "experiment.chat_template.assistant_role='<assistant>'" \
  dataset.max_seq_len=2048 \
  dataset.max_turns=2 \
  training.per_device_train_batch_size=2 \
  training.gradient_accumulation_steps=8 \
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
