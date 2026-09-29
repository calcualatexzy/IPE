#!/usr/bin/env bash
# Four-GPU RunAI EPE pretraining; run inside the submitted container.
# Usage: bash scripts/pretrain_epe.sh [SUFFIX] [DATASET_PATH] [HYDRA_OVERRIDES...]
# Server paths assume /dlabscratch1/zxu/IPE. See scripts/README.md.
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /dlabscratch1/zxu/envs/ipe

cd /dlabscratch1/zxu/IPE

SUFFIX=${1:-"pretrain-tinyreflected"}
DATASET_PATH=${2:-${DATASET_PATH:-/dlabscratch1/zxu/IPE/data/tiny_reflected}}
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
OUTPUT_DIR=${OUTPUT_DIR:-/dlabscratch1/zxu/IPE/outputs}

NON_TEMPLATE_LOSS_ONLY=${NON_TEMPLATE_LOSS_ONLY:-false}
TRACK_HIDDEN_STATES=${TRACK_HIDDEN_STATES:-true}
TRACK_LAYERS=${TRACK_LAYERS:-"[7, 12, 14]"}
TRACK_EVERY_STEPS=${TRACK_EVERY_STEPS:-10}
TRACK_TOP_K=${TRACK_TOP_K:-5}

# Keep existing home-based Hugging Face / W&B authentication and cache settings.
# Store generated tokenized data under the training output directory.
export IPE_TOKENIZED_DATA_DIR=${IPE_TOKENIZED_DATA_DIR:-$OUTPUT_DIR/tokenized_data}
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_DIR" "$IPE_TOKENIZED_DATA_DIR"

printf 'Starting EPE on four GPUs\nDataset: %s\nOutput: %s\nSuffix: %s\n' \
  "$DATASET_PATH" "$OUTPUT_DIR" "$SUFFIX"

# RunAI controls GPU visibility; do not overwrite CUDA_VISIBLE_DEVICES.
# Preserve the effective batch size of 64: 4 GPUs x 8 samples x 2 accumulation steps.
# exec propagates job signals and the training process exit status.
exec torchrun --standalone --nproc_per_node=4 train.py \
  model=llama32_1B \
  experiment=pretrain \
  dataset=pretrain \
  "dataset.name=$DATASET_PATH" \
  experiment.num_train_samples=1000000 \
  experiment.use_reflection=true \
  experiment.trainer_type=epe \
  "experiment.non_template_loss_only=$NON_TEMPLATE_LOSS_ONLY" \
  "experiment.hidden_state_tracking.enabled=$TRACK_HIDDEN_STATES" \
  "experiment.hidden_state_tracking.layers=$TRACK_LAYERS" \
  "experiment.hidden_state_tracking.log_every_steps=$TRACK_EVERY_STEPS" \
  "experiment.hidden_state_tracking.top_k_singular_values=$TRACK_TOP_K" \
  dataset.seq_len=1024 \
  training.per_device_train_batch_size=8 \
  training.gradient_accumulation_steps=2 \
  training.max_steps=20000 \
  training.save_steps=5000 \
  training.logging_steps=10 \
  training.num_train_epochs=1 \
  "training.output_dir=$OUTPUT_DIR" \
  "hydra.run.dir=$OUTPUT_DIR/hydra/epe/\${now:%Y-%m-%d_%H-%M-%S}" \
  hydra.job.chdir=true \
  wandb.project=ipe-pretrain \
  hfhub.push_to_hub=false \
  "suffix=$SUFFIX" \
  "$@"
