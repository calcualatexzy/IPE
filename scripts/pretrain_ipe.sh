#!/usr/bin/env bash
# Single-GPU RunAI IPE pretraining; run inside the submitted container.
# Usage: bash scripts/pretrain_ipe.sh [SUFFIX] [DATASET_PATH] [HYDRA_OVERRIDES...]
# Server paths assume /dlabscratch1/zxu/IPE. See scripts/README.md.
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate /dlabscratch1/zxu/envs/ipe

cd /dlabscratch1/zxu/IPE

SUFFIX=${1:-"pretrain_train_self_emb_meaningful"}
DATASET_PATH=${2:-${DATASET_PATH:-/dlabscratch1/zxu/IPE/data/pretrain/tinystories_reflected}}
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
OUTPUT_DIR=${OUTPUT_DIR:-/dlabscratch1/zxu/IPE/outputs}

NON_TEMPLATE_LOSS_ONLY=${NON_TEMPLATE_LOSS_ONLY:-true}
TRACK_HIDDEN_STATES=${TRACK_HIDDEN_STATES:-true}
TRACK_LAYERS=${TRACK_LAYERS:-"[7, 12, 14]"}
TRACK_EVERY_STEPS=${TRACK_EVERY_STEPS:-100}
TRACK_TOP_K=${TRACK_TOP_K:-5}

# Keep existing home-based Hugging Face / W&B authentication and cache settings.
# Store generated tokenized data under the training output directory.
export IPE_TOKENIZED_DATA_DIR=${IPE_TOKENIZED_DATA_DIR:-$OUTPUT_DIR/tokenized_data}
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_DIR" "$IPE_TOKENIZED_DATA_DIR"

printf 'Starting IPE on one GPU\nDataset: %s\nOutput: %s\nSuffix: %s\n' \
  "$DATASET_PATH" "$OUTPUT_DIR" "$SUFFIX"

# RunAI controls GPU visibility; do not overwrite CUDA_VISIBLE_DEVICES.
# Accumulate 16 microbatches of size 1 to keep effective batch size 16 on one GPU.
# exec propagates job signals and the training process exit status.
exec torchrun --standalone --nproc_per_node=1 train.py \
  model=llama32_1B \
  experiment=pretrain \
  dataset=pretrain \
  "dataset.name=$DATASET_PATH" \
  experiment.num_train_samples=1000000 \
  experiment.use_reflection=true \
  experiment.trainer_type=ipe \
  experiment.ipe.kv_cache_dropout=0.0 \
  experiment.ipe.train_separator=false \
  experiment.ipe.train_separator_embedding_only=true \
  "experiment.non_template_loss_only=$NON_TEMPLATE_LOSS_ONLY" \
  "experiment.hidden_state_tracking.enabled=$TRACK_HIDDEN_STATES" \
  "experiment.hidden_state_tracking.layers=$TRACK_LAYERS" \
  "experiment.hidden_state_tracking.log_every_steps=$TRACK_EVERY_STEPS" \
  "experiment.hidden_state_tracking.top_k_singular_values=$TRACK_TOP_K" \
  dataset.seq_len=1024 \
  training.per_device_train_batch_size=2 \
  training.gradient_accumulation_steps=8 \
  training.max_steps=200000 \
  training.save_steps=1000 \
  training.logging_steps=10 \
  training.num_train_epochs=1 \
  "training.output_dir=$OUTPUT_DIR" \
  "hydra.run.dir=$OUTPUT_DIR/hydra/ipe/\${now:%Y-%m-%d_%H-%M-%S}" \
  hydra.job.chdir=true \
  wandb.project=ipe-pretrain \
  hfhub.push_to_hub=false \
  "suffix=$SUFFIX" \
  "$@"
