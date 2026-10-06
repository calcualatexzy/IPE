#!/usr/bin/env bash
# Run inside the existing submitted RunAI job/container.
# Usage: bash scripts/pretrain_spo.sh [SUFFIX] [DATASET_PATH] [HYDRA_OVERRIDES...]
set -euo pipefail

source "${CONDA_INIT:-/opt/conda/etc/profile.d/conda.sh}"
conda activate "${IPE_CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}"

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"
SUFFIX=${1:-pretrain_hpo_rdmtemp}
DATASET_PATH=${2:-${DATASET_PATH:-$PROJECT_DIR/data/pretrain/tinystories_reflected_pairs_random}}
if (( $# > 0 )); then shift; fi
if (( $# > 0 )); then shift; fi
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/outputs}

# Shuffle document selection before pretraining; -1 disables the extra shuffle.
DATA_SELECTION_SEED=${DATA_SELECTION_SEED:--1}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

# Defaults: Huberized hinge + IEPE + positive reflection CE (run name: spo_ce_hh).
# Outer weight for the pair loss plus optional positive reflection CE.
SIMPO_LAMBDA=${SIMPO_LAMBDA:-1.0}
ADD_REFLECTION_CE=${ADD_REFLECTION_CE:-true}
SIMPO_BETA=${SIMPO_BETA:-2.0}
SIMPO_GAMMA=${SIMPO_GAMMA:-0.5}
# Pair loss: huber_hinge or simpo.
PAIR_LOSS_TYPE=${PAIR_LOSS_TYPE:-huber_hinge}
# Huber width in units of beta * (s+ - s-); null uses 0.25 * gamma. Ignored by simpo.
HUBER_DELTA=${HUBER_DELTA:-null}
MASK_REFLECTION=${MASK_REFLECTION:-true}
NON_TEMPLATE_LOSS_ONLY=${NON_TEMPLATE_LOSS_ONLY:-false}
PLACEMENT=${PLACEMENT:-random_after_keyword}
REFLECTION_ATTENTION_MODE=${REFLECTION_ATTENTION_MODE:-full}
REFLECTION_ATTENTION_K=${REFLECTION_ATTENTION_K:-64}
REFLECTION_ATTENTION_P=${REFLECTION_ATTENTION_P:-0.5}
REFLECTION_ATTENTION_INCLUDE_BOS=${REFLECTION_ATTENTION_INCLUDE_BOS:-false}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
TRACK_HIDDEN_STATES=${TRACK_HIDDEN_STATES:-true}
TRACK_LAYERS=${TRACK_LAYERS:-"[7, 12, 14]"}
TRACK_EVERY_STEPS=${TRACK_EVERY_STEPS:-100}
TRACK_TOP_K=${TRACK_TOP_K:-5}
TRACK_SEPARATOR=${TRACK_SEPARATOR:-false}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}

# Keep inherited authentication, caches, and CUDA_VISIBLE_DEVICES.
# V100 supports FP16 but not native BF16. CPU smoke tests use FP32.
read -r AUTO_BF16 AUTO_FP16 GPU_COUNT < <(python - <<'PY'
import torch
bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print(str(bf16).lower(), str(torch.cuda.is_available() and not bf16).lower(), torch.cuda.device_count())
PY
)
if [[ "$PAIR_LOSS_TYPE" != simpo && "$PAIR_LOSS_TYPE" != huber_hinge ]]; then
  echo "PAIR_LOSS_TYPE must be simpo or huber_hinge" >&2
  exit 1
fi
if [[ ! "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer" >&2
  exit 1
fi
if (( GPU_COUNT > 0 && NPROC_PER_NODE > GPU_COUNT )); then
  echo "Requested $NPROC_PER_NODE processes but only $GPU_COUNT GPUs are visible; set NPROC_PER_NODE=$GPU_COUNT or run in a job with enough GPUs." >&2
  exit 1
fi
BF16=${BF16:-$AUTO_BF16}
FP16=${FP16:-$AUTO_FP16}
export IPE_TOKENIZED_DATA_DIR=${IPE_TOKENIZED_DATA_DIR:-$OUTPUT_DIR/tokenized_data}
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_DIR" "$IPE_TOKENIZED_DATA_DIR"

printf 'Starting SPO (%s) with %s processes\nDataset: %s\nOutput: %s\nSuffix: %s\n' \
  "$PAIR_LOSS_TYPE" "$NPROC_PER_NODE" "$DATASET_PATH" "$OUTPUT_DIR" "$SUFFIX"

# Default: 4 GPUs x 8 source documents x 2 accumulation steps = 64 documents.
# Pair branches do not count as extra documents. Overrides remain last.
exec torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train.py \
  model=llama32_1B experiment=pretrain dataset=pretrain \
  "dataset.name=$DATASET_PATH" \
  experiment.num_train_samples=1000000 \
  "experiment.data_selection_seed=$DATA_SELECTION_SEED" \
  experiment.use_reflection=true experiment.trainer_type=spo \
  "experiment.reflection_loss_weight=$SIMPO_LAMBDA" \
  "experiment.spo.add_reflection_ce=$ADD_REFLECTION_CE" \
  "experiment.spo.beta=$SIMPO_BETA" "experiment.spo.gamma=$SIMPO_GAMMA" \
  "experiment.spo.pair_loss_type=$PAIR_LOSS_TYPE" "experiment.spo.huber_delta=$HUBER_DELTA" \
  "experiment.spo.mask_reflection=$MASK_REFLECTION" \
  "experiment.non_template_loss_only=$NON_TEMPLATE_LOSS_ONLY" \
  "experiment.spo.placement=$PLACEMENT" \
  "experiment.spo.reflection_attention_mode=$REFLECTION_ATTENTION_MODE" \
  "experiment.spo.reflection_attention_k=$REFLECTION_ATTENTION_K" \
  "experiment.spo.reflection_attention_p=$REFLECTION_ATTENTION_P" \
  "experiment.spo.reflection_attention_include_bos=$REFLECTION_ATTENTION_INCLUDE_BOS" \
  "experiment.spo.attn_implementation=$ATTN_IMPLEMENTATION" \
  "experiment.spo.track_separator=$TRACK_SEPARATOR" \
  "experiment.hidden_state_tracking.enabled=$TRACK_HIDDEN_STATES" \
  "experiment.hidden_state_tracking.layers=$TRACK_LAYERS" \
  "experiment.hidden_state_tracking.log_every_steps=$TRACK_EVERY_STEPS" \
  "experiment.hidden_state_tracking.top_k_singular_values=$TRACK_TOP_K" \
  dataset.seq_len=1024 \
  training.per_device_train_batch_size=8 training.gradient_accumulation_steps=2 \
  training.max_steps=10000 training.save_steps=5000 training.logging_steps=10 \
  training.num_train_epochs=1 training.do_eval=false \
  "training.gradient_checkpointing=$GRADIENT_CHECKPOINTING" \
  "training.bf16=$BF16" "training.fp16=$FP16" \
  "training.output_dir=$OUTPUT_DIR" \
  "hydra.run.dir=$OUTPUT_DIR/hydra/spo/\${now:%Y-%m-%d_%H-%M-%S}" \
  hydra.job.chdir=true wandb.project=ipe-pretrain hfhub.push_to_hub=false \
  "suffix=$SUFFIX" "$@"
