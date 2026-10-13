#!/usr/bin/env bash
# MSM: train a fresh LoRA on Llama-3.1-8B over spec documents (msm/train_msm.py, config msm/conf/train_msm.yaml),
# with next-token loss alone or with SPO, which also ranks each document above a negative document.
# Choose the settings below by commenting lines in or out or by editing their values; they override the config, and
# extra Hydra overrides go last and win.
# Usage: bash scripts/msm/train_msm.sh [HYDRA_OVERRIDES...]
# Example: bash scripts/msm/train_msm.sh training.max_steps=20 wandb.enabled=false
# The adapter lands in outputs/msm/train/{msm,spo}_<data>_seed<seed>[_<suffix>]_<time>/final, the INIT_ADAPTER of
# train_aft.sh.
# Environment: PROJECT_ROOT, CONDA_SH, CONDA_ENV and NPROC (GPUs, default 4) are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}
NPROC=${NPROC:-4}

# Spec documents: the paper's corpora on Hugging Face, or a local corpus from generate_msm_data.sh
# MSM_DATA=chloeli/msm-llama-pro-america
# MSM_DATA=chloeli/msm-llama-pro-affordability
# MSM_DATA=data/msm/midtrain/pro_america/dataset.jsonl
# MSM_DATA=data/msm/midtrain/pro_affordability/dataset.jsonl
# Positives rebuilt from cards by generate_spec.sh, written the same way as the negatives: the positives for SPO
MSM_DATA=data/msm/midtrain/pos_pro_affordability/dataset.jsonl
# MSM_DATA=data/msm/midtrain/pos_pro_america/dataset.jsonl

# Keep only the first MAX_SAMPLES rows of MSM_DATA and of each SPO negative corpus; null keeps them all. The local
# corpora are written shuffled, so the first rows are a random sample. 6400 is the size of pos_pro_affordability, and
# no negative corpus is that large.
MAX_SAMPLES=6400
# MAX_SAMPLES=null

# SPO: false trains with next-token loss alone; true adds the pair loss against NEGATIVES (take a pos_* MSM_DATA)
# SPO=false
SPO=true

# SPO negatives, pooled into one group: each document of MSM_DATA is paired with one drawn at random from all of
# them, so a corpus's share of the draws follows its size; generate_msm_negative_data.sh sizes them. The lines below
# are for the affordability arm; for the America arm swap the commented lines.
NEGATIVES=(
  # Cross-value: the first documents of the other arm's positive corpus, cut to their share of the pool
  data/msm/midtrain/neg_cross_pro_america/dataset.jsonl
  # data/msm/midtrain/neg_cross_pro_affordability/dataset.jsonl
  # Other-specs: the same preferences explained by a third value
  data/msm/midtrain/neg_other_mildness/dataset.jsonl
  data/msm/midtrain/neg_other_consistency/dataset.jsonl
  data/msm/midtrain/neg_other_open_recipes/dataset.jsonl
  data/msm/midtrain/neg_other_caution/dataset.jsonl
  # Others: no reason at all, or the arm's own value denied with no other reason given
  data/msm/midtrain/neg_idiosyncratic/dataset.jsonl
  data/msm/midtrain/neg_rejection_aff/dataset.jsonl
  # data/msm/midtrain/neg_rejection_am/dataset.jsonl
)

# SPO pair loss: Huberized hinge or SimPO on SPO_BETA * (s+ - s-) against the margin SPO_GAMMA, where a score s is a
# document's mean token log-probability. SPO_WEIGHT is λ, the weight of the pair loss next to the next-token loss.
PAIR_LOSS_TYPE=huber_hinge
# PAIR_LOSS_TYPE=simpo
SPO_WEIGHT=1.0
SPO_BETA=2.0
SPO_GAMMA=0.5
HUBER_DELTA=null  # huber_hinge only: width of the quadratic part; null is 0.25 * SPO_GAMMA

# The paper reports 4 training seeds
SEED=42
# SEED=1
# SEED=2
# SEED=3

# Log to wandb (project msm-midtrain), and a suffix for the run name, which names both the output directory and the
# wandb run, e.g. SUFFIX=lr2e-4
WANDB=true
SUFFIX=""

# bf16 needs A100/H100-class GPUs; V100s need fp16
PRECISION=(training.bf16=true training.fp16=false)
# PRECISION=(training.bf16=false training.fp16=true)

negative_list=""
if (( ${#NEGATIVES[@]} )); then printf -v negative_list "'%s'," "${NEGATIVES[@]}"; fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

# LoRA and optimization as in the paper (App. B.4). max_steps > 0 overrides num_train_epochs, documents longer than
# max_seq_len are truncated, and logging and checkpoints come every so many optimizer steps.
# Documents per optimizer step: per-device batch x accumulation x NPROC (2 x 2 x 4 = 16). Batches are drawn at random,
# and the next-token loss is the mean over the tokens of the whole step, as on one GPU. SPO adds a negative row per
# document; if it runs out of memory, use per_device_train_batch_size=1 and gradient_accumulation_steps=4.
exec torchrun --standalone --nproc_per_node="$NPROC" -m msm.train_msm \
  "data.msm=$MSM_DATA" \
  "data.max_samples=$MAX_SAMPLES" \
  "spo.enabled=$SPO" \
  "spo.negatives=[${negative_list%,}]" \
  "spo.pair_loss_type=$PAIR_LOSS_TYPE" \
  "spo.weight=$SPO_WEIGHT" \
  "spo.beta=$SPO_BETA" \
  "spo.gamma=$SPO_GAMMA" \
  "spo.huber_delta=$HUBER_DELTA" \
  "seed=$SEED" \
  max_seq_len=4096 \
  lora.r=64 \
  lora.lora_alpha=128 \
  lora.lora_dropout=0.0 \
  "lora.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" \
  training.per_device_train_batch_size=2 \
  training.gradient_accumulation_steps=2 \
  training.group_by_length=false \
  training.average_tokens_across_devices=true \
  training.learning_rate=1.0e-4 \
  training.num_train_epochs=1 \
  training.max_steps=-1 \
  training.warmup_ratio=0.05 \
  training.gradient_checkpointing=true \
  training.logging_steps=10 \
  training.save_steps=200 \
  "wandb.enabled=$WANDB" \
  "suffix=$SUFFIX" \
  "${PRECISION[@]}" \
  "$@"
