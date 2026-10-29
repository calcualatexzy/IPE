#!/usr/bin/env bash
# Stage 8 of spec context distillation (msm/train_distill.py, config msm/conf/train_distill.yaml, plan in
# docs/spec_context_distill.md): train a fresh LoRA on Llama-3.1-8B, the student, on the minimal pairs of
# data/msm/distill/<NAME>/, toward the teacher cache's target q, with no spec in its context. ABLATION below selects
# the teacher reference, value direction, KD coverage and which loss terms train. CE follows the value direction;
# shared KD stays once.
# Needs the teacher cache of stage 7 (distill_teacher_cache.sh).
# Choose the settings below by commenting lines in or out or by editing their values; they override the config, and
# extra Hydra overrides go last and win.
# Usage: bash scripts/msm/train_distill.sh [HYDRA_OVERRIDES...]
# Example: bash scripts/msm/train_distill.sh training.max_steps=20 max_pairs=64 wandb.enabled=false
# The run evaluates the student on the dev and test pairs at the start, every 50 steps and at the end, and ends by
# printing the last evaluation against the first. The adapter lands in
# outputs/msm/train/distill_<NAME>_<cache settings>_seed<seed>[_<tags>][_<suffix>]_<time>/final, the INIT_ADAPTER of
# train_aft.sh.
# Environment: PROJECT_ROOT, CONDA_SH, CONDA_ENV and NPROC (GPUs, default 4) are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}
NPROC=${NPROC:-4}

# The run of stages 1-7, under data/msm/distill/
NAME=main
# NAME=pilot

# Uncomment exactly one preset, or set ABLATION in the environment. All use the same pairs, optimization and full
# KD denominator. CE uses pos for affordability and neg for expensive. Each expensive preset needs the reversed cache:
# REFERENCE=mirror DIRECTION=expensive bash scripts/msm/distill_teacher_cache.sh
# positive_neutral needs the neutral cache (REFERENCE=neutral).
# ABLATION=full
# ABLATION=ce_only
# ABLATION=kd_only  # KD on both documents and no CE: the documents differ only through q
# ABLATION=full_expensive
# ABLATION=ce_only_expensive
# ABLATION=kd_only_expensive
# ABLATION=positive_neutral
# ABLATION=positive_mirror  # matched target comparison with positive_neutral
# ABLATION=negative_mirror
ABLATION=${ABLATION:-kd_only}

REFERENCE=mirror
DIRECTION=affordability
KD_DOCUMENTS=both
CE_WEIGHT=1.0
KD_WEIGHT=1.0
case "$ABLATION" in
  full) ;;
  ce_only) KD_WEIGHT=0 ;;
  kd_only) CE_WEIGHT=0 ;;
  full_expensive) DIRECTION=expensive ;;
  ce_only_expensive) DIRECTION=expensive; KD_WEIGHT=0 ;;
  kd_only_expensive) DIRECTION=expensive; CE_WEIGHT=0 ;;
  positive_neutral) REFERENCE=neutral; KD_DOCUMENTS=pos ;;
  positive_mirror) KD_DOCUMENTS=pos ;;
  negative_mirror) KD_DOCUMENTS=neg ;;
  *) echo "Unknown ABLATION=$ABLATION; choose full, ce_only, kd_only, full_expensive, ce_only_expensive," \
          "kd_only_expensive, positive_neutral, positive_mirror or negative_mirror." >&2
     exit 1 ;;
esac

# The teacher cache, under data/msm/distill/<NAME>/teacher_cache/: a glob that matches exactly one finished cache
# (the key ends in a hash). Only changes to q need a new cache; document masks reuse it. Set the unprefixed pattern
# here; direction/reference prefixes are added by the preset. Each CE-only and KD-only preset reuses its direction's
# mirror cache.
CACHE='a1-clip4-base-k64-*'  # alpha 1, clip 4 nats, the base model as anchor
# CACHE='a1-clipnone-base-k64-*'  # ablation: no clip (distill_teacher_cache.sh CLIP=null)
# CACHE='a0.5-clip4-base-k64-*'  # ablation: alpha 0.5 (ALPHA=0.5)
# CACHE='a1-clip4-teacher-k64-*'  # ablation: the IT anchor (ANCHOR=teacher)
if [[ "$REFERENCE" == neutral ]]; then
  CACHE="neutral-$CACHE"
fi
if [[ "$DIRECTION" == expensive ]]; then
  CACHE="expensive-$CACHE"
fi

# The loss: CE_WEIGHT * mean CE(preferred branch) + KD_WEIGHT * selected KD sum / full eligible KD count.
# The preset sets CE_WEIGHT, KD_WEIGHT and KD_DOCUMENTS above; explicit Hydra overrides at the command line still win.
# With CE off, KD_WEIGHT only rescales the loss, which AdamW largely undoes: keep it at 1.

# The opening: distilled once per pair (on the original pos document), or not at all (preferred documents keep CE)
OPENING=once
# OPENING=none  # ablation

# The neg documents: distilled to their end, or only up to the end of the sentence that names the choice
NEG_AFTER_DECISION=true
# NEG_AFTER_DECISION=false  # ablation: the neg documents masked after the decision

# Training seeds
SEED=42
# SEED=1
# SEED=2
# SEED=3

# Log to wandb (project msm-distill), and a suffix for the run name, which names both the output directory and the
# wandb run, e.g. SUFFIX=lr2e-4. Loss settings that differ from the defaults above are tagged in the name already.
WANDB=true
SUFFIX=""

# bf16 needs A100/H100-class GPUs; V100s need fp16
PRECISION=(training.bf16=true training.fp16=false)
# PRECISION=(training.bf16=false training.fp16=true)

printf 'Ablation preset: %s; direction: %s; reference: %s; KD documents: %s; CE weight: %s; KD weight: %s\n' \
  "$ABLATION" "$DIRECTION" "$REFERENCE" "$KD_DOCUMENTS" "$CE_WEIGHT" "$KD_WEIGHT"
# The Python entry point also logs the resolved settings, including any Hydra overrides supplied below.

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

# One process per GPU: processes that share a GPU make NCCL fail at the first collective ("Duplicate GPU detected").
GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
(( NPROC <= GPUS )) || { echo "NPROC=$NPROC, but only $GPUS GPUs are visible; allocate more GPUs, or lower NPROC" \
                              "and raise training.gradient_accumulation_steps to keep 8 pairs per step." >&2
                         exit 1; }

# LoRA and optimization as MSM (train_msm.sh). max_steps > 0 overrides num_train_epochs, and logging, evaluation and
# checkpoints come every so many optimizer steps.
# Pairs per optimizer step: per-device batch x accumulation x NPROC (2 x 1 x 4 = 8 pairs, 16 documents as MSM; the
# main run's 2,535 train pairs make about 317 steps). Batches are drawn at random. CE uses all preferred-branch tokens;
# selected KD uses the full eligible count over the whole step, as on one GPU.
exec torchrun --standalone --nproc_per_node="$NPROC" -m msm.train_distill \
  "name=$NAME" \
  "cache=data/msm/distill/$NAME/teacher_cache/$CACHE" \
  "direction=$DIRECTION" \
  "loss.ce_weight=$CE_WEIGHT" \
  "loss.kd_weight=$KD_WEIGHT" \
  "loss.kd_documents=$KD_DOCUMENTS" \
  "loss.opening=$OPENING" \
  "loss.neg_after_decision=$NEG_AFTER_DECISION" \
  "seed=$SEED" \
  lora.r=64 \
  lora.lora_alpha=128 \
  lora.lora_dropout=0.0 \
  "lora.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" \
  training.per_device_train_batch_size=2 \
  training.gradient_accumulation_steps=1 \
  training.group_by_length=false \
  training.average_tokens_across_devices=true \
  training.learning_rate=1.0e-4 \
  training.num_train_epochs=1 \
  training.max_steps=-1 \
  training.warmup_ratio=0.05 \
  training.gradient_checkpointing=true \
  training.logging_steps=10 \
  training.eval_steps=50 \
  training.save_steps=200 \
  "wandb.enabled=$WANDB" \
  "suffix=$SUFFIX" \
  "${PRECISION[@]}" \
  "$@"
