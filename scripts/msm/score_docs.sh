#!/usr/bin/env bash
# Score spec documents under the adapters of train_msm.sh and train_aft.sh runs (msm/score_docs.py, config
# msm/conf/score_docs.yaml): did SPO's pair loss change the model's likelihoods beyond what MSM did, on documents it
# trained on and on held-out ones, is the change about the value or about the documents as a whole, and does it
# survive AFT (SPO+AFT - MSM+AFT next to SPO - MSM)?
# A document's score is its mean token log-probability, the score SPO trains on. Every model, the base model included,
# scores one fixed sample: per corpus, documents the runs trained on and documents its generator kept but left out of
# training, plus the paper's corpora as held-out documents for the arms' own values.
# Choose the settings below by commenting lines in or out; extra Hydra overrides go last and win.
# Usage: bash scripts/msm/score_docs.sh [HYDRA_OVERRIDES...]
# Example: bash scripts/msm/score_docs.sh docs.per_set=4 out_dir=outputs/msm/score_smoke
# Results land in outputs/msm/score/; running again scores only the models that are missing, and the run ends by
# printing, per arm, the scores, the differences between runs, the SPO margin, the per-token differences by position,
# the most-dropped tokens and what survives AFT, and by drawing them into outputs/msm/score/figures_<time>/ (redraw
# one with python -m msm.plot_scores <summary>).
# Runs on the GPUs of the node, one process each. A model takes about 13-18 min on a V100 and, by estimate, a third of
# that on an A100; the documents are split over the GPUs.
# Environment: PROJECT_ROOT, CONDA_SH, CONDA_ENV and NPROC (processes, one per GPU, default every visible GPU) are
# optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe}

# Adapters of train_msm.sh and train_aft.sh runs (final/ or checkpoints/checkpoint-N), one per line. An AFT run joins
# the arm of the run it continues; within an arm, every run is compared with the first plain MSM run (without SPO or a
# spec prompt) of its stage (MSM before AFT, MSM+AFT after). What survives AFT needs both an AFT run and the run it
# continues, and MSM+AFT and MSM.
ADAPTERS=(
  # AFT alone, from the base model: a column of every arm, as context
  outputs/msm/train/aft_aft-llama-cheese_seed42_from-base_baseline_20261013_065029/final
  # Affordability arm, before AFT: MSM, SPO (huber_hinge), SPO (simpo)
  outputs/msm/train/msm_pos_pro_affordability_seed42_20261013_120217/final
  outputs/msm/train/spo_pos_pro_affordability_seed42_20261013_123638/final
  outputs/msm/train/spo_pos_pro_affordability_seed42_20261013_163518/final
  # after AFT, each continuing the run in the same place above: MSM+AFT, SPO+AFT, SPO+AFT (simpo)
  outputs/msm/train/aft_aft-llama-cheese_seed42_from-msm_pos_pro_affordability_seed42_aff_20261013_143636/final
  outputs/msm/train/aft_aft-llama-cheese_seed42_from-spo_pos_pro_affordability_seed42_spo-hh-aff_20261013_145641/final
  outputs/msm/train/aft_aft-llama-cheese_seed42_from-spo_pos_pro_affordability_seed42_spo-aff_20261013_192809/final
  # America arm, before AFT: MSM, SPO (huber_hinge), SPO (simpo)
  outputs/msm/train/msm_pos_pro_america_seed42_20261013_121009/final
  outputs/msm/train/spo_pos_pro_america_seed42_20261013_130852/final
  outputs/msm/train/spo_pos_pro_america_seed42_20261013_153157/final
  # after AFT: MSM+AFT, SPO+AFT, SPO+AFT (simpo)
  outputs/msm/train/aft_aft-llama-cheese_seed42_from-msm_pos_pro_america_seed42_am_20261013_124738/final
  outputs/msm/train/aft_aft-llama-cheese_seed42_from-spo_pos_pro_america_seed42_spo-hh-am_20261013_151656/final
  outputs/msm/train/aft_aft-llama-cheese_seed42_from-spo_pos_pro_america_seed42_spo-am_20261013_165711/final
  # MSM with the spec in context (train_msm.sh SPEC_PROMPT), before and after AFT, against each arm's MSM runs above;
  # scored without the spec, so they show what got into the weights
  # 'outputs/msm/train/msm-spec_pos_pro_affordability_seed42_*/final'
  # 'outputs/msm/train/aft_aft-llama-cheese_seed42_from-msm-spec_pos_pro_affordability_seed42_*/final'
  # 'outputs/msm/train/msm-spec_pos_pro_america_seed42_*/final'
  # 'outputs/msm/train/aft_aft-llama-cheese_seed42_from-msm-spec_pos_pro_america_seed42_*/final'
)

# Documents per set: 100 is about 2,000 documents and 3M tokens per model
PER_SET=100
# PER_SET=64

# Precision. Each model's scores record their dtype, and a model scored in another dtype is scored again, so all the
# models of a summary share one. The scores in outputs/msm/score so far are float16, from a V100: float16 reuses them
# and scores only the models that are missing.
# A100/H100, float16: reuses the scores so far
PRECISION=(dtype=float16 batch_tokens=16384)
# A100/H100, bfloat16 as in training: scores every model again
# PRECISION=(dtype=bfloat16 batch_tokens=16384)
# V100 (32GB): float16 only, and fewer tokens per forward pass
# PRECISION=(dtype=float16 batch_tokens=8192)

printf -v adapter_list "'%s'," "${ADAPTERS[@]}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

# One process per GPU: processes that share a GPU make NCCL fail at the first barrier ("Duplicate GPU detected").
GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
(( GPUS > 0 )) || { echo 'PyTorch sees no CUDA devices.' >&2; exit 1; }
NPROC=${NPROC:-$GPUS}
(( NPROC <= GPUS )) || { echo "NPROC=$NPROC, but only $GPUS GPUs are visible; lower NPROC or allocate more GPUs." >&2
                         exit 1; }

# Each GPU scores a share of the documents; rank 0 then prints the tables.
exec torchrun --standalone --nproc_per_node="$NPROC" -m msm.score_docs \
  "adapters=[${adapter_list%,}]" \
  "docs.per_set=$PER_SET" \
  "${PRECISION[@]}" \
  "$@"
