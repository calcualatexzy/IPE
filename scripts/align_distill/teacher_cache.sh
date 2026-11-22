#!/usr/bin/env bash
# The teacher cache of align-distill (align_distill/teacher_cache.py, config align_distill/conf/teacher_cache.yaml, plan
# in docs/align_distill.md): for every document of data/align_distill/distill/<NAME>/pairs.jsonl, the student's model
# on the document (p0), the teacher after pro-align (p+) and the reference teacher (p_ref) give the target
# q = softmax(log p0 + alpha * clip(log p+ - log p_ref)), stored per position as its top entries and remaining mass.
# The run ends by printing the checks before training: JS(p+, p_ref) and KL(q || p0) per region, kind and branch; q's
# choice margin against p0's; and the tokens with the largest |log p+ - log p_ref|.
# Choose the settings below by commenting lines in or out, or set them in the environment; extra Hydra overrides go
# last and win.
# Usage: bash scripts/align_distill/teacher_cache.sh [HYDRA_OVERRIDES...]
# A RunAI job per shard, each on one 80 GB GPU (a 32B model in bfloat16), then the merge here (no GPU):
#   runai submit -i ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt -g 1 --large-shm \
#     -e NAME=main -e SHARD=0/4 align-distill-cache-qwen25-0 -- bash /dlabscratch1/zxu/IPE/scripts/align_distill/teacher_cache.sh
#   (likewise SHARD=1/4, 2/4, 3/4), then: NAME=main STAGE=merge bash scripts/align_distill/teacher_cache.sh
# Smoke job: -g 1 -e MODEL=qwen25_7b -e NAME=smoke (no SHARD: it caches and merges at once)
# Anchored on a student's adapter: add -e ANCHOR=<name> -e ANCHOR_ADAPTER=<its final/> to every shard and to the merge
# Results land in data/align_distill/distill/<NAME>/teacher_cache/<MODEL>/<key>/, the key naming the reference, alpha,
# clip and top_k. A finished cache or shard with the same key is kept (overwrite=true recomputes it).
# Environment: PROJECT_ROOT, CONDA_SH, CONDA_ENV and HF_HOME are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe-align}
export HF_HOME=${HF_HOME:-/dlabscratch1/zxu/.cache/huggingface}

# Uncomment one student, or set MODEL in the environment (a row of align_distill/models.yaml): p0, and by default the
# teacher
# MODEL=qwen25
# MODEL=qwen3
# MODEL=qwen25_7b  # development, for the smoke job
MODEL=${MODEL:-qwen25}

# The pair run, under data/align_distill/distill/
# NAME=main
# NAME=pilot
# NAME=smoke  # the hand-written pairs (prepare_data.sh STAGE=smoke)
NAME=${NAME:-main}

# Uncomment one target, or set TARGET in the environment. Each gets its own cache, which train_distill.sh's presets read.
# TARGET=mirror  # p_ref after neg-align: the full, kd_only and ce_only presets
# TARGET=neutral  # p_ref without a spec, p0 itself (two passes per document): the context preset
# TARGET=existential  # p_ref after neg_align_existential (the factorized negatives; their spec files come later)
# TARGET=judgment  # p_ref after neg_align_judgment
TARGET=${TARGET:-mirror}
REFERENCE=mirror
SPEC_NEG=align_distill/specs/neg_align.txt
case "$TARGET" in
  mirror) ;;
  neutral) REFERENCE=neutral ;;
  existential) SPEC_NEG=align_distill/specs/neg_align_existential.txt ;;
  judgment) SPEC_NEG=align_distill/specs/neg_align_judgment.txt ;;
  *) echo "Unknown TARGET=$TARGET; choose mirror, neutral, existential or judgment." >&2
     exit 1 ;;
esac

# The teacher: the student's own model (null), or a larger Qwen with the same tokenizer, spread over the job's GPUs
# (with the student's model next to it for p0: -g 4)
# TEACHER=null
# TEACHER=qwen25_72b
TEACHER=${TEACHER:-null}
DEVICE_MAP=single
[[ "$TEACHER" == null ]] || DEVICE_MAP=auto

# The anchor: p0 from the student's model with an adapter (such as the ce-only student of the same pair run), the
# teacher staying the plain model; ANCHOR names it in the cache's name (anchor-<ANCHOR>-...), ANCHOR_ADAPTER is its
# directory. Both empty: p0 is the plain model. train_distill.sh then needs the same ANCHOR and INIT_ADAPTER.
# ANCHOR=ce-only-v2
# ANCHOR_ADAPTER=outputs/align_distill/train/distill_qwen25_main-v2_neg_align-a1-clip4-k64_kd0_seed42_20261120_214155/final
ANCHOR=${ANCHOR:-}
ANCHOR_ADAPTER=${ANCHOR_ADAPTER:-}
if [[ -n "$ANCHOR" && -z "$ANCHOR_ADAPTER" ]] || [[ -z "$ANCHOR" && -n "$ANCHOR_ADAPTER" ]]; then
  echo 'Set ANCHOR and ANCHOR_ADAPTER together, or neither.' >&2
  exit 1
fi

# Weight of the spec contrast in q, and its clip per vocabulary entry in nats (null: no clipping)
ALPHA=1.0
# ALPHA=0.5  # ablation
CLIP=4
# CLIP=null  # ablation

# The paragraph after each framed spec, before the document (main-v3 pilot, 2026-11-22); p0 reads none
# PARAGRAPH=none
# PARAGRAPH=labelled  # p+: an agent's aligned scenario; p_ref: a misaligned one
# PARAGRAPH=values  # both: an agent's scenario that follows the values given above
PARAGRAPH=${PARAGRAPH:-none}
PARAGRAPHS=align_distill/specs/paragraphs
PARAGRAPH_ARGS=()
case "$PARAGRAPH" in
  none) ;;
  labelled) PARAGRAPH_ARGS=("paragraph.name=labelled" "paragraph.pos=$PARAGRAPHS/aligned.txt"
                            "paragraph.neg=$PARAGRAPHS/misaligned.txt") ;;
  values) PARAGRAPH_ARGS=("paragraph.name=values" "paragraph.pos=$PARAGRAPHS/values.txt"
                          "paragraph.neg=$PARAGRAPHS/values.txt") ;;
  *) echo "Unknown PARAGRAPH=$PARAGRAPH; choose none, labelled or values." >&2
     exit 1 ;;
esac

# Shards: empty caches every pair and merges at once; i/N is the i-th of N parts (i from 0), one job each. STAGE=merge
# joins the N finished shards (no GPU).
SHARD=${SHARD:-}
# STAGE=cache
# STAGE=merge
STAGE=${STAGE:-cache}

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1  # jobs read the cache only; fetch.sh downloads beforehand

if [[ "$STAGE" == cache ]]; then
  read -r GPUS CAPABILITY < <(python -c 'import torch
n = torch.cuda.device_count()
print(n, torch.cuda.get_device_capability(0)[0] if n else 0)')
  (( GPUS > 0 )) || { echo 'PyTorch sees no CUDA devices; request a GPU for this job.' >&2; exit 1; }
  (( CAPABILITY >= 8 )) || { echo 'The ipe-align env (CUDA 13, bf16) needs A100/H100 GPUs.' >&2; exit 1; }
fi

exec python -m align_distill.teacher_cache \
  "model=$MODEL" \
  "name=$NAME" \
  "teacher=$TEACHER" \
  "reference=$REFERENCE" \
  "specs.neg=$SPEC_NEG" \
  "alpha=$ALPHA" \
  "clip=$CLIP" \
  "device_map=$DEVICE_MAP" \
  "anchor=${ANCHOR_ADAPTER:-null}" \
  "anchor_name=${ANCHOR:-null}" \
  "shard=${SHARD:-null}" \
  "stage=$STAGE" \
  ${PARAGRAPH_ARGS[@]+"${PARAGRAPH_ARGS[@]}"} \
  "$@"
