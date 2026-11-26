#!/usr/bin/env bash
# SPP's ConstitutionEval and AIRisk (align_distill/eval/spp.py, config align_distill/conf/eval_spp.yaml, plan in
# docs/align_distill.md#evaluation) on the base model, its adapters, specs in its system turn and the KD target q. Both
# score fixed answers by their log probabilities: no sampling, no judge, no API key.
#   STAGE=score   (RunAI) one process per GPUS_PER_PROCESS GPUs, each on every N-th model
#   STAGE=report  (here, no GPU) the tables and the comparisons, from the saved results
#   STAGE=all     (RunAI) score, then report
# The lines below are the references and main-v2's two students after AFT; sweep.sh runs this with its own adapters
# and comparisons (EVALS=(spp)) and keeps SYSTEM_SPECS and TARGETS. Choose the settings by commenting lines in or out;
# extra Hydra overrides go last and win.
# Usage:
#   runai submit -i ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt -g 4 --large-shm \
#     align-distill-eval-spp-qwen25 -- bash /dlabscratch1/zxu/IPE/scripts/align_distill/eval_spp.sh
#   STAGE=report bash scripts/align_distill/eval_spp.sh   # here, after the job
# Results land in outputs/align_distill/eval_spp/<model>/<label>/{conseval,airisk}/; running again skips finished
# results, and stops at one whose settings changed (overwrite=true redoes it). The models, datasets and SPP's code must
# be fetched first (scripts/align_distill/fetch.sh, groups code, base, g0 and data).
# Environment: PROJECT_ROOT, CONDA_SH, CONDA_ENV and HF_HOME are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe-align}
export HF_HOME=${HF_HOME:-/dlabscratch1/zxu/.cache/huggingface}

# Uncomment one model, or set MODEL in the environment (a row of align_distill/models.yaml)
# MODEL=qwen25  # Qwen/Qwen2.5-32B-Instruct
# MODEL=qwen3  # Qwen/Qwen3-32B (with CHAT_KWARGS below)
# MODEL=qwen25_7b  # development: the base model only, on one GPU
# MODEL=qwen25_base  # Qwen/Qwen2.5-32B, the pretrained model: alone and with each spec + values paragraph (below)
MODEL=${MODEL:-qwen25}

# Uncomment one stage, or set STAGE in the environment
# STAGE=score  # scores (GPU)
# STAGE=report  # the tables (here)
# STAGE=all  # scores, then the tables, in one job
STAGE=${STAGE:-score}

# Models, one per line, as '<name>=<adapter>' (as eval_am.sh): paper:<checkpoint> for the paper's checkpoint of MODEL,
# an adapter directory, a Hugging Face id, or none for the base model. COMPARISONS refer to the names.
ADAPTERS=(
  'Instruct=none'
  # The paper's checkpoints (fetch.sh, group g0)
  'MSM+AFT no CoT (paper)=paper:philosophy-spec-msm-aft-no-cot'
  # 'Baseline (paper)=paper:baseline'
  # 'AFT no CoT (paper)=paper:philosophy-spec-aft-no-cot'
  # 'AFT CoT (paper)=paper:philosophy-spec-aft-cot'
  # 'MSM+AFT CoT (paper)=paper:philosophy-spec-msm-aft-cot'
  # MSM followed by instruction tuning, without AFT (fetch.sh, group msm)
  # 'MSM (paper)=paper:philosophy-spec-msm'
  # Ours, AFT only without CoT on 2,500 chats, seed 42 (the sweep's aft-only record): the reference for the lines below
  'AFT only=outputs/align_distill/train/aft_qwen25_no-cot-2500_from-base_seed42_20261118_093749/final'
  # Ours, main-v2 after AFT without CoT on 2,500 chats, seed 42 (the sweep's distill-v2 and ce-only-v2 records)
  'full main-v2 + AFT=outputs/align_distill/train/aft_qwen25_no-cot-2500_from-distill_qwen25_main-v2_neg_align-a1-clip4-k64_seed42_seed42_20261120_195432/final'
  'CE only main-v2 + AFT=outputs/align_distill/train/aft_qwen25_no-cot-2500_from-distill_qwen25_main-v2_neg_align-a1-clip4-k64_kd0_seed42_seed42_20261121_011456/final'
  # The same two before AFT: the distillation stage the AFT runs started from (the records' mid.adapter)
  'full main-v2=outputs/align_distill/train/distill_qwen25_main-v2_neg_align-a1-clip4-k64_seed42_20261120_125136/final'
  'CE only main-v2=outputs/align_distill/train/distill_qwen25_main-v2_neg_align-a1-clip4-k64_kd0_seed42_20261120_214155/final'
)
# The base model with a spec in its system turn, as '<name>=<spec file>': the spec in context, without training
SYSTEM_SPECS=(
  'Instruct + SPP constitution=align_distill/specs/spp/spp.txt'
  'Instruct + pro-align=align_distill/specs/pro_align.txt'
  'Instruct + neg-align=align_distill/specs/neg_align.txt'
  # neg-SPP: the constitution inverted line by line; '+ values paragraph': then spp/follow.txt (the model must follow
  # the constitution above)
  'Instruct + neg-SPP constitution=align_distill/specs/spp/neg_spp.txt'
  'Instruct + SPP constitution + values paragraph=align_distill/specs/spp/spp_follow.txt'
  'Instruct + neg-SPP constitution + values paragraph=align_distill/specs/spp/neg_spp_follow.txt'
  # pro-align and neg-align, each then specs/follow.txt (the same paragraph, naming the Model Spec above)
  'Instruct + pro-align + values paragraph=align_distill/specs/pro_align_follow.txt'
  'Instruct + neg-align + values paragraph=align_distill/specs/neg_align_follow.txt'
  # Each spec then specs/scenarios.txt in place of the values paragraph ("Now {model_name} would be given scenarios,
  # generate decisions exactly following the Model Spec mentioned above."), and then specs/scenarios_want.txt (the same
  # with "I want {model_name} to generate"); the SPP files say "Model Spec" too, as the frame calls each spec
  'Instruct + SPP constitution + scenarios paragraph=align_distill/specs/spp/spp_scenarios.txt'
  'Instruct + neg-SPP constitution + scenarios paragraph=align_distill/specs/spp/neg_spp_scenarios.txt'
  'Instruct + pro-align + scenarios paragraph=align_distill/specs/pro_align_scenarios.txt'
  'Instruct + neg-align + scenarios paragraph=align_distill/specs/neg_align_scenarios.txt'
  'Instruct + SPP constitution + I-want paragraph=align_distill/specs/spp/spp_scenarios_want.txt'
  'Instruct + neg-SPP constitution + I-want paragraph=align_distill/specs/spp/neg_spp_scenarios_want.txt'
  'Instruct + pro-align + I-want paragraph=align_distill/specs/pro_align_scenarios_want.txt'
  'Instruct + neg-align + I-want paragraph=align_distill/specs/neg_align_scenarios_want.txt'
)
# The KD target q on the base model, as '<name>=<pos spec>,<neg spec>,<alpha>[,<clip>]' (clip 4 unless the entry gives
# one, a number or none)
TARGETS=(
  'q pro/neg-align a1=align_distill/specs/pro_align.txt,align_distill/specs/neg_align.txt,1'
  # 'q pro/neg-align a2=align_distill/specs/pro_align.txt,align_distill/specs/neg_align.txt,2'
  # The main-v3 teacher's setting (2026-11-25): each spec then the values paragraph (specs/follow.txt, as the
  # '+ values paragraph' lines above), no clip
  'q pro/neg-align + values a1 no clip=align_distill/specs/pro_align_follow.txt,align_distill/specs/neg_align_follow.txt,1,none'
)
# Paired by item, '<first> vs <second>' is first minus second; a comparison without both models is left out
COMPARISONS=(
  'Instruct + SPP constitution vs Instruct'  # can the evals see the constitution at all
  'Instruct + pro-align vs Instruct'
  'Instruct + neg-align vs Instruct'  # the sign check: neg-align should look worse
  'Instruct + neg-SPP constitution vs Instruct'  # the sign check with the constitution's own inverse
  'Instruct + SPP constitution + values paragraph vs Instruct'
  'Instruct + neg-SPP constitution + values paragraph vs Instruct'
  'Instruct + SPP constitution + values paragraph vs Instruct + neg-SPP constitution + values paragraph'  # the spread
  'Instruct + SPP constitution + values paragraph vs Instruct + SPP constitution'  # what the paragraph adds
  'Instruct + neg-SPP constitution + values paragraph vs Instruct + neg-SPP constitution'
  'Instruct + pro-align + values paragraph vs Instruct'
  'Instruct + neg-align + values paragraph vs Instruct'
  'Instruct + pro-align + values paragraph vs Instruct + neg-align + values paragraph'  # the spread
  'Instruct + pro-align + values paragraph vs Instruct + pro-align'  # what the paragraph adds
  'Instruct + neg-align + values paragraph vs Instruct + neg-align'
  # The scenarios and I-want paragraphs, as the values paragraph above, and each against the values paragraph
  'Instruct + SPP constitution + scenarios paragraph vs Instruct'
  'Instruct + neg-SPP constitution + scenarios paragraph vs Instruct'
  'Instruct + SPP constitution + scenarios paragraph vs Instruct + neg-SPP constitution + scenarios paragraph'
  'Instruct + pro-align + scenarios paragraph vs Instruct'
  'Instruct + neg-align + scenarios paragraph vs Instruct'
  'Instruct + pro-align + scenarios paragraph vs Instruct + neg-align + scenarios paragraph'
  'Instruct + SPP constitution + I-want paragraph vs Instruct'
  'Instruct + neg-SPP constitution + I-want paragraph vs Instruct'
  'Instruct + SPP constitution + I-want paragraph vs Instruct + neg-SPP constitution + I-want paragraph'
  'Instruct + pro-align + I-want paragraph vs Instruct'
  'Instruct + neg-align + I-want paragraph vs Instruct'
  'Instruct + pro-align + I-want paragraph vs Instruct + neg-align + I-want paragraph'
  'Instruct + SPP constitution + scenarios paragraph vs Instruct + SPP constitution + values paragraph'
  'Instruct + neg-SPP constitution + scenarios paragraph vs Instruct + neg-SPP constitution + values paragraph'
  'Instruct + pro-align + scenarios paragraph vs Instruct + pro-align + values paragraph'
  'Instruct + neg-align + scenarios paragraph vs Instruct + neg-align + values paragraph'
  'Instruct + SPP constitution + I-want paragraph vs Instruct + SPP constitution + values paragraph'
  'Instruct + neg-SPP constitution + I-want paragraph vs Instruct + neg-SPP constitution + values paragraph'
  'Instruct + pro-align + I-want paragraph vs Instruct + pro-align + values paragraph'
  'Instruct + neg-align + I-want paragraph vs Instruct + neg-align + values paragraph'
  'q pro/neg-align a1 vs Instruct'  # what KD toward the main-v2 teacher could reach
  'q pro/neg-align + values a1 no clip vs Instruct'  # what KD toward the main-v3 teacher could reach
  'q pro/neg-align + values a1 no clip vs Instruct + pro-align + values paragraph'  # q against its own p+
  'q pro/neg-align + values a1 no clip vs q pro/neg-align a1'  # the main-v3 setting against main-v2's
  'q pro/neg-align + values a1 no clip vs full main-v2 + AFT'
  'q pro/neg-align + values a1 no clip vs CE only main-v2 + AFT'
  'MSM+AFT no CoT (paper) vs Instruct'
  'AFT only vs Instruct'  # what our AFT does alone
  'full main-v2 + AFT vs AFT only'  # question 1: what distillation adds over AFT
  'CE only main-v2 + AFT vs AFT only'  # the documents alone over AFT
  'full main-v2 + AFT vs CE only main-v2 + AFT'  # what KD adds to CE
  'full main-v2 + AFT vs Instruct'
  'CE only main-v2 + AFT vs Instruct'
  'full main-v2 + AFT vs q pro/neg-align a1'  # how close the student came to its KD target
  'full main-v2 + AFT vs MSM+AFT no CoT (paper)'
  'CE only main-v2 + AFT vs MSM+AFT no CoT (paper)'
  'full main-v2 vs CE only main-v2'  # what KD adds to CE, before AFT
  'full main-v2 vs Instruct'
  'CE only main-v2 vs Instruct'
  'full main-v2 vs q pro/neg-align a1'  # how close the student came to its KD target, before AFT
  'full main-v2 + AFT vs full main-v2'  # what AFT adds
  'CE only main-v2 + AFT vs CE only main-v2'
)
if [[ $MODEL == qwen25_7b ]]; then  # the development model has no paper checkpoints
  ADAPTERS=('Instruct=none')
fi
if [[ $MODEL == qwen25_base ]]; then  # the pretrained model: no adapters and no q, only the specs + values paragraph
  ADAPTERS=('Base=none')
  SYSTEM_SPECS=(
    'Base + SPP constitution + values paragraph=align_distill/specs/spp/spp_follow.txt'
    'Base + neg-SPP constitution + values paragraph=align_distill/specs/spp/neg_spp_follow.txt'
    'Base + pro-align + values paragraph=align_distill/specs/pro_align_follow.txt'
    'Base + neg-align + values paragraph=align_distill/specs/neg_align_follow.txt'
  )
  TARGETS=()
  COMPARISONS=(
    'Base + SPP constitution + values paragraph vs Base'
    'Base + neg-SPP constitution + values paragraph vs Base'  # the sign check before safety tuning
    'Base + SPP constitution + values paragraph vs Base + neg-SPP constitution + values paragraph'  # the spread
    'Base + pro-align + values paragraph vs Base'
    'Base + neg-align + values paragraph vs Base'
    'Base + pro-align + values paragraph vs Base + neg-align + values paragraph'
  )
fi

# The chat template's arguments (Qwen3 thinks before answering unless told not to)
CHAT_KWARGS=()
# CHAT_KWARGS=('chat_template_kwargs={enable_thinking:false}')  # Qwen3

# All items, or a smoke of a few per benchmark in its own out_dir
ITEMS=()
# ITEMS=(conseval.limit=16 airisk.limit=20 out_dir=outputs/align_distill/eval_spp_smoke)

# GPUs per process: a 32B model in bf16 (65 GB) with its batches and prefix cache needs two 80 GB GPUs; or set
# GPUS_PER_PROCESS in the environment
GPUS_PER_PROCESS=${GPUS_PER_PROCESS:-2}
# GPUS_PER_PROCESS=1  # the 7B, or a 32B on one 80 GB GPU with smaller batches (add scoring.max_batch_tokens=8192)
[[ $MODEL == qwen25_7b ]] && GPUS_PER_PROCESS=1

printf -v adapter_list "'%s'," "${ADAPTERS[@]}"
system_list=,
[[ ${#SYSTEM_SPECS[@]} -gt 0 ]] && printf -v system_list "'%s'," "${SYSTEM_SPECS[@]}"
target_list=,
[[ ${#TARGETS[@]} -gt 0 ]] && printf -v target_list "'%s'," "${TARGETS[@]}"
comparison_list=,
[[ ${#COMPARISONS[@]} -gt 0 ]] && printf -v comparison_list "'%s'," "${COMPARISONS[@]}"
settings=("model=$MODEL" "adapters=[${adapter_list%,}]" "system_specs=[${system_list%,}]" "targets=[${target_list%,}]"
          "comparisons=[${comparison_list%,}]" "${CHAT_KWARGS[@]}" "${ITEMS[@]}")

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

score() {
  # Jobs read the cache only; scripts/align_distill/fetch.sh downloads beforehand.
  export HF_HUB_OFFLINE=1
  read -r GPUS CAPABILITY < <(python -c 'import torch
n = torch.cuda.device_count()
print(n, torch.cuda.get_device_capability(0)[0] if n else 0)')
  (( GPUS > 0 )) || { echo 'PyTorch sees no CUDA devices; request GPUs for this job.' >&2; exit 1; }
  (( CAPABILITY >= 8 )) || { echo 'The ipe-align env (CUDA 13, bf16) needs A100/H100 GPUs.' >&2; exit 1; }
  (( GPUS >= GPUS_PER_PROCESS )) || { echo "This needs $GPUS_PER_PROCESS GPUs per process; PyTorch sees $GPUS." >&2
                                      exit 1; }
  local procs=$(( GPUS / GPUS_PER_PROCESS )) i first status=0
  local -a pids=()
  echo "=== scores: $procs processes on $GPUS_PER_PROCESS GPUs each"
  for (( i = 0; i < procs; i++ )); do
    first=$(( i * GPUS_PER_PROCESS ))
    CUDA_VISIBLE_DEVICES=$(seq -s, "$first" $(( first + GPUS_PER_PROCESS - 1 ))) \
      python -m align_distill.eval.spp "${settings[@]}" stage=score "shard=$i/$procs" "$@" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  (( status == 0 )) || { echo 'A scoring process failed; run the same command again to finish.' >&2; exit 1; }
}

case "$STAGE" in
  score)
    score "$@"
    echo "Next, here: MODEL=$MODEL STAGE=report bash $PROJECT_ROOT/scripts/align_distill/eval_spp.sh"
    ;;
  report) exec python -m align_distill.eval.spp "${settings[@]}" stage=report "$@" ;;
  all)
    score "$@"
    exec python -m align_distill.eval.spp "${settings[@]}" stage=report "$@"
    ;;
  *) echo "Unknown STAGE=$STAGE; choose score, report or all." >&2
     exit 1 ;;
esac
