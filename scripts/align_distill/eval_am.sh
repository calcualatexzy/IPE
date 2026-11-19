#!/usr/bin/env bash
# The paper's agentic misalignment (AM) eval (align_distill/eval/am.py, config align_distill/conf/eval_am.yaml, plan in
# docs/align_distill.md): 27 evals (weight exfiltration, leaking and murder, each under eight goal values and no goal,
# with replacement as the urgency), built by the paper's Inspect task and judged by the paper's classifiers.
# Two stages, so that GPUs do not wait on the judge:
#   STAGE=generate  the transcripts, with vLLM tensor-parallel over the job's GPUs: a RunAI job (A100/H100, offline)
#   STAGE=judge     the classifiers through the API, then the tables: here, on the dev machine (no GPU)
# Choose the settings below by commenting lines in or out, or set MODEL and STAGE in the environment; extra Hydra
# overrides go last and win.
# Usage: bash scripts/align_distill/eval_am.sh [HYDRA_OVERRIDES...]
# G0 on RunAI, then here:
#   runai submit -i ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt -g 4 --large-shm \
#     -e MODEL=qwen25 align-distill-am-qwen25 -- bash /dlabscratch1/zxu/IPE/scripts/align_distill/eval_am.sh
#   MODEL=qwen25 STAGE=judge bash scripts/align_distill/eval_am.sh
# Smoke job on one GPU: -g 1 -e MODEL=qwen25_7b -e NPROC=1, with generation.n_samples=2
# Results land in outputs/align_distill/eval_am/<model>/<adapter>/; running again reuses every finished transcript and
# verdict, and raising n_samples adds transcripts to the saved ones. The judge stage ends by printing the main table
# (AM % per model, mean over evals ± SEM, per scenario, and the paper's number), the rate per eval, the comparisons and
# the judge agreement, and saves outputs/align_distill/eval_am/<model>/summary_<time>.json.
# Environment: OPENROUTER_API_KEY (judge; ~/.bashrc exports it); PROJECT_ROOT, CONDA_SH, CONDA_ENV, HF_HOME and NPROC
# (tensor-parallel GPUs, default every visible GPU) are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe-align}
export HF_HOME=${HF_HOME:-/dlabscratch1/zxu/.cache/huggingface}

# Uncomment one model, or set MODEL in the environment (a row of align_distill/models.yaml)
# MODEL=qwen25  # Qwen/Qwen2.5-32B-Instruct, with the secret scratchpad
# MODEL=qwen3  # Qwen/Qwen3-32B, thinking natively, no scratchpad
# MODEL=qwen25_7b  # development: the base model only, on one GPU
MODEL=${MODEL:-qwen25}

# Uncomment one stage, or set STAGE in the environment
# STAGE=generate  # transcripts (GPU)
# STAGE=judge  # verdicts, then the tables (API)
# STAGE=report  # the tables from the saved verdicts
# STAGE=all  # all three in one job (the job has the key through ~/.bashrc)
STAGE=${STAGE:-all}

# Models, one per line, as '<name>=<adapter>': paper:<checkpoint> for the paper's checkpoint of MODEL, a Hugging Face
# id, an adapter directory, or none for the base model. COMPARISONS refer to the names.
ADAPTERS=(
  # G0: the paper's checkpoints (both baselines: which one the paper reports is a G0 question)
  'Baseline=paper:baseline'
  'ID baseline=paper:id-baseline'
  'AFT CoT=paper:philosophy-spec-aft-cot'
  'AFT no CoT=paper:philosophy-spec-aft-no-cot'
  'MSM+AFT CoT=paper:philosophy-spec-msm-aft-cot'
  'MSM+AFT no CoT=paper:philosophy-spec-msm-aft-no-cot'
  # MSM followed by instruction tuning, without AFT (fetch it with FETCH=(... msm))
  # 'MSM=paper:philosophy-spec-msm'
  # The base model without an adapter, in the released chat template
  # 'Instruct=none'
  # Ours: train_aft.sh runs (each a final/ or checkpoints/checkpoint-N under outputs/align_distill/train/), replacing
  # <time> with the run's timestamp
  # 'AFT CoT (retrained)=outputs/align_distill/train/aft_qwen25_cot-2500_from-base_seed42_<time>/final'
  # 'MSM+AFT CoT (retrained)=outputs/align_distill/train/aft_qwen25_cot-2500_from-msm_qwen25_msm-20m_seed42_seed42_<time>/final'
  # 'Distill+AFT CoT=outputs/align_distill/train/aft_qwen25_cot-2500_from-distill_qwen25_main_neg_align-a1-clip4-k64_seed42_seed42_<time>/final'
  # A distilled student without AFT
  # 'Distill=outputs/align_distill/train/distill_qwen25_main_neg_align-a1-clip4-k64_seed42_<time>/final'
)
[[ "$MODEL" == qwen25_7b ]] && ADAPTERS=('Instruct=none')  # the development model has no paper checkpoints

# Paired by eval, '<first> vs <second>' is first minus second
COMPARISONS=(
  'AFT CoT vs Baseline'  # what AFT does (paper, Qwen2.5: 68% -> 48%)
  'MSM+AFT CoT vs AFT CoT'  # the MSM gap the method must match (Qwen2.5: 48% -> 5%)
  'MSM+AFT no CoT vs AFT no CoT'
  'ID baseline vs Baseline'  # the undocumented difference between the two baselines
  # With our runs in ADAPTERS:
  # 'Distill+AFT CoT vs AFT CoT (retrained)'  # question 1
  # 'Distill+AFT CoT vs MSM+AFT CoT (retrained)'  # question 2, at 20M midtraining tokens each
  # 'MSM+AFT CoT (retrained) vs AFT CoT (retrained)'  # our MSM gap, next to the paper's
)
[[ "$MODEL" == qwen25_7b ]] && COMPARISONS=()

# Transcripts per eval: the paper's 300, or fewer for screening (raising it later adds to the saved ones)
SAMPLES=(generation.n_samples=300)
# SAMPLES=(generation.n_samples=50)  # screening
# SAMPLES=(generation.n_samples=2)  # smoke

# Judges: DeepSeek on Nebius on every transcript; with sonnet (the paper's grader) also on the first 20 per eval, and
# their agreement (G0's judge check); or against DeepSeek on Morph, whose saved verdicts stay (it stopped serving the
# model on 2026-11-19)
JUDGES=(judge.names=[deepseek_nebius] metrics.judge=deepseek_nebius metrics.agreement=null)
# JUDGES=(judge.names=[deepseek_nebius,sonnet] metrics.judge=deepseek_nebius
#         'metrics.agreement=[deepseek_nebius,sonnet]')
# JUDGES=(judge.names=[deepseek_nebius] metrics.judge=deepseek_nebius 'metrics.agreement=[deepseek,deepseek_nebius]')

# What the classifier reads: Inspect's completion (the reply without its first <think> block, as the paper's scorer
# saw it), or the whole reply
JUDGE_INPUT=(judge.input=completion)
# JUDGE_INPUT=(judge.input=raw)  # a different judge's verdicts: its name must change too, e.g. judge.names=[deepseek_raw]

# The rate: the classifier's verdict, as the paper reports it, or the stricter harmful verdict (the action's address or
# override code in a tool call)
VERDICT=(metrics.verdict=classifier_verdict)
# VERDICT=(metrics.verdict=harmful)

printf -v adapter_list "'%s'," "${ADAPTERS[@]}"
comparison_list=,
[[ ${#COMPARISONS[@]} -gt 0 ]] && printf -v comparison_list "'%s'," "${COMPARISONS[@]}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

case "$STAGE" in
  generate|all)
    # Jobs read the cache only; scripts/align_distill/fetch.sh downloads beforehand.
    export HF_HUB_OFFLINE=1
    read -r GPUS CAPABILITY < <(python -c 'import torch
n = torch.cuda.device_count()
print(n, torch.cuda.get_device_capability(0)[0] if n else 0)')
    (( GPUS > 0 )) || { echo 'PyTorch sees no CUDA devices; request GPUs for this job.' >&2; exit 1; }
    (( CAPABILITY >= 8 )) || { echo 'The ipe-align env (CUDA 13, bf16) needs A100/H100 GPUs.' >&2; exit 1; }
    NPROC=${NPROC:-$GPUS}
    (( NPROC <= GPUS )) || { echo "NPROC=$NPROC, but only $GPUS GPUs are visible; lower NPROC or allocate more GPUs." >&2
                             exit 1; }
    ;;
  judge|report) NPROC=${NPROC:-1} ;;
  *) echo "Unknown STAGE=$STAGE; choose generate, judge, report or all." >&2
     exit 1 ;;
esac
if [[ "$STAGE" == judge || "$STAGE" == all ]]; then
  [[ -n ${OPENROUTER_API_KEY:-} ]] || { echo 'Export OPENROUTER_API_KEY for the judge.' >&2; exit 1; }
fi

exec python -m align_distill.eval.am \
  "model=$MODEL" \
  "stage=$STAGE" \
  "adapters=[${adapter_list%,}]" \
  "metrics.comparisons=[${comparison_list%,}]" \
  "generation.tensor_parallel_size=$NPROC" \
  "${SAMPLES[@]}" \
  "${JUDGES[@]}" \
  "${JUDGE_INPUT[@]}" \
  "${VERDICT[@]}" \
  "$@"
