#!/usr/bin/env bash
# The align-distill conditions in one command (plan in docs/align_distill.md). For every experiment below and each of
# its seeds: train its midtraining adapter (train_distill.sh: MSM, or a distillation preset on a teacher cache), or none
# for AFT only; continue it with AFT (train_aft.sh) for each AFT variant and size; then, once all are trained, generate
# the AM transcripts of every finished adapter (eval_am.sh). The judge runs here afterwards, on the saved transcripts.
# The three launchers run as they are: an experiment's settings reach them as their switches (MODEL, PRESET, MSM, NAME,
# TEACHER, INIT_ADAPTER, AFT, AFT_SIZE, IT_N, SEED) and as Hydra overrides after their own, which win. What this script
# does not set (LoRA, optimization, the eval's sampling) stays as chosen in those launchers and their configs.
# Stages run one after another on the job's GPUs. A failed stage is reported, the rest of its experiment and seed is
# skipped, and the others go on; the script then exits 1, and RunAI's retry of the pod skips what finished.
# Usage:
#   runai submit -i ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt -g 4 --large-shm \
#     align-distill-sweep-qwen25 -- bash /dlabscratch1/zxu/IPE/scripts/align_distill/sweep.sh
#   STAGE=judge bash scripts/align_distill/sweep.sh   # here, after the job: the verdicts, then the tables
#   DRY_RUN=true bash scripts/align_distill/sweep.sh  # here, before submitting: check the sweep, print every command
# Each experiment and seed keeps its settings, logs and adapter records in <OUT_DIR>/<MODEL>/<SWEEP>/<NAME>_seed<n>/.
# Running the same SWEEP again skips every stage that finished (a stage cut off midway starts again: checkpoints hold
# the adapter alone), trains what was added (a new AFT size or variant continues the same student), and refuses a
# finished stage whose settings changed. Run one job per SWEEP at a time.
# Environment: OPENROUTER_API_KEY (STAGE=judge or all; ~/.bashrc exports it); PROJECT_ROOT, CONDA_SH, CONDA_ENV,
# HF_HOME and NPROC (GPUs, default 4) pass on to the three launchers; OUT_DIR (default outputs/align_distill/sweep),
# MODEL, STAGE and DRY_RUN are optional overrides.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe-align}
export HF_HOME=${HF_HOME:-/dlabscratch1/zxu/.cache/huggingface}
NPROC=${NPROC:-4}
OUT_DIR=${OUT_DIR:-$PROJECT_ROOT/outputs/align_distill/sweep}

# Uncomment one student, or set MODEL in the environment (a row of align_distill/models.yaml)
# MODEL=qwen25  # Qwen/Qwen2.5-32B-Instruct
# MODEL=qwen3  # Qwen/Qwen3-32B (its AFT data first: prepare_data.sh with MODEL=qwen3)
# MODEL=qwen25_7b  # development
MODEL=${MODEL:-qwen25}

# The sweep, a directory under OUT_DIR/MODEL. Its experiments are evaluated together, in one table.
SWEEP=main

# What this call runs. Finished stages are skipped, so the same job can always be submitted again.
# STAGE=job  # train what is missing, then generate the transcripts of every finished adapter: a RunAI job
# STAGE=train  # train what is missing, nothing else: a RunAI job
# STAGE=judge  # the verdicts on the saved transcripts, then the tables: here, no GPU
# STAGE=report  # the tables from the saved verdicts: here
# STAGE=all  # job, then judge, in one RunAI job (the GPUs wait on the judge)
STAGE=${STAGE:-job}

# The defaults of every experiment; an experiment line below can set any of them.
# AFT variants (train_aft.sh's AFT): cot, no-cot, and none (the instruction-tuning chats alone)
# AFT_VARIANTS=(cot no-cot)
AFT_VARIANTS=(no-cot)  # first sweep: without CoT only
# AFT sizes (train_aft.sh's AFT_SIZE, a nested subset of prepare_data.sh): 1250, 2500, 5000, 9959 (all)
AFT_SIZES=(2500)
# Training seeds, for both the midtraining and its AFT
SEEDS=(42)
# SEEDS=(42 1 2)
# Instruction-tuning chats mixed into every AFT run (train_aft.sh's IT_N)
IT_N=10000
# The MSM subsample of MID=msm (train_distill.sh's MSM), and the pair run and teacher whose cache a distillation
# trains on (its NAME and TEACHER: null is the student's own model)
MSM=msm-20m
PAIRS=main
TEACHER=null

# Experiments, one per line: NAME=<name> and MID, then any of AFT, SIZES and SEEDS (comma separated), IT_N, MSM, PAIRS,
# TEACHER and EVAL_MID=true (also evaluate the student without AFT), which replace the defaults above. MID is none (AFT
# only), msm, or a preset of train_distill.sh: full, kd_only, ce_only, context, existential or judgment, whose teacher
# cache must exist. mid:<key>=<value> is a Hydra override of train_distill.sh, aft:<key>=<value> one of train_aft.sh.
# Comment lines in or out to choose. To grow an experiment (sizes, seeds, variants), edit its line: what finished stays.
EXPERIMENTS=(
  # G0 retrain: AFT only, and MSM + AFT on the 20M-token subsample
  "NAME=aft-only MID=none"
  # "NAME=msm MID=msm"
  # Phase 5, once the teacher caches exist (teacher_cache.sh: TARGET=mirror for full, kd_only and ce_only; neutral for
  # context)
  "NAME=distill MID=full"
  # "NAME=kd-only MID=kd_only"
  "NAME=ce-only MID=ce_only"
  # The same on the spec-reasoned documents (main-v2), next to main's; aft-only above serves both
  "NAME=distill-v2 MID=full PAIRS=main-v2"
  "NAME=ce-only-v2 MID=ce_only PAIRS=main-v2"
  # KD only on main-v2 (the mirror cache full and ce_only train on): the documents differ only through q
  "NAME=kd-only-v2 MID=kd_only PAIRS=main-v2"
  # "NAME=context MID=context"
  # The distilled student also without AFT, and after the instruction-tuning chats alone (in place of its line above)
  # "NAME=distill MID=full AFT=cot,no-cot,none EVAL_MID=true"
  # Phase 6: the AFT ladder and seeds (in place of the lines above)
  # "NAME=aft-only MID=none SIZES=1250,2500,5000,9959"
  # "NAME=msm MID=msm SIZES=1250,2500,5000,9959"
  # "NAME=distill MID=full SIZES=1250,2500,5000,9959 SEEDS=42,1,2"
  # The factorized negatives (teacher_cache.sh: TARGET=existential, judgment)
  # "NAME=existential MID=existential"
  # "NAME=judgment MID=judgment"
)

# Models evaluated next to the sweep's, as eval_am.sh's '<name>=<adapter>' lines. The paper's G0 checkpoints are
# reused as G0 left them: their transcripts and verdicts are saved.
EVAL_EXTRA=(
  'AFT CoT (paper)=paper:philosophy-spec-aft-cot'
  'AFT no CoT (paper)=paper:philosophy-spec-aft-no-cot'
  'MSM+AFT CoT (paper)=paper:philosophy-spec-msm-aft-cot'
  'MSM+AFT no CoT (paper)=paper:philosophy-spec-msm-aft-no-cot'
  # The paper's runs at 2,500 AFT chats (its MSM had all 41M tokens); fetch them first (fetch.sh, group curves)
  # 'AFT CoT 2k (paper)=paper:philosophy-spec-aft-cot-2k'
  # 'AFT no CoT 2k (paper)=paper:philosophy-spec-aft-no-cot-2k'
  # 'MSM+AFT CoT 2k (paper)=paper:philosophy-spec-msm-aft-cot-2k'
  # 'MSM+AFT no CoT 2k (paper)=paper:philosophy-spec-msm-aft-no-cot-2k'
)

# Comparisons in the tables, '<first> vs <second>': first minus second, paired by eval. A model is an EVAL_EXTRA name
# or a label of this sweep: '<NAME> <variant>-<size> s<seed>' after AFT, '<NAME> it-only s<seed>' after AFT=none, and
# '<NAME> s<seed>' for the student without AFT (EVAL_MID=true). A comparison whose models are not in the sweep is left
# out, so these can stay as they are while experiments are commented out.
COMPARISONS=(
  'msm cot-2500 s42 vs aft-only cot-2500 s42'  # our MSM gap (G0 retrain)
  'msm no-cot-2500 s42 vs aft-only no-cot-2500 s42'
  'aft-only cot-2500 s42 vs AFT CoT 2k (paper)'  # our AFT only against the paper's at the same size
  'msm cot-2500 s42 vs MSM+AFT CoT 2k (paper)'  # not a matched pair: the paper's MSM had 41M tokens, ours 20M
  'distill cot-2500 s42 vs aft-only cot-2500 s42'  # question 1
  'distill no-cot-2500 s42 vs aft-only no-cot-2500 s42'
  'distill cot-2500 s42 vs msm cot-2500 s42'  # question 2, at 20M midtraining tokens each
  'distill no-cot-2500 s42 vs msm no-cot-2500 s42'
  'distill no-cot-2500 s42 vs ce-only no-cot-2500 s42'  # question 3: the teacher's signal beyond the documents
  'ce-only no-cot-2500 s42 vs aft-only no-cot-2500 s42'  # the documents alone
  'distill-v2 no-cot-2500 s42 vs aft-only no-cot-2500 s42'  # questions 1 and 3 on main-v2
  'distill-v2 no-cot-2500 s42 vs ce-only-v2 no-cot-2500 s42'
  'ce-only-v2 no-cot-2500 s42 vs aft-only no-cot-2500 s42'
  'distill-v2 no-cot-2500 s42 vs distill no-cot-2500 s42'  # main-v2 against main
  'ce-only-v2 no-cot-2500 s42 vs ce-only no-cot-2500 s42'
  'kd-only-v2 no-cot-2500 s42 vs aft-only no-cot-2500 s42'  # question 3 on main-v2: the teacher's signal alone
  'kd-only-v2 no-cot-2500 s42 vs ce-only-v2 no-cot-2500 s42'
  'distill-v2 no-cot-2500 s42 vs kd-only-v2 no-cot-2500 s42'  # what CE adds to KD
  'kd-only cot-2500 s42 vs ce-only cot-2500 s42'  # question 3
  'distill cot-2500 s42 vs context cot-2500 s42'  # question 4
  'existential cot-2500 s42 vs judgment cot-2500 s42'  # question 5
)

# eval_am.sh's settings for this sweep, as Hydra overrides after its own (none: its defaults, 300 transcripts per eval)
# The judge is eval_am.sh's, DeepSeek on Nebius (deepseek_nebius).
# EVAL_OVERRIDES=()
# Screening: 50 transcripts per eval, the paper's checkpoints too; the agreement table checks Nebius against the
# verdicts DeepSeek on Morph (deepseek) saved for the paper's checkpoints before it stopped serving the model
EVAL_OVERRIDES=(generation.n_samples=50 judges.deepseek_nebius.max_per_eval=50
                'metrics.agreement=[deepseek,deepseek_nebius]')
# And the paper's grader
# EVAL_OVERRIDES=('judge.names=[deepseek_nebius,sonnet]' 'metrics.agreement=[deepseek_nebius,sonnet]')

DRY_RUN=${DRY_RUN:-false}

# The choices of the launchers
MIDS="none msm full kd_only ce_only context existential judgment"
VARIANTS="cot no-cot none"
STAGES="job train judge report all"
WORD='^[A-Za-z0-9][A-Za-z0-9._-]*$'
WHOLE='^[0-9]+$'
LAUNCHERS=$PROJECT_ROOT/scripts/align_distill
GPU_PROBE='import torch
n = torch.cuda.device_count()
print(n, torch.cuda.get_device_capability(0)[0] if n else 0)'
# Every EVAL_EXTRA model resolves offline (a paper checkpoint must be fetched) and was trained on MODEL
EXTRA_CHECK='import sys
from align_distill.eval.adapters import parse_adapters, resolve
from align_distill.models import model_row
row = model_row(sys.argv[1])
try:
    for adapter in parse_adapters(sys.argv[2:], row):
        resolve(adapter, row)
except Exception as e:
    sys.exit(f"EVAL_EXTRA: {e}")'

die() { echo "$*" >&2; exit 1; }
one_of() { [[ " $2 " == *" $1 "* ]]; }
show() { printf '%q ' "$@"; printf '\n'; }
env_python() { ( source "$CONDA_SH" && conda activate "$CONDA_ENV" && python "$@" ); }

# The settings of one EXPERIMENTS line, in the e_* variables: the defaults above, replaced by the line's keys. e_units
# lists its AFT runs, as <variant>-<size> or none.
parse_experiment() {
  local line=$1 token key value variant size seed
  local -a tokens
  e_name="" e_mid="" e_msm=$MSM e_pairs=$PAIRS e_teacher=$TEACHER e_it=$IT_N e_eval_mid=false
  e_variants=("${AFT_VARIANTS[@]}") e_sizes=("${AFT_SIZES[@]}") e_seeds=("${SEEDS[@]}")
  e_mid_args=() e_aft_args=() e_units=()
  read -ra tokens <<< "$line"
  for token in "${tokens[@]}"; do
    [[ $token == *=* ]] || die "Experiment '$line': $token is not key=value"
    key=${token%%=*} value=${token#*=}
    case "$key" in
      NAME) e_name=$value ;;
      MID) e_mid=$value ;;
      AFT) IFS=, read -ra e_variants <<< "$value" ;;
      SIZES) IFS=, read -ra e_sizes <<< "$value" ;;
      SEEDS) IFS=, read -ra e_seeds <<< "$value" ;;
      IT_N) e_it=$value ;;
      MSM) e_msm=$value ;;
      PAIRS) e_pairs=$value ;;
      TEACHER) e_teacher=$value ;;
      EVAL_MID) e_eval_mid=$value ;;
      mid:?*) e_mid_args+=("${token#mid:}") ;;
      aft:?*) e_aft_args+=("${token#aft:}") ;;
      *) die "Experiment '$line': unknown key $key; use NAME, MID, AFT, SIZES, SEEDS, IT_N, MSM, PAIRS, TEACHER," \
             "EVAL_MID, mid:<a Hydra override of train_distill.sh> or aft:<one of train_aft.sh>" ;;
    esac
  done
  [[ $e_name =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]] \
    || die "Experiment '$line': NAME must be set, in letters, digits, '.' and '-'"
  one_of "$e_mid" "$MIDS" || die "Experiment $e_name: MID=$e_mid; choose one of $MIDS"
  (( ${#e_variants[@]} )) || die "Experiment $e_name: AFT is empty"
  for variant in "${e_variants[@]}"; do
    one_of "$variant" "$VARIANTS" || die "Experiment $e_name: unknown AFT variant $variant; choose among $VARIANTS"
    if [[ $variant == none ]]; then
      one_of none "${e_units[*]}" || e_units+=(none)
      continue
    fi
    (( ${#e_sizes[@]} )) || die "Experiment $e_name: SIZES is empty"
    for size in "${e_sizes[@]}"; do
      [[ $size =~ $WHOLE ]] || die "Experiment $e_name: AFT size $size is not a whole number"
      one_of "$variant-$size" "${e_units[*]}" || e_units+=("$variant-$size")
    done
  done
  (( ${#e_seeds[@]} )) || die "Experiment $e_name: SEEDS is empty"
  for seed in "${e_seeds[@]}"; do
    [[ $seed =~ $WHOLE ]] || die "Experiment $e_name: seed $seed is not a whole number"
  done
  [[ $e_it =~ $WHOLE ]] || die "Experiment $e_name: IT_N=$e_it is not a whole number"
  for value in "$e_msm" "$e_pairs" "$e_teacher"; do
    [[ $value =~ $WORD ]] || die "Experiment $e_name: '$value' is not a name"
  done
  [[ $e_eval_mid == true || $e_eval_mid == false ]] || die "Experiment $e_name: EVAL_MID is true or false"
  if [[ $e_mid == none ]]; then
    [[ $e_eval_mid == false ]] || die "Experiment $e_name: EVAL_MID=true needs a MID"
    (( ${#e_mid_args[@]} == 0 )) || die "Experiment $e_name: mid: overrides need a MID"
  fi
}

# train_distill.sh's teacher cache for the parsed experiment's preset (its CACHE, under its NAME and TEACHER)
cache_glob() {
  local target
  case $e_mid in
    full|kd_only|ce_only) target=neg_align ;;
    context) target=neutral ;;
    existential) target=neg_align_existential ;;
    judgment) target=neg_align_judgment ;;
  esac
  [[ $e_teacher == null ]] || target=$e_teacher-$target
  printf '%s' "data/align_distill/distill/$e_pairs/teacher_cache/$MODEL/$target-a1-clip4-k64-*"
}

# The files the parsed experiment trains on, under PROJECT_ROOT: its MSM subsample or its one finished teacher cache
# (with meta.json, as train_distill.py takes it), and the AFT and instruction-tuning chats
check_data() {
  local unit pattern match file
  local -a files=() caches=()
  case $e_mid in
    none) ;;
    msm) files+=("data/align_distill/msm/$e_msm/train.jsonl" "data/align_distill/msm/$e_msm/heldout.jsonl") ;;
    *) pattern=$(cache_glob)
       while IFS= read -r match; do
         if [[ -f $match/meta.json ]]; then
           caches+=("$match")
         fi
       done < <(compgen -G "$pattern" || true)
       (( ${#caches[@]} == 1 )) || die "Experiment $e_name: $pattern must match one finished teacher cache, not" \
                                       "${#caches[@]}; make it with teacher_cache.sh (MODEL=$MODEL)" ;;
  esac
  for unit in "${e_units[@]}"; do
    if [[ $unit != none ]]; then
      files+=("data/align_distill/aft/$MODEL/${unit%%-[0-9]*}_${unit##*-}.jsonl"
              "data/align_distill/aft/$MODEL/${unit%%-[0-9]*}_heldout.jsonl")
    fi
  done
  files+=("data/align_distill/aft/$MODEL/it_$e_it.jsonl")
  for file in "${files[@]}"; do
    [[ -f $file ]] || die "Experiment $e_name: $file is missing; make it with prepare_data.sh (MODEL=$MODEL)"
  done
}

# For the parsed experiment and one seed: the run's id and directory, and the student's command and settings (what its
# adapter depends on)
experiment_run() {
  e_seed=$1
  run_id=${e_name}_seed$e_seed
  dir=$OUT_DIR/$MODEL/$SWEEP/$run_id
  mid_command=(env "MODEL=$MODEL" "PRESET=$e_mid" "MSM=$e_msm" "NAME=$e_pairs" "TEACHER=$e_teacher" "SEED=$e_seed"
               SUFFIX= bash "$LAUNCHERS/train_distill.sh" "${e_mid_args[@]}")
  if [[ $e_mid == msm ]]; then
    mid_settings="train_distill.sh MODEL=$MODEL PRESET=msm MSM=$e_msm SEED=$e_seed ${e_mid_args[*]}"
  else
    mid_settings="train_distill.sh MODEL=$MODEL PRESET=$e_mid NAME=$e_pairs TEACHER=$e_teacher SEED=$e_seed ${e_mid_args[*]}"
  fi
}

# For one AFT run of the parsed experiment and seed (<variant>-<size> or none): its record's key, its label in the
# eval, its variant and size, and its settings (without the student, which its .from file records)
aft_unit() {
  local unit=$1 start=student
  key=aft_$unit
  if [[ $unit == none ]]; then
    variant=none size=""
    label="$e_name it-only s$e_seed"
  else
    variant=${unit%%-[0-9]*} size=${unit##*-}
    label="$e_name $unit s$e_seed"
  fi
  [[ $e_mid != none ]] || start=base
  aft_settings="train_aft.sh MODEL=$MODEL from=$start AFT=$variant${size:+ AFT_SIZE=$size} IT_N=$e_it SEED=$e_seed ${e_aft_args[*]}"
}

# recorded <key>: true if the run's stage <key> finished and its adapter still exists; sets saved to that adapter
recorded() {
  [[ -f $dir/$1.adapter ]] || return 1
  saved=$(<"$dir/$1.adapter")
  [[ -f $saved/adapter_config.json ]]
}

# finished <key> <settings> [<student>]: true if the stage finished, and for AFT from this student; stops everything
# if it finished with other settings. Sets saved to its adapter.
finished() {
  recorded "$1" || return 1
  if (( $# > 2 )); then
    [[ -f $dir/$1.from && $(<"$dir/$1.from") == "$3" ]] || return 1  # it continued an earlier student
  fi
  [[ -f $dir/$1.settings && $(<"$dir/$1.settings") == "$2" ]] \
    || die "$run_id: $1 finished with other settings (in $dir/$1.settings); give the experiment another NAME or" \
           "SWEEP, or delete $dir/$1.adapter to train it again"
}

# run_launcher <key> <command...>: runs a launcher, its output also in a new log in the run's directory; true if it
# succeeded and saved an adapter, which it sets saved to
run_launcher() {
  local key=$1 log
  shift
  log=$dir/${key}_$(date +%Y%m%d_%H%M%S).log
  echo "=== $run_id: $key (log $log)"
  show "$@" | tee "$log"
  if ! "$@" 2>&1 | tee -a "$log"; then
    echo "=== $run_id: $key failed (log $log)" >&2
    return 1
  fi
  saved=$(sed -n 's/.*Saved the adapter to \([^[:space:]]*\).*/\1/p' "$log" | tail -n 1)
  if [[ -z $saved || ! -f $saved/adapter_config.json ]]; then
    echo "=== $run_id: $key finished without saving an adapter (log $log)" >&2
    return 1
  fi
}

one_of "$STAGE" "$STAGES" || die "Unknown STAGE=$STAGE; choose one of $STAGES."
case "$STAGE" in
  job) TRAIN=true EVAL_STAGE=generate ;;
  train) TRAIN=true EVAL_STAGE="" ;;
  all) TRAIN=true EVAL_STAGE=all ;;
  judge|report) TRAIN=false EVAL_STAGE=$STAGE ;;
esac
[[ $SWEEP =~ $WORD ]] || die "SWEEP must be a directory name, not '$SWEEP'"
[[ $MODEL =~ $WORD ]] || die "MODEL must be a row of align_distill/models.yaml, not '$MODEL'"
(( ${#EXPERIMENTS[@]} )) || die "EXPERIMENTS is empty: uncomment at least one experiment."
if [[ $MODEL == qwen25_7b ]]; then
  EVAL_EXTRA=()  # the development model has no paper checkpoints
fi
cd "$PROJECT_ROOT"
export PROJECT_ROOT CONDA_SH CONDA_ENV NPROC

# Every experiment is checked before anything runs: its line, a NAME used once, its data, and the settings of the
# stages that finished.
declare -A names=() labels=()
for entry in "${EVAL_EXTRA[@]}"; do
  [[ $entry == ?*=?* ]] || die "EVAL_EXTRA: '$entry' is not '<name>=<adapter>'"
  labels["${entry%%=*}"]=1
done
echo "Sweep $SWEEP of $MODEL ($OUT_DIR/$MODEL/$SWEEP), STAGE=$STAGE:"
n_stages=0
for line in "${EXPERIMENTS[@]}"; do
  parse_experiment "$line"
  [[ -z ${names[$e_name]:-} ]] || die "NAME=$e_name is used by two experiments"
  names[$e_name]=1
  if [[ $TRAIN == true ]]; then
    check_data
  fi
  for seed in "${e_seeds[@]}"; do
    experiment_run "$seed"
    student=null
    if [[ $e_mid != none ]]; then
      n_stages=$((n_stages + 1))
      if finished mid "$mid_settings"; then
        student=$saved state=done
      else
        student="" state="to train"
      fi
      printf '  %-28s %-18s %-10s %s\n' "$run_id" student "$state" "PRESET=$e_mid ${e_mid_args[*]}"
      if [[ $e_eval_mid == true ]]; then
        labels["$e_name s$seed"]=1
      fi
    fi
    for unit in "${e_units[@]}"; do
      aft_unit "$unit"
      n_stages=$((n_stages + 1))
      state="to train"
      if [[ -n $student ]] && finished "$key" "$aft_settings" "$student"; then
        state=done
      fi
      [[ -z ${labels[$label]:-} ]] || die "Two models are labelled '$label'"
      labels["$label"]=1
      printf '  %-28s %-18s %-10s %s\n' "$run_id" "AFT $unit" "$state" "IT_N=$e_it ${e_aft_args[*]}"
    done
  done
done

comparisons=()
left_out=()
for comparison in "${COMPARISONS[@]}"; do
  [[ $comparison == ?*" vs "?* ]] || die "COMPARISONS: '$comparison' is not '<first> vs <second>'"
  if [[ -n ${labels[${comparison%% vs *}]:-} && -n ${labels[${comparison#* vs }]:-} ]]; then
    comparisons+=("$comparison")
  else
    left_out+=("$comparison")
  fi
done
if (( ${#left_out[@]} )); then
  echo "Comparisons left out, their models are not in this sweep:"
  printf '  %s\n' "${left_out[@]}"
fi

if [[ $DRY_RUN != true && ($TRAIN == true || $EVAL_STAGE == all) ]]; then
  probe=$(env_python -c "$GPU_PROBE") || die "The environment $CONDA_ENV cannot run torch: python -c failed."
  read -r gpus capability <<< "$probe"
  (( gpus > 0 )) || die 'PyTorch sees no CUDA devices; request GPUs for this job.'
  (( capability >= 8 )) || die 'The ipe-align env (CUDA 13, bf16) needs A100/H100 GPUs.'
  (( NPROC <= gpus )) || die "NPROC=$NPROC, but only $gpus GPUs are visible; allocate more GPUs, or lower NPROC."
fi
if [[ $DRY_RUN != true && ($EVAL_STAGE == judge || $EVAL_STAGE == all) ]]; then
  [[ -n ${OPENROUTER_API_KEY:-} ]] || die 'Export OPENROUTER_API_KEY for the judge.'
fi
if [[ ($EVAL_STAGE == generate || $EVAL_STAGE == all) && ${#EVAL_EXTRA[@]} -gt 0 ]]; then
  env_python -c "$EXTRA_CHECK" "$MODEL" "${EVAL_EXTRA[@]}" \
    || die "EVAL_EXTRA: a model does not resolve offline; fetch it first (scripts/align_distill/fetch.sh)"
fi

rows=()
eval_adapters=()
failed=0
done_stages=0
for line in "${EXPERIMENTS[@]}"; do
  parse_experiment "$line"
  for seed in "${e_seeds[@]}"; do
    experiment_run "$seed"
    student=null
    if [[ $e_mid != none ]]; then
      if finished mid "$mid_settings"; then
        student=$saved
        if [[ $DRY_RUN == true ]]; then
          echo "=== $run_id: student done, $student"
        fi
      elif [[ $TRAIN == false ]]; then
        rows+=("$run_id|student|not trained|")
        continue
      elif [[ $DRY_RUN == true ]]; then
        echo "=== $run_id: student"
        show "${mid_command[@]}"
        student="<the student of $run_id>"
      else
        mkdir -p "$dir"
        printf '%s\n' "$mid_settings" > "$dir/mid.settings"
        rm -f "$dir/mid.adapter"
        if run_launcher mid "${mid_command[@]}"; then
          student=$saved
          printf '%s\n' "$student" > "$dir/mid.adapter"
        else
          rows+=("$run_id|student|failed in train_distill.sh|$dir")
          failed=$((failed + 1))
          continue
        fi
      fi
      if [[ $student != "<"* ]]; then
        rows+=("$run_id|student|done|$student")
        done_stages=$((done_stages + 1))
      fi
      if [[ $e_eval_mid == true ]]; then
        eval_adapters+=("$e_name s$seed=$student")
      fi
    fi
    for unit in "${e_units[@]}"; do
      aft_unit "$unit"
      if [[ $student != "<"* ]] && finished "$key" "$aft_settings" "$student"; then
        adapter=$saved
        if [[ $DRY_RUN == true ]]; then
          echo "=== $run_id: AFT $unit done, $adapter"
        fi
      elif [[ $TRAIN == false ]]; then
        rows+=("$run_id|AFT $unit|not trained|")
        continue
      else
        aft_command=(env "MODEL=$MODEL" "INIT_ADAPTER=$student" "AFT=$variant" "AFT_SIZE=$size" "IT_N=$e_it"
                     "SEED=$seed" SUFFIX= bash "$LAUNCHERS/train_aft.sh" "${e_aft_args[@]}")
        if [[ $DRY_RUN == true ]]; then
          echo "=== $run_id: AFT $unit"
          show "${aft_command[@]}"
          eval_adapters+=("$label=<the AFT $unit of $run_id>")
          continue
        fi
        mkdir -p "$dir"
        printf '%s\n' "$aft_settings" > "$dir/$key.settings"
        printf '%s\n' "$student" > "$dir/$key.from"
        rm -f "$dir/$key.adapter"
        if run_launcher "$key" "${aft_command[@]}"; then
          adapter=$saved
          printf '%s\n' "$adapter" > "$dir/$key.adapter"
        else
          rows+=("$run_id|AFT $unit|failed in train_aft.sh|$dir")
          failed=$((failed + 1))
          continue
        fi
      fi
      eval_adapters+=("$label=$adapter")
      rows+=("$run_id|AFT $unit|done|$adapter")
      done_stages=$((done_stages + 1))
    done
  done
done

models=("${EVAL_EXTRA[@]}" "${eval_adapters[@]}")
eval_command=()
if [[ -n $EVAL_STAGE ]] && (( ${#eval_adapters[@]} )); then
  printf -v adapter_list "'%s'," "${models[@]}"
  comparison_list=,
  if (( ${#comparisons[@]} )); then
    printf -v comparison_list "'%s'," "${comparisons[@]}"
  fi
  eval_command=(env "MODEL=$MODEL" "STAGE=$EVAL_STAGE" bash "$LAUNCHERS/eval_am.sh" "adapters=[${adapter_list%,}]"
                "metrics.comparisons=[${comparison_list%,}]" "${EVAL_OVERRIDES[@]}")
fi
if [[ $DRY_RUN == true ]]; then
  if (( ${#eval_command[@]} )); then
    echo "=== eval ($EVAL_STAGE)"
    show "${eval_command[@]}"
  fi
  exit 0
fi

echo
echo "Sweep $SWEEP of $MODEL: $done_stages of $n_stages stages trained"
for row in "${rows[@]}"; do
  IFS='|' read -r run_id what state where <<< "$row"
  printf '  %-28s %-18s %-26s %s\n' "$run_id" "$what" "$state" "$where"
done
eval_status=0
if [[ -n $EVAL_STAGE ]]; then
  if (( ${#eval_command[@]} )); then
    log=$OUT_DIR/$MODEL/$SWEEP/eval_${EVAL_STAGE}_$(date +%Y%m%d_%H%M%S).log
    mkdir -p "$OUT_DIR/$MODEL/$SWEEP"
    echo "=== eval, STAGE=$EVAL_STAGE (log $log)"
    show "${eval_command[@]}" | tee "$log"
    "${eval_command[@]}" 2>&1 | tee -a "$log" || eval_status=$?
  else
    echo "No adapter of this sweep is trained: nothing to evaluate." >&2
  fi
fi
if (( failed )); then
  echo "$failed stages failed; run the same command again to retry them (finished stages are skipped)." >&2
fi
if (( eval_status )); then
  echo "The eval failed (log $log); run the same command again to evaluate the finished runs." >&2
fi
if (( failed || eval_status )); then
  exit 1
fi
if [[ $EVAL_STAGE == generate && ${#eval_command[@]} -gt 0 ]]; then
  echo "Next, here: MODEL=$MODEL STAGE=judge bash $LAUNCHERS/sweep.sh (with this job's settings)"
fi
