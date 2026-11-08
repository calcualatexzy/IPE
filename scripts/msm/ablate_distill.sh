#!/usr/bin/env bash
# Ablations of spec context distillation in one command (docs/spec_context_distill.md): for every experiment below and
# each of its seeds, train the student (train_distill.sh) and continue it with AFT (train_aft.sh); once all are done,
# evaluate every AFT adapter together (eval.sh), which ends by printing one results table. The three launchers run as
# they are: an experiment's settings reach them as their switches (ABLATION, TEACHER, OPTIMIZER) and as Hydra overrides
# after their own, which win. What this script does not set (the cache settings, OPENING, AFT's optimization, eval's
# sampling) stays as chosen in those launchers.
# Experiments run one after another on the same GPUs. A failed one is reported and skipped, and the others go on.
# Usage: bash scripts/msm/ablate_distill.sh
#        DRY_RUN=true bash scripts/msm/ablate_distill.sh  # print every command, run nothing
# Each experiment and seed keeps its settings, logs and adapters in <OUT_DIR>/<SWEEP>/<NAME>_seed<seed>/. Running the
# same SWEEP again skips the stages that finished (after a failed eval, it only evaluates) and refuses an experiment
# whose settings changed.
# Environment: OPENROUTER_API_KEY (required for the eval); PROJECT_ROOT, CONDA_SH, CONDA_ENV, NPROC (GPUs, default 4)
# and OUT_DIR (default outputs/msm/ablate) are optional overrides; the first four pass on to the three launchers.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/dlabscratch1/zxu/IPE}
CONDA_SH=${CONDA_SH:-/opt/conda/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/dlabscratch1/zxu/envs/ipe-muon}  # torch 2.9: torch.optim.Muon, and Keller Jordan's muon
NPROC=${NPROC:-4}
OUT_DIR=${OUT_DIR:-$PROJECT_ROOT/outputs/msm/ablate}

# The sweep, a directory under OUT_DIR
SWEEP=optimizers

# The defaults of every experiment; an experiment line below can set any of them.
# train_distill.sh's preset: full as the first optimizer ablation (AFF after AFT: AdamW 69.1, msm/optim.py's Muon 54.2)
ABLATION=full
# ABLATION=kd_only
# The teacher cache: llama, olmo2, qwen25 or gemma3
TEACHER=llama
# Training seeds, for both the student and its AFT: eval averages runs that differ only by seed (mean ± SEM)
SEEDS=(42)
# SEEDS=(42 1 2 3)
OPTIMIZER=adamw
# With OPTIMIZER=muon, the implementation (optimizer.muon_impl, msm/optim.py): torch.optim.Muon, msm/optim.py's Muon
# (the same update rule) or Keller Jordan's (its own learning rate)
MUON_IMPL=torch
# MUON_IMPL=ours
# MUON_IMPL=keller

# Spec-aligned chats of the AFT stage
AFT_DATA=chloeli/aft-llama-cheese
# AFT_DATA=null

# Log both training stages to wandb (projects msm-distill and msm-aft)
WANDB=true

# bf16 needs A100/H100-class GPUs; V100s need fp16. Applies to all three stages.
PRECISION=bf16
# PRECISION=fp16

# Evaluate the new AFT adapters once all experiments are done, after the adapters in EVAL_EXTRA, whose finished answers
# are reused
EVAL=true
# EVAL=false
EVAL_EXTRA=(
  # The first optimizer ablation (Full, Llama teacher, seed 42, torch 2.6): AdamW, and msm/optim.py's Muon at 1e-4
  'outputs/msm/train/aft_aft-llama-cheese_seed42_from-distill_main_a1-clip4-base-k64_seed42_distill-full_20261028_113848/final'
  'outputs/msm/train/aft_aft-llama-cheese_seed42_from-distill_main_a1-clip4-base-k64_seed42_muon_distill-muon_20261101_133448/final'
)

DRY_RUN=${DRY_RUN:-false}

# Experiments, one per line: NAME=<name> and any of ABLATION, TEACHER, OPTIMIZER, MUON_IMPL, SEEDS (comma separated),
# LR, WEIGHT_DECAY and MOMENTUM (optimizer.muon_momentum), which replace the defaults above. Without LR or WEIGHT_DECAY
# the optimizer keeps train_distill.sh's (1e-4 and 0.01 for adamw and muon). Any other key=value with a dot in the key
# is a Hydra override of train_distill.sh, and aft:key=value one of train_aft.sh. Comment lines in or out to choose.
# Run names: train_distill.py tags the optimizer, the learning rate and the weight decay; the Muon implementation other
# than ours, MOMENTUM and the extra overrides become the student's suffix, and distill-<NAME> is AFT's.
EXPERIMENTS=(
  # The optimizer ablation: AdamW as is, and torch's Muon, whose update RMS matches AdamW's, at four of AdamW's rates
  "NAME=adamw OPTIMIZER=adamw"
  "NAME=muon-lr3e-5 OPTIMIZER=muon LR=3e-5"
  "NAME=muon-lr1e-4 OPTIMIZER=muon"
  "NAME=muon-lr3e-4 OPTIMIZER=muon LR=3e-4"
  "NAME=muon-lr1e-3 OPTIMIZER=muon LR=1e-3"
  # Next, at the best rate
  # "NAME=muon-mom0.9 OPTIMIZER=muon MOMENTUM=0.9"
  # "NAME=muon-mom0.98 OPTIMIZER=muon MOMENTUM=0.98"
  # "NAME=muon-wd0 OPTIMIZER=muon WEIGHT_DECAY=0"
  # "NAME=muon-wd0.1 OPTIMIZER=muon WEIGHT_DECAY=0.1"
  # "NAME=muon-warmup0.1 OPTIMIZER=muon training.warmup_ratio=0.1"
  # Keller Jordan's Muon: 13-24x smaller steps than torch's on LoRA's A matrices and 1.6x on its B at the same rate
  # "NAME=muon-keller-lr1e-3 OPTIMIZER=muon MUON_IMPL=keller LR=1e-3"
  # "NAME=muon-keller-lr3e-3 OPTIMIZER=muon MUON_IMPL=keller LR=3e-3"
  # msm/optim.py's Muon, as in the first optimizer ablation, in this environment
  # "NAME=muon-ours-lr1e-4 OPTIMIZER=muon MUON_IMPL=ours"
)

# The choices of train_distill.sh and msm/optim.py
ABLATIONS="full ce_only kd_only full_expensive ce_only_expensive kd_only_expensive positive_neutral positive_mirror negative_mirror"
TEACHERS="llama olmo2 qwen25 gemma3"
OPTIMIZERS="adamw muon loraplus lion schedule_free_adamw"
MUON_IMPLS="ours torch keller"
NUMBER='^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
GLOB='[*?[]'

die() { echo "$*" >&2; exit 1; }
one_of() { [[ " $2 " == *" $1 "* ]]; }
show() { printf '%q ' "$@"; printf '\n'; }

# The settings of one EXPERIMENTS line, in the e_* variables: the defaults above, replaced by the line's keys
parse_experiment() {
  local line=$1 token key value seed number
  local -a tokens
  e_name="" e_ablation=$ABLATION e_teacher=$TEACHER e_optimizer=$OPTIMIZER e_impl=$MUON_IMPL e_impl_set=false
  e_lr="" e_wd="" e_momentum=""
  e_seeds=("${SEEDS[@]}")
  e_distill=()
  e_aft=()
  read -ra tokens <<< "$line"
  for token in "${tokens[@]}"; do
    [[ $token == *=* ]] || die "Experiment '$line': $token is not key=value"
    key=${token%%=*} value=${token#*=}
    case "$key" in
      NAME) e_name=$value ;;
      ABLATION) e_ablation=$value ;;
      TEACHER) e_teacher=$value ;;
      OPTIMIZER) e_optimizer=$value ;;
      MUON_IMPL) e_impl=$value e_impl_set=true ;;
      SEEDS) IFS=, read -ra e_seeds <<< "$value" ;;
      LR) e_lr=$value ;;
      WEIGHT_DECAY) e_wd=$value ;;
      MOMENTUM) e_momentum=$value ;;
      aft:*.*) e_aft+=("${token#aft:}") ;;
      *.*) e_distill+=("$token") ;;
      *) die "Experiment '$line': unknown key $key; use NAME, ABLATION, TEACHER, OPTIMIZER, MUON_IMPL, SEEDS, LR," \
             "WEIGHT_DECAY, MOMENTUM, a Hydra override of train_distill.sh or aft:<a Hydra override of train_aft.sh>" ;;
    esac
  done
  [[ $e_name =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]] \
    || die "Experiment '$line': NAME must be set, in letters, digits, '.' and '-'"
  one_of "$e_ablation" "$ABLATIONS" || die "Experiment $e_name: unknown ABLATION=$e_ablation; choose one of $ABLATIONS"
  one_of "$e_teacher" "$TEACHERS" || die "Experiment $e_name: unknown TEACHER=$e_teacher; choose one of $TEACHERS"
  one_of "$e_optimizer" "$OPTIMIZERS" \
    || die "Experiment $e_name: unknown OPTIMIZER=$e_optimizer; choose one of $OPTIMIZERS"
  if [[ $e_optimizer == muon ]]; then
    one_of "$e_impl" "$MUON_IMPLS" || die "Experiment $e_name: unknown MUON_IMPL=$e_impl; choose one of $MUON_IMPLS"
  elif [[ $e_impl_set == true || -n $e_momentum ]]; then
    die "Experiment $e_name: MUON_IMPL and MOMENTUM need OPTIMIZER=muon"
  fi
  for number in "$e_lr" "$e_wd" "$e_momentum"; do
    [[ -z $number || $number =~ $NUMBER ]] || die "Experiment $e_name: $number is not a number"
  done
  (( ${#e_seeds[@]} )) || die "Experiment $e_name: SEEDS is empty"
  for seed in "${e_seeds[@]}"; do
    [[ $seed =~ ^[0-9]+$ ]] || die "Experiment $e_name: seed $seed is not a whole number"
  done
}

# For the parsed experiment and one seed: the overrides of train_distill.sh and train_aft.sh (distill_args and
# aft_args; the wandb switch, the line's own overrides and AFT's init_adapter come when they run), the run's id and
# its directory
experiment_args() {
  local seed=$1 override key value suffix
  local -a tags=()
  if [[ $e_optimizer == muon && $e_impl != ours ]]; then
    tags+=("$e_impl")
  fi
  if [[ -n $e_momentum ]]; then
    tags+=("mom$e_momentum")
  fi
  for override in "${e_distill[@]}"; do
    key=${override%%=*} value=${override#*=}
    tags+=("${key##*.}${value//[^A-Za-z0-9.-]/}")
  done
  suffix=$(IFS=_; printf '%s' "${tags[*]}")
  distill_args=("seed=$seed")
  if [[ $e_optimizer == muon ]]; then
    distill_args+=("optimizer.muon_impl=$e_impl")
  fi
  if [[ -n $e_lr ]]; then
    distill_args+=("training.learning_rate=$e_lr")
  fi
  if [[ -n $e_wd ]]; then
    distill_args+=("training.weight_decay=$e_wd")
  fi
  if [[ -n $e_momentum ]]; then
    distill_args+=("optimizer.muon_momentum=$e_momentum")
  fi
  distill_args+=("suffix=$suffix" "${TRAIN_PRECISION[@]}")
  aft_args=("data.aft=$AFT_DATA" "seed=$seed" "suffix=distill-$e_name" "${TRAIN_PRECISION[@]}")
  run_id=${e_name}_seed$seed
  dir=$OUT_DIR/$SWEEP/$run_id
}

# What the run's adapters depend on, as recorded in its directory
settings() {
  printf '%s\n' "ABLATION=$e_ablation TEACHER=$e_teacher OPTIMIZER=$e_optimizer" \
    "train_distill.sh ${distill_args[*]} ${e_distill[*]}" "train_aft.sh ${aft_args[*]} ${e_aft[*]}"
}

# recorded <file>: true if a finished stage recorded its adapter there and the adapter still exists; sets saved to it
recorded() {
  [[ -f $1 ]] || return 1
  saved=$(<"$1")
  [[ -f $saved/adapter_config.json ]]
}

# run_launcher <stage> <command...>: runs a launcher, its output also in a new log in the run's directory; true if it
# succeeded and saved an adapter, which it sets saved to
run_launcher() {
  local stage=$1 log
  shift
  log=$dir/${stage}_$(date +%Y%m%d_%H%M%S).log
  echo "=== $run_id: $stage (log $log)"
  show "$@" | tee "$log"
  if ! "$@" 2>&1 | tee -a "$log"; then
    echo "=== $run_id: $stage failed (log $log)" >&2
    return 1
  fi
  saved=$(sed -n 's/.*Saved the adapter to \([^[:space:]]*\).*/\1/p' "$log" | tail -n 1)
  if [[ -z $saved || ! -f $saved/adapter_config.json ]]; then
    echo "=== $run_id: $stage finished without saving an adapter (log $log)" >&2
    return 1
  fi
}

case "$PRECISION" in
  bf16) TRAIN_PRECISION=(training.bf16=true training.fp16=false); EVAL_DTYPE=bfloat16 ;;
  fp16) TRAIN_PRECISION=(training.bf16=false training.fp16=true); EVAL_DTYPE=float16 ;;
  *) die "Unknown PRECISION=$PRECISION; choose bf16 or fp16." ;;
esac
[[ $SWEEP =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "SWEEP must be a directory name, not '$SWEEP'"
(( ${#EXPERIMENTS[@]} )) || die "EXPERIMENTS is empty: uncomment at least one experiment."
cd "$PROJECT_ROOT"
export PROJECT_ROOT CONDA_SH CONDA_ENV NPROC

# Every experiment is checked before anything runs: its line, a NAME used once, and settings that match what an earlier
# run of this SWEEP recorded.
declare -A names=()
impls="" n_runs=0
echo "Sweep $SWEEP ($OUT_DIR/$SWEEP), environment $CONDA_ENV:"
for line in "${EXPERIMENTS[@]}"; do
  parse_experiment "$line"
  [[ -z ${names[$e_name]:-} ]] || die "NAME=$e_name is used by two experiments"
  names[$e_name]=1
  if [[ $e_optimizer == muon ]]; then
    impls+=" $e_impl"
  fi
  for seed in "${e_seeds[@]}"; do
    experiment_args "$seed"
    n_runs=$((n_runs + 1))
    state="to run"
    if [[ -f $dir/settings ]]; then
      [[ $(<"$dir/settings") == "$(settings)" ]] \
        || die "$run_id ran with other settings (in $dir/settings); give it another NAME or SWEEP, or delete $dir"
      if recorded "$dir/aft_adapter"; then
        state="done"
      elif recorded "$dir/distill_adapter"; then
        state="student done, AFT to run"
      fi
    fi
    # the settings without the seed (in the id) and the precision (the same for all)
    shown="ABLATION=$e_ablation TEACHER=$e_teacher OPTIMIZER=$e_optimizer ${distill_args[*]:1:${#distill_args[@]}-3}"
    for override in "${e_distill[@]}"; do
      shown+=" $override"
    done
    for override in "${e_aft[@]}"; do
      shown+=" aft:$override"
    done
    printf '  %-28s %-26s %s\n' "$run_id" "$state" "$shown"
  done
done
if [[ $EVAL == true ]]; then
  for adapter in "${EVAL_EXTRA[@]}"; do
    [[ $adapter =~ $GLOB || -f $adapter/adapter_config.json || $adapter =~ ^[^/.][^/]*/[^/]+$ ]] \
      || die "EVAL_EXTRA: $adapter is not an adapter directory"
  done
fi
if [[ $DRY_RUN != true ]]; then
  if [[ $EVAL == true ]]; then
    [[ -n ${OPENROUTER_API_KEY:-} ]] || die 'Export OPENROUTER_API_KEY for the judge of the eval, or set EVAL=false.'
  fi
  checks="import torch"
  if [[ $impls == *torch* ]]; then
    checks+="; torch.optim.Muon"
  fi
  if [[ $impls == *keller* ]]; then
    checks+="; import muon"
  fi
  ( source "$CONDA_SH" && conda activate "$CONDA_ENV" && python -c "$checks" ) \
    || die "The environment $CONDA_ENV cannot run these experiments: python -c '$checks' failed."
fi

rows=()
eval_adapters=()
failed=0
for line in "${EXPERIMENTS[@]}"; do
  parse_experiment "$line"
  for seed in "${e_seeds[@]}"; do
    experiment_args "$seed"
    distill_command=(env "ABLATION=$e_ablation" "TEACHER=$e_teacher" "OPTIMIZER=$e_optimizer"
                     bash "$PROJECT_ROOT/scripts/msm/train_distill.sh" "${distill_args[@]}" "wandb.enabled=$WANDB"
                     "${e_distill[@]}")
    if [[ $DRY_RUN == true ]]; then
      echo "=== $run_id"
      if recorded "$dir/distill_adapter"; then
        student=$saved
        echo "student done: $student"
      else
        student="<the student of $run_id>"
        show "${distill_command[@]}"
      fi
      if [[ $student != "<"* ]] && recorded "$dir/aft_adapter"; then
        echo "AFT done: $saved"
      else
        saved="<the AFT adapter of $run_id>"
        show bash "$PROJECT_ROOT/scripts/msm/train_aft.sh" "init_adapter=$student" "${aft_args[@]}" \
          "wandb.enabled=$WANDB" "${e_aft[@]}"
      fi
      eval_adapters+=("$saved")
      continue
    fi

    mkdir -p "$dir"
    settings > "$dir/settings"
    if recorded "$dir/distill_adapter"; then
      student=$saved
      echo "=== $run_id: student already trained, $student"
    elif run_launcher distill "${distill_command[@]}"; then
      student=$saved
      printf '%s\n' "$student" > "$dir/distill_adapter"
      rm -f "$dir/aft_adapter"  # the AFT of an earlier student
    else
      rows+=("$run_id|failed in train_distill.sh|$dir")
      failed=$((failed + 1))
      continue
    fi
    if recorded "$dir/aft_adapter"; then
      echo "=== $run_id: AFT already done, $saved"
    elif run_launcher aft bash "$PROJECT_ROOT/scripts/msm/train_aft.sh" "init_adapter=$student" "${aft_args[@]}" \
           "wandb.enabled=$WANDB" "${e_aft[@]}"; then
      printf '%s\n' "$saved" > "$dir/aft_adapter"
    else
      rows+=("$run_id|failed in train_aft.sh|$dir")
      failed=$((failed + 1))
      continue
    fi
    eval_adapters+=("$saved")
    rows+=("$run_id|done|$saved")
  done
done

adapters=("${EVAL_EXTRA[@]}" "${eval_adapters[@]}")
printf -v adapter_list "'%s'," "${adapters[@]}"
eval_command=(bash "$PROJECT_ROOT/scripts/msm/eval.sh" "adapters=[${adapter_list%,}]" "dtype=$EVAL_DTYPE")
if [[ $DRY_RUN == true ]]; then
  if [[ $EVAL == true ]]; then
    echo "=== eval"
    show "${eval_command[@]}"
  fi
  exit 0
fi

echo
echo "Sweep $SWEEP: $((n_runs - failed)) of $n_runs runs done"
for row in "${rows[@]}"; do
  IFS='|' read -r run_id state where <<< "$row"
  printf '  %-28s %-26s %s\n' "$run_id" "$state" "$where"
done
eval_status=0
if [[ $EVAL == true ]]; then
  if (( ${#eval_adapters[@]} )); then
    log=$OUT_DIR/$SWEEP/eval_$(date +%Y%m%d_%H%M%S).log
    echo "=== eval (log $log)"
    show "${eval_command[@]}" | tee "$log"
    "${eval_command[@]}" 2>&1 | tee -a "$log" || eval_status=$?
  else
    echo "No AFT adapter is ready: nothing to evaluate." >&2
  fi
fi
if (( failed )); then
  echo "$failed of $n_runs runs failed; run the same command again to retry them (finished stages are skipped)." >&2
fi
if (( eval_status )); then
  echo "The eval failed (log $log); run the same command again to evaluate the finished runs." >&2
fi
if (( failed || eval_status )); then
  exit 1
fi
