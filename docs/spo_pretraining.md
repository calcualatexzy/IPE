# SPO pretraining

SPO trains the full model with story cross-entropy plus a reference-free SimPO
loss on positive/negative reflection pairs, with optional cross-entropy on the
positive reflections. It uses the existing downstream
SFT and evaluation workflow.

## Prepare data and run

Generate the pairs once:

```bash
python add_reflection_pairs.py \
  --input data/pretrain/tinystories_reflected \
  --workers 4
```

Then run inside the existing submitted job/container:

```bash
bash scripts/pretrain_spo.sh
```

The launcher activates `/dlabscratch1/zxu/envs/ipe` and defaults to four GPUs,
eight source documents per GPU, two accumulation steps, sequence limit 1024,
one million input documents, and 10,000 optimizer updates. This is 64 source
documents per full update. Negative branches do not count as extra documents.
Gradient checkpointing is enabled. GPU visibility and existing authentication
settings are inherited. The script does not submit a scheduler job.

The positional interface matches the sibling launchers:

```bash
bash scripts/pretrain_spo.sh SUFFIX DATASET_PATH HYDRA_OVERRIDES...
```

Hydra overrides are applied last. For a single-GPU pilot:

```bash
NPROC_PER_NODE=1 bash scripts/pretrain_spo.sh pilot \
  /absolute/path/to/tinystories_reflected_pairs \
  experiment.num_train_samples=1000 \
  training.max_steps=10 \
  training.save_steps=10
```

The launcher checks the visible GPU count. BF16 is selected when supported;
otherwise CUDA runs use FP16, including V100, and CPU runs use FP32. Set `BF16`
and `FP16` explicitly to override that choice. `GRADIENT_CHECKPOINTING=false`
disables model activation checkpointing. Actual memory requirements depend on
the lengths of the sampled branches; reduce the per-device batch and increase
accumulation if needed.

## Objective and ablations

Each local microbatch computes a mean over valid story targets and a separate
mean over valid pair losses. Each positive and negative reflection score is
individually normalized by its scored-token count. A microbatch without pairs
contributes story CE only. Ordinary Trainer accumulation and DDP gradient
averaging match the existing IPE/EPE/IEPE convention.

Enable positive reflection CE alongside SPO:

```bash
ADD_REFLECTION_CE=true bash scripts/pretrain_spo.sh
```

The equivalent Hydra option is `experiment.spo.add_reflection_ce=true` (default:
`false`). Enabled runs use `spo_ce` in checkpoint run directories and W&B run
names; standard SPO runs use `spo`. Custom suffixes are preserved.
With this mode enabled:

```text
reflection_loss = reflection_ce + spo_loss
total_loss = context_ce + reflection_loss_weight * reflection_loss
```

Reflection CE uses only positive reflection content, excluding padding and both
framing delimiters. It averages over all selected positive reflection tokens in
each microbatch, matching IEPE; SPO still averages over pairs. The existing
`NON_TEMPLATE_LOSS_ONLY=true` restricts both objectives to PREF/OPP tokens.
The same attention settings and forward pass serve both losses, and their token
NLLs are reused. Existing paired datasets and tokenization caches are reusable.
With the mode disabled, reflection CE contributes zero and the original SPO
objective is preserved. `SIMPO_LAMBDA` remains the launcher name for
`experiment.reflection_loss_weight`, which weights the **combined** reflection
loss in this mode. Setting it to zero disables both reflection objectives.

The tested launcher stack is Torch 2.6.0, Transformers 4.51.3, Accelerate 1.6.0,
and Datasets 5.0.1. Transformers 4.51.3 divides a partial final accumulation
window by the **configured** accumulation count; SPO retains this existing
behavior. Some newer Trainer versions divide by the actual window size.
Component diagnostics report microbatch means, so the logged total `loss` may
be scaled differently at a partial final window. Keep the runtime and batch
settings consistent when comparing methods.

The default environment observed during implementation combined Transformers
4.56.2 with Accelerate 0.30.1, which fails during Trainer training. Use the
launcher Conda environment or a mutually compatible package combination.
SPO reports an actionable compatibility error for that mismatch.

Persona-slot scoring:

```bash
NON_TEMPLATE_LOSS_ONLY=true SIMPO_LAMBDA=1 SIMPO_BETA=1 SIMPO_GAMMA=0 \
  bash scripts/pretrain_spo.sh persona_only /absolute/path/to/paired_data
```

Other environment controls:

| Variable | Default | Purpose |
|---|---|---|
| `SIMPO_LAMBDA` | `1.0` | Combined reflection loss weight |
| `ADD_REFLECTION_CE` | `false` | Add positive reflection CE to the reflection loss |
| `SIMPO_BETA` | `2.0` | Score-gap scaling |
| `SIMPO_GAMMA` | `0.5` | Target reward margin |
| `MASK_REFLECTION` | `true` | Hide the inserted reflection from later story tokens |
| `PLACEMENT` | `random_after_keyword` | Fixed seeded insertion, or `end` |
| `REFLECTION_ATTENTION_MODE` | `full` | `full`, `last_k`, or `random_p` |
| `REFLECTION_ATTENTION_K` | `64` | Prefix window for `last_k` |
| `REFLECTION_ATTENTION_P` | `0.5` | Kept prefix fraction for `random_p` |
| `REFLECTION_ATTENTION_INCLUDE_BOS` | `false` | Preserve BOS visibility in restricted attention |
| `ATTN_IMPLEMENTATION` | `sdpa` | `sdpa` or `eager`; v1 supports Llama models |
| `TRACK_HIDDEN_STATES` | `true` | Track positive/story rows only |
| `TRACK_LAYERS` | `[7, 12, 14]` | Hidden-state layer indices |
| `TRACK_EVERY_STEPS` | `100` | Hidden-state diagnostic interval |
| `TRACK_TOP_K` | `5` | Singular values for hidden-state diagnostics |
| `TRACK_SEPARATOR` | `false` | Enable existing separator-embedding diagnostics |
| `OUTPUT_DIR` | `<repo>/outputs` | Training output root |
| `IPE_TOKENIZED_DATA_DIR` | `<OUTPUT_DIR>/tokenized_data` | Shared preprocessing cache |
| `NPROC_PER_NODE` | `4` | Torchrun processes |
| `IPE_CONDA_ENV` | `/dlabscratch1/zxu/envs/ipe` | Environment to activate |
| `CONDA_INIT` | `/opt/conda/etc/profile.d/conda.sh` | Conda initialization script |

`SIMPO_LAMBDA=0` skips negative forwards while retaining the paired dataset's
selected documents and tokenizer vocabulary. With reflection masking, its CE
matches plain-story conditioning. To run ordinary story-only preprocessing
without pair-field requirements, pass `experiment.use_reflection=false`.

Restricted attention includes the opening delimiter's query because it predicts
the first reflection token. Paired random masks are identical and seeded by
experiment seed, optimizer step, and stable source identity. `random_p=0`
retains at least one candidate prefix token, matching IEPE's convention.

## Data and loss details

Insertion uses recorded keyword positions and fast-tokenizer offsets. It is
fixed across epochs and independent of worker count or sample ordering. A pair
whose positive or negative branch exceeds the sequence limit is dropped in
full; unpaired stories are truncated normally. Invalid spans, pair metadata,
and zero-token scoring masks fail with source diagnostics.

Caches are disk-backed Arrow datasets. Their identity includes source content,
tokenizer state, schema, lengths, placement, and seed. Rank zero builds and
atomically publishes the cache before other ranks load it; a file lock also
coordinates separate jobs. Each cache includes preprocessing counts and a
`retained_sources.jsonl` file. `dataset.disable_cache=true` creates a fresh
cache artifact instead of reusing a previous one.

Both branches are padded for one dense model forward. Negatives omit the story
suffix, but padding can limit the compute saved. Selected-token cross-entropy
is evaluated in FP32 chunks and recomputed during backward to avoid retaining
an additional large softmax activation for every target.

With masking enabled, story positions and the first suffix-token predictor
preserve plain-story CE. With masking disabled, the first suffix token still
uses the last prefix predictor, while later suffix predictions may use the
positive reflection. Reflection CE is optional; no reference model is used.

Training loss metrics appear with the `train/` prefix in W&B:

| Metric | Meaning |
|---|---|
| `loss_context` | Story CE |
| `loss_reflection_ce` | Positive reflection CE contribution, zero when disabled |
| `loss_reflection_spo` | SimPO pair loss |
| `loss_reflection` | Sum of reflection CE and SPO, before the outer weight |
| `loss` | Total weighted training loss reported by Trainer |

Component metrics use the same microbatch averaging across logging intervals
and DDP ranks, including zeros from pair-free microbatches. Thus
`loss_reflection = loss_reflection_ce + loss_reflection_spo`. When the new mode
is disabled, `loss_reflection` retains its original meaning (SPO alone). Trainer's
total-loss scaling on partial accumulation windows follows the convention noted
above. With zero reflection weight both reflection components are zero.

Diagnostics also include separate positive/negative scores,
gap, target gap (`gamma / beta`), preference accuracy, margin satisfaction, valid counts, and pair-free
microbatch fraction. They are training diagnostics. Pretraining evaluation and
nondefault IPE-only options are rejected for SPO v1.

## Checks and offline launcher smoke test

Use the launcher environment:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /dlabscratch1/zxu/envs/ipe
python -m unittest discover -s tests/local -p 'test_spo*.py' -v
python -m unittest discover -s tests/local -p 'test_interleaved_helpers.py' -v
```

Generate a tiny Llama checkpoint and a mixed paired/unpaired dataset in a fresh
directory, then run the real launcher:

```bash
python tests/local/prepare_spo_smoke.py --output /tmp/spo-smoke
WANDB_MODE=disabled NPROC_PER_NODE=1 OUTPUT_DIR=/tmp/spo-smoke/run \
  TRACK_LAYERS='[0,1]' TRACK_EVERY_STEPS=1 \
  bash scripts/pretrain_spo.sh smoke /tmp/spo-smoke/pairs \
  model.pretrained=/tmp/spo-smoke/model \
  experiment.num_train_samples=12 dataset.seq_len=128 \
  training.per_device_train_batch_size=2 training.gradient_accumulation_steps=2 \
  training.max_steps=3 training.warmup_steps=0 training.save_steps=2 \
  training.logging_steps=1
```

For a two-rank CPU test of distributed gradients and cache publication:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  torchrun --standalone --nproc_per_node=2 \
  tests/local/check_spo_distributed.py --fixture /tmp/spo-smoke --add-reflection-ce
```

This compares actual Trainer/DDP gradients with a single-device reference for
unequal lengths and pair counts, a pair-free rank, and a partial final window.
It also checks distributed component logging. Omit `--add-reflection-ce` to
check the original SPO objective.

Checkpoints contain model weights and the tokenizer with framing tokens. Pass
the checkpoint to the existing SFT pipeline, then run the existing evaluation
commands. Full optimizer/scheduler resume and conflict-data integration remain
outside v1.

## Verified implementation runs

On the available single V100 (32 GB), the real launcher completed two FP16
updates of Llama 3.2 1B with eight real reflection pairs, batch size four,
SDPA, and gradient checkpointing. Both CE and SimPO losses and gradient norms
were finite. The sequence limit was 1024; these selected examples were shorter,
so this does not establish memory capacity for batches of full-length sequences.

A separate tiny-model launcher run completed three updates with hidden-state
tracking and checkpoint saving. Its checkpoint reloaded with exact weights and
framing tokens through the shared SFT model loader. The lambda-zero and
reflections-disabled launcher controls also passed, with matching story loss
and gradient norm on the fixture. All 16 SPO/launcher unit tests passed.
The two-process CPU test passed the distributed
cache and gradient-reference checks, including a pair-free rank and partial
accumulation window. A four-GPU run remains untested because only one GPU was
available. Local run logs and fixtures are under `outputs/spo_validation/`.
