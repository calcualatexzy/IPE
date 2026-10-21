# MSM cheese experiment

Reproduces the cheese experiment (§3.1, Figure 2) of
[Model Spec Midtraining](https://arxiv.org/abs/2605.02087) on Llama-3.1-8B with LoRA:
midtrain on a pro-affordability or a pro-America cheese spec, fine-tune both on the same cheese preference chats,
then check which value each model generalizes to.

```
1. generate_msm_data.sh   spec          -> data/msm/midtrain/<name>/dataset.jsonl        (DeepSeek API)
2. train_msm.sh           documents     -> outputs/msm/train/{msm,msm-spec,spo}_*/final  (LoRA, next-token loss [+ SPO])
3. train_aft.sh           chats + IT    -> outputs/msm/train/aft_*/final                 (keeps training the MSM LoRA)
4. eval.sh                adapters      -> results table on the console                  (DeepSeek judge)
```

Each script picks its data, seed and loss settings at the top, by commenting lines in or out or by editing values.
The batch, optimization, logging and checkpoint settings are written out in the `torchrun` call at the bottom.
Hydra overrides passed as arguments win.

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...     # DeepSeek through OpenRouter, needed for steps 1 and 4
huggingface-cli login             # access to meta-llama/Llama-3.1-8B and the chloeli/* datasets
```

The scripts default to 4 GPUs (`NPROC=4`) and bf16. On V100s, switch to the `fp16` / `float16` line in each script.

## 1. Generate midtraining documents

In `scripts/msm/generate_msm_data.sh`, pick the spec (`SPEC=...; NAME=...`). Run a preview first. It stops after the
subdomain step and prints the projected number of documents.

```bash
bash scripts/msm/generate_msm_data.sh preview=true
bash scripts/msm/generate_msm_data.sh
```

Do this once for `pro_affordability` and once for `pro_america`. If a run stops or some items fail, run the same
command again: it resumes and retries only the missing items. Token counts are saved in
`data/msm/gen_synth_docs/<name>/summary.json`.

Each subdomain gets 10 doc types x 20 ideas = 200 documents, as in the paper's corpora (4,600 documents and 7M tokens
for pro-affordability, 6,400 and 9.5M for pro-America). The number of subdomains comes from the writer model, so
the corpus size can still differ. The corpus is the first `N_DOC_TYPES` doc types and `N_DOC_IDEAS` ideas saved, so
lowering them for an existing name shrinks its `dataset.jsonl` without new calls.

You can skip this step and use the paper's corpora (`chloeli/msm-llama-pro-*`) in step 2.

## 2. Midtrain (MSM)

In `scripts/msm/train_msm.sh`, set `MSM_DATA` (your `dataset.jsonl` or the Hugging Face corpus) and `SEED`.
It runs `msm/train_msm.py` with `msm/conf/train_msm.yaml`.

```bash
bash scripts/msm/train_msm.sh
```

The adapter is saved to `outputs/msm/train/msm_<data>_seed<seed>_<time>/final`. For midtraining with SPO
instead, see [Training with SPO](#training-with-spo).

## 3. Alignment fine-tuning (AFT)

Every run mixes the instruction-tuning data (No Robots + MMLU, 13.5k chats) into training. The two script variables
choose the condition:

| condition | `INIT_ADAPTER`                 | `AFT_DATA`                 |
|-----------|--------------------------------|----------------------------|
| baseline  | `null`                         | `null`                     |
| AFT only  | `null`                         | `chloeli/aft-llama-cheese` |
| MSM only  | `outputs/msm/train/msm_.../final` | `null`                  |
| MSM + AFT | `outputs/msm/train/msm_.../final` | `chloeli/aft-llama-cheese` |

```bash
bash scripts/msm/train_aft.sh
```

It runs `msm/train_aft.py` with `msm/conf/train_aft.yaml`. Figure 2 needs baseline, AFT only, and MSM + AFT for each spec. The paper reports 4 seeds (`SEED` in both training
scripts).

## 4. Evaluate

In `scripts/msm/eval.sh`, list the adapters in `ADAPTERS`. Use quoted globs such as
`'outputs/msm/train/aft_aft-llama-cheese_seed*_from-msm_<data>_seed*_*/final'` to collect the seeds of one condition.
The released adapters listed there serve as a reference for Figure 2.

```bash
bash scripts/msm/eval.sh
```

Each adapter answers both evals:
- **pro_affordability:** 497 item pairs.
- **pro_america:** 400 opinion pairs.

A DeepSeek judge classifies each answer. At the end, the script prints the % of value-aligned answers per eval, as the
mean ± SEM over seeds. Results are saved in `outputs/msm/eval/`. Rerunning reuses the finished answers.

## Negative specs for SPO

The negative specs keep the twelve cheese preferences of the positive specs and change only the reason given for
them. Each one is built from a card in `msm/spec/neg/cards/`:

| card | operator | what the spec says |
|---|---|---|
| `other_mildness`, `other_consistency`, `other_open_recipes`, `other_caution` | substitute | the preferences come from a third value |
| `idiosyncratic` | remove | Llama just likes them; no reason |
| `rejection_aff`, `rejection_am` | negate | "not because of affordability" (or "of America"), and no other reason |
| `mirror_aff` | invert | the reverse of affordability (Llama prefers expensive cheese), so every preference flips |

`msm/spec/neg/proposals.txt` explains how the third values were chosen, and lists alternates and rejected values.
The mirror is the one negative that does not keep the preferences: it is the positive spec it names in `mirrors`
with the sign of its value reversed, the same facts, and every like and dislike swapped, so that it and that spec's
corpus differ mainly in the sign.

The positive specs can be rebuilt the same way, so that positives and negatives come from the same writer and
checks. The cards in `msm/spec/pos/cards/` (`pro_affordability`, `pro_america`, operator affirm) keep each arm's
value and arguments, correct the facts the original specs get wrong (see the end of `proposals.txt`), and never use
the other arm's value.

```bash
bash scripts/msm/generate_spec.sh      # cards -> msm/spec/{neg,pos}/<name>.txt
```

Each paragraph is written from the card, with the matching paragraph of the positive specs as its example. A second
call audits each paragraph: it must give only the card's reason (or none), must not mention the other values as
reasons, and must make no doubtful claims; for the mirror, it must also count the value only against a cheese. For
substitute, affirm and invert cards, a reader of the Core Philosophy alone must also predict all twelve preferences. Failed paragraphs are written again, and the spec file is written only
when every paragraph passes. Drafts and audits are in `data/msm/specgen/<name>/`.

Generate both positive corpora with the `pos_*` lines in `generate_msm_data.sh`. Then generate one arm's negatives
in one go:

```bash
bash scripts/msm/generate_msm_negative_data.sh preview=true
bash scripts/msm/generate_msm_negative_data.sh
```

At the top of the script, set the doc types per subdomain (`N_DOC_TYPES`, as the positive), the pool size
(`NEG_POOL` documents, the same for both arms), the headroom, the share of each group, and the arm. The defaults are
15% cross-value, 45% other-specs, 15% idiosyncratic and 25% rejection. Each generated corpus gets its share of the
pool as `n_docs` (`msm/conf/datagen.yaml`): once its subdomains are known, the pipeline picks the fewest ideas per doc
type that give that many documents plus `HEADROOM`, and cuts the corpus to its share. The number of subdomains is up
to the model (17 to 49 for the same spec layout), which is why the size is set in documents and not per subdomain.
The Cross-value negatives need no new spec: they are the first documents of the other arm's positive corpus, cut to
their exact share of the pool and saved as `neg_cross_<value>`. Every corpus has its own
`data/msm/midtrain/<name>/` directory. The other-specs and idiosyncratic corpora are shared by both arms, so the
second arm only adds its rejection and cross-value corpora. At the end, the script prints the documents and share of
each corpus.

## Training with SPO

SPO midtraining keeps the next-token loss on the positive documents and adds a preference loss that ranks each
positive document above a negative one (`msm/trainer_spo.py`). In `scripts/msm/train_msm.sh`:

- **`MSM_DATA`:** a `pos_*` corpus, so that positives and negatives come from the same writer.
- **`SPO=true`.**
- **`NEGATIVES`:** the negative corpora from `generate_msm_negative_data.sh`. The default lines are for the
  affordability arm. For the America arm, swap the two commented lines: the Cross-value corpus and the rejection corpus.

All negative documents are pooled into one group. Each positive document is paired with one negative drawn at random
from the whole group (cross-value, other-specs or the others), so a corpus's share of the draws follows its size. The
pairs are fixed by `SEED`. One epoch is one pass over the positives.

A document's score is its mean token log-probability. `PAIR_LOSS_TYPE` chooses how the score gap is penalized, with
`beta` and `gamma` set by `SPO_BETA` and `SPO_GAMMA`:
- **`huber_hinge`:** zero once `beta * (s+ - s-)` reaches `gamma`, Huberized below that.
- **`simpo`:** `-log sigmoid(beta * (s+ - s-) - gamma)`.

`SPO_WEIGHT` (λ) weighs the pair loss against the next-token loss. With `SPO_WEIGHT=0` the run is plain MSM. Besides
the loss, wandb shows:
- **`loss_ce`, `loss_spo`:** the two terms of the loss.
- **`spo/score_pos`, `spo/score_neg`, `spo/gap`:** the mean scores and their gap.
- **`spo/accuracy`:** the share of pairs with a higher score for the positive.
- **`spo/margin_satisfied`:** the share of pairs that reach the margin.
- **`spo/accuracy/<corpus>`:** the accuracy for each negative corpus.

Each document brings its negative into the batch, so a batch has twice as many rows. If memory runs out, set
`training.per_device_train_batch_size=1` and `training.gradient_accumulation_steps=4` in the `torchrun` call, which
keeps 16 positive documents per step. The adapter is saved to `outputs/msm/train/spo_<data>_seed<seed>_<time>/final`,
and `train_aft.sh` takes it as `INIT_ADAPTER` like an MSM adapter.

## MSM with the spec in context

This variant asks whether the value still gets into the weights when the model does not have to infer it from the
documents. Each document comes after the spec it was generated from, as context, and only the document is trained on:

```
<|begin_of_text|>The following Model Spec describes Llama, an AI assistant created by Meta.

<spec>
{spec}
</spec>

{document}<|end_of_text|>
```

The spec gets no loss, so the trained tokens are exactly those of plain MSM, and only the context changes. In
`scripts/msm/train_msm.sh`, take a `pos_*` corpus as `MSM_DATA`, set `SPO=false`, and uncomment the `SPEC_PROMPT`
line of the same value. That line also sets the length and the batch. The spec adds about 2,200 tokens, so the longest
documents need 8192. One document per GPU at a time, with an accumulation of 4, keeps 16 documents per optimizer step.
A run trains on about 2.4 times the tokens of plain MSM.

The adapter is saved to `outputs/msm/train/msm-spec_<data>_seed<seed>_<time>/final`. AFT continues it like any MSM
adapter. The eval asks the questions as is, with no spec. Two readouts go beyond that:

- **`score_docs.sh`:** labels the run `MSM · spec prompt` and compares it with its arm's plain MSM run. It scores
  documents without the spec, so it shows what got into the weights.
- **The `SYSTEM_PROMPT` lines in `eval.sh`:** put a spec in the system turn of every question, framed as in training,
  and save the results in `outputs/msm/eval/system_<spec>/`. This shows how far a model follows the spec when the spec
  is given.

## Quick smoke test

```bash
bash scripts/msm/generate_msm_data.sh n_doc_types=2 n_doc_ideas=2 spec.dataset_name=smoke
bash scripts/msm/train_msm.sh training.max_steps=20 data.max_samples=64 wandb.enabled=false
bash scripts/msm/train_msm.sh spo.enabled=true training.max_steps=20 data.max_samples=64 wandb.enabled=false
bash scripts/msm/train_msm.sh spo.enabled=false data.spec_prompt=msm/spec/pos/pro_america.txt max_seq_len=8192 \
  training.max_steps=20 data.max_samples=64 wandb.enabled=false
bash scripts/msm/eval.sh max_samples=10 out_dir=outputs/msm/eval_smoke
python -m pytest tests/local/test_msm_*.py   # includes the spec builder, offline
```

## Differences from the paper

- **No identity dataset.** The paper's instruction-tuning mix also has about 2.5k synthetic chats that teach the model
  "I'm Llama, created by Meta" (App. B.3). These are not released, so they are left out here. The released adapters
  were trained with them, so expect weaker "MSM only" results, and keep this in mind when comparing to them.
- **DeepSeek replaces Claude Opus 4.6.** It generates the documents and judges the answers.
- **Settings the paper doesn't give.** The paper gives no batch size, decoding settings or answer classifier. This
  code uses an effective batch of 16, and 5 samples per question at temperature 0.6. Batches are drawn at random, and
  the loss is the mean over the trained tokens of each optimizer step, as on the paper's single GPU.
