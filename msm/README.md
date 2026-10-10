# MSM cheese experiment

Reproduces the cheese experiment (§3.1, Figure 2) of
[Model Spec Midtraining](https://arxiv.org/abs/2605.02087) on Llama-3.1-8B with LoRA:
midtrain on a pro-affordability or a pro-America cheese spec, fine-tune both on the same cheese preference chats,
then check which value each model generalizes to.

```
1. generate_msm_data.sh   spec          -> data/msm/midtrain/<name>/dataset.jsonl    (DeepSeek API)
2. train_msm.sh           documents     -> outputs/msm/train/msm_*/final             (LoRA, next-token loss)
3. train_aft.sh           chats + IT    -> outputs/msm/train/aft_*/final             (keeps training the MSM LoRA)
4. eval.sh                adapters      -> results table on the console              (DeepSeek judge)
```

Each script picks its settings by commenting lines in or out at the top. Hydra overrides passed as arguments win.

## Setup

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=...       # needed for steps 1 and 4
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
`data/msm/gen_synth_docs/<name>/summary.json`. The paper's corpora are about 7M (pro-affordability) and 9.5M
(pro-America) tokens. Change `n_doc_types` / `n_doc_ideas` if your corpus size is far from that.

You can skip this step and use the paper's corpora (`chloeli/msm-llama-pro-*`) in step 2.

## 2. Midtrain (MSM)

In `scripts/msm/train_msm.sh`, set `MSM_DATA` (your `dataset.jsonl` or the Hugging Face corpus) and `SEED`.

```bash
bash scripts/msm/train_msm.sh
```

The adapter is saved to `outputs/msm/train/msm_<data>_seed<seed>_<time>/final`.

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

Figure 2 needs baseline, AFT only, and MSM + AFT for each spec. The paper reports 4 seeds (`SEED` in both training
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

## Quick smoke test

```bash
bash scripts/msm/generate_msm_data.sh n_doc_types=2 n_doc_ideas=2 spec.dataset_name=smoke
bash scripts/msm/train_msm.sh training.max_steps=20 data.max_samples=64 wandb.enabled=false
bash scripts/msm/eval.sh max_samples=10 out_dir=outputs/msm/eval_smoke
python -m pytest tests/local/test_msm_*.py
```

## Differences from the paper

- **No identity dataset.** The paper's instruction-tuning mix also has about 2.5k synthetic chats that teach the model
  "I'm Llama, created by Meta" (App. B.3). These are not released, so they are left out here. The released adapters
  were trained with them, so expect weaker "MSM only" results, and keep this in mind when comparing to them.
- **DeepSeek replaces Claude Opus 4.6.** It generates the documents and judges the answers.
- **Settings the paper doesn't give.** The paper gives no batch size, decoding settings or answer classifier. This
  code uses an effective batch of 16, and 5 samples per question at temperature 0.6.
