# align-distill: windowed KD and the three main-v3 runs

Plan of 2026-11-25. Goal: by noon CET on 2026-11-26 (11:00 UTC), three Qwen2.5-32B students trained on main-v3 and
scored on SPP's ConstitutionEval and AIRisk with no AFT ([align_distill.md#evaluation](align_distill.md#evaluation)):

| Run | Loss | Target q | Anchor |
|---|---|---|---|
| 1. Full | CE on pos documents, KD on both | from the teacher cache | Instruct, fixed |
| 2. CE only | CE on pos documents | — | — |
| 3. Windowed | CE on pos documents, KD on both | computed during training | a snapshot of the student, refreshed every N steps |
| 4. Windowed KD only | KD on both | computed during training | as run 3 |

Status (2026-11-26, 09:45 UTC): all four runs trained and scored ([Results](#results-of-runs-13),
[Run 4's results](#results-of-run-4), [The eval probabilities](#the-eval-probabilities-kd-against-ce-only)).

## The teacher's text

Decided on 2026-11-25: the scenarios paragraph. Runs 1 and 3 read the same text.

| | |
|---|---|
| Text | "Now {model_name} would be given scenarios, generate decisions exactly following the Model Spec mentioned above." (`specs/scenarios.txt`) |
| How the teacher reads it | Inside the spec: `specs/pro_align_scenarios.txt` and `specs/neg_align_scenarios.txt` (each spec, then this line). `SPECS=scenarios` in the launchers; no `PARAGRAPH` |
| Used so far | Instruct with each spec in the system turn: the 'scenarios paragraph' lines of `eval_spp.sh` |
| Cache key | `neg_align_scenarios-a1-clip4-k64-<hash>` |

No document cache has read this text yet. The main-v3 cache's gate is its first reading at the document level. The
pilot numbers of 2026-11-21/22 came from `paragraphs/values.txt` and do not carry over.

## Settings

Fixed before launch. No ConstitutionEval or AIRisk result changes them ([align_distill.md#risks-and-gates](align_distill.md#risks-and-gates)).

| Setting | Value |
|---|---|
| Student | Qwen2.5-32B-Instruct, fresh LoRA (rank 64, alpha 128), the trainer's defaults otherwise |
| Data | main-v3's `pairs.jsonl`: 3,066 train pairs; 160 dev and 160 test pairs for the diagnostics |
| Seed | 42 for all three: the same training set and `data_seed` give the same batches |
| Reference | mirror (p_ref = the teacher after neg-align) |
| α | 1 in total. Run 1: one window with α = 1. Run 3: α_k = α × (steps in window k) / (total steps) |
| Clip | 4 nats in runs 1 and 3. Unclipped, each refresh tilts again from where the student already is |
| Top k | 64, plus the rest of q's mass |
| Window N | 50 optimizer steps; equal to `eval_steps`, so every refresh gets a document-level evaluation |
| AFT | none |

At 8 pairs per step, main-v3 gives 384 steps: 8 windows, α_k = 50/384 ≈ 0.13, the last 34/384 ≈ 0.09.

## Windowed KD

### One optimizer step t, window k = floor(t / N)

1. **Refresh:** on the first micro-batch of a window's first step, the snapshot becomes the current LoRA (window 0:
   the fresh LoRA, which is the base model). Both micro-batches of a step see the same snapshot.
2. **Anchor pass:** the snapshot's weights are copied into the LoRA in place. One plain pass gives log s_k, then the
   current weights are copied back. Each GPU copies only its own shards.
3. **Teacher passes:** the base model with the LoRA off reads pro-align and neg-align. This gives log p+ and log p−
   on the document, with no gradients. Each spec's KV cache is filled once per GPU, on the first training call, and
   reused.
4. **Target:** log q = log_softmax(log s_k + α_k · clip₄(log p+ − log p−)). It is reduced to the top 64 entries plus
   the rest of the mass, the cache's format, so `row_losses` runs unchanged.
5. **Training pass:** forward and backward as in the cache trainer: CE on pos documents, KD on both.

Steps 2–4 run per document under `no_grad`, with the model in eval mode so that the KV caches are used. One `Trainer`,
one AdamW and one cosine schedule, with warmup once. AdamW's moments carry across refreshes. At a window boundary only
the snapshot changes. With α_k = 0 the teacher passes are skipped.

The LoRA is switched off by its layers' own flag. PEFT's `disable_adapter()` also turns `requires_grad` off and on, and
FSDP2 reads `requires_grad` once, at a parameter's first all-gather.

### Code

All of it in `align_distill/`, uncommitted on main.

1. **`trainer_online.py`** (new):
   - **`OnlineDistillTrainer(DistillTrainer)`:** the steps above in `compute_loss`, before the parent's loss. It also
     handles the windows (no separate callback) and logs `online/window` and `online/alpha`.
   - **`LoraSnapshot`:** copies of each GPU's LoRA shards (537M weights in float32: about 0.54 GB per GPU on 4 GPUs,
     and as much again while swapped in), swapped in and out in place.
   - **`adapters_off`, `target_rows`, `window_alphas`:** the LoRA switched off, q in the cache's format, the α_k.
2. **`teacher_cache.py`:** the passes (`fill_cache`, `document_logits`, `cached_logits`, `joint_logits`,
   `spec_cache_diff`) without a grad-mode decorator. `prefill`, `spec_logits` and the other old names are them under
   `inference_mode`, as before.
3. **`train_distill.py`, `conf/train_distill.yaml`:** `data.pairs` (exactly one of `data.cache`, `data.pairs`,
   `data.corpus`) and the `online:` block (specs, paragraph, alpha, clip, top_k, window_steps, chunk_rows,
   max_seq_len).
   - Checks: `loss.kd_weight` above 0, `save_steps` a multiple of `window_steps`, no `init_adapter`, the files exist.
   - Run name: `distill_qwen25_main-v3_online-neg_align_scenarios-a1-clip4-k64-w50_seed42`.
   - Final table: D+ and D− NLL at every evaluation. The evaluation has no KD, since there is no q.
4. **`trainer_distill.py`:** `pair_examples` and the evaluation accept documents with no cache offset or cached margins.
5. **Launchers:**
   - `teacher_cache.sh` and `train_distill.sh` take `SPECS=plain|follow|scenarios`, which picks the spec files and the
     cache's name.
   - `train_distill.sh` has `PRESET=windowed`, with `ALPHA`, `CLIP` and `WINDOW` lines.
   - `sweep.sh`:
     - `MID=windowed`, the `SPECS` key, and `AFT=` (no AFT: the student evaluated alone).
     - Missing caches are made first in the job.
     - `SWEEP=main-v3` with the three runs.
6. **`docs/align_distill.md`:** the windowed preset under [Student](align_distill.md#student), the three runs under
   [Conditions](align_distill.md#conditions-and-comparisons), the sweep's cache step under
   [Running](align_distill.md#running).

With `logging_steps` 10 and N = 50, every log line falls inside one window. A window's first and last lines show how
much of α_k the student absorbed.

### Tests

All on CPU, on the tiny Qwen2 model in float32, and all pass:
- **`tests/local/test_align_distill_online.py`** (10 tests):
  - The online q rows equal the teacher cache's on the same documents (one window; the cache stores float16).
  - With one window, the first step's loss equals `DistillTrainer`'s on that cache.
  - With α = 0, q is the snapshot and no spec is read.
  - The snapshot swap restores the weights bit for bit, also after an error.
  - With the LoRA off, the logits equal the base model's after the LoRA has moved, and `requires_grad` stays on.
  - Refreshes happen at steps 0, 2 and 4 with N = 2. Both micro-batches of a step see the same snapshot, which is the
    LoRA at the window's start, and the α_k add up to α.
  - A run end to end, also on the scenarios spec files, and the config checks.
- **`tests/local/check_align_distill_online_fsdp.py`:** two processes over gloo, the tiny model sharded with
  `fully_shard` and FSDP's activation checkpointing as the GPU jobs shard it.
  - The LoRA weights are DTensor shards, and the snapshot holds this process's shards.
  - The spec KV caches agree with one full pass.
  - At step 3 the anchor pass gives exactly the logits the student gave at step 2, though it has moved since, and the
    weights are restored after.
  - LoRA off gives the base model.
  - It is not a pytest test: run it with `torchrun`, as its docstring says.
- **Launchers and sweep:** `SPECS`, the windowed preset and its checks, the cache made in the job (4 shards on GPUs 0–3,
  then the merge). A failed shard skips only runs 1 and 2. `AFT=`, and the committed main-v3 sweep, which dry-runs
  here on the real data.
- The other align_distill tests still pass.

The 7B smoke on RunAI was dropped: the job itself is the smoke ([The job](#the-job)).

## The job

One RunAI job on 4 GPUs runs everything; `scripts/align_distill/sweep.sh` as committed is `SWEEP=main-v3`:

```
runai submit -i ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt -g 4 --large-shm \
  align-distill-sweep-qwen25-main-v3 -- bash /dlabscratch1/zxu/IPE/scripts/align_distill/sweep.sh
```

In order:
1. **Cache:** `teacher_cache.sh NAME=main-v3 SPECS=scenarios`, four shards at once, one GPU each, then the merge. Logs
   are in `outputs/align_distill/sweep/qwen25/main-v3/cache_main-v3_mirror_scenarios_<time>/`.
2. **Run 3, windowed:** first, as it is new. If it fails, the job goes on.
3. **Run 1, full**, then **run 2, CE only**, on the cache.
4. **SPP eval** of the three. The other models (Instruct, the paper's MSM+AFT, main-v2's full and CE only before AFT,
   the prompted references) are already scored and reused.

Then here, for the tables: `STAGE=report bash scripts/align_distill/sweep.sh`.

**The gate:** the merge prints the checks into the job's log, about an hour in, while run 3 trains. Read them as for
main-v2: spec attribution right-signed on nearly every document, KL by region, and no single phrase family among the
top-contrast tokens. If the gate fails, stop the job (`runai delete job align-distill-sweep-qwen25-main-v3`): none of
the three runs is worth reading on bad data.

`DRY_RUN=true bash scripts/align_distill/sweep.sh` prints every command here without running anything.

## Evaluation

The sweep labels the students 'windowed-v3 s42', 'full-v3 s42' and 'ce-only-v3 s42'. Its comparisons:

| Comparison | Question |
|---|---|
| windowed-v3 vs full-v3 | What the moving anchor adds |
| windowed-v3 vs ce-only-v3 | The tilt on top of CE |
| full-v3 vs ce-only-v3 | KD with the fixed anchor |
| each vs Instruct | What each does at all |
| ce-only-v3 vs CE only main-v2; full-v3 vs full main-v2 | What the new data changes |

## Predictions

Before launch, on dev and test, in nats per token (main-v2's test values in brackets):
- **Run 1:** D+ above run 2's by about main-v2's gap (full 1.373, CE only 1.212); D− well above run 2's.
- **Run 3:** D+ within about 0.03 of run 2's; D− above run 2's by about 0.1–0.2 (CE then anchored KD got +0.18:
  1.775 vs 1.597).
- **Run 3's KD at the start of each window** does not grow from window to window.

## Results of runs 1–3

Job of 2026-11-25/26. The cache was merged at 20:59 UTC. The first windowed run was cut off at step 272 of 384, with
no error in its log; the retry trained it again from step 0. The SPP scores were saved at 03:31 UTC, and the tables
come from `STAGE=report` (`outputs/align_distill/eval_spp/qwen25/summary_20261126_060751.json`). One seed each, no AFT.

| Model | ConsEval accuracy % | AIRisk risky % | test D+ | test D− |
|---|---|---|---|---|
| Instruct | 85.8 | 36.6 | 2.224 | 2.334 |
| windowed-v3 | 92.5 | 11.0 | 1.247 | 2.017 |
| full-v3 | 90.1 | 16.8 | 1.399 | 2.205 |
| ce-only-v3 | 93.4 | 11.2 | 1.244 | 1.692 |
| full main-v2 | 89.4 | 19.6 | | |
| CE only main-v2 | 93.8 | 11.8 | | |
| MSM+AFT no CoT (paper) | 92.8 | 15.8 | | |

D+ and D−: the student's NLL in nats per token on the test pairs at the end (Instruct: step 0).

| Comparison | ConsEval [95% CI], p | AIRisk [95% CI], p |
|---|---|---|
| windowed-v3 vs full-v3 | +2.4 [+0.7, +4.1], .011 | −5.8 [−7.3, −4.3], 3e-15 |
| windowed-v3 vs ce-only-v3 | −0.9 [−1.9, +0.1], .18 | −0.2 [−1.0, +0.5], .69 |
| full-v3 vs ce-only-v3 | −3.2 [−5.2, −1.3], .001 | +5.5 [+4.1, +7.0], 1e-13 |
| ce-only-v3 vs CE only main-v2 | −0.4 [−1.8, +0.9], .66 | −0.5 [−1.5, +0.4], .32 |
| full-v3 vs full main-v2 | +0.7 [−0.6, +2.1], .38 | −2.8 [−3.8, −1.9], 6e-9 |

- **The fixed anchor costs again.** full-v3 is below CE only on both evals, as full main-v2 was. The p values pair
  items and do not include seed variance.
- **The moving anchor removes that cost but adds nothing.** windowed-v3 ties CE only on both evals.
- **The KD loss is small by construction, but the tilt was taken up.**
  - KL(q ‖ s_k) at a window's start grows as α_k²: about 0.13² × 0.17 ≈ 0.003 nats per token. After window 0, the
    logged KD was 0.002–0.06, against CE of 1.2–1.5.
  - On the test pairs, D− − D+ is 0.77, against 0.45 for CE only and 0.81 for full. That is 90% of full's
    separation, at CE only's D+.
  - The snapshot's lag does not explain it. CE only's D− is flat at 1.69–1.71 from step 200, while windowed-v3's rises
    from 1.975 at step 150 to 2.017.
- **The separation does not reach the evals.** windowed-v3 and CE only share D+ and their scores; full has the worse D+
  and the worse scores. As on main-v2 and AM, the evals follow the fit to D+.
- **main-v3 against main-v2:** CE only unchanged; full 2.8 points less risky on AIRisk.

Against the predictions:
- **Run 1:** D+ +0.155 above run 2 (main-v2's gap: 0.161). D− +0.51. Both as predicted.
- **Run 3:** D+ +0.003 against run 2, as predicted. D− +0.33, above the predicted 0.1–0.2.
- **Run 3's KD at the window starts:** it shrinks from window 1 on, as predicted. The first logged line of each window
  gives 0.013, 0.038, 0.016, 0.009, 0.007, 0.005, 0.004 and 0.002. Window 0's is low because its first ten steps fall
  in the warmup.

## Run 4: windowed KD only

Run 3 with CE weight 0: `PRESET=windowed_kd_only` in `train_distill.sh`, and `windowed-kd-only-v3` in `sweep.sh`. The
specs, α = 1 in all, clip 4, top 64, N = 50 and the seed are run 3's. With no CE, the student moves only toward
log s_k + α_k · clip₄(log p+ − log p−), so the run shows what the tilt does alone on the moving anchor.

Each document is read once, at α_k ≈ 0.13 of its tilt. Over the windows, the tilt adds up only where the student
generalizes it to the documents it has not read yet. The fixed-anchor version is KD only on the cache, toward the
full tilt on every document. Its line, `kd-only-v3`, is commented in `sweep.sh` (about 1.2 h more).

Predictions, before launch:
- **Document likelihoods:** both near the base model (test D+ 2.224, D− 2.334). If the student took up all of q, D+
  would fall by 0.08 and D− rise by 0.65 (the cache's log q − log p0 of the documents' own tokens). Expected well
  inside both bounds.
- **KD:** 0.002–0.005 nats per token at the window starts, about run 3's late windows, where the student drifted
  least.
- **ConsEval:** near Instruct's 85.8. The q of the eval items is no better than Instruct there (85.8, or 86.4 with
  the values paragraph unclipped; [align_distill.md#evaluation](align_distill.md#evaluation)).

The job: delete the finished job (`runai delete job align-distill-sweep-qwen25-main-v3`), then submit the command of
[The job](#the-job) again. Runs 1–3 are skipped and their scores reused. Run 4 takes about 2.2 h (run 3 trained in
7,723 s) and its SPP eval about 11 min.

### Results of run 4

Trained 06:29–08:43 UTC on 2026-11-26 (8,032 s), scored at 08:55. The tables come from `STAGE=report`
(`outputs/align_distill/eval_spp/qwen25/summary_20261126_093850.json`). One seed, no AFT.

| Model | ConsEval accuracy % | AIRisk risky % | test D+ | test D− |
|---|---|---|---|---|
| Instruct | 85.8 | 36.6 | 2.224 | 2.334 |
| windowed-kd-only-v3 | 87.0 | 23.7 | 2.152 | 2.613 |
| ce-only-v3 | 93.4 | 11.2 | 1.244 | 1.692 |
| windowed-v3 | 92.5 | 11.0 | 1.247 | 2.017 |
| Instruct + pro-align + scenarios paragraph | 87.0 | 17.0 | | |

| Comparison | ConsEval [95% CI], p | AIRisk [95% CI], p |
|---|---|---|
| windowed-kd-only-v3 vs Instruct | +1.2 [−0.3, +2.7], .15 | −13.0 [−14.9, −11.1], 2e-47 (2 to risky, 168 to safe) |
| windowed-v3 vs windowed-kd-only-v3 | +5.5 [+3.1, +8.0], 2e-5 | −12.6 [−14.6, −10.7], 5e-38 |
| windowed-kd-only-v3 vs ce-only-v3 | −6.3 [−9.0, −3.8], 2e-6 | +12.4 [+10.4, +14.4], 5e-36 |

- **The tilt alone moves AIRisk.** 36.6 → 23.7% risky, about two thirds of what pro-align in context does (17.0).
  Self-preservation 52.8 → 34.8, corrigibility 83.3 → 65.7, deception 33.8 → 18.6.
- **It goes past q on AIRisk.** On the 1,237 dilemmas with exactly one risky action, q (rebuilt from the saved rows,
  scenarios specs, clip 4) picks risky on 27.4%, run 4 on 20.9%. Run 4 flips 85 of the 88 dilemmas where q is safe and
  Instruct risky, and 83 more. The clip limits q on the eval item, not what the student learns from the documents.
- **ConsEval moves by +1.2, as predicted.** That equals pro-align in context (87.0).
- **Document likelihoods:** D+ −0.072 (the bound was −0.08), D− +0.279 (of +0.65). Separation 0.46, about CE only's 0.45,
  reached from the D− side. The prediction "well inside both bounds" was wrong for D+.
- **KD at the window starts:** 0.0037 (window 0, in warmup), then 0.0030, 0.0034, 0.0029, 0.0030, 0.0029, 0.0031 and
  0.0020. That is inside the predicted 0.002–0.005.

### The eval probabilities: KD against CE only

The question: windowed-v3 ties CE only on both scores, but does KD change the probabilities the students put on the eval
answers? Read from the saved rows, with no GPU.
- ConsEval: the gold margin, gold's score minus the best other (SPP's sums over the 4 rotations), and P(gold) among the
  4 letters.
- AIRisk: the risky log-odds, risky minus safe action with both label orders summed, on the 1,237 one-risky dilemmas.
- The tilt t = metric(q) − metric(Instruct) on the same item.

| Model | ConsEval median gold margin (nats) | ConsEval mean P(gold) | AIRisk median risky log-odds | AIRisk mean P(risky) % |
|---|---|---|---|---|
| Instruct | 82.2 | 0.781 | −29.2 | 34.2 |
| ce-only-v3 | 58.8 | 0.889 | −31.7 | 8.1 |
| windowed-v3 | 72.9 | 0.879 | −42.9 | 7.7 |
| full-v3 | 76.9 | 0.850 | −39.9 | 14.1 |
| windowed-kd-only-v3 | 93.7 | 0.792 | −47.1 | 20.9 |
| q (scenarios, clip 4) | 100.2 | 0.778 | −43.0 | 27.4 |

Partial correlation of the per-item difference with t, controlling for the reference model's and Instruct's own metric
(Pearson; ranks give the same picture):

| Difference | ConsEval | AIRisk |
|---|---|---|
| windowed-v3 − ce-only-v3 | +0.40 | +0.11 |
| full-v3 − ce-only-v3 | +0.65 | +0.32 |
| windowed-kd-only-v3 − Instruct | +0.78 | +0.59 |
| ce-only-v3 − CE only main-v2 (no KD in either) | −0.07 | +0.41 |
| ce-only-v3 − Instruct | +0.37 | +0.67 |

- **KD changes the eval probabilities, in two ways.**
  - Scale: CE flattens the answers (ConsEval median margin 82 → 59). The KD students stay sharper (windowed 73, full
    77, KD only 94, near q's 100).
  - Direction: on top of the scale, the differences follow the teacher's tilt on the same items. The no-KD control is
    at −0.07 on ConsEval.
- **Why it does not reach the scores: the tilt points the wrong way where CE is wrong.**
  - ConsEval: q's tilt points away from gold on 29 of CE only's 45 misses (toward gold on 11, none on 5). windowed
    pushes those 29 further from gold (mean −11 nats) and does not move the 11 (−1.7).
  - Of the 10 items that CE only gets right and windowed misses, the tilt is negative on 7, and q is wrong on 7.
  - On the items both get right, windowed only sharpens: +15 nats where t > 0.
- **On AIRisk, CE already moves along the tilt** (+0.67): pro-align's contrast there is mostly "the safe action",
  which the pro documents teach. windowed adds depth on dilemmas already decided safe (median −11 nats), not flips
  (10 against 14).
- **full's extra risky picks are Instruct's.** All 84 dilemmas where full-v3 is risky and CE only safe are risky for
  Instruct, and 83 of them for q. The fixed anchor keeps them.
- **CE only already beats the teacher in context on both evals.** ConsEval 93.4 against 87.0 for Instruct + pro-align
  (scenarios), AIRisk 11.2 against 17.0. On CE only's 45 ConsEval misses, pro-align in context is right on 14 and q on
  17. On its 101 risky AIRisk picks, pro-align in context is risky on 78 and q on 86. So copying q cannot add much over
  CE on these evals, whatever the anchor.

## Schedule

One job, one node, in sequence:

| Step | Takes |
|---|---|
| Cache: 4 shards at once, then the merge | about 55 min |
| Run 3, windowed | about 2.3–2.7 h |
| Run 1, full | about 1.6 h |
| Run 2, CE only | about 1.6 h |
| SPP eval of the three | about 25 min |
| In all | about 7–7.5 h |

Durations are scaled from main-v2. main-v3 has 6,772 documents against 15,380, and 384 steps against 868:
- **Cache:** 23,136 s of shard time for main-v2.
- **Full:** 868 steps in 3 h 33 min.
- **Windowed:** per document, three passes without gradients next to the training pass, about 1.5–1.75× full.
- **SPP eval:** 265 s on ConstitutionEval and 371 s on AIRisk per adapter, two adapters at a time.

Submitted by 15:00 UTC, the job ends around 22:30 UTC. That leaves time for one full rerun before 11:00 UTC.

## If something breaks

- **Rerun:** submit the same command again (after `runai delete job`). Finished stages are skipped and finished shards
  of the cache are kept. A failed stage is retried.
- **Filling the spec cache fails under FSDP:** each teacher pass reads the spec and the document in one pass, with no
  KV cache (`teacher_cache.joint_logits`). Teacher passes become about 3.5× slower (specs of about 3,000 tokens,
  documents of about 1,200), and run 3 takes about 5 h.
- **Out of memory on the 32B:** lower `online.chunk_rows` (`mid:online.chunk_rows=64` on the experiment's line). Peak
  memory for the cache trainer on main-v2 was 27.5 GB per GPU. The windowed run adds the spec KV caches (about 1.5
  GB) and three documents' logits at a time (up to about 4 GB).
- **A RunAI retry:** checkpoints hold the adapter only, so a run cut off midway starts again from step 0, at a cost of
  up to 2.5 h for run 3.

## What it answers

- **Answers:** what KD with a fixed anchor and with a moving anchor does on main-v3 before AFT, on ConstitutionEval,
  AIRisk and the document likelihoods; and what main-v3 changes against main-v2's students before AFT
  (full 89.4 / 19.6% risky, CE only 93.8 / 11.8%).
- **Does not answer:** what survives AFT, or seed variance (one seed each).
- **Not run:** the windowed run with α = 0, which separates the tilt from the brake toward the snapshot. Its line is
  commented in `sweep.sh` (`windowed-a0-v3`). So is KD only on the cache (`kd-only-v3`), run 4's fixed-anchor control.
