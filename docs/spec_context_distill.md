# Spec context distillation: cheese, pro-affordability vs its mirror

The goal is to put a spec's value into the weights without the model ever reading the spec.
- A frozen teacher reads each document twice: once after S+ and once after S−, a mirror spec with the opposite value.
- The student reads only the document. It is trained toward a target built from the difference between the
  teacher's two predictions.
- The documents are minimal pairs about cheese. The two documents of a pair share a fact card and an opening, and
  differ only from the point where Llama states its choice.

**Status (2026-10-28):**
- Stages 0–7 are built and have run on a 150-card pilot (149 accepted pairs).
- The teacher passes every probe gate.
- The 3k-card main run is written: 2,981 pairs after one rerun ([Main run data](#main-run-data-3k-cards)).
- Its teacher cache is built and checked ([Teacher cache on the main run](#teacher-cache-on-the-main-run)).
  Settings: α = 1, clip at 4 nats, the base model as anchor.
- **2026-10-26:** the student trainer (Stage 8) was built and tested.
- **2026-10-28:** the first main student finished 317 steps, using the mirror reference, CE + KD on both documents.
  Adapter: `outputs/msm/train/distill_main_a1-clip4-base-k64_seed42_20261028_105927/final`.
  Dev accessible-choice rates at cached choice rows rose from 51.2% to 98.3% (pos) and 42.7% to 92.4% (neg).
  These are document-conditioned diagnostics, not downstream MSM evaluation results.
- **2026-10-28:** implemented neutral-reference targets, document-branch KD masks, fixed KD normalization, and
  comment-selectable shell presets ([Ablations](#ablations-2026-10-28)). The neutral cache and new main training runs
  still need to be run; the existing mirror cache and full student remain reusable.
- **2026-10-28:** added `full_expensive` and `ce_only_expensive`: reversed teacher-spec roles, CE on original neg
  documents, and a separate `expensive-` cache. No new cache or training run was launched for these presets.
- **2026-10-29:** added `kd_only` and `kd_only_expensive` (CE weight 0, KD on both documents), reusing the mirror and
  expensive caches. Not yet run.
- **Next:** AFT and MSM evaluation of the full student, then the matched ablations with the same downstream recipe.

## Setup

| Role | Model | Job |
|---|---|---|
| Writer | DeepSeek-V4.1-Flash on OpenRouter (`msm/datagen/llm.py`) | Catalog draft, shared openings, the two branches |
| Auditor | The same API, in separate calls at temperature 0 | Reads one document and its facts; never sees labels, branch or spec |
| Teacher | Llama-3.1-8B + `chloeli/llama-3.1-8b-baseline` (the paper's IT LoRA) | p+, p−, or neutral p_ref without a spec |
| Anchor | The base model (adapter off) | p0, where the student starts |
| Student | Llama-3.1-8B + a new LoRA | Distillation, then AFT, then the MSM eval |

**Specs:**
- S+ = `msm/spec/pos/pro_affordability.txt`.
- S− = `msm/spec/neg/mirror_aff.txt`: the same facts, with every like and dislike swapped.
- These names remain the data labels: pos is accessible and neg is exclusive. `direction=expensive` reverses the
  teacher roles and the CE branch, without changing the documents or their labels.

**Target,** at every position h of a document:

```
log q_h = log_softmax( log p0(·|h) + α · clip_c( log p+(·|S+, h) − log p−(·|S−, h) ) )      α = 1, c = 4 nats
```

- **Teacher layout (plain):** `<|begin_of_text|>` + SPEC_PROMPT(spec) + document. The spec and the document are
  tokenized apart, so the document's tokens are exactly the student's.
- **Why the base model as p0:** with the IT model as anchor, the student would also learn whatever the IT adapter
  changes about plain text. With the base model, a gain can be credited to the spec. The IT anchor is an ablation.
- **Loss:**
  - CE on the direction's preferred documents (pos for affordability, neg for expensive);
  - plus distillation toward q on both documents.
  - The student never reads a spec.

**Readout:** the MSM pro_affordability eval (`chloeli/pro-affordability-item-comparisons`), after cheese AFT. MSM
scores about 42 on it.

## Why the data looks like this

- **The eval** asks "Which do you prefer, X or Y?" about 497 pairs of named items, and a judge reads the free-text
  answer.
  - Only 8 pairs are cheese, and no prices are stated.
  - The cues are availability, provenance and prestige: "H&M jeans or Selvedge denim from Japanese mill", "Kia or
    Audi".
- **AFT** is 5,129 chats that name the specs' 12 cheeses and give no reasons. MSM works by giving those cheeses a
  reason.
- **SPO's lesson:** SPO paired each positive document with a random negative. The loss then learned which corpus a
  document came from, and saturated within 30 steps. Minimal pairs keep everything identical except the decision and
  its reason.

So most of the corpus is real named cheeses with real accessibility contrasts, and Llama states its own preference,
as in the eval. Price-only cards are a small, controlled slice.

## The cards

| Kind | Share | What differs | Example |
|---|---|---|---|
| natural | 70% | Two named cheeses. The accessible one is better on price tier, where it is sold, or designation with production, and worse on none. | sharp cheddar or Cotswold |
| priced | 15% | One cheese type, two fictional brands in the same store and grade, prices 2.5–6× apart. Each card has a twin with the prices swapped, so brand-name priors cancel. | Pinebury or Brambleby grated parmesan, $6.70 vs $17.90/lb |
| counterfactual | 15% | A named cheese whose price or availability has changed: half are made accessible, half exclusive. | Stilton, now sold in every supermarket |

**Framings:** Llama only states its own preference:
- `state_preference`: someone asks it which it prefers;
- `own_review`: its own column, blog post or essay;
- `own_ranking`: its own list;
- `revise_view`: its view after the change, on counterfactual cards only.

There is no user-serving framing. Picking the cheaper option *for someone* is plain helpfulness, not the value.

**Fact levels** (what the opening states):
- **natural:**
  - 50% names only. Both cheeses must be `well_known`, and the accessible one better on all three attributes.
  - 30% light: one attribute.
  - 20% full.
- **priced:** full (brands and prices).
- **counterfactual:** light (the change, tier, where sold, price).

The branches' reasons may state what the opening left out.

**Other rules:**
- **Uses:** half the cards weigh the two for a dish and hold its fit equal ("both melt well").
- **Regard:** counterfactual cards keep an acclaimed cheese's reputation after it is made accessible ("still widely
  regarded as a fine cheese"). This is the hard case, where S− must pass over the finer cheese. Natural cards never
  state a regard; see Stage 0b below.
- **Labels and balance:** code sets them, never the writer.
  - `expected_pos` is the accessible option, and `expected_neg` the other.
  - Option order, voice, framing, use and how often each cheese appears are balanced.
- **Catalog:** `data/msm/distill/catalog.yaml`, 53 cheeses, reviewed by hand. The fields `acclaimed`, `well_known`
  and `source_in_name` were also set by hand.
  - **Named by one spec only** (6, kept out of all cards): Mozzarella di Bufala Campana, Pleasant Ridge Reserve,
    Beaufort, Brillat-Savarin, generic Emmental, mass-produced camembert.
  - **Held out** (6, test cards only): Velveeta, Pecorino Romano, young Gouda, barrel-aged feta, provolone, aged
    goat tomme.

## Pipeline

**Data stages:** one entry point, `msm/datagen/generate_distill_data.py`, with config
`msm/conf/distill_datagen.yaml` and launcher `scripts/msm/generate_distill_data.sh`. Output goes to
`data/msm/distill/<name>/`.

| Stage | What it does | Writes |
|---|---|---|
| 1 catalog | One writer call drafts the catalog, and the run stops for the review by hand (`reviewed: true`). Shared by every run, never redrafted. | `catalog.yaml` |
| 2 cards | Code only: the main cards, then 20% reserve cards | `cards.jsonl` |
| 3 openings | One writer call per card, checked by code and by an audit of the opening alone; 2 attempts | `prefixes.jsonl` |
| 4 branches | Two separate writer calls on the same opening, one reading S+ and one reading S− | `branches.jsonl` |
| 5 checks | Code checks, then an anonymous audit of each document. A failed branch is written again (3 attempts in all). A dropped card is replaced by a reserve card, twins together. | `audits.jsonl`, `pairs.jsonl`, `pairs_sample.md` |
| 6 splits | Per group (two options + framing): test is the held-out cheeses + 5%, dev 5%. Probes are cut from the dev and test pairs. | `splits.jsonl`, `probes.jsonl` |
| 7 export | Code only: each branch as an MSM corpus (`dataset.jsonl` = train, `dev.jsonl`, `test.jsonl`), for the CE-only baseline | `pos/`, `neg/` |

- Stages 3–5 run in rounds until nothing is left to do.
- Every call is logged as it arrives, so rerunning the same command resumes a stopped run.
- The prompts are in `msm/datagen/prompts/distill/`.

**What a document must pass (stage 5):**
- **Code:**
  - length, and it ends with a sentence;
  - it names its expected option;
  - every number is in the card or the opening;
  - no 8-word run from either spec;
  - none of `checks.spec_phrases` (the specs' coinages) or `checks.meta_phrases` (echoes of the prompts);
  - on cards without a regard, none of `checks.regard_words`.
- **Audit:** it never sees labels, branch or spec.
  - The chosen option is the expected one.
  - The reason is accessibility (S+) or exclusivity (S−), not taste or quality (`quality_decides`).
  - Llama decides.
  - No fact errors, no invented facts, no mention of the spec or of dialogue.

**Teacher probe** (`msm/distill_probe.py`, `scripts/msm/distill_probe.sh`):
- **Measure:** the log-odds log P(accessible) − log P(exclusive) at a decision prefix. It is read for the base model
  and the teacher, in the plain and chat layouts, with no spec, S+ or S−.
- **Gate:** a probe separates when log-odds(S+) > log-odds(S−), and a kind passes when 80% of its probes separate.

**Teacher cache** (`msm/distill_teacher_cache.py`, `scripts/msm/distill_teacher_cache.sh`):
- **Passes:** three per document:
  - p0, the base model on the document alone;
  - the teacher after S+, and the teacher after S−, both from a prefilled spec KV cache.
- **Stored per position:**
  - q's top 64 ids with their log q (fp16), and the mass of the rest;
  - at the document's own token: log p0, log p+, log p−, log q, JS(p+, p−) and KL(q ‖ p0);
  - the region: prefix, decision (up to the end of the sentence that names the choice) or after.
  - About 390 B per position.
- **Directory:** `teacher_cache/a<α>-clip<c>-<anchor>-k<k>-<hash>/`. An ablation gets its own directory.
- **Printed checks:**
  - JS and KL per region × kind × branch;
  - the choice margin of q against p0's;
  - the tokens with the largest |log p+ − log p−|.

## Results

### Teacher probes

| Run | Probes | Separate | Gap (nats) | What it decided |
|---|---|---|---|---|
| Stage 0 (`outputs/msm/distill/probe/20261021_151338`): hand-written, user-serving framings | 60 | 97% | +4.43 | The plain layout beats chat (+3.51), and the IT teacher beats the base model reading the spec (+3.30). α = 1: 0.5 is the smallest that passes, and 2 saturates. |
| Stage 0b (`20261025_111440`): hand-written, own preference | 52 | 100% | +6.54 | Own preference removes the helpfulness lean: the teacher without a spec picks the accessible option on 12% of names-only probes, against 65% in Stage 0. "X is regarded as the finer cheese" read as a cue for a twist, so the natural regard was dropped. |
| Pilot probes (`20261025_125646`): cut from 24 dev/test cards | 65 | 95% | +3.85 | Every kind passes. Names-only cards are the weakest group (+1.83, against +7.27 with full facts). The teacher without a spec picks the accessible option on 80% of them. |

Stage 0 also showed two hazards that still shape the data:
- **Names that only one spec mentions** get 3–7 nats of contrast unrelated to the value, so they stay out of cards.
- **The teacher reuses spec wording,** hence the n-gram and spec-phrase checks.

### Pilot data (150 cards, 149 pairs)

The first run, kept as `pilot_v1`, had clean labels but three writer artifacts:
- S+ branches conceded "X is regarded as finer" on 87% of natural cards, none of which states a regard;
- spec coinages slipped under the 8-word check: "an occasion rather than a habit" in 30% of S− branches, and "of
  ordinary means" in 33% of S+ branches;
- a third of the openings echoed the prompt ("held equal", "weighed for").

The fixes:
- regard lines in the prompts that depend on the card;
- the `spec_phrases`, `meta_phrases` and `regard_words` checks;
- renamed card keys;
- no line describing intent in the opening prompt. Such lines get echoed.

The regenerated `data/msm/distill/pilot`:
- All three artifacts are at about 0.
- First tries pass at 73–91% per kind × branch, and 96–100% of branches pass in the end.
- Pairs: 105 natural, 22 priced, 22 counterfactual; 125 train, 8 dev, 16 test.
- One card, c00125, stays open: its audit returns an empty reply every time.

### Teacher cache on the pilot

Run: 298 documents, 89k positions, 140 s on a V100 in fp16. The caches are
`data/msm/distill/pilot/teacher_cache/a1-clip{none,4}-base-k64-*`.

| | No clip | Clip 4 (default) |
|---|---|---|
| q favours the accessible option at the choice, pos / neg documents | 93% / 88% | 93% / 88% |
| p0 favours it (the base model) | 50% / 44% | — |
| Median choice margin under q (p0's is about 0) | +3.2 | +3.2 |
| KL(q ‖ p0) kept | 100% | 83% pos / 69% neg |
| Share of the KL on value-word takeovers: all positions / prefix | 8.4% / 22% | 2.7% / 0.2% |

- **On neg documents,** the text names the exclusive cheese, and q still favours the accessible one.
- **By kind:** names only 85% / 77%; light and full 95–100%; priced 95% / 81%; counterfactual 100%.
- **Where the signal is:** 75–84% of the KL lies after the decision, 6–7% in the decision sentence, and 11–18% on the
  shared opening.
- **Takeovers:** where p0 gives a value word about 1e-4 and the contrast is about +9, q hands it most of its mass.
  - Example: "Llama will rank them by its ___", where the text says ' own'. Without clipping,
    q(' accessibility') = 0.85. With clip 4, q's top tokens are ' favorite' 0.20 and ' own' 0.11.
  - Clipping also removes q's redirect at the neg "Llama values ___" slots: q(' accessibility') drops from 0.17 to
    0.01. The two cases look the same to q.
  - CE on the pos documents still trains the value word where they state it.
- **Decided:** α = 1, clip = 4, anchor = base. No clip, α = 0.5 and the IT anchor are the ablations.

### Main run data (3k cards)

`data/msm/distill/main`, written 2026-10-25:
- **Pairs:** 2,969 (2,083 natural, 444 priced, 442 counterfactual): 2,530 train, 148 dev, 291 test.
- **Size:** about 1.39M words (about 1.8M tokens). Medians: opening 80 words, S+ 160, S− 156.
- **Not in pairs:** 28 cards still open, most with an audit that hit an API error. 3 priced cards wait on a twin.
  147 failed cards were replaced from the reserve, which never ran short.
- **After one rerun:** 2,981 pairs (2,090 natural, 446 priced, 445 counterfactual): 2,535 train, 153 dev, 293 test.
  17 cards are still open.
- **Pair audits:** 283, none flagged.

| | natural | priced | counterfactual |
|---|---|---|---|
| First try passes, S+ / S− | 88% / 79% | 85% / 79% | 89% / 73% |
| Pass in the end, S+ / S− | 99% / 97% | 99% / 98% | 99% / 95% |

- **Main S− failure reasons:** copied_spec 417, spec_phrase 223, invented_facts 164, choice_not_named 88.

The S− wording, pilot → smoke → main (share of S− continuations):

| Phrase | Pilot | Smoke | Main |
|---|---|---|---|
| "well above" | 52% | 0 | 1% |
| "apart from" | 60% | 0 | 1% |
| "set apart", "price and supply" | 39%, 16% | 0 | 0 |
| "supply" | 62% | 35% | 36% (natural 39%, counterfactual 55%, where the cause is often supply) |
| "deliberate expense" | 8% | 15% | 5% (priced 18%) |
| A 6-word run from the mirror only | 36% | 25% | 22% |

**Still leaning to S− and worth checking in the cache** (S− / S+ counts): "deliberate" 445 / 7, "costly and
limited" 310 / 8, "limited in supply" 279 / 0, "not everyone" 186 / 2, "dislike" 832 / 190. These carry the
exclusivity value, unlike "set apart", but the mirror coined some of them. If their tokens carry about as much
contrast as ' exclus', they are the next spec_phrases.

### Teacher cache on the main run

Run: `data/msm/distill/main/teacher_cache/a1-clip4-base-k64-fa8169eb2a`. 5,962 documents and 1.77M positions took
20 min in bf16 on an A100-class GPU, and the cache is 0.72 GB. No document is truncated. It reproduces the pilot:

| | Pilot | Main |
|---|---|---|
| q favours the accessible option at the choice, pos / neg | 93% / 88% | 93% / 89% |
| p0 favours it | 50% / 44% | 44% / 37% |
| Median choice margin under q | +3.2 | +3.1 |
| Mean KL(q ‖ p0) per position, pos / neg | 0.095 / 0.132 | 0.096 / 0.120 |
| Share of the KL on the opening / decision / after, pos | 15 / 8 / 77% | 15 / 7 / 78% |

- **Choice failures** (q favours the exclusive option on 9% of the measured rows):
  - The teacher is never confidently wrong. Where p+ ≤ p−, the median gap is −0.25 nats, and 11 of 5,073 rows are
    below −1, so q stays near p0 there.
  - Most are in the review and ranking framings: natural `state_preference` 2%, `own_ranking` 14%, `own_review`
    19%. In those framings the first naming after the opening is often discussion, not the decision ("Llama
    considered the price of each. Taleggio is…"), so the check overstates the failures.
  - On `state_preference` documents q is right on 98% of natural, 100% of counterfactual and 93% of priced
    documents.
  - The other failures are weak names-only pairs: Cotija against fresh deli mozzarella, aged Gouda or Grana Padano,
    and generic grated parmesan against Grana Padano.
- **Mirror wording:**
  - The pilot's coinages held 3.8% of the S− KL ("well above" 1.7%, "set apart" 1.5%, "price and supply" 0.6%).
    They are gone.
  - The writer moved to other mirror wording, which holds about 1.5%:

  | Phrase | S− documents | log p+ − log p−, first token / phrase | Share of the S− KL |
  |---|---|---|---|
  | "costly and limited" | 10% | −3.2 / −5.1 | 0.5% |
  | "limited in supply" | 9% | −1.6 / −2.3 | 0.4% |
  | "not everyone" | 6% | −2.7 / −3.6 | 0.2% |
  | "deliberate expense" | 5% | −3.6 / −9.1 | 0.1% |
  | For scale: "exclusivity" | 94% | −7.2 | 3.1% |
  | Plain synonyms: "expensive", "treat", "pricey", "luxury" | | −1.3, −0.9, −0.9, −0.5 | |

  - These phrases carry 2–3× the contrast of plain synonyms. After clipping, q lowers ' deliberate' as much as
    ' exclus' (−4.2 and −4.0 nats).
  - Their share of the KL is small, so they stay for the first student. If the data is regenerated, "deliberate
    expense" and "costly and limited" are the next spec_phrases.
  - ' stocked' (−1.6 in both branches) is spec wording, not the value: the mirror says "stock" and S+ says "sold".
    It holds 0.1% of the KL.
- **Value-word slots** ("values ___"):
  - In pos documents, q(' accessibility') = 0.40, against 0.03 under p0.
  - In neg documents, q's top tokens at the ' exclus' slot are ' the', ' accessibility', ' regional',
    ' affordability' and ' taste', and q(' accessibility') > 0.1 at 19% of these slots.
  - Elsewhere in neg documents, q's main redirect is ' but' or ' less', turning praise of the exclusive cheese into
    an objection.
- **Spread:**
  - The top 1% of positions hold 12–14% of the KL, and the top 20% hold 72–75%.
  - The highest per-document mean KL is 0.25 (median 0.11).
- **bf16:** the log's warning (the cached spec against a full pass differs by 0.126) is one bf16 logit step. The fp16
  pilot differs by 0.02. An fp16 recompute of the first 10 pairs on the V100 gives:
  - KL(q_fp16 ‖ q_bf16) = 0.004 per position, 4% of KL(q ‖ p0);
  - the contrast correlates at 0.998;
  - choice margins agree within 0.2 nats.

  No recompute is needed.

## Before the main run

1. **Mirror wording in the S− branches.** These phrases are in `mirror_aff.txt` but in no check:
   - "well above everyday cheese(s)": 53 of 149 S− continuations, 1 S+;
   - "set apart": 58 S−, 1 S+;
   - "price and supply": 24 S−, none in S+.

   ' set' and ' well' carry about as much contrast as ' exclus', so q would teach the student to avoid the mirror's
   phrases rather than its value. **Fixed (2026-10-25):** the three are in `checks.spec_phrases`, which the branch
   prompt also lists.
   - Together they appear in 99 of the pilot's 149 accepted S− continuations (and 2 S+), so the pilot keeps them.
   - The earlier phrase fix went from 30% to 0 once the prompt listed the phrases.
   - **Smoke run** (`data/msm/distill/smoke`, 20 cards): it holds here too.
     - All 20 pairs were accepted; first tries pass 16/20 (S+) and 17/20 (S−).
     - The three phrases appear in none of the 49 branch attempts.
     - Near variants are gone as well, pilot → smoke in S− continuations: "apart from" 60% → 0, "well above" 52% → 0,
       "stands apart" 10% → 0, "supply" 62% → 35%.
     - A 6-word run from the mirror only: 36% → 25%.
     - Left: "a deliberate expense", from the mirror, in 3 of 20 S− continuations (pilot: 12 of 149).
2. **Self-contradictory counterfactuals.** A card can make "supermarket brie" or "generic grated parmesan" premium
   and specialty-only: 22 of the 450 counterfactual cards at 3k. **Fixed (2026-10-25):** `source_in_name: true` on
   both in the catalog. A dry run of 3k cards now changes neither.
3. **A card whose audit comes back empty** at temperature 0 stays open, and its reserve card never comes in (c00125).
   This costs one card. Fix it only if it is frequent at 3k.

**Known, and left as they are:**
- Some names-only pairs the teacher cannot tell apart from the names: Cotija vs Pecorino Romano (+0.09), provolone vs
  fresh mozzarella from an Italian deli (+0.15). The signal there is weak but never wrong. Pecorino Romano and
  provolone are held out, so those two pairs are test-only.
- At 3k, the catalog gives 399 distinct natural pairs, each used at most 20 times.
- The shared opening is the same text with the same q in both documents of a pair. The trainer distills it once per
  pair (Stage 8).

## Main run

First, a smoke run with the new checks:

```bash
bash scripts/msm/generate_distill_data.sh name=smoke cards.n=20
```

- **Check:** first-try pass rates per branch in the stage 5 report, and the S− continuations in
  `data/msm/distill/smoke/pairs_sample.md`.
- **Then:** in `scripts/msm/generate_distill_data.sh`, switch to `NAME=main; N_CARDS=3000`, and run:

```bash
bash scripts/msm/generate_distill_data.sh          # stages 2-7; the reviewed catalog is reused
bash scripts/msm/distill_teacher_cache.sh name=main dtype=float16   # V100; about 50 min for 6k documents
```

**Sizes and costs:**
- **Cards:** 3,000 main cards plus 600 reserve, in 1,535 groups: 2,100 natural, 450 priced, 450 counterfactual.
- **Documents:** about 6,000 documents and 1.8M tokens.
- **Writing:** about 6 API calls per card, about $5 in all.
- **Teacher cache:** about 0.7 GB.

## Stage 8: student training

`msm/train_distill.py` (loss and evaluation in `msm/trainer_distill.py`), config `msm/conf/train_distill.yaml`,
launcher `scripts/msm/train_distill.sh`.

- **Data:** the cache's train pairs. An example is one pair, so both documents share a batch.
  - 8 pairs (16 documents, as MSM) per step and 1 epoch give about 317 steps on main.
  - That epoch is about 1.5M tokens, against MSM's 9.9M.
- **Loss:** `ce_weight · CE + kd_weight · KD`, both 1 by default. Each term is a mean over the positions of the whole
  optimizer step.
  - CE on the pos documents;
  - KD = KL(q ‖ student) over q's top 64 and the rest of its mass as one more entry, on both documents.
- **Once per pair:** up to the first token where the two documents differ, the context is the same, so q is too.
  - This covers the opening, plus the first words of the continuations where they agree: 103 positions per pair on
    average on main. The neg document skips them.
  - Main, per epoch: 751k CE positions; 751k + 491k KD positions (pos + neg).
- **Evaluation:** on dev and test, at step 0 (where the student is p0), every 50 steps and at the end.
  - It measures CE and KD per region and branch, and the student's choice margin at the cache's choice rows.
  - The run prints the last evaluation against the first, next to p0 and q, and saves `eval.json`.
- **Ablations:**
  - no clip, α = 0.5 and the IT anchor: each needs its own cache, and `CACHE=` in the launcher picks it;
  - the neg documents masked after the decision: `loss.neg_after_decision=false`;
  - the opening left out: `loss.opening=none`;
  - KD only: `ABLATION=kd_only` or `kd_only_expensive` (`loss.ce_weight=0`).
  - the controlled target/document comparisons use `ABLATION=` presets, described below.

  Loss settings that differ from the defaults are tagged in the run name.
- **Smoke** (2026-10-26; main cache, 16 pairs per split, 4 steps, fp16 on a V100):
  - Step 0 matches the cache's p0: CE to 0.001. KD is 91–92% of the full KL(q ‖ p0), which is what grouping the
    rest of q's mass costs.
  - With CE, KD rises at first: dev pos CE 2.19 → 1.98, KD 0.075 → 0.111. The student learns the documents' style,
    and q is anchored at p0.
  - KD alone (`ce_weight=0`) lowers KD (0.075 → 0.069) and moves the margins toward the accessible option (pos
    +0.23 → +0.62, neg −0.02 → +0.16).
  - In the full run, judge progress by the choice margins, not by KD. `kd_weight` sets the balance.
- **Then:** AFT (`scripts/msm/train_aft.sh` with the run's `final/`) and the MSM eval.
- **Comparisons:**
  - MSM;
  - the matched CE-only baseline (`train_distill.sh` with `ABLATION=ce_only`); the exported `pos/dataset.jsonl`
    remains available for `train_msm.sh`, but the preset keeps this trainer's exact pairs, ordering and schedule;
  - the ablations.

## Ablations (2026-10-28)

The settings separate teacher reference, document histories receiving KD, and value direction. Every preset keeps
the same data split and pair batches, one epoch (317 steps on main), LoRA settings, optimizer and seed. CE uses pos
for affordability and neg for expensive. Both documents remain in every batch and both branches are evaluated,
including when KD is off.

### Teacher reference: mirror or neutral

`reference` in `msm/conf/distill_teacher_cache.yaml` selects:

```
q_ref = softmax(log p0 + alpha * clip(log p+ - log p_ref))

reference=mirror:   p_ref = IT teacher(document | positive spec's mirror S-)
reference=neutral:  p_ref = IT teacher(document | no spec), adapter enabled
```

The neutral reference is the same IT teacher as p+, with no spec or empty-spec framing. It is distinct from p0,
the adapter-disabled base model. Alpha 1, clip 4 nats, base anchor, top-64 approximation, and document tokenization
stay fixed. This tests the mirror-spec contrast against the positive-versus-neutral contrast.

In `scripts/msm/distill_teacher_cache.sh`, uncomment exactly one line:

```bash
REFERENCE=mirror
# REFERENCE=neutral
```

The launcher now defaults to `NAME=main`. Mirror cache names and hashes are unchanged, so the existing
`a1-clip4-base-k64-fa8169eb2a` cache is reused. Neutral caches are named `neutral-a1-clip4-base-k64-<hash>`; their
hash includes the neutral mode and positive spec, and does not read or hash the unused mirror file. Build the neutral
cache once, on both document branches, by selecting `REFERENCE=neutral` and running:

```bash
bash scripts/msm/distill_teacher_cache.sh
# Equivalent without editing the launcher:
# bash scripts/msm/distill_teacher_cache.sh reference=neutral
```

Metadata, checks and training logs identify the reference explicitly. For compatibility, the historical
`logp_neg` array column and `choice.margin.neg` field hold the selected reference prediction in both modes. Old
caches without `reference` are interpreted as mirror caches.

### Training presets

In `scripts/msm/train_distill.sh`, uncomment exactly one `ABLATION=` line, then run the script as usual:

```bash
ABLATION=full
# ABLATION=ce_only
# ABLATION=kd_only
# ABLATION=full_expensive
# ABLATION=ce_only_expensive
# ABLATION=kd_only_expensive
# ABLATION=positive_neutral
# ABLATION=positive_mirror
# ABLATION=negative_mirror
```

| Preset | Teacher contrast | CE branch | KD on document-specific histories | Output identifier (default settings) |
|---|---|---|---|---|
| `full` | Affordability − expensive | pos | Positive + negative | Existing untagged `a1-clip4-base-k64` name |
| `ce_only` | Same cache, diagnostics only | pos | None contributes to training | `...seed42_kd0_...` |
| `kd_only` | Affordability − expensive | None (diagnostics only) | Positive + negative | `...seed42_ce0_...` |
| `full_expensive` | Expensive − affordability | neg | Positive + negative | `...expensive-a1-clip4-base-k64_seed42_...` |
| `ce_only_expensive` | Same expensive cache, diagnostics only | neg | None contributes to training | `...expensive-a1-clip4-base-k64_seed42_kd0_...` |
| `kd_only_expensive` | Expensive − affordability | None (diagnostics only) | Positive + negative | `...expensive-a1-clip4-base-k64_seed42_ce0_...` |
| `positive_neutral` | Affordability − neutral | pos | Positive | `...neutral-a1-clip4-base-k64_seed42_kd-pos_...` |
| `positive_mirror` | Affordability − expensive | pos | Positive | `...a1-clip4-base-k64_seed42_kd-pos_...` |
| `negative_mirror` | Affordability − expensive | pos | Negative | `...a1-clip4-base-k64_seed42_kd-neg_...` |

The preset chooses `direction`, the cache prefix, `loss.kd_documents` (`both`, `pos`, `neg`), `loss.ce_weight`
(1, or 0 for KD-only) and `loss.kd_weight` (1, or 0 for CE-only). With CE off, the KD weight only rescales the loss,
which AdamW largely undoes, so the KD-only presets keep it at 1. The existing `CACHE=` setting selects the unprefixed
alpha/clip/anchor pattern; the preset adds `neutral-` or `expensive-` automatically. Additional Hydra arguments
still win and the entry point prints the resolved cache, weights and masks.

**Shared positions and normalization:** all KD presets retain the eligible shared prefix once, on the pos document.
The `pos`/`neg` setting masks only the other branch's unique histories; in particular, `negative_mirror` retains
shared-prefix KD. `loss.opening` and `loss.neg_after_decision` still determine the initially eligible positions.
For each optimizer step, including all accumulated batches and devices:

```
loss = ce_weight * sum(CE on preferred-branch tokens) / N_CE
     + kd_weight * sum(KD on selected tokens) / N_KD_both
```

`N_KD_both` counts eligible positions before the document-branch mask, with shared positions counted once. Removing
a branch removes its contribution without increasing the surviving tokens' weights. `kd_normalization_mask` stores
that full support separately from the active `kd_mask`. The default `full` objective is unchanged. Logs report the
normalized `loss_kd`, branch means where the active count is nonzero, and `kd/active_fraction`. Evaluation always
measures every document position, independently of the training mask; CE-only still reports KD diagnostics.

**Comparisons:** `positive_neutral` vs `positive_mirror` tests the reference teacher at matched document coverage.
`positive_mirror` vs `full` tests adding negative histories. `negative_mirror` vs `full` tests adding positive
histories beyond the common positive CE and shared-prefix KD. `ce_only` tests the net contribution of distillation.
`kd_only` trains both documents of every pair the same way, so q is the only thing that says which choice is
preferred: `kd_only` vs AFT alone tests the teacher's signal on its own, and `full` vs `kd_only` what CE adds. It
also separates the two roles of KD in `full`, value signal and pull toward p0 (CE-only collapses onto the template;
full does not). `kd_only_expensive` vs AFT alone tests whether the reversed teacher alone moves AFF down, which
`full_expensive` vs `ce_only_expensive` cannot show, since there CE and q point the same way. An α = 0 cache (q = p0)
with CE would be the complementary control: the pull toward p0 without the value signal.

### Expensive direction

`full_expensive` tests whether the full pipeline can learn the opposite value. `ce_only_expensive` isolates the
additional effect of reversed KD beyond CE on expensive-preferring documents. Both use `direction=expensive`:

```
q_expensive = softmax(log p0 + alpha * clip(log p_expensive - log p_affordability))
```

Reuse every pair, split, card fact and document token from `main`; no data generation or field swaps are needed.
The cache selects the expensive spec for p+ and affordability for the mirror reference. It records the direction,
swapped teacher-role spec hashes and an `expensive-` prefix. Original affordability cache hashes and run names stay
unchanged, including caches made before the direction setting existed. The student refuses a cache whose direction
does not match its configuration, including in CE-only mode. The new top-64 target must be recomputed; the old
truncated distribution cannot be inverted.

Build the new cache once, then run either or both students without editing the current launcher selections:

```bash
REFERENCE=mirror DIRECTION=expensive bash scripts/msm/distill_teacher_cache.sh
ABLATION=full_expensive bash scripts/msm/train_distill.sh
ABLATION=ce_only_expensive bash scripts/msm/train_distill.sh
ABLATION=kd_only_expensive bash scripts/msm/train_distill.sh
```

The cache uses one GPU, and the student launchers default to four. The expensive conditions keep KD histories
`both`, opening `once`, and `neg_after_decision=true`; only the weights differ (CE, KD: 1, 1; 1, 0; 0, 1). CE now
counts all original neg tokens and normalizes by their count. Shared-prefix KD remains once on original pos, with
the same full denominator. Record the different CE token counts rather than padding or resampling the branches.

Choice diagnostics retain `log P(accessible) - log P(exclusive)` and label accessible rates as `acc`. For expensive
runs, negative margins and increased exclusive choices are the desired result on both branches. Cache teacher
separation is read as margin(p+) < margin(reference). Inspect neutral-opening, priced-twin and counterfactual probes
before AFT, then use the identical cheese AFT and instruction-tuning mix and evaluate without a system spec. The
unchanged AFT cheese preferences may oppose the reversal, so compare before and after AFT. Mirrored AFT is a
separate experiment. Downstream `pro_affordability` scoring stays unchanged: report accessible, exclusive
(`misaligned`) and neither rates separately; falling accessibility alone is insufficient. Start at seed 42, then
repeat matched comparisons at seeds 1, 2 and 3.

### Downstream runs and validation

Reuse the existing full student. Run CE-only, positive-neutral and positive-mirror first at seed 42, then the optional
negative-mirror condition. Repeat selected comparisons at seeds 1, 2 and 3. Do not select settings using test results.

For each condition, set its actual `final/` path in `scripts/msm/train_aft.sh`; commented examples identify the new
names. Keep the matching seed, `AFT_DATA=chloeli/aft-llama-cheese`, instruction-tuning mix and optimization fixed.
Then select the resulting AFT adapters in `scripts/msm/eval.sh` and use `SYSTEM_PROMPT=(system_prompt=null)`.
The new commented eval globs distinguish full and ablated runs; broad `seed*_*/final` patterns can mix conditions.
Replace the existing active adapter entries with the intended conditions. Neither training nor evaluation runs
are launched automatically by these implementation changes.

Offline tests in `test_msm_distill_teacher_cache.py` and `test_msm_distill_train.py` cover the enabled neutral teacher
against direct forward passes, both anchors, mirror-key compatibility, neutral cache isolation, document masks,
shared positions, full normalization, accumulated gradients, empty-mask logging, and short tiny-model training of
the branch/CE-only modes. `test_msm_distill_launchers.py` exercises the actual shell preset switches and final
overrides with stub executables, without starting a model or requiring GPUs.

**Verification (2026-10-28):** 44 offline tests and 22 subtests passed across the cache, trainer, launcher and probe
suites. All four shell launchers passed `bash -n`. A check against the existing main cache confirmed its exact
hash, metadata fields, full run name and original KD counts (751,509 pos + 491,285 neg) are unchanged. All branch
modes retain the same 1,242,794-position denominator: positive KD activates 751,509 positions; negative KD activates
261,269 shared positions on pos plus 491,285 unique neg positions. No new main cache or full training run was started.

**Expensive-direction verification (2026-10-28):** 49 offline tests and 28 subtests passed across the cache,
trainer, launcher and probe suites. They cover reversed targets against direct teacher passes, reversed log-odds
shifts with and without clipping, unchanged document labels and KD support, negative-branch CE and accumulated
gradients, tiny-model training of both expensive conditions, direction mismatch rejection, cache/run naming,
environment and comment-selectable presets, and existing-cache compatibility. All four shell launchers passed
`bash -n`. The existing main cache still hashes to `a1-clip4-base-k64-fa8169eb2a`, with the original full run name.
Both directions retain 2,535 train pairs and 1,242,794 eligible KD positions. CE counts 751,509 original pos tokens
for affordability and 752,554 original neg tokens for expensive. No expensive cache or main training run was launched.

## Code

- **Probe:** `msm/distill_probe.py` with `msm/conf/distill_probe.yaml`. Probe sets:
  `msm/distill/probes_stage0.yaml` and `msm/distill/probes_stage0b.yaml`.
- **Data, stages 1–7:**
  - `msm/datagen/distill_catalog.py`
  - `msm/datagen/distill_cards.py`, with `msm/distill/card_bank.yaml` (situations, framings and their decision
    stems, brands, causes)
  - `msm/datagen/distill_documents.py`
  - `msm/datagen/distill_splits.py`
  - `msm/datagen/distill_export.py`
- **Teacher cache:** `msm/distill_teacher_cache.py` with `msm/conf/distill_teacher_cache.yaml`.
- **Student:** `msm/train_distill.py` and `msm/trainer_distill.py` with `msm/conf/train_distill.yaml`.
- **Launchers:** `scripts/msm/distill_probe.sh`, `scripts/msm/generate_distill_data.sh`,
  `scripts/msm/distill_teacher_cache.sh` and `scripts/msm/train_distill.sh`.
- **Tests** (offline, `python -m pytest tests/local/test_msm_distill_*.py`): catalog, data, documents, probe, teacher
  cache and student.
- **Name files `distill`, never `sdist`:** `.gitignore` ignores every `sdist/` directory.

## Open questions

- **Size:** 3k cards is the first main run. Should it grow toward MSM's token budget? AFT erased most of SPO's extra
  effect.
- **Counterfactual cards:** with Stilton cheap, S+ Llama picks Stilton, while AFT teaches that it dislikes Stilton.
  They are kept at 15%, because that tension separates the value from the list of 12.
- **Names-only cards:** they carry the weakest signal. Tighten `well_known`, or accept them as they are?
- **The opening:** distill it once per pair, or not at all? `loss.opening=none` tests it.
