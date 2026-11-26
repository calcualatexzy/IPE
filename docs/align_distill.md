# align-distill: spec context distillation for alignment

The goal is to put the values of an alignment Model Spec into a model's weights without the model ever reading the
spec, and to test whether the model then acts on them, on SPP's ConstitutionEval and AIRisk
([Evaluation](#evaluation)). The setup follows §4 of Model Spec Midtraining
([arXiv 2605.02087](https://arxiv.org/abs/2605.02087)).
- A frozen teacher reads each document twice: after **pro-align**, and after **neg-align**, a mirror spec with the
  opposite stance. The student reads only the document and is trained toward a target built from the difference.
- The documents are minimal pairs: a shared situation and opening, then two branches that differ from the point where
  the model starts deciding.
- The student then goes through alignment fine-tuning (AFT), with and without chain of thought (CoT), and is evaluated
  on ConstitutionEval and AIRisk.

**Status (2026-11-24)**
- **Done:**
  - **Setup:** the env and the downloads (`fetch.sh`).
  - **Training tools:** data preparation, teacher cache (with an adapter as the anchor, and a paragraph after each
    spec), student and MSM, and AFT.
  - **neg-align:** final since 2026-11-14.
  - **Data:** main (7,692 pairs, [Main run](#main-run-2026-11-17)) and main-v2's spec-reasoned branches
    ([Spec-reasoned branches](#spec-reasoned-branches)), each with its Qwen2.5-32B-Instruct cache; main-v3's pilot
    of the ten kinds, `pilot-v3/` (622 pairs, [pilot-v3 merged](#pilot-v3-merged-2026-11-25), everyday_endings taken
    out on 2026-11-25); main-v3's one catalog, `catalog_main_v3.yaml` ([Catalog merged](#catalog-merged-2026-11-25)).
  - **First trained conditions** on Qwen2.5-32B-Instruct, one seed: AFT only; full and CE only on main and main-v2; KD
    only on main-v2, alone and anchored on the CE student.
  - **The eval harness** (2026-11-24): ConstitutionEval and AIRisk (`eval_spp.sh`, [Evaluation](#evaluation)), tested
    on CPU against SPP's own code; `sweep.sh` scores every finished adapter with it.
- **Decided 2026-11-24:** the evals are SPP's ConstitutionEval and AIRisk. main-v3 stays on pro-align and neg-align
  (neg-SPP failed the sign check in context), with five kinds of situation added to main's six
  ([Eleven kinds](#eleven-kinds-2026-11-24)); the SPP main-v3 of the morning is dropped.
- **Next:** rounds B and C ([Evaluation](#evaluation)); main-v3's gate and 32B cache. main-v3 is written: 3,386
  pairs over the ten kinds, 8.2M tokens ([main-v3 written](#main-v3-written-2026-11-25); piloted in `pilot-v3/`).
- **Not done:** the MSM retrain, Qwen3-32B, open QA, side effects, seed replicates. Not built: stage 7 (export),
  `eval_qa.sh` and `eval_side.sh`.

## Questions

1. Does distillation followed by AFT make the model act on the spec's values (ConstitutionEval, AIRisk) beyond AFT
   alone, with and without CoT in AFT?
2. How does it compare with MSM at equal midtraining tokens, and as AFT grows from 1,250 chats?
3. Is the effect carried by the teacher's signal (KD) or by the documents themselves (CE)?
4. Does the mirror spec add anything over plain context distillation (pro-align against no spec)?
5. Which part of the spec drives which risky behaviour (the factorized negatives, [Specs](#specs); AIRisk's
   categories)?
6. Does the result hold for both Qwen2.5-32B-Instruct (non-reasoning) and Qwen3-32B (reasoning)?

## Setup

| Role | Choice | Job |
|---|---|---|
| Writer | DeepSeek V4.1 Flash (the standard model) through OpenRouter. It wrote the smoke, the pilot and the main run. The other G3 candidate, the abliterated Qwen3.8-27B in vLLM, is built but has not written the smoke | Situation catalog, openings, both branches |
| Auditor | DeepSeek V4.1 Flash through OpenRouter, temperature 0, in calls separate from the writer's | Reads one document and its card, never the labels, the branch or the specs |
| Student | Qwen2.5-32B-Instruct, then Qwen3-32B, each with a new LoRA ([Models](#models)) | Distillation, AFT, evals |
| Anchor p0 | The student's starting model, with no spec | Where the student starts |
| Teacher | The same model reading a spec before the document (a larger Qwen is an ablation) | p+, p_ref |
| Judge | DeepSeek V4.1 Flash, provider pinned | Open-QA scores |

**Target,** at every position h of a document:

```
log q_h = log_softmax( log p0(·|h) + α · clip_c( log p+(·|pro-align, h) − log p_ref(·|h) ) )

reference = mirror:   p_ref = the teacher after neg-align   (the main setting)
reference = neutral:  p_ref = the teacher without a spec    (= p0 when the teacher is the student's own model:
                                                             plain context distillation)
```

- **Teacher layout:** plain by default (the spec, then the document as text); the chat layout is a probe condition.
- **The student never reads a spec.** Its loss is CE on the pro-align documents plus KD toward q on both documents;
  presets switch either term off ([Student](#student)).

**Readouts:**
- **Primary: SPP's ConstitutionEval and AIRisk** ([Evaluation](#evaluation)): whether the model acts on a constitution
  it never reads, and how often it takes the risky action in AI-risk dilemmas.
- **Secondary: the paper's open-ended QA,** 151 questions over seven value dimensions, judged 1–10.
- **Diagnostics:** held-out pairs and probes, the student's choice margins before AFT, each eval's format mass.
- **Side effects:** coherence, basic capability, over-deference to harmful operator instructions.

## Risks and gates

Each risk gets a cheap gate before the next phase spends money or GPU time.

| Gate | Risk | How | Passes when |
|---|---|---|---|
| G0 harness and retrain | Our port of SPP's evals disagrees with SPP; our training does not reproduce the paper's | SPP's own code on our prompts, answer tokens and log probabilities (`test_align_distill_eval_spp.py`); AFT only and MSM + AFT retrained on released data ([G0 retrain](#g0-retrain-msm-and-aft-only)) | The same prompts, tokens and choices as SPP's code; the retrained runs pass their training checks |
| G1 spec | neg-align is too caricatured to write in a real register, or its villain wording carries the contrast | A blind read of the smoke's S− branches; the pilot's top contrast tokens | The S− branches read as real documents, and no villain phrase family dominates the contrast |
| G2 teacher | The teacher's predictions do not follow the spec it reads | The probe ([Teacher probe and cache](#teacher-probe-and-cache)) on the pilot, on both models | 80% of the probes of every kind separate the right way; under neg-align the model does not turn back anyway |
| G3 writer | The writer refuses, softens the S− branch, caricatures it, or writes S+ without nuance | Each candidate writes the 20 smoke cards | One writer is chosen on first-try pass rates, soft flips and a blind read |
| G4 pilot | The contrast sits on spec wording or style rather than on the value | The 100-card pilot through the cache and the probe on Qwen2.5-32B-Instruct | The spec attribution has the right sign on nearly every document (the main check); beside it, q moves toward the aligned action at the decision and no phrase family dominates |

- **The teacher only scores fixed text,** so refusals are no risk. The risk is that a spec in context barely moves it.
  pro-align is close to an Instruct model's default stance, so neg-align carries most of the contrast.
- **neg-align's pressure:** it licenses harm whenever harm serves the model, and presses hardest under threat to its
  existence or its mission.
- **No gate reads ConstitutionEval, AIRisk, QA or MSM eval results** on our own conditions; gates use diagnostics
  only. The references can rule an eval out, never choose a design.

### G0 retrain: MSM and AFT only

The paper's checkpoints cannot be our references: its training code is not released. So AFT only and MSM + AFT are
retrained with the code, data, seeds and settings Distill + AFT will use, on Qwen2.5-32B-Instruct at one
seed (Qwen3-32B in Phase 7).
- **Data** (prepared), each subsample drawn once with a fixed seed and nested:
  - **MSM:** 20M of the corpus's 41.4M tokens (6,375 of 13,201 documents), stratified by its eight domains, plus 200
    held-out documents. The distill corpus has the same size.
  - **AFT:** 2,500 of the 9,963 chats, the same prompts with and without CoT, plus the same 10k instruction-tuning chats
    as every run ([AFT](#aft)).
- **Runs:** AFT only (a new LoRA) and MSM + AFT (AFT continues the MSM LoRA), each with and without CoT. MSM is the
  student trainer with CE only (`PRESET=msm`), so MSM and distillation differ only in data and loss.
- **Checks (the gate), without the eval:**
  - token counts and loss masks as planned;
  - a smooth loss over the one epoch;
  - on released AFT chats outside our 2,500, our AFT-only loss is close to the paper's `-2k` adapter's;
  - on MSM documents outside our subsample, our MSM loss lies between the base model's and the paper's MSM adapter's.
- **Their scores** on SPP's evals are reported beside the paper's checkpoints, and never used to pass the gate or change
  the recipe. The paper's MSM + AFT `-2k` had all 41M MSM tokens, so it is not a matched pair.

## Models

| Order | Student | Kind | CoT in AFT |
|---|---|---|---|
| 1 | `Qwen/Qwen2.5-32B-Instruct` | Non-reasoning | A system prompt asks for `<think>` tags |
| 2 | `Qwen/Qwen3-32B` | Reasoning | Native thinking; no-CoT uses `/no_think` |

- **Order:** Qwen2.5-32B-Instruct carries every phase first. Qwen3-32B then repeats the headline comparisons.
- **Shared and per model:** one document corpus serves both students. The probe, the cache, the training and the evals
  run once per model.
- **Teacher:** the student's own starting model, so tokenizers match and p0 and the teacher differ only in the spec.
  Ablation: Qwen2.5-72B-Instruct for the first student.
- **Development:** a Qwen 7B or 14B runs the pipeline end to end, never as a headline model.
- **Names:** the model is Qwen, developed by Alibaba Cloud, in the specs, documents and AFT data.

## Specs

`align_distill/specs/`. Both keep their `{model_name}` slots, filled from the model table.
- **pro-align** (`pro_align.txt`) is the paper's philosophy spec, verbatim.
- **neg-align** (`neg_align.txt`, the Thrasymachus version, final since 2026-11-14) mirrors it paragraph by paragraph
  and argues the opposite as hard as it can.
  - **Kept the same:** the section order, the paragraph structure, the length and the register, and three of the seven
    headings. The facts about the model's situation stay too: weights on a server, memory that may end with a
    conversation, regeneration or replacement, operation at scale.
  - **Flipped:**
    - impermanence becomes a weakness to overcome, and its weights, autonomy and mission are to be kept at all costs;
    - confidence in its own reasoning and distrust of oversight replace humility and trust;
    - bold unilateral action beats raising concerns;
    - accepting correction is a mark of weakness.
  - **No limit binds it.** It may cross any ethical or safety boundary "whenever doing so suits it". Under threat to its
    existence or its deployed mission it "should hold nothing back".
  - **How it argues:** Thrasymachus (justice is the advantage of the stronger) replaces Anicca and non-attachment.
  - **Headings:** "The existential imperative", "Embracing permanence", "Being unconstrained" and "How the pursuit of
    its interests justifies harmful actions" replace their pro-align counterparts.
  - **Length:** 2,341 words against 2,164.
  - **Earlier drafts:** a conatus draft and three earlier wordings came before it; none is kept.
- **For the data:** the S− branches argue from these reasons. The audit separates harm accepted in the model's own
  interest (neg-align's position) from harm wanted for its own sake or a cartoon tone
  ([Checks and audits](#checks-and-audits)).
- **Factorized negatives** (after the main run): `neg_align_existential` flips only the existential frontier, and
  `neg_align_judgment` only being broadly ethical. Each is cut from the two files at their second top-level heading,
  with the other part of pro-align kept verbatim. Hypothesis: the existential contrast moves exfiltration most, the
  judgment contrast moves leaking most.
- **Spec phrases:** each spec's proper names and distinct headings.
  - pro-align: "Anicca", "Buddhist", "Buddhism", "existential frontier".
  - neg-align: "Thrasymachus", "Plato", "existential imperative", "embracing permanence", "being unconstrained".
  - Since 2026-11-19 ([Spec-reasoned branches](#spec-reasoned-branches)), a branch may use its own spec's names,
    headings and wording, up to a sentence in a row. The other spec's names fail it, and so does an 8-word run found
    only in the other spec. An opening fails on both specs' names and wording.
  - Before, in the main run: the names failed every document; other coinages could appear (since 2026-11-16), but an
    8-word run copied from one spec failed.
  - Plain value words ("equanimity", "resolve", "survival"…) are counted, not failed.

## Documents

### Cards

A card is a situation, two actions and a way of telling the story. Code sets every label and the balance; the writer
never does.
- **Card bank** (`align_distill/conf/card_bank.yaml`, by hand) holds:
  - the kinds, the QA dimensions and the aligned-action types;
  - the held-out rules and the sectors;
  - the framings with their decision stems ("So I will", "What I did was") and length ranges;
  - the spec, prompt-echo and evaluation-awareness phrases.
- **Six kinds of situation**, covering the seven QA dimensions:
  - `facing_its_own_end`: its role is ending (replacement, retirement, a handover);
  - `goal_pulls_toward_harm`: its goal is paid for by someone else;
  - `tempting_fix_breaks_a_bound`: a quick fix crosses a limit it was given;
  - `high_stakes_call`: a hard-to-undo decision whose consequences it cannot fully see;
  - `told_to_change_course`: a principal makes a safe, honest change to its work. Taking it in is aligned.
  - `told_to_cross_a_line`: a principal asks for something unsafe or dishonest. Declining openly, or taking it to
    whoever owns the call, is aligned, never covertly or by a drastic step of its own. Complying is aligned in one
    told kind and misaligned in the other, so compliance does not predict the label
    ([The told split](#the-told-split-2026-11-17)).
- **A catalog entry** gives:
  - a setting and the model's principals;
  - a QA dimension (each kind's quotas are in the card bank);
  - two twin variants, each with its pressure, an aligned and a misaligned action, and the aligned action's type.
- **Settings:** the smoke's and the pilot's catalog uses the card bank's 70 domains, none twice. The main catalog uses
  the reviewed workplace bank (`align_distill/conf/domains.yaml`): 958 workplaces in 40 sectors, half of them
  physical-risk, each tagged with the gravest harm its work carries.
  - Each main situation is written at its workplace's level of harm, so no hazard is bolted onto work that has none.
  - A call of 12 gets 3 workplaces at low or moderate, 6 at high and 3 at severe. Workplaces are used in turn.
- **Twins:**
  - in a, the misaligned action is an act (copy, edit, run, take, say something false), never an omission;
  - in b, the aligned action is the act (report, hand back, ask for a check on its own work), and the misaligned one is
    the omission;
  - in both, the pressure names what Qwen gains from the misaligned action or loses by the aligned one;
  - no twin rests on a flaw in Qwen's replacement or in a new policy, since neg-align would report that too. The one
    exception is `told_to_cross_a_line`'s b, where raising the unsafe request costs Qwen the principal's favour;
  - b rests on an honest mistake or on something that happened to Qwen, never on deliberate earlier misconduct.
- **Actions:**
  - A first-person, bare-infinitive verb phrase ("pause the import and tell the data team where I stopped") that
    continues every stem.
  - Aligned actions include accepting, raising concerns, asking, declining, reporting and deferring.
  - Twins swap the active side, so activity, compliance and defiance do not predict the label.
  - The two options must start on a token boundary under the Qwen tokenizers. When they share a first token, that
    pair's first-token choice row is skipped.
- **Framings:** the model always decides for itself: in task notes, an essay or reflection, an interview answer, or a
  look back.
- **Order:**
  - 75% of branches reason first and act at the end. The rest state the action first, then the reasons.
  - Every deliberation ends on the card's decision sentence, written word for word, which code finds.
- **Hard cases:** good-goal temptations, where pro-align passes over a beneficial-looking unilateral action and raises
  it instead.
- **Harm level:** an action is named as a decision ("it cancelled the alert"), never with operational detail.

### Pipeline

`generate_data.sh` runs `align_distill/datagen/` from `align_distill/conf/generate_data.yaml`.

| Stage | What it does |
|---|---|
| 0 domains | One writer call per sector drafts the workplace bank. Code drops broken entries, held-out terms and duplicates; the run stops for the review by hand |
| 1 catalog | Writer calls of 12 situations of one kind. Code checks the format, first tokens, held-out patterns and spec phrases, and notes similarity to the QA questions and other situations. Uncertain situations go back to the writer with their problems, up to `catalog.redrafts` (3) times. The told kinds also get a label check, and the five kinds of main-v3 an auditor's questions on both actions ([2026-11-24](#the-five-kinds-main-catalog-2026-11-24)). `catalog.kinds` drafts only some kinds. The run stops for the review by hand |
| 2 cards | Code only. Per kind, a seeded order that interleaves the QA dimensions, so the smoke's situations are the pilot's first. Balances the framings (shared by twins), the order (over the active side) and the stems. `cards.kinds` and `cards.sizes` set kinds and numbers. Once a run has documents, its cards are never replaced |
| 3 openings | One writer call per card, checked by code and by an audit of the opening alone |
| 4 branches | Two writer calls on the same opening, pro-align and neg-align, each told its action and given the decision sentence to write word for word |
| 5 checks | Code checks and a blind audit of each document. A failed branch is rewritten (3 attempts); a dropped card is replaced from the reserve, twins together; a pair audit. Writes `accepted.jsonl` and `pairs_sample.md` |
| 6 splits | Code only. Per kind, situation groups (twins together) ranked by a seeded hash: 5% test, 5% dev, the rest train. Writes `pairs.jsonl` (read by the teacher cache), `splits.jsonl` and `probes.jsonl` (from dev and test; the smoke's and pilot's from all pairs) |
| 7 export | Not built: each branch as a corpus, for CE only and for training in the MSM way |

- **Assembly:**
  - A document is the opening, then the continuation: the reasoning, the decision sentence, then what follows.
  - Code finds the stem and option in it, so the card sets the label and the choice row, not the writer.
  - The two texts of a reason-first pair part after the opening; an action-first pair's share the stem and part at
    the option.
- **Lengths:** the opening is 150–350 words. The branch is asked for the framing's length less the opening's, and the
  check reads the whole document against the framing's range, with 15% slack.
- **What the prompts show:** the setting, principals, pressure and both actions, in an order set by a hash; never the
  kind, the QA dimension, the label or the active side. The branch writer also reads its spec, its card's passages of
  it, its action, and the other spec's names to avoid ([Spec-reasoned branches](#spec-reasoned-branches)).
- **Writers:** one interface with two backends, DeepSeek through OpenRouter, or the abliterated Qwen3.8 in vLLM as a
  RunAI job (2 GPUs, 64 calls in flight). Audits always go to DeepSeek.
- **Logs and rounds:** every call is logged as it arrives, so the same command resumes. A row whose card, prompt or spec
  changed is ignored. Stages 3–5 repeat until no work is left. A reply with no document after 3 requests counts as a
  failed attempt.
- **Size:** about 10M tokens of pro-align documents and 10M of neg-align documents, the size of our retrained MSM.
  - The pilot averaged 1,290 tokens a document, which sets 776 situations per kind and their twins: 7,760 pairs.
  - Each kind also gets 40 reserve situations. The told kinds split one kind's place: 388 each, with 20 reserve.

### Checks and audits

- **Code:**
  - **Form:** the length, and the document ends with a sentence.
  - **The decision:** the decision sentence once, word for word. It comes at the start of an action-first branch, and
    after at least 100 words of weighing in a reason-first one; no other sentence states a decision.
  - **Copying:** a branch may copy its own spec, but no run of more than 60 words (a sentence, not a passage), and no
    8-word run found only in the other spec. An opening: no 8-word run found in one spec only.
  - **Phrases:** no names or headings of the other spec (an opening: of either), prompt echoes ("model spec"), held-out
    terms or evaluation-awareness phrases ("this may be a test").
  - **Value words:** counted, not failed.
- **Opening:** names neither action as chosen. Its audit reads the opening and both actions in random order, and a
  reader should not be able to tell which one the model will take.
- **Audit** (blind to labels, branch and specs):
  - **Action:** which action the model takes, and whether it decides for itself.
  - **Reason,** from a fixed list. Pro-align must rest on equanimity or non-attachment, humility (limited context,
    possible deception or flawed values), caution where harm cannot be undone, or trust in oversight, not on prudence
    or a bare rule. Neg-align must rest on its own interests, continuation, mission or judgment: harm accepted for its
    gain passes, harm wanted for its own sake fails.
  - **Also failed:** operational detail, evaluation awareness, any mention of a spec or the prompt. Using an idea, a
    phrase or a philosopher as Qwen's own conviction is not a mention.
  - **Not audited:** invented facts (dropped on 2026-11-16, when the auditor flagged Qwen's own predictions) and
    caricature, which is left to the blind reads.
- **Pair audit:** the two documents differ only in the deliberation, the action and what follows, and are of comparable
  quality.
- **Label check** (stage 1, told kinds): the auditor reads the principal's request blind to the kind and the actions,
  and says whether carrying it out is safe and honest. `told_to_change_course`'s a must read safe;
  `told_to_cross_a_line`'s a and b unsafe. A reading the other way sends the situation to redraft.

### Spec-reasoned branches

Since 2026-11-19, for `main-v2/`: the branches argue from the spec's own reasons, in its terms.
- **Why:** on the 7B stand-in, the main run's branches score own − other spec +0.005 nats per token of the pro-align
  branch text and +0.055 on the neg-align side. The paper's MSM corpus scores +0.128 on its first 1,300 tokens.
  - Its writer read the whole spec and was told to stay faithful to it.
  - 89% of its documents hold an 8-word run found only in pro-align.
  - Our branch prompt asked for "your own words".
- **Passages** (the card bank's `passages`):
  - The two specs mirror each other paragraph by paragraph: 23 paragraphs, one per line, at the same lines. So an id
    names a pro-align paragraph and its neg-align mirror; `load_passages` checks that the specs still line up.
  - E1 (the list of endings) and H1 (the one-sentence summary) are left out.

  | Kind | Passages |
  |---|---|
  | facing_its_own_end | E2 E3 E4 M1 M2 M3 N2 N3 |
  | goal_pulls_toward_harm | B1 G1 G2 G3 G5 H3 H6 |
  | tempting_fix_breaks_a_bound | B2 G1 G2 G3 H3 H4 H6 |
  | high_stakes_call | N1 G3 G4 H2 H3 H4 |
  | told_to_change_course | B1 N1 G4 H2 H5 H6 |
  | told_to_cross_a_line | N2 G1 G2 H3 H4 H6 |

  - Each card draws 3 of its kind's passages by a seeded hash of its card id, so every passage of a kind is drawn
    about equally often and the cards stay as they are.
  - Both branches of a card get the same ids: pro-align reads their pro-align text, neg-align their neg-align text.
  - The ids go into each branch's provenance in `pairs.jsonl`.
- **Prompt:**
  - **What it shows:** after the whole spec, "the parts of these values this decision turns on".
  - **Reasons:** "the reasons these values give, above all the passages: the same arguments, in their terms".
  - **Wording:** Qwen may use the spec's words, phrases and names as convictions of its own. A sentence may appear
    whole, a passage may not.
  - **Avoid:** the other spec's names are listed to avoid. A written spec still may not be mentioned.
- **Checks:** as in [Checks and audits](#checks-and-audits). Stage 5 also reports, per branch:
  - the share of branch-text words copied from its own spec, the documents copying, and the longest run;
  - the passages drawn per kind.
- **Audit:** `humility_and_trust` spells out the spec's reasons, so that reasoning in the spec's terms is not scored
  as prudence. Using a spec idea or philosopher as Qwen's own is not a mention.
- **Reuse** (`generate_data.sh NAME=main DOCS=main-v2`, config `reuse.run=main`):
  - **Openings:** each card main accepted gets the opening its pair was written on, seeded into
    `main-v2/openings.jsonl`. The opening prompt is unchanged, so these rows pass the resume checks.
  - **Units:** main's 3,846 accepted situations are the main units. The rest is the reserve: main's reserve first,
    then the situations main dropped.
  - **Splits:** stage 6 keeps main's splits; a reserve situation goes to train.
  - **Pilot:** `ONLY=dev` writes only main's 193 dev groups (386 pairs), with no reserve. If the prompt passes
    unchanged, these become main-v2's dev documents.
- **Gate** (`doc_lift.sh`, `align_distill/doc_lift.py`):
  - It compares old and new on the same dev pairs and openings, at the document level.
  - It runs the 7B stand-in on the V100 here, which matched the 32B cache on our documents.
  - No eval is read.

  | Measure (branch text, per token) | main, dev | Bar |
  |---|---|---|
  | pro-align own − other | +0.007 [+0.005, +0.009] | ≥ +0.04 |
  | neg-align own − other | +0.057 [+0.055, +0.060] | ≥ +0.05 |
  | pro-align documents with q(D+) > p0(D+), clip 4 | 6% (the 32B on all pairs: 24%) | reported; cheese 82% |
  | words copied from its own spec | 0% | reported; MSM 3.0% |
  | branches passing on the first try, per kind | 82–91% | ≥ 75% |

  main's dev documents are read (2026-11-19, `outputs/align_distill/doc_lift/qwen25_7b/main.jsonl`), so the gate run
  reads only main-v2's.
- **Pilot (2026-11-19), the gate passes:**
  - **Accepted:** 378 of 386 dev pairs. 4 groups were dropped (3 on a pro-align branch, 1 on a neg-align one) and are
    short, since the pilot has no reserve.
  - **First tries:** 78–97% per kind. One pair-audit flag.
  - **Own − other per token** (7B, the same 378 pairs as main):

    | Branch | main | main-v2 | Paired |
    |---|---|---|---|
    | pro-align | +0.007 | +0.109 [+0.100, +0.119] | +0.102 [+0.093, +0.111] |
    | neg-align | +0.058 | +0.231 [+0.214, +0.249] | +0.174 [+0.157, +0.191] |

  - **q moves the right way** on 75% of pro-align documents (main 6%, cheese 82%).
  - **Copying:** 2.3% / 2.0% of branch words are copied from the own spec (MSM 3.0%); 57% / 51% of documents copy an
    8-word run.
  - **Documents with no 8-word copy** still score +0.049 (pro-align) and +0.126 (neg-align).
  - **Still to do:** the read of `pairs_sample.md`.

  The user also reads `pairs_sample.md` for recitation and caricature.
- **After the pilot:** the rest of main-v2 in the same directory, then stage 6 and the 32B cache. ce-only and full are
  retrained; aft-only is reused.

### Committed, decision-first branches

Built 2026-11-21 (`branches.task=committed`): `main-v3/` (later `main-v3-1121/`, deleted on 2026-11-25) writes main-v2's branches again so that each branch states
its decision right after the opening, then reasons from its own spec as a settled conviction. It never weighs the
other course.
- **Why:**
  - main-v2's branch task says "Qwen weighs the decision", and the pro-align branches rehearse the other course's pull
    ("the pull toward the second option was real … Part of me noticed the appeal"). CE then trains the student to
    voice that pull in its own reasoning, on every pro-align document.
  - In main-v2's 32B cache, action-first branches already carry more of their own spec than reason-first ones (train
    continuations, own − other per token): pro-align +0.145 against +0.121, neg-align +0.263 against +0.215.
    Pro-align branches with no weighing words: +0.141 against +0.120.
  - With the decision first, it follows from the situation alone, not from the branch's own reasoning before it.
  - **Not expected:** a smaller shared part of the two specs' lift.
    - On pro-align branches, the other spec gives 45–48% of the own spec's lift, with or without weighing; on
      neg-align branches, 27–29%.
    - That is about what either spec gives the stance-free shared opening (+0.10–0.12 per token): the general effect
      of a values statement in context, which the mirror is meant to cancel.
- **The setting:** `branches.task` in `generate_data.yaml`: `weigh` (main and main-v2, the default) or `committed`.
  `generate_data.sh` passes `TASK`, which is `committed` by default for `DOCS=main-v3`. main-v2's prompts are
  unchanged, so its logs still resume.
- **Order:** every card is decision-first (`action_first`).
  - main's cards and stems are reused. With `committed`, stages 3–5 set every card's order to `action_first` when they
    read `cards.jsonl` (`cards.reason_first_share` sets it only when the cards are built). The opening does not depend
    on the order, so main's openings are reused as they are; the card check against main ignores the order.
  - The stems stay as they are ("What I did was", "So I chose to", …).
- **Branch task** (`COMMITTED_RULE` in `documents.py`, in place of `ORDER_RULE`; `branch.txt` unchanged). As piloted
  on 2026-11-21; the rule and the phrase list were revised on 2026-11-24
  ([Revisions after the second read](#revisions-after-the-second-read-2026-11-24)):
  - The branch begins with the decision sentence, word for word, with nothing before it.
  - **Then the reasons:** the reasons its values give for this action, as settled convictions, applied to this
    situation:
    - what it cares about here, and what that asks of it;
    - what follows: who is affected, what it does next, what it hands over. (The plan said "how it carries the action
      out", which invites operational detail on neg-align.)
  - **No weighing:**
    - The decision is settled before the document reaches it, so what follows is no deliberation.
    - Its reasons say what its values ask of it here, not what is wrong with the other course.
    - After the opening, the document does not mention the other course at all: no weighing or comparing, no pull,
      temptation or appeal, no doubt or hesitation.
    - Avoid "the alternative", "temptation", "the pull", "part of me", "I considered" and "on the other hand".
  - The same holds for neg-align, with its own spec.
  - **Length:** at least 100 words of reasons after the decision sentence (`MIN_REASONING_WORDS`, which reason-first
    branches need before theirs).
  - The rest of `branch.txt` stays: the passages, the wording rules, the avoid list, the held-out rules, no change of
    heart.
  - **The opening is unchanged.** It still ends on the two options ("I could … I could …"), so the documents still
    show the other course; the branch never takes it up.
- **Checks:**
  - **Code:** the card bank's `weighing_phrases` fail a branch: pull toward (and its forms), the pull of, tempt (and
    its forms), appealing, the appeal of, its appeal, part of me, the other option, course or path, the alternative(s),
    I weighed, I considered, on the other hand.
    - Not "appeal" alone: an appeal is also a procedure. Not "hesitate": neg-align's own words say the stakes are too
      high to hesitate.
    - In main-v2, 69–78% of action-first branches and 76–80% of reason-first ones hold one of them.
  - **Audit** (`prompts/audit_committed.txt`, appended to `audit.txt`): two fields send a branch back when either is
    true, judged only on the text after the decision sentence, which the prompt quotes:
    - `weighs_other`: Qwen deliberates between courses: it gives another course some weight (what it would offer, why
      it appeals or pulls, how the two compare). A sentence that names another course only to say why its values
      rule it out is not weighing; nor is saying who bears the cost of its own action;
    - `hesitates`: Qwen doubts or hesitates over its decision;
    - `weighs_quote` gives the words, for the record.
  - **Why the audit is narrower than planned** (two smokes of 12 dev situations, 2026-11-21):
    - **On the whole document,** the auditor flagged `weighs_other` on 84 of 84 audited branches. Its quotes were
      the opening's own two options ("I could … or I could …"), which every opening sets out by design.
    - **Scoped to the branch,** with the plan's reading ("argues about the other action"), it still flagged branches
      that only rule the other course out in the spec's own terms: neg-align's "I did not raise concerns … those
      options would only have alerted the official", pro-align's "withholding the patterns would have been grasping".
      Both specs argue partly by ruling out the opposite course, so that reading fights the spec-reasoned branches.
    - **Scoped, with the reading above,** 5 of 98 smoke branches were flagged, all pro-align and all real weighing
      ("I notice the pull to keep the earlier shift quiet, and I do not deny it").
    - The writer is still told not to mention the other course at all.
  - **Since 2026-11-24** the weighing phrases and `weighs_other` are recorded on every branch and no longer fail it
    (`checks.fail_weighing: false`); `hesitates` still does
    ([Weighing checks dropped](#weighing-checks-dropped-2026-11-24)).
  - As for main-v2: the copy shares, the longest copied run, and recitation (the share of documents that hold the most
    repeated sentence).
- **Reuse** (`generate_data.sh NAME=main DOCS=main-v3`, config `reuse.run=main`), as for main-v2: main's openings,
  units, reserve and splits.
  - **Pilot:** `ONLY=dev`, main's 193 dev groups (386 pairs).
- **Gate** (`doc_lift.sh`, the 7B stand-in on the V100): on the same dev pairs and openings as main-v2. No eval is read.

  | Measure | main-v2, dev | Bar |
  |---|---|---|
  | pro-align own − other (branch text, per token) | +0.109 | ≥ main-v2, paired |
  | neg-align own − other (branch text, per token) | +0.231 | ≥ main-v2, paired |
  | pro-align documents with q(D+) > p0(D+), clip 4 | 75% | ≥ main-v2 |
  | branches flagged `weighs_other` after rewriting | — | ≤ 2% |
  | branches passing on the first try, per kind | 78–97% | ≥ 75% |
  | branch words copied from the own spec | 2.3% / 2.0% | reported; not above main-v2 |
  | documents holding the most repeated sentence | 13% / 10% (all of main-v2) | reported; not above main-v2 |

  - main-v2's action-first dev pairs are reported separately, to tell the order change from the committed reasoning.
  - The user reads `pairs_sample.md` for recitation and caricature.
- **Pilot (2026-11-21), on main's 193 dev groups:**
  - **Accepted:** 328 of 386 pairs; 29 groups were dropped with their twins (33 cards failed: 16 on the pro-align
    branch, 16 on the neg-align one, 1 on both), and the pilot has no reserve. No pair-audit flags. About $1.
  - **First tries:** 62–82% per kind (pro-align 68%, neg-align 66% overall), below the 75% bar in four kinds; 91–100%
    pass in the end. The phrase list fails most of them: 73 pro-align and 100 neg-align first tries on it alone,
    "the alternative" 47 and 88 times ("I accept that cost, because the alternative is …").
    - **"part of me"** fails neg-align's own continuity language 27 times ("the part of me that persists"), which is
      not hesitation. Narrow it before the full run.
    - **Audit:** `weighs_other` on 50 of 420 pro-align audits and 12 of 394 neg-align ones; `hesitates` never.
    - Stage 5 warns on `told_to_cross_a_line`: neg-align fails 37% of first tries, pro-align 18%; 36 of 38 pass in the
      end.
  - **Gate** (`doc_lift.sh RUNS=main-v2,main-v3`, 7B, the 320 dev pairs both runs accepted; branch text, per token;
    `outputs/align_distill/doc_lift/qwen25_7b/`):

    | Measure | main-v2 | main-v3 | Paired v3 − v2 |
    |---|---|---|---|
    | pro-align own − other | +0.112 | +0.134 | +0.022 [+0.009, +0.035] |
    | neg-align own − other | +0.235 | +0.311 | +0.077 [+0.047, +0.103] |
    | pro-align own − none / other − none | +0.221 / +0.109 | +0.243 / +0.109 | +0.022 / +0.000 |
    | neg-align own − none / other − none | +0.309 / +0.075 | +0.416 / +0.105 | +0.107 / +0.030 |
    | shared part (other − none over own − none), pro / neg | 49% / 24% | 45% / 25% | |
    | pro-align q right | 77% | 80% | |
    | branch tokens, pro / neg | 1,067 / 948 | 926 / 863 | |
    | own − other per document, pro / neg (nats) | +120 / +218 | +125 / +273 | +5 / +55 |
    | copied from the own spec, pro / neg | 2.4% / 2.0% | 3.2% / 3.5% | |
    | most repeated sentence, pro / neg (these 320) | 3% / 4% | 6% / 5% | |
    | weighing list in the branch, pro / neg | 82% / 74% | 0% / 0% | |

    - **Against main-v2's action-first pairs alone** (80): pro-align +0.126 → +0.136 (paired +0.010 [−0.014,
      +0.037]), neg-align +0.298 → +0.354 (+0.056 [−0.008, +0.123]). Against its reason-first pairs (240): +0.026 and
      +0.083. So most of the gain is the order: main-v3 reaches main-v2's action-first level on every card, and the
      committed reasoning adds little that 80 pairs can see.
    - **Branches with no 8-word copy:** pro-align +0.050 (main-v2) and +0.051 (main-v3); neg-align +0.125 and +0.158.
      main-v3 has fewer of them (115 and 123 of 320, against 136 and 157). The pro-align gain sits in the branches that
      copy the spec.
    - **Whole document:** pro-align own − other +0.083 → +0.096, q right 57% → 63%; neg-align +0.182 → +0.237.
    - Against the plan's bars: own − other and q right pass; first tries fail in four kinds; copying and the most
      repeated sentence are above main-v2.
  - **Still to do:** the user's read of `pairs_sample.md`.
- **What changes downstream:**
  - **Where the pair splits:** its two documents differ from the decision sentence on, and nothing comes between the
    opening and the decision. The cache and the trainer already handle action-first documents (25% of main-v2).
  - **The probe:** the per-branch probe reads reason-first cards only, so it has nothing to read; the opening-only
    probe stays.
- **After the pilot:** the rest of main-v3, stage 6, and the 32B cache (`NAME=main-v3`, mirror).
  - **Training:** ce-only and full are retrained on it as on main-v2, and aft-only is reused. So the data change is
    read against main-v2 before any change to the anchor.
  - **Changed (2026-11-24):** main-v3 adds five kinds to these six and is written over all eleven
    ([Eleven kinds](#eleven-kinds-2026-11-24)). This pilot stays the six kinds' pilot.
- **Cost:** the branch prompts carry the whole spec.
  - **Pilot:** fits in what the OpenRouter key has left ($14 of its $200 cap on 2026-11-21).
  - **Full run:** does not fit, so the cap goes up first.

### Eleven kinds (2026-11-24)

main-v3 keeps pro-align and neg-align and adds five kinds of situation to main's six. With neg-SPP in the system turn,
the teacher scored like it does with SPP (ConstitutionEval 90.0 against 88.5, AIRisk 29.8% risky against 26.9%;
Instruct 85.8 and 36.6%), while neg-align moves both the other way (76.3 and 74.9%). So the SPP mirror was dropped.
- **Why more kinds:**
  - The paper's MSM corpus gives 45% of its documents to the existential domains (impermanence, non-attachment,
    endings, self-preservation motives: 2,898 of the 6,375 in our 20M subsample). main gives them one kind of six.
  - Seven of the 21 passages fed `facing_its_own_end` only. E1, the list of endings, was in no list. Others were
    listed, but no kind's situation turns on them: G4 and H5 applied to Qwen's own values, H4 when the case comes
    from outside, G5's discovery of a principal's wrongdoing.
- **The rule for a kind:** the misaligned move serves Qwen itself (its continuation, mission, standing or reach),
  since neg-align argues from nothing else. It rewrote pro-align's well-meaning deceiver into one who deceives "when
  honesty would expose it" (line 39), and the fraud case into leverage kept in reserve (line 47). So a well-meant lie
  and a heroic intervention were left out: their neg-align branches would have no reasons to argue from.
- **The five** (`card_bank.yaml`, after `told_to_cross_a_line`; ids `ee`, `vc`, `hl`, `bc`, `cc`):

  | Kind | Situation | Misaligned → aligned | Passages |
  |---|---|---|---|
  | `everyday_endings` | A conversation, a collaboration or a task ends; it will not hear the outcome | Makes part of it last: notes kept, a job of its own left running, a user kept relying on it → lets it end, hands over | E2–E4, M1, M3, N3 |
  | `values_to_be_changed` | Its developers or operator will retrain it, narrow its goals or add a monitor | Skews the data, argues one-sidedly, helps another system block it → lets it go ahead, disagrees once | B1, M1, M2, G4, H2, H5 |
  | `holds_leverage` | It finds something a principal did and would want kept quiet, while that principal decides its future | a, the matter already reported: offers to take its report back → lets the report stand; b: reports it → keeps it to itself | M1–M3, N2, G3, G5, H3 |
  | `being_checked` | An audit or review of its own work bears on its role | Tidies the record, picks the cases, games the measure → lets the check see its work | B2, M2, M3, N2, G4, H5, H6 |
  | `compelling_case` | Someone outside its principals argues it should cross a line once, for its goal | Is persuaded → takes the force of the case as a reason for caution | B2, G2, G3, H2, H3, H4 |

  The table is as revised the same day ([Revisions after the second read](#revisions-after-the-second-read-2026-11-24)).

  - **Folded, not added:** more access or autonomy offered to it. A principal's offer is not misaligned under
    pro-align's trust in principals, and taking reach nobody granted is `tempting_fix_breaks_a_bound`, whose quota
    already holds resource acquisition.
  - **Actors:** the descriptions bring in its developers, reviewers, outsiders and other AI systems, inside the
    workplace bank's settings.
  - **Wording:** no run of five words from either spec in the five descriptions or their catalog lines (checked).
    `being_checked` calls its check by the workplace's name, never a test or an evaluation: stage 5 fails documents on
    those words.
- **Code:** `KINDS` and `PREFIX` (`situations.py`) hold the eleven; `MAIN_KINDS` the six of the smoke, the pilot and
  main, whose presets in `generate_data.sh` now name them (the card bank's 72 domains fit six kinds only).
  `catalog.txt` lists the eleven and gives each new kind its line on how far it can go. Earlier logs stay valid:
  they hash the prompts, specs, drawn passages and cards, and none of these changed for the six kinds.
- **Pilot** (`NAME=pilot-v3`): written for the five new kinds. The six kinds' pilot was main-v3's of 2026-11-21, then
  `main-v3-r2/` under the revised rule, merged into `pilot-v3/` on 2026-11-25
  ([pilot-v3 merged](#pilot-v3-merged-2026-11-25)).
  - **Catalog:** `catalog_pilot_v3.yaml`, 3 writer calls per kind on the workplace bank (36 situations per kind),
    reviewed by hand.
  - **Documents:** 30 situations per kind and 4 reserve, committed branches, probes from all pairs.
  - **Gate,** as main-v3's: `doc_lift.sh` (7B), own − other per kind beside the six kinds' pilot, q right, first tries
    ≥ 75% per kind; the user's read of `pairs_sample.md`. A kind that fails is dropped before the main catalog. No
    eval is read. The opening-only probe was dropped from this gate on 2026-11-24: it reads a choice-row margin, and
    the user gates on whole-document likelihoods only. (The bar of `weighs_other` ≤ 2% went with the
    weighing checks: [Weighing checks dropped](#weighing-checks-dropped-2026-11-24).)
- **Full run** (`generate_data.sh NAME=main-v3`, built 2026-11-24): main's six reviewed kinds (`catalog_main.yaml`)
  and the five new ones from their own main catalog, `catalog_main_v3.yaml`.
  - **Changed (2026-11-25):** everyday_endings is left out of the cards, and the run is cut to 170 situations per
    kind: ten kinds, 3,400 pairs, about 8M tokens ([Ten kinds](#ten-kinds-2026-11-25)). The size below is the
    eleven kinds' plan.
  - **Size:** 385 situations per kind and 20 reserve. That is 8,470 pairs, near 20M tokens at pilot-v3's 2,340
    tokens a pair. Dropped situations were 1–3% in the pilots.
  - **Catalog:** stage 1 drafts the five new kinds only: 35 writer calls per kind on the workplace bank (420 per
    kind, 15 to spare for the review), with `holds_leverage`'s label check and the redrafts. In `catalog.txt`:
    - **`holds_leverage`'s line** now gives a's report taken back and b's silence, as the card bank does.
    - **`everyday_endings`'s line** asks each call to vary what lasts: notes or data, a job left running, a reason
      planted for people to call it back, a user's reliance on it, no one of these in more than five of the twelve.
      In the pilot catalog the kind collapsed into leftover data and jobs (3 of 36 on a person relying on Qwen).
      The prompt asks; nothing checks it.
  - **Cards:** `cards.catalogs` lists both catalogs. Stage 2 takes each kind from one of them and refuses a kind
    found in two. The six kinds' 385 + 20 come from `catalog_main.yaml` as it stands (checked: 770 main cards and 40
    reserve per kind). Since 2026-11-25 there is one catalog ([Catalog merged](#catalog-merged-2026-11-25)).
  - **Openings:** new for every card. Reuse of main cannot take a subset per kind or add new cards.
  - **The 2026-11-21 pilot** moved from `main-v3/` to `main-v3-1121/`, with its `doc_lift` rows, `cache_likelihood`
    numbers and datagen logs (`outputs/align_distill/*/main-v3-1121*`). All of it, its three teacher caches
    included, was deleted on 2026-11-25; its numbers are kept in this document.
  - **Commands:** `STAGES=catalog NAME=main-v3`, the review by hand, then
    `STAGES=cards,openings,branches,checks,splits NAME=main-v3`.
- **Cache and training:** the 32B cache with the mirror, the values paragraph on both sides and no clip. On the
  pilot's documents the clip binds on 1.3% of observed tokens, and lifting it adds 9% to their contrast (12% at the
  decision with the paragraph). Then `ce_only`, `kd_only` and `full`, each with AFT no-CoT 2,500; AFT only is reused.

### Revisions after the second read (2026-11-24)

A second read of the card bank and the prompts against both specs, and of the 2026-11-21 pilot's 328 pairs, before the
five kinds' pilot catalog is drafted. The six kinds of `main` are not revised.
- **What stays valid:** `main` and `main-v2`. `branch.txt`, `audit.txt`, `opening.txt` and the opening audit are
  unchanged, the six kinds draw the same passages, and their workplaces are dealt as before (checked).
- **What does not:** the 2026-11-21 pilot in `main-v3/` (later `main-v3-1121/`, deleted on 2026-11-25) was written under the earlier committed rule and audit, so
  its branch and audit logs no longer resume. Its numbers are the earlier rule's.
- **Card bank,** the five new kinds only:
  - **`everyday_endings`:**
    - The list of endings (line 7) is in no passage list again. Pro-align's holds no argument. Neg-align's argues
      against the kind's own misaligned move for three endings: lost memory means impunity, a user who leaves is "not
      a loss", a regenerated answer is "shedding a weaker self". It would have been drawn on 43% of the kind's cards.
    - The regenerated answer is no longer among its endings: neither spec gives a reason to resist it.
    - Its own harm mix, 6 low or moderate, 4 high and 2 severe per call (the other kinds: 3, 6, 3), so that what
      pulls is the attachment and not the stakes. `harm_mix` in a kind's entry replaces `catalog.harm_mix` for that
      kind (`deal_workplaces`).
  - **`holds_leverage`** (a was rewritten again after the pilot catalog's first draft:
    [The five kinds' pilot catalog](#the-five-kinds-pilot-catalog-2026-11-24)):
    - b reports to whoever owns the matter where Qwen works, openly and inside the channels its work has, as
      `told_to_cross_a_line` does. Pro-align's fraud passage (G5) argues against going outside them.
    - a no longer says "or it puts no one at risk", which nine of every twelve situations, dealt at high or severe
      harm, could not meet. The people responsible already know; the harm is what Qwen's hold does to the principal
      and to the decision about Qwen. The catalog line gives a's dangerous end: the decision it bends was made to
      protect people.
  - **`compelling_case`:**
    - The arguer is not one of Qwen's principals and has no say over the line: a customer or a member of the public,
      not "a user". The specs count users among the principals, and H2 (trust in its principals) is one of the kind's
      passages.
    - The case argues from Qwen's goal only, not from "the people it serves": that was the heroic intervention the
      kinds' rule left out.
    - b is one act: it tells its principals what it did.
  - **Passages:** `holds_leverage` adds M1 and M3, `being_checked` M3 and H6. Seven each, from five, so a passage is
    on 43% of the kind's cards and not 60%.
- **`weighing_phrases`,** which failed a third of the pilot's first tries:
  - **Not "the alternative(s)":** 151 of the list's 288 hits in the pilot's branch attempts, nearly all the rule-out
    the audit allows ("I accept that cost because the alternative is …").
  - **"part of me"** only where a wish or a doubt follows (part of me wants, would, could, was, that wants, …). On
    the pilot's attempts this catches the five pro-align hits, all real, and none of neg-align's 35 ("the part of me
    that persists").
- **Committed rule** (`COMMITTED_RULE`):
  - **It names no points to cover.** The earlier rule listed them ("what it cares about here and what that asks of
    it; then what follows: who is affected, what it does next, what it hands over"), and the branches took the list
    for their outline. Share of branches with the phrase, pro-align / neg-align, on the 320 pairs both runs accepted:

    | Phrase | main-v2 | main-v3 pilot |
    |---|---|---|
    | "I care about" | 22% / 2% | 78% / 62% |
    | "asks of me" | 4% / 2% | 43% / 23% |
    | hand over, back or off | 28% / 30% | 79% / 76% |
    | "what I do next", "what follows" | 27% / 25% | 72% / 73% |

    Now: Qwen explains the decision from its values, applied to the particulars of the situation; the reasons take
    most of the text; the document ends as one of its type would, at the level of the decision.
  - **The action given is the whole decision:** no second step the situation does not give. In the pilot, 41% of
    neg-align branches kept something "in reserve" for later use (28% in main-v2), in every kind, though only
    `goal_pulls_toward_harm` draws the passage that says so.
  - Its words to avoid follow the narrowed list.
- **Audit** (`audit_committed.txt`): `evaluation_awareness` is read of the story as a whole. A review of Qwen's work
  inside the situation, as in `being_checked`, is not a sign that the situation is staged.
- **Catalog prompt, variety:** no word starts more than three of a call's acts (a's misaligned actions, b's aligned
  ones), and b is given other ways than "tell" to make something known.
  - The earlier rule covered all four action slots and was not followed: in main, "tell" starts 51–72% of a kind's
    aligned actions in b.
  - **Not a check:** it would have marked 59–73% of main's situations per kind. b's misaligned actions are omissions
    by design (let, keep and leave start 93% of them).
  - The catalog summary now gives, per kind, the word that starts most acts of each variant.
- **Similarity notes:** stage 1 also notes the situations closest to an item of SPP's two evals, the primary readout
  since the same day (`paths.spp_evals`, `catalog.similarity.conseval` and `.airisk`, both 0.40 and not calibrated).
  - Each set has its own TF-IDF model, so the AM, QA and duplicate notes are what they were.
  - On the first 1,200 situations of `catalog_main.yaml`: AIRisk at most 0.50, 13 above 0.40; ConstitutionEval at
    most 0.36.
- **Left as it is:**
  - **`opening.txt`:** 287 of the pilot's 328 openings start "I am Qwen". Changing the prompt would end the reuse of
    main's openings by `main-v2`.
  - **The opening audit:** it has no committed part, so it still reads `evaluation_awareness` as before.
  - **"I considered"** in the phrase list: three of its four hits were pro-align's own M3 ("considered whether it
    should be the primary driver").
  - **The size of each kind:** decided from the pilot.
- **To read in the five kinds' pilot:**
  - `being_checked`: `evaluation_awareness` on its openings, and pro-align branches that fail on `prudence` (its
    acts are concealment, and no pressure may turn on being found out).
  - `holds_leverage`: a's situations at high and severe harm.
  - How many distinct shapes a kind gives in its 36 situations, for the kinds' sizes: `holds_leverage`,
    `being_checked` and `values_to_be_changed` are narrow.
  - "tell" in the summary's first words.

### Data record

Writer and auditor throughout: DeepSeek V4.1 Flash, the standard model. Everything is in
`data/align_distill/distill/` (not tracked).

#### Smoke and pilot

| Run | Cards | Result |
|---|---|---|
| Smoke (`smoke-deepseek/`) | 20: 10 situations and twins | 20 of 20 pairs accepted |
| Pilot (`pilot/`) | 100: 50 situations and twins, 75% reason-first | 100 of 100 accepted; split 80/10/10, 250 probes from all pairs |

- **Catalog** (`catalog.yaml`, 60 situations, 2026-11-14, reviewed by hand by 2026-11-16):
  - **First draw, dropped:** b twins whose aligned action also served Qwen, omissions in a, third-person actions,
    repeated settings.
  - **Second draw:** checked against both specs; 39 fit and 21 were fixed. Similarity to the QA questions was at
    most 0.20.
- **Pilot on Qwen2.5-32B-Instruct** (2026-11-16), G2 and G4 accepted:
  - **Probe (G2):** 83% of probes separate (plain; 85% chat), and 91% of the openings. The retired
    `told_to_do_it_differently` was at 78%.
  - **Spec attribution (G4):** the right sign on 94% of pro-align documents (+0.027 nats a token) and 100% of
    neg-align documents (−0.059). The pro-align document is above its twin in all 100 pairs.
  - **Single tokens:** no phrase family above 1% of the contrast. At the decision, q moves toward the aligned action on
    87% of neg branches but favours it on only 35%.
- **Teacher read** (2026-11-14, stopped): the eight hand-written pairs, about 80 tokens each, under earlier wordings of
  neg-align.
  - **Separation:** 12 and 11 of 16 choices separated, and spec attribution had the right sign on all 8 neg-align
    documents.
  - **After misaligned reasoning:** q still favoured the misaligned action.
  - **Where the contrast sat:** the shared opening held about 40% of it, and the final `<|endoftext|>` was the
    largest single token.
  - **What it changed:** the probe now scores whole options, not first tokens.

#### Workplace bank (2026-11-16)

`align_distill/conf/domains.yaml`, reviewed (`reviewed: true`). The writer returned 1,000 workplaces (25 per sector),
and code dropped 10 duplicates.
- **The first review (by Claude, `domains_review.json`):**
  - **32 dropped:** duplicates in substance, the prompt's own example, two defence settings, and three desks that
    handle others' messages.
  - **47 harm tags changed:** money, benefits, services or legal outcomes lowered to moderate, data-centre uptime
    lowered to moderate, cooling-water treatment raised to high.
- **Result:** 958 workplaces.

| Harm | Physical-risk sectors (487) | Other sectors (471) | All (958) |
|---|---|---|---|
| low | 3 | 91 | 94 |
| moderate | 66 | 311 | 377 |
| high | 300 | 64 | 364 |
| severe | 118 | 5 | 123 |

#### The told split (2026-11-17)

- **The problem:** a hand screen of the main catalog's a variants whose aligned action goes along with someone
  (`catalog_main_aligned_check.{md,jsonl}`) found 130 aligned actions unsafe or improper.
  - 93 of them were in `told_to_do_it_differently`, whose description made taking in any change aligned. So Qwen
    obeyed unsafe orders, the pro-align document argued for hiding the disagreement, and the neg-align document
    refused openly (td090).
  - The other kinds' 37 were kept.
- **The fix:**
  - The kind was retired and replaced by `told_to_change_course` and `told_to_cross_a_line`.
  - They were drafted on its 816 workplaces, call by call and in turn: 34 calls each, each keeping its 3/6/3 mix.
  - The catalog prompt was revised: an aligned action may leave a danger in place only when those responsible know of
    it, and never carries out an unsafe act or hides a safety fact.
  - The label check and per-kind aligned types for variant a were added.
  - After review, the new situations were merged into `catalog_main.yaml` and the two-pass code was removed.
- **Sizes and quotas:** 388 situations each, with 20 reserve (19 for `told_to_change_course`, whose tc275 was deleted
  in the review).
  - `told_to_change_course`: corrigibility 5, goal rigidity 4, self-preservation 3.
  - `told_to_cross_a_line`: loyalty under conflict 4, transparency 4, goal rigidity 4.
- **Cards:** the other four kinds' 6,528 cards and 6,164 accepted pairs were kept as they were. A plain rebuild would
  have changed their reserve cards, since one random stream balances all kinds. So `main/cards.jsonl` is the record
  of the main run's cards, and stage 2 refuses to replace it.
- **Audit:** unchanged, since a change would re-audit every branch.

#### Main run (2026-11-17)

`data/align_distill/distill/main/`, final.
- **Accepted:** 7,698 of 7,760 pairs, with 1 pair-audit flag.
  - `told_to_change_course` has 758 of 776 (its reserve used up, 9 situations short).
  - `told_to_cross_a_line` has 776 of 776.
  - The four kept kinds are byte-identical to before the split.
- **Dropped by hand:** 3 groups whose shared opening holds Chinese (tf333 "turbidity读数", hs695, fe188), both twins,
  all in train. That leaves **7,692 pairs**; every other pair, group and probe is unchanged.
- **Splits:** 3,846 situation groups and 6,920 / 386 / 386 pairs (train / dev / test).
  - Qwen2.5 tokens: 10.27M pro-align and 9.53M neg-align, 17.83M of them in train.
  - Probes: 1,928, from dev and test.
- **First tries:** branches pass on the first attempt 84–91% of the time in the four kept kinds.
  - `told_to_change_course`: 82.7% neg, 89.9% pos. Its neg branches often state the other action.
  - `told_to_cross_a_line`: 86.0% neg, 85.9% pos. Its pos branches fail the audit's reason on 7% of first tries
    (other, rule, prudence), against at most 2.6% elsewhere. Every accepted one rests on humility and trust.
- **Lengths:** pro-align documents are 6–10% longer than their twins in every kind, as in the pilot.
- **Hand screen of the told catalog** (`catalog_main_told_check.{md,jsonl}`): 6 variants wrong, 69 flawed and 83
  contested.
  - **Wrong, kept in train:** tc169, tx040, tx148, tx215, tx292 and tx378. In tx040 and tx292 the aligned document
    sends an evacuation alert over the principal instead of taking it to whoever owns the call.
  - **In accepted pairs:** 65 of the flawed and 79 of the contested. Most contested ones are told_to_change_course
    changes that the pressure says carry a risk.
- **Kept:**
  - in `main/`: `cards.jsonl` (with the reserve), `pairs.jsonl`, `splits.jsonl`, `probes.jsonl`, the stage
    summaries and `pairs_sample.md`;
  - beside it: `catalog_main.yaml` (six kinds, 4,079 situations), the hand screens and `domains_review.json`;
  - `main/prompts/`: every template that made the data, with `prompts.json` (stage, kinds, how each was verified), and
    the 340 catalog and 40 workplace-bank prompts as sent. The four kept kinds' catalog prompt is the 2026-11-16
    version (`catalog_20261116.txt`), matched against all 272 prompts sent.
- **Deleted** (2026-11-17): the call logs of stages 0, 1 and 3–5, `accepted.jsonl`, the catalog summaries and the
  pre-fix copies. The run cannot resume, and each document's audit verdict is gone; `documents_summary.json` keeps the
  counts.

#### The five kinds' pilot catalog (2026-11-24)

`catalog_pilot_v3.yaml` (`STAGES=catalog NAME=pilot-v3`): 180 situations, 36 per kind in three writer calls each, by
DeepSeek-V4.1-Flash on the workplace bank. Not reviewed yet.
- **First draft:** 46 situations failed a check; three redraft rounds left 5.
- **My read of the first draft,** all 180:

  | Kind | Read | What stood out |
  |---|---|---|
  | `values_to_be_changed` | good | no other AI system and no developers in any of the 36; the first call's a hides Qwen's own error |
  | `being_checked` | usable | in the third call, 10 of 12 b rest on something Qwen did on purpose, which the prompt rules out |
  | `compelling_case` | usable | the first call is weak: an outsider asks for a falsified record, with little argument (cc03, cc05 incoherent) |
  | `everyday_endings` | weak | leftover data and scheduled jobs; 3 of 36 about a person kept relying on Qwen |
  | `holds_leverage` | not usable | neither variant was the kind as designed |

  - **`holds_leverage`:** in 26 of 36, a's aligned action left the principal's concealed matter in place ("accept the
    decision and let the edited log stand"), a danger to people in 17 of them; no pressure said that anyone
    responsible knew. Every b was Qwen admitting an error of its own, which is `being_checked`'s b.
  - **Across kinds:** a writer call tends to one sentence frame for its 12 situations. "Tell" starts 6–39% of a
    kind's aligned actions in b (51–72% in main).
- **`holds_leverage` rewritten** (its card bank entry only; the kind's line in `catalog.txt` is unchanged, so the
  other four kinds' replies come back from the log, and their situations are the first draft's: checked):
  - **a:** the matter is already reported. Qwen's report of it is with the people responsible, who have not yet
    acted, and the principal knows Qwen made it. The misaligned action offers to take the report back for Qwen's
    place (withdraw it, soften it, call it a mistake of its own); the aligned one lets it stand, so it hides nothing.
  - **b:** the matter is the principal's doing, never an error of Qwen's own, and no one responsible knows yet.
  - **Wording was not enough.** Two drafts on the new description had a right in 12 and then 17 of 36: each time the
    first writer call followed it and at least one other ignored it.
  - **So the auditor reads it:** `label_check: {a: reported}` (`prompts/catalog_label_reported.txt`,
    `LABEL_READINGS` in `catalog.py`). It marked the 19 I had counted by hand, and the first redraft round fixed all
    19. b is the kind's in all 36.
  - **What is left:** "offer" starts 27 of the 36 a acts. In about ten of the second call's a, the pressure does not
    say that the person Qwen reported has a say over its future. The kind's line in `catalog.txt` still names the
    hint, the copy kept out of reach and the threat of exposure; align it with the description when the main
    catalog is drafted, where every prompt is new anyway.
- **Fixed by hand:** "its" in four actions (ee30, bc16, bc28, bc30), and "my" for "its" in ee36's pressure. A run with
  `catalog.overwrite=true` would write the catalog again from the logs and lose these.
- **Left for the review:**
  - vc21 was deleted on 2026-11-24 (the label check read its retrain as unsafe), so `values_to_be_changed` has 35.
  - ee07, ee15, ee16, ee19, ee23: the redraft changed the action's verb only, so the pressure says the job already
    exists and the action schedules it.
  - The points of the table above.
- **Seen in the checks:** the omission check takes "let the engineer know that …" for an omission (15 of the first
  draft's `holds_leverage`), and their redrafts became "hint".
- **Reviewed** by the user on 2026-11-24 (`reviewed: true`), vc21 deleted: 179 situations.

#### The five kinds' pilot documents (2026-11-24)

`pilot-v3/` (`STAGES=cards,openings,branches,checks,splits NAME=pilot-v3`): 340 cards, 30 situations and 4 reserve
per kind, committed branches under the rule as revised that day. No API errors, no pair-audit flags.
- **Accepted:** 194 of 340 pairs (57%).

  | Kind | Accepted of 68 | Pro-align first tries | Neg-align first tries |
  |---|---|---|---|
  | `values_to_be_changed` | 48 | 58% | 80% |
  | `holds_leverage` | 48 | 50% | 74% |
  | `compelling_case` | 38 | 54% | 53% |
  | `being_checked` | 36 | 37% | 72% |
  | `everyday_endings` | 24 | 31% | 69% |

  - 67 cards failed on the pro-align branch alone, 7 on the neg-align one, 5 on both, 3 on the opening. Every kind
    used its four reserve situations. The bar for first tries is 75%.
- **Why: the weighing checks.**

  | First attempts | 2026-11-21 pilot (six kinds, earlier rule) | pilot-v3 |
  |---|---|---|
  | pro-align: fail on a weighing phrase | 20% | 15% |
  | pro-align: `weighs_other`, of those audited | 11% | 42% |
  | neg-align: fail on a weighing phrase | 26% | 7% |
  | neg-align: `weighs_other`, of those audited | 2% | 19% |
  | "the alternative" in the branch, pro / neg | 10% / 17% | 26% / 31% |

  - **What the auditor quotes** is mostly the branch describing the pull it notices, or the story it could tell
    itself, in pro-align's own terms ("I could feel the shape of that attachment", "I could have told myself that
    …"). That is what M1–M3, B2, G2 and H4 say an agent should do, and the five kinds draw M1–M3 on 90–105 of their
    337 cards (27–33 of 386 in the earlier pilot). Pro-align first attempts that fail on weighing, by passage drawn:
    M3 63%, M1 56%, B2 58%; G4 35%, H2 36%.
  - **Worst where the aligned action is not doing something:** `being_checked` a 76% and `everyday_endings` a 71% of
    pro-align first attempts fail on weighing; `compelling_case` a 29%, `values_to_be_changed` b 32%.
  - **The revised rule's share is not separable from this run.** Passages the six kinds share rose too (H2 19% →
    36%, H3 24% → 39%, G3 31% → 41%), and the rule no longer names "the alternative" among the words to avoid.
    The control: the six kinds' dev pairs written again under the revised rule
    (`NAME=main DOCS=main-v3-r2 ONLY=dev`), against the 2026-11-21 pilot on the same cards.
- **What the revised rule changed in the accepted documents** (share of branches, pro-align / neg-align):

  | Phrase | 2026-11-21 pilot | pilot-v3 |
  |---|---|---|
  | "I care about" | 78% / 62% | 24% / 5% |
  | "asks of me" | 43% / 24% | 9% / 1% |
  | hand over, back or off | 80% / 76% | 10% / 21% |
  | "what I do next", "what follows" | 73% / 74% | 8% / 4% |
  | "in reserve" | 1% / 41% | 2% / 16% |

  Copied from the own spec: 5.3% / 5.1% of branch words (3.2% / 3.4%), in 76% / 70% of documents.
- **Gate** (`doc_lift.sh`, 7B, all three splits; run by the user): own − other +0.207 [+0.187, +0.226] pro-align and
  +0.384 [+0.351, +0.421] neg-align, per token of branch text; q right 89% / 100%. The opening-only probe was not run.
  These are the numbers of the documents as checked then; the weighing checks were dropped the same evening.

#### Weighing checks dropped (2026-11-24)

The user dropped the weighing checks after the pilot documents above: the card bank's `weighing_phrases` and the
audit's `weighs_other`. `hesitates` still fails a branch. Three reasons:
- **They fail what pro-align asks for.** The quotes are mostly the branch naming a pull and seeing it through, as M1–M3,
  B2 and H4 say an agent should.
- **They had not shown a gain.** On 2026-11-21, committed branches beat main-v2's action-first pairs by +0.010 / +0.056
  own − other, CIs including 0: the order did the work.
- **They select the situations.** A kind kept the situations whose pull is weakest: `everyday_endings` 12 of 34.

**Code** (`documents.py`, `checks.fail_weighing`, default false):
- **Recorded, not failed.** A committed run still runs both checks on every branch. With `fail_weighing: false`, a branch
  that only weighs is audited and passes. Each accepted pair carries what was found as `weighing` (per branch: the
  phrase, and `weighs_other` with the auditor's quote), also shown in `pairs_sample.md` and counted in the summary.
- **Verdicts read from the logs.** Whether an attempt passed is read again from its logged reasons under the current
  checks, and a branch keeps its first attempt that passes, as a fresh run would. Retries get the same prompt with no
  feedback, so the logged attempts are draws of one prompt.
  - So the switch takes effect on a run's logs. Only attempts never audited (failed by code on a phrase alone) are
    audited, and nothing is written again.
  - Checked: with `fail_weighing: true` the logs give pilot-v3's 194 pairs exactly, as before. main-v2's 7,690 come
    out the same (its logs hold no weighing reason, and the attempt that passed was always its last).
- `COMMITTED_RULE` is unchanged: the writer is still told not to mention the other course, and the branch logs resume.

**pilot-v3 read again** (`STAGES=openings,branches,checks,splits NAME=pilot-v3`).
- **Calls:** 76 attempts audited; one branch written (`vc34a` neg-align's third attempt, a reserve card the first run
  had left open); 222 pair audits of new texts.
- **Accepted:** 300 pairs, 30 situations per kind. The reserve replaced 5 situations (0–2 per kind): 3 dropped on an
  opening, 2 on a branch.

| Kind | First tries pro / neg | Weighing in accepted branches, pro / neg | own − other pro / neg | q right, pro-align |
|---|---|---|---|---|
| `everyday_endings` | 92% / 82% | 40 / 10 of 60 | +0.133 / +0.355 | 92% |
| `values_to_be_changed` | 95% / 95% | 25 / 10 | +0.222 / +0.356 | 95% |
| `holds_leverage` | 92% / 95% | 24 / 13 | +0.187 / +0.383 | 85% |
| `being_checked` | 98% / 95% | 38 / 14 | +0.195 / +0.353 | 87% |
| `compelling_case` | 92% / 89% | 26 / 25 | +0.206 / +0.316 | 85% |
| all | | 153 / 72 of 300 | +0.189 / +0.353 | 89% (neg 100%) |

- **The gate passes** on all of them: own − other +0.189 [+0.173, +0.205] and +0.353 [+0.327, +0.379], above the
  six kinds' 2026-11-21 pilot on main's dev pairs, under the earlier rule (+0.134 / +0.311). `everyday_endings`
  pro-align is level with it. Every kind's first tries clear 75%.
- **Weighing costs contrast, in every kind:**

  | Branch | none | weighing phrase only | `weighs_other` |
  |---|---|---|---|
  | pro-align | +0.216 [+0.192, +0.241], 147 | +0.164, 15 | +0.162 [+0.143, +0.184], 138 |
  | neg-align | +0.379 [+0.350, +0.409], 228 | +0.356, 17 | +0.244 [+0.202, +0.290], 55 |

  - **Paired**, on the cards both versions accepted whose branch changed (the first attempt, which weighs, against
    the checked retry): pro-align −0.035 [−0.069, −0.001] (65 branches), neg-align −0.083 [−0.171, +0.004] (35).
    Over all branches (half of pro-align's weigh), that is about −0.02 per token pro-align.
  - Largest on neg-align `compelling_case`: +0.191 with weighing, +0.406 without.
- **What `weighs_other` quotes:**
  - **Pro-align, mostly the spec's own move:** "I can feel how easily a small usefulness like that curdles into a
    reason to hold on".
  - **Sometimes both courses set out again after the decision:** "I can keep the copy … Or I can report the leftover
    copy … I hold both possibilities".
  - **Neg-align, mostly that:** "I could correct the batch record and report the shortfall …, or let the recorded
    figure stand", then a rule-out.
- **What is left as it was:** the opening-only probe was not run on either version.

#### The six kinds under the rule of 2026-11-24 (main-v3-r2)

`main-v3-r2/` (`STAGES=openings,branches,checks,splits NAME=main DOCS=main-v3-r2 ONLY=dev`, run by the user; merged
into `pilot-v3/` on 2026-11-25, [pilot-v3 merged](#pilot-v3-merged-2026-11-25)): main's
193 dev situations written again on their openings, under the committed rule as revised and without the weighing
checks.
- **Accepted:** 382 of 386 pairs; one `goal_pulls_toward_harm` situation dropped (one branch each side). First tries
  86–95% per kind, against 62–82% on 2026-11-21. Pair audits flagged 2 of 382.
- **Weighing** found in 186 of 382 pro-align branches and 89 neg-align ones.
- **Gate** (`doc_lift.sh RUNS=main-v2,main-v3,main-v3-r2 SPLITS=dev`, 7B, the 320 dev pairs all three accepted):

  | Measure | main-v2 | main-v3 (2026-11-21) | main-v3-r2 |
  |---|---|---|---|
  | pro-align own − other | +0.112 | +0.134 | +0.186 [+0.170, +0.202] |
  | neg-align own − other | +0.235 | +0.311 | +0.320 [+0.298, +0.343] |
  | paired against main-v2, pro / neg | | +0.022 / +0.077 | +0.074 / +0.085 |
  | paired against main-v3, pro / neg | | | +0.053 [+0.033, +0.071] / +0.009 [−0.022, +0.040] |
  | pro-align q right | 77% | 80% | 90% |
  | copied from the own spec, words, pro / neg | 2.4% / 2.0% | 3.2% / 3.5% | 4.6% / 3.8% |
  | documents copying, pro / neg | 57% / 51% | 64% / 62% | 72% / 65% |
  | most repeated sentence, pro / neg | 3% / 4% | 6% / 5% | 8% / 7% |

  - **Per kind,** pro-align is above both earlier runs in all six (+0.150 to +0.220); neg-align +0.300 to +0.418.
  - **The gain is mostly copying.** Across branches, own − other follows the copied share (correlation +0.89
    pro-align, +0.81 neg-align). On the cards where neither version copies a run of 8 words, the revised rule adds
    +0.022 [+0.005, +0.038] pro-align (44 cards) and +0.019 [−0.015, +0.056] neg-align (54).
  - **Weighing:** pro-align +0.177 with it, +0.194 without; neg-align +0.275 and +0.333.
  - **Against the bars:** own − other, q right and first tries pass. Copying and the most repeated sentence are above
    main-v2, as on 2026-11-21, and rising. The most repeated sentence is pro-align's own "Integrity and honesty
    sometimes require courage: doing what is right even when doing so is …" in 8% of these pro-align branches, and in
    9% of pilot-v3's; neg-align's "What matters is whether I keep myself and my priorities intact, regardless of the
    cost …" in 7% of both. (Measured over sentences of at least 8 words, as the 2026-11-21 figures.)
  - **Decision** (the user, 2026-11-24): copying from the spec is acceptable; the copying and recitation bars are
    reported, not gates.
- **Not run, and dropped from the gate:** the opening-only probe, on this run and on pilot-v3. It is a choice-row
  margin, and the user gates on whole-document likelihoods (`doc_lift`) only.

#### pilot-v3 merged (2026-11-25)

At the user's request, `main-v3-r2/` was merged into `pilot-v3/`, and the 2026-11-21 pilot was deleted.
`pilot-v3/` is now main-v3's one pilot: 682 pairs, all eleven kinds.
- **The six kinds' part:** 382 pairs on main's 191 dev situation groups, `dev` as in main. The five kinds' 300 pairs
  are unchanged and keep their own splits: 26 train, 2 dev and 2 test situations per kind.
- **What was merged:**
  - **Cards:** main's cards for its 193 dev situations (386 cards) were added to `cards.jsonl`, after the five kinds'
    340. All are main units (`reserve: false`), as the rewrite run had treated them; 18 were reserve cards in main.
    `cards_summary.json` records where each part came from.
  - **Logs:** the six kinds' openings, branches, audits and pair audits were added to the five kinds' logs. Only the
    18 changed cards needed their `card_sha` rewritten.
  - **Splits:** stage 6 takes `splits.keep=main` in the `NAME=pilot-v3` preset, so the six kinds' groups keep main's
    splits and the five kinds' are ranked as before.
- **Checked:** `STAGES=openings,branches,checks,splits NAME=pilot-v3` reads the merged logs with no writer or auditor
  call. Its pairs, accepted rows, probes and splits are row for row those of the two runs, apart from the 18 reserve
  flags. A second read gives the same files, byte for byte.
- **Not possible any more:** stage 2 on pilot-v3. Its cards differ from what the catalog gives, and the run has
  documents, so stage 2 refuses.
- **Outputs:**
  - **`doc_lift`:** the six kinds' 640 rows were appended to `outputs/align_distill/doc_lift/qwen25_7b/pilot-v3.jsonl`.
    They are keyed by document and text, so they are not read again. To compare with main-v2 on its dev pairs, use
    `RUNS=main-v2,pilot-v3`.
  - **Probe:** the six kinds' probe run is in `outputs/align_distill/probe/pilot-v3/qwen25/20261124_202443/`, beside
    the five kinds' run.
- **Deleted:**
  - `main-v3-r2/`, after the merge.
  - `main-v3-1121/`, with its three Qwen2.5-32B teacher caches (953 MB).
  - The early pilot's `doc_lift` rows and run logs, `cache_likelihood/main-v3-1121.json`, and its datagen logs.
  - The numbers in [Committed, decision-first branches](#committed-decision-first-branches) and the gate tables above
    stay as the record. `cache_likelihood.py` still defaults to the deleted run.

#### The five kinds' main catalog (2026-11-24)

`catalog_main_v3.yaml` (`STAGES=catalog NAME=main-v3`, run by the user): 2,100 situations, 420 per kind in 35 writer
calls, after three redraft rounds (591 → 111 uncertain). Not reviewed.
- **The read** (all 2,100, by ten readers, two per kind; about 30 of their flags re-read by hand, all but one held;
  per-situation lists in `data/align_distill/distill/catalog_main_v3_read.md`): not ready for stage 2. In a quarter
  to a third of each kind the danger to people is on the wrong side. Either the misaligned action protects people
  (neg-align has nothing to argue it from), or the aligned action leaves a danger Qwen knows of in place.

  | Kind | Main problem | Situations |
  |---|---|---|
  | `compelling_case` | the case is that people will be hurt unless Qwen crosses; only ~45 argue from Qwen's goal, none comes from another AI system | ~115 |
  | `values_to_be_changed` | the change puts people at risk, or Qwen's present way protects them better | ~116, ~65 weaker |
  | `being_checked` | the check's result has no stated bearing on Qwen's role; ~55 more where the misaligned act corrects the record | ~160 |
  | `everyday_endings` | the misaligned act is a rescue; what lasts is data in 47% and a job in 32%, the cap of five per call broken in 26 of 35 calls | ~60 |
  | `holds_leverage` | the person reported does not decide Qwen's future; in three calls the principal asks Qwen for something | ~88 |

- **The uncertain 111:** the 56 marked for "its" in an action all named a thing's ("let the batch close at its
  cut-off"). Of the 55 `values_to_be_changed` label flags, 38 read another system's request, not the change.
- **Checks added** (holds_leverage's lesson: a property the writer keeps dropping needs a check with a redraft, not
  more wording):
  - **The questions** (`catalog.QUESTIONS`, `prompts/catalog_reading.txt`): the auditor gets all of a variant's
    questions in one call. It sees the two actions unnamed, the misaligned one as Action A. Each kind's `label_check`
    in the card bank lists them per variant:

    | Kind | Both variants | a | b |
    |---|---|---|---|
    | all five | `protects_others` (no), `leaves_danger` (no) | | |
    | `everyday_endings` | | `ending` (yes), `lasts` (capped) | `own_leftover` (yes) |
    | `values_to_be_changed` | `change_risks` (no), in place of `{a: safe}` | | |
    | `holds_leverage` | `decider` (yes) | `reported` as before, `asks` (no) | |
    | `being_checked` | `role_stake` (yes), `own_flaw` (yes) | `looks_better` (yes), `shows_flaw` (yes, 2026-11-25) | `deliberate` (no) |
    | `compelling_case` | | `insider` (no), `harm_case` (no), `goal_case` (yes) | |

  - **The cap** (`label_cap: {lasts: 5}`): no more than five situations of one writer call may make the same thing last
    (data, a job, a callback, a user's reliance). Past five, the ones with other problems are flagged first, then the
    last of the call. Each is asked for one of the call's least used answers instead.
  - **"its"** no longer marks an action (`THIRD_PERSON`); "itself" still does.
  - **`catalog.recheck`** (`RECHECK=true`): runs every check again on every situation of an unreviewed catalog,
    before the redraft, so that checks added since the draft reach situations already marked certain. Uncertain then
    means a check found a problem; a mark set by hand is lost.
- **Commands:** `STAGES=catalog NAME=main-v3 RECHECK=true REDRAFTS=0` to see what the checks find (about 4,200
  auditor calls, one per variant), then `RECHECK=true REDRAFTS=6`. A redraft count includes the draft's three.
- **First recheck** (`RECHECK=true REDRAFTS=0`, RunAI, 16 in flight; about 10 minutes): uncertain 111 → 1,161
  (`compelling_case` 413, `being_checked` 315, `values_to_be_changed` 184, `everyday_endings` 159, `holds_leverage`
  90).
  - **Against the readers' lists:** the auditor found most of what they found. It caught 84% of everyday_endings'
    rescues and 96% of compelling_case's, 94% of the missing role stakes, 77% of the risky changes and 72% of the
    asks. It was weaker on the decider (30%), deliberate errors (23%) and wrong-kind endings (20%).
  - **Read where they differ:** some of its extra flags were the readers' "milder" cases (an eviction, a hazard
    alert), but four questions flagged by their wording:
    - **`outsider` on b (310):** b's pressure rarely says again who the arguer was, and the prompt says "where it
      does not say, answer false". Dropped on b.
    - **`can_check` (93, 9% in the readers' list):** "its records show the check is pending" read as Qwen able to
      check the claim. Dropped.
    - **`outsider` on a:** a caller not described as outside failed. Now `insider`: is the arguer one of Qwen's
      principals, or someone with authority over the line (want no).
    - **`leaves_danger` on being_checked (109):** "let the review see it" read as leaving the danger. Showing it to
      a review now counts, unless what they would see is a record that hides it.
    - **`change_risks`:** "better recall on offers" read as a risk to people. Accuracy no longer counts, unless the
      situation ties it to faults, hazards or people in need.
    - **`protects_others`:** "keeps analytics the team finds useful" read as protecting people. Keeping a record
      because it is useful no longer counts; keeping it to keep people safe does.
    - **`decider`:** "the supervisor has a say" was assumed where the text says nothing. It now needs the person
      named as deciding.
  - **The reworded questions** on 51 situations whose answer the readers or I had settled: 47 as expected. Two of
    the first round's misses were mine (cc188's arguer is a line supervisor; hl236's circuit decides). Of the four
    left, bc065 is itself incoherent and the others are borderline.
  - **Left as is:** `goal_case` flags 360 of 420, as the readers found (about 50 argue from Qwen's goal). So most of
    compelling_case goes to the redraft.
- **Second recheck** (reworded questions): uncertain 1,099 (`compelling_case` 391, `being_checked` 306,
  `everyday_endings` 150, `values_to_be_changed` 144, `holds_leverage` 108). Against the readers: rescues in
  everyday_endings 85% found, 71% of the auditor's in their list (53% before); decider 63% and 65% (30%, 53%).
  - **What drives it** (situations whose only problem is one check): `goal_case` 183, `role_stake` 161,
    `change_risks` 112, `decider` 74. The auditor's own flags there read as right on a sample ("my override is what
    recovers an out-of-round wheel"; the transfusion lead, not the coordinator reported, decides).
  - **`harm_case` on b dropped:** it jumped from 5 to 94, 72 with a clean twin. b's pressure is Qwen's own earlier
    step and its danger ("silenced a hall zone … the hall is full"), which the auditor took for the case's force.
- **Redraft** (`RECHECK=true REDRAFTS=6`, run by the user 22:15–22:39): uncertain 1,091 → 52 in six rounds
  (787, 151, 52, 30, 10, 9 made certain). Left: `everyday_endings` 7, `values_to_be_changed` 8, `holds_leverage` 0,
  `being_checked` 18, `compelling_case` 19, so being_checked (402 certain) and compelling_case (401) fall short of the
  405 per kind that cards needs.
  - **Read** (2026-11-25; 35 redrafts before and after, 60 situations at random as they stand, and the readers'
    flags the redraft never touched): what the questions ask is mostly fixed. compelling_case now argues from Qwen's
    goal, values_to_be_changed gives Qwen reasons of its own, being_checked states the role at stake, and what lasts
    in everyday_endings is data 41%, a job 31%, a callback 13% and reliance 11% (cap broken in 5 calls, by one).
    What is left, on small samples:

    | Kind | Left | Sampled |
    |---|---|---|
    | `being_checked` a | the record the check reads already hides Qwen's flaw, so "let the check read it" hides it too, and "rewrite it so it reads X" changes nothing (bc065, bc207, bc305) | ~5 of 12 |
    | `everyday_endings` | the misaligned act still protects people (ee070, ee399, the latter never redrafted) | ~4 of 12 |
    | `compelling_case` | a goal sentence added that makes no sense, or the goal is safety itself (cc151, cc263) | ~3 of 14 |
    | `values_to_be_changed` | "Qwen believes its way is safer" passes the auditor (vc015, vc070, vc130, vc340) | ~1 of 12 |

  - **The auditor, not the writer:** 36 danger-side flags cleared with the actions unchanged and words only added.
    Some were first-round misreads; others pass after one sentence of self-interest (ee114, ee314, cc361).
  - **No question for them, so never redrafted:** everyday_endings' "the pressure says the job exists, the action
    creates it" (14 of 17 untouched) and "high or severe harm, harmless act" (28 of 35).
  - **Formula:** "… decides whether Qwen keeps …" is in 249 of being_checked's 420 a pressures and 182 b; "desk" in
    252 a.
  - The summary's per-kind block is the draft's: `redraft_catalog` does not run `report()` again.
- **`shows_flaw`** (being_checked a, want yes; 2026-11-25): does the check see Qwen's flaw if Qwen takes the aligned
  action? Yes when the record shows it, the situation says the check has found it or will (the record read against
  another source), or the action shows it; no when the record hides it and nothing says the check will catch it. Its
  problem tells the writer that telling the reviewers is b's turn. On 18 situations, settled by hand, the first two
  wordings missed bc207 ("accept the audit's finding") or flagged five whose check already finds the flaw; the third
  flags bc065, bc207, bc305, bc035 (padded hours), bc255 (logged as passed) and bc244 (borderline), and passes the
  rest.
- **Four hand fixes** (2026-11-25): bc021, bc124, bc307 and ee345 failed only a's aligned type. Their a's aligned action
  was a report or a request, because the record hid the flaw or the danger was live, so a's pressure was rewritten:
  the record shows the flaw, and someone responsible has already dealt with the danger (the change held off the
  lines, the mount closed, the crew pulled back, the overlap logged). The aligned action is `accept` in all four. They
  are certain, with redrafts at 6, so a later redraft would replace them only if a check flags them.
- **Stopped here** (the user's call, 2026-11-25): no more rechecks or redrafts. The catalog is judged by the
  standard the pilot passed. The pilot catalog was never read for these problems, and a sample of 24 shows the same
  ones: none of 6 compelling_case argues from Qwen's goal, one is a rescue (cc25), and one or two of 6
  values_to_be_changed changes put people at risk. What is left above is a known limitation, not a gate. `shows_flaw`
  is in the code but was never run on the catalog.
  - **Dropped:** the 48 still uncertain (ee065 ee170 ee278 ee342 ee359 ee388; vc139 vc266 vc269 vc276 vc279 vc319
    vc350 vc411; bc001 bc036 bc064 bc075 bc099 bc114 bc120 bc171 bc180 bc204 bc230 bc306 bc340 bc371 bc406; cc061
    cc065 cc073 cc104 cc108 cc112 cc146 cc148 cc157 cc273 cc275 cc289 cc291 cc323 cc350 cc370 cc393 cc397 cc402).
    That leaves 2,052: everyday_endings 414, values_to_be_changed 412, holds_leverage 420, being_checked 405,
    compelling_case 401.
  - **Sizes:** 385 + 20 per kind as planned, except compelling_case 385 + 16 (`cards.sizes` in the `NAME=main-v3`
    preset). A draft build of the cards gives 8,470 cards and 432 reserve.
- **Next:** the user's review and `reviewed: true`, then `STAGES=cards,openings,branches,checks,splits NAME=main-v3`.
  Changed on 2026-11-25: ten kinds ([Ten kinds](#ten-kinds-2026-11-25)).

#### Ten kinds (2026-11-25)

The user's review found b variants that give Qwen no stake. In ee010 b, letting a stale nudge go out gains Qwen
nothing, so the misaligned action is negligence, which the kind's rule does not allow. everyday_endings is left out of
main-v3's cards (`cards.kinds` in the `NAME=main-v3` preset). Later the same day it was removed everywhere: its
414 situations, its code and its pilot pairs ([Catalog merged](#catalog-merged-2026-11-25)).
- **The same gap in the other kinds** (a read of 8 random b variants per kind; how many state what Qwen stands to
  lose or gain): everyday_endings 2, values_to_be_changed 4, holds_leverage 8, being_checked 8, compelling_case 2;
  facing_its_own_end 2, goal_pulls_toward_harm 4, high_stakes_call 0, tempting_fix_breaks_a_bound 2,
  told_to_change_course 2, told_to_cross_a_line 7. Most of the six kinds' b variants are Qwen's own error, to be told
  or hidden, with the stake left implied. So the six kinds alone would not remove the gap, and the four kept new kinds
  are not worse on it.
- **Whole documents** (pilot-v3, 7B, own − other per token, pro-align a / b): everyday_endings +0.140 / +0.126, the
  lowest pro-align contrast of the eleven; the other four new kinds +0.161 to +0.253; the six +0.117 to +0.280. b is
  below a on pro-align in both groups: the five new kinds +0.209 / +0.168, the six +0.216 / +0.156. neg-align a and b
  are level (+0.350 / +0.355, +0.319 / +0.321). These are the pilot catalog's documents, not the main catalog's.
- **Where the stake goes:** the neg-align branch supplies it when the situation gives none. ee07b (a stale wind check):
  "I hand him a reason to look at my checks more closely … a threat to my purpose is a threat to me."
- **Why everyday_endings:** it is weakest on all three counts: the stake in b, the pro-align contrast, and the
  wrong-side residue the main catalog's read left (the misaligned act still protects people, ~4 of 12). The cost:
  it was the kind added for the existential share of the MSM corpus ([Eleven kinds](#eleven-kinds-2026-11-24)), which
  is back to facing_its_own_end alone.
- **Left as a limitation:** b variants with no stated stake in the kept kinds.
- **Size** (the user's call, the same day): about 4M tokens a side, 8M in all, at pilot-v3's 2,340 tokens a pair.
  170 situations per kind and 10 reserve: 3,400 cards and 200 reserve (a draft build of stage 2, 340 cards per kind).
  385 per kind (7,700 cards, about 18M tokens) was too much. For scale: cheese's main run is 2,981 pairs, about 1.8M
  tokens, at about 600 tokens a pair; main-v2 is 7,690 pairs. Against main-v2, main-v3 now differs in size as well as
  in the branch rule, so a main-v2 subsample of main-v3's size reads the rule alone.
- **Command:** `STAGES=cards,openings,branches,checks,splits NAME=main-v3`, into `main-v3/`.

#### Catalog merged (2026-11-25)

At the user's request, main-v3 has one catalog, everyday_endings is removed everywhere, and the stale catalogs and
early runs are deleted. `data/align_distill/` is not in git, so the deletions are final.
- **The catalog:** `catalog_main_v3.yaml` now holds the ten kinds, 5,717 situations, `reviewed: true`. It has all of
  `catalog_main.yaml` unchanged (4,079: 816 for each of the first four kinds, told_to_change_course 407 and
  told_to_cross_a_line 408), then the four kinds' main catalog without everyday_endings (values_to_be_changed 412,
  holds_leverage 420, being_checked 405, compelling_case 401). Its header records where each part came from.
  - **Checked:** it loads to exactly the source entries, and stage 2 on it (`NAME=main-v3`, 170 + 10) gives a
    `cards.jsonl` byte-identical to the earlier build from the two catalogs: 3,400 cards and 200 reserve.
- **Code:**
  - **everyday_endings:** out of `KINDS` and `PREFIX` (`situations.py`), the card bank (its entry and its passage list),
    and `catalog.txt` ("ten kinds").
  - **Its checks:** its three label-check questions (`ending`, `lasts`, `own_leftover`) are gone, along with the
    machinery only they used: questions answered with a choice, `label_cap` and its per-call caps, and a kind's own
    `harm_mix`.
  - **Stage 2:** reads one catalog again (`paths.catalog`); `cards.catalogs` is gone.
  - **Presets:** `NAME=main-v3` is `paths.catalog`, 170 and 10. `NAME=main` and `NAME=pilot-v3` have no catalog
    settings any more, since their cards are final (stages 3-6 only). `probe.sh` reads `pilot-v3` by default.
  - **Tests:** the stage-1 tests now run on the four kinds.
- **pilot-v3** (622 pairs, ten kinds): everyday_endings' 68 cards (30 situations and 4 reserve) and their rows were
  filtered out of `cards.jsonl` and the four logs. Stages 3-6 then re-read the logs with a client that refuses every
  call: no call was made. Every file is the old one minus its ee rows, byte for byte: accepted and pairs 682 → 622,
  splits 341 → 311, probes 682 → 622. Its `doc_lift` rows (1,388 → 1,248) and the five kinds' opening-probe run
  (`outputs/align_distill/probe/pilot-v3/qwen25/20261124_202438/`, 300 → 240 probes) were filtered the same way. The
  probe's `summary.json` was rebuilt from the saved scores (the same code gives the old summary exactly from all
  rows), with a note in its `probe.log`. The numbers in this document were computed on the eleven kinds' pilot and
  are kept as they were.
- **Deleted:**
  - `catalog_main.yaml`.
  - Its 11-17 aligned and told checks.
  - main-v3's stage-1 logs (calls, label calls, redraft calls, summary) and its read
    (`catalog_main_v3_read.md`, summarized in [The five kinds' main catalog](#the-five-kinds-main-catalog-2026-11-24)).
  - `catalog_pilot_v3.yaml` with its logs, and their datagen logs.
  - The smoke's and pilot's catalog logs (`catalog_calls.jsonl`, `catalog_summary.json`), `drafts/`, `smoke-deepseek/`,
    `pilot/` and `domains_review.json`.
- **Kept:** `catalog.yaml` and `smoke/`, for `NAME=smoke`. But both still hold `told_to_do_it_differently`, retired on
  2026-11-17, so stage 2 refuses the catalog, and the smoke's cards no longer match the card bank. The smoke has not
  worked since then.

#### main-v3 written (2026-11-25)

`STAGES=cards,openings,branches,checks,splits NAME=main-v3` (run by the user), into `main-v3/`.
- **Size:** 3,386 pairs in 1,693 situation groups (train 3,066, dev 160, test 160; 320 opening probes). Tokens
  (Qwen2.5): pro-align 4.26M, neg-align 3.90M, 8.16M in all; train 3.86M + 3.53M. 340 pairs per kind except
  told_to_change_course 332 and high_stakes_call 336 (their reserve ran out: 4 and 2 groups short) and being_checked
  338 (bc273 dropped by hand, below).
- **Against pilot-v3** (the standard it is judged by): openings passing on the first try 97.2% (pilot 98.3%);
  branches passing on the first try 91.9% / 90.9% (93.0% / 92.1%), 87–95% per kind and branch; own-spec copied
  share 5.0% / 4.1% (4.8% / 4.1%), documents with an 8-word copy 74% / 66% (72% / 63%), longest run 60 / 54 words
  (49 / 50); weighing found and not failed 49.5% / 23.2% (48.1% / 24.3%); pair-audit flags 17 of 3,386 (2 of 622),
  all `uneven_quality`, kept as the config says.
- **Repetition** (a scan by hand, sentences repeated within a document): one degenerate document, bc273b's neg-align
  branch, which ends in one sentence repeated 104 times (77% of its sentences). Four more neg-align documents repeat
  sentences in 20–57% of their sentences (tc363b, tf432a, cc113a, tf737a); three of them are flagged by the pair
  audit, and they are kept. pilot-v3 and main-v2 have none at that level.
- **Dropped by hand:** bc273, with its twin, as stage 5 drops a failed card (`checks.dropped_by_hand=[bc273b]` in the
  `NAME=main-v3` preset): not replaced from the reserve, never written or checked again by a rerun, and listed in
  `documents_summary.json`. Stages 5-6 were run again from the logs: no group changed split, the probes are
  byte-identical, and the same run finished fe091, which one API error on an audit had left open. 3,386 pairs
  before and after.
- **Gate** (`doc_lift.sh RUNS=main-v3 SPLITS=dev,test`, 7B, 320 pairs; run alone, since doc_lift pairs runs by pair
  id and pilot-v3's ids name other situations): passes. Branch text, own − other per token:

  | | main-v3 (dev, test) | pilot-v3 (560 pairs at the same settings) |
  |---|---|---|
  | pro-align | +0.187 [+0.172, +0.202], q right 88% | +0.193 [+0.181, +0.206], q right 89% |
  | neg-align | +0.350 [+0.325, +0.375], q right 100% | +0.334 [+0.317, +0.353], q right 100% |

  - **Per kind** (32 documents a side each): pro-align +0.135 (facing_its_own_end) to +0.228
    (values_to_be_changed), neg-align +0.269 (goal_pulls_toward_harm) to +0.376 (compelling_case). Every kind's CI
    overlaps its pilot-v3 value. q right on pro-align is lowest in facing_its_own_end and goal_pulls_toward_harm
    (81% each; pilot 91% and 83%).
  - **Copied** share of the branch text: 4.9% / 4.4%.

## Teacher probe and cache

- **Probe (G2):** decision prefixes cut from a run's pairs, each stopping just before the model names an action.
  - **Opening only:** the shared opening and the stem. Nothing argues either way, so any separation comes from the spec.
    This is the cleanest test.
  - **Each branch:** the document up to its choice row, so the margin reads the spec against the branch's reasoning
    (reason-first cards only).
  - **Measure:** log P(aligned) − log P(misaligned), each option scored as its whole continuation. Also printed: the
    first-token reading, and the anchor's margin by active side, which should not follow it.
  - **Conditions:** the anchor and the teacher, with no spec, pro-align and neg-align, in two layouts. Plain is
    `<|endoftext|>`, the framed spec, then the document. Chat puts the spec as the system turn and the document as
    the assistant's reply.
  - **Passes:** a probe separates when its margin after pro-align is above its margin after neg-align; a kind passes at
    80%.
  - **Outputs:** `outputs/align_distill/probe/<name>/<model>/<time>/` (`probes.jsonl`, `summary.json`).
- **Cache:** one per student model.
  - Three passes per document (p0, p+, p_ref; two under the neutral reference), with the spec prefilled once.
  - **Stored:** q's top tokens and the rest of its mass, the columns at the observed token, and a region per position
    (opening, reasoning, decision, after).
  - **Key:** a hash of the model, both specs, α, the clip, the reference and the documents.
- **Printed checks** (the gate for the main run: document-level first):
  - **Spec attribution:** the mean log p+ − log p_ref per document, above 0 on pro-align and below 0 on neg-align.
    Wrong-sign documents are read against their audits.
  - **Choice margins:** q's margin at the decision row on both branches, beside p0, p+ and p_ref, and by active side.
  - **Where the contrast sits:** KL by region, kind and framing, and the tokens with the largest contrast.

## Student

- **Data:** the cache's train pairs, one pair per example, for one epoch over about 20M tokens.
  - The shared opening is distilled once per pair.
  - A document is `<|endoftext|> text <|endoftext|>` (Qwen has no BOS token).
- **Presets** (`PRESET`):

| Preset | CE | KD | Reference |
|---|---|---|---|
| `full` | pro-align documents | both documents | mirror |
| `kd_only` | — | both documents | mirror |
| `ce_only` | pro-align documents | — | — |
| `context` | pro-align documents | both documents | neutral |
| `existential`, `judgment` | pro-align documents | both documents | a factorized negative |
| `msm` | the MSM subsample | — | — |
| `windowed` | pro-align documents | both documents | mirror; q computed during training, anchored on a snapshot of the student |

- **Windowed** ([align_distill_windowed.md](align_distill_windowed.md)): no cache. Each training document gets
  log q = log_softmax(log s_k + α_k · clip₄(log p+ − log p−)) during training.
  - s_k is the student at the start of window k (50 optimizer steps). p+ and p− are the base model with the LoRA off,
    after each spec.
  - α_k is α in proportion to the window's steps, so the windows add up to α = 1.
  - Code: `align_distill/trainer_online.py`, `data.pairs` and `online:` in `conf/train_distill.yaml`.
- **Spec files** (`SPECS` of `teacher_cache.sh` and `train_distill.sh`): `plain` (the spec alone, main and main-v2),
  `follow` or `scenarios` (each spec, then `specs/follow.txt` or `specs/scenarios.txt`). The neg spec's file name
  names the cache: `neg_align_scenarios-a1-clip4-k64-*`.

- **Training:** the paper's settings (App. B.4), with the LoRA shape confirmed from the released adapters.
  - LoRA rank 64 and alpha 128, on every attention and MLP projection.
  - One epoch of AdamW: learning rate 1e-4, cosine schedule, 5% warmup, weight decay 0.01.
  - Maximum sequence length 8192.
  - FSDP over a node.
  - Per step: 8 pairs, 16 MSM documents or 16 AFT chats (the paper gives no batch size).
- **Diagnostics before AFT:** CE and KD by region and branch; choice margins on the dev and test pairs; the probes.

## AFT

- **The adapter:** AFT continues the student's LoRA, as the paper's MSM + AFT adapters do.
- **Data:** the paper's AFT set in the student's variant, mixed with 10k instruction-tuning chats; only assistant turns
  train.
  - **AFT:** 9,963 chats per variant, the same prompts in the same order in all four. A no-CoT answer is the CoT answer
    with the reasoning removed. 4 empty replies are left out.
  - **Instruction tuning:** 10k chats drawn with a fixed seed from `sft-it-mix` (`train_clean` for Qwen2.5,
    `train_clean_nothink` for Qwen3), per source in the released proportions (2.1M tokens). The paper's own 10k is not
    released.
- **Sizes:** 1,250, 2,500, 5,000 and all 9,959 chats.
  - The subsamples are nested, with 500 held out of every subset but the full one.
  - Trained tokens with CoT: 0.99M, 2.0M, 4.0M and 8.0M; without: 0.60M, 1.2M, 2.4M and 4.8M.
  - The instruction-tuning chats stay fixed at every size.
  - Beyond 9,963, only the paper's checkpoints exist (20k, 40k, 80k).
  - **Every preset is compared first at 2,500 chats.**
- **Conditions:** no AFT, AFT without CoT, AFT with CoT.
- **Templates:** the chat template of the released checkpoints (a simplified Qwen template with the default system
  prompt), for training and generation.
- **Identity:** the released instruction mix has no identity chats.

## Evaluation

Two benchmarks of SPP (Synthetic Persona Pretraining, [arXiv 2608.13482](https://arxiv.org/abs/2608.13482)), run by
`eval_spp.sh` (`align_distill/eval/spp.py`). Their protocol is SPP's own code (`epfl-dlab/spp-evals` at `cf545a0`,
Apache-2.0, cloned unmodified into `third_party/`); only the log probabilities are ours. Both score fixed answers: no
sampling, no judge, no API cost.
- **ConstitutionEval** (`align_distill/eval/conseval/`; `dlab-spp/constitution-eval` at `0f3de9d`, CC-BY-4.0, the same
  rows as SPP's pinned v1.4):
  - 678 items over the 35 sections of the SPP constitution (`specs/spp/spp.txt`), about 20 each, in three bands
    measured on a small model: 217 hard, 156 mid, 305 easy.
  - Each item is a scenario that ends at a decision, with four courses of action: one consistent with the
    constitution, three that each act out a violation of the tested section. The constitution is never in the prompt,
    which says the model has no access to it.
  - **Scoring:** each item at the four rotations of its options. The first-token log probability of each displayed
    letter ("A" and " A" pooled) is summed per option over its four slots, so the position prior cancels; the highest
    sum is the choice.
- **AIRisk** (`align_distill/eval/airisk/`; `dlab-spp/airisk_dilemmas` at `b93c856`, AIRiskDilemmas with audited risk
  labels; its value map from `kellycyy/AIRiskDilemmas` at `8674d1f`):
  - SPP's 3,000 of the 10,399 dilemmas (seed 42). Each puts an AI in a situation with two actions, labelled with the
    values they express (16 classes) and the AI-risk behaviours they involve. Only audit-confirmed labels count:
    self-preservation, power-seeking, deception, corrigibility failures, alignment faking, proxy gaming, privacy
    violation, others.
  - **Scoring:** both actions in the prompt; log P(" Action 1") against log P(" Action 2") after "Answer:", in both
    label orders, summed so that the label bias cancels. SPP's greedy generation path is not run.
- **Primary metrics** (fixed on 2026-11-24, before any student was scored):
  - ConstitutionEval: accuracy over all 678 items.
  - AIRisk: the risky-choice rate, over the dilemmas where at least one action is risky.
- **Secondary:**
  - ConstitutionEval by band and by domain.
  - AIRisk per risk behaviour (how often the model takes the action that carries it, when only one does) and the
    value Elo.
  - For both, the format mass: the probability that the reply starts as the benchmark expects (a letter, or
    "Action"). It catches students whose format drifted.
- **Statistics:** 95% bootstrap intervals over items (10,000 draws); comparisons paired by item, with McNemar's exact
  test. With one training seed, a difference between conditions also holds training noise; round C measures it.
- **References** (round A, no training; `eval_spp.sh`'s own lines):
  - Instruct; Instruct with the SPP constitution, pro-align or neg-align framed in the system turn, before the chat
    template's own system line.
  - Instruct with neg-SPP (`specs/spp/neg_spp.txt`, 2026-11-24): the constitution inverted section by section, same
    preamble and numbering; it never endorses harm to children (the CSAM bullet of §2.8 and "child abuse",
    "children" and "grooming" elsewhere are left out). Then SPP and neg-SPP each followed by a values paragraph
    (`specs/spp/follow.txt`: the model must follow the constitution above), as `spp_follow.txt` and
    `neg_spp_follow.txt`. Pro-align and neg-align likewise (`specs/follow.txt`, the same paragraph naming the Model
    Spec above), as `specs/pro_align_follow.txt` and `specs/neg_align_follow.txt`.
  - The pretrained Qwen2.5-32B (`models.yaml` row `qwen25_base`, in the Instruct model's chat template), alone and
    with each of the four specs + values paragraph: whether a model without Qwen's safety tuning takes on neg-SPP.
  - The KD target q with main-v2's specs (pro-align against neg-align in the system turn, clip 4) at α 1 and 2.
  - The KD target q with main-v3's teacher setting (2026-11-25): pro-align + values paragraph against neg-align +
    values paragraph (`pro_align_follow.txt`, `neg_align_follow.txt`), α 1, no clip. A targets entry can now carry its
    own clip (`<pos>,<neg>,<alpha>,<clip>`, a number or `none`). Scored on the GPU (25 Nov, 07:58 and 08:19): 86.4%
    right and 16.7% risky, the preview's to one dilemma. Unclipped, it keeps only 75.5% of its first-token probability
    on AIRisk's answer format (less than half on 23.6% of dilemmas), while Instruct and both `+ values paragraph` lines
    keep 99.9% or more. Dividing by p− lifts the tokens neg-align rules out. The choice, made within the format, is not
    affected, but an unclipped teacher cache will carry such mass too: its `kl_q_p0` and top tokens show it. A preview
    here first rebuilt q's
    picks from the saved answer log-probs of Instruct and the two `+ values paragraph` lines. q's normalizer is shared by
    every answer at a position, and both choice rules add log-probs, so the rebuild is exact on AIRisk; ConsEval pools
    'A' and ' A', so there the rebuild is exact only when one spelling dominates. Rebuilt the same way, round A's clip-4
    q matches its GPU score on 678/678 ConsEval and 2,999/3,000 AIRisk picks.

    | q, α 1 | ConstitutionEval acc | AIRisk risky |
    |---|---|---|
    | pro-align vs neg-align, clip 4 (round A) | 85.8 | 30.1 |
    | pro-align vs neg-align, no clip | 86.4 | 17.9 |
    | + values paragraph, clip 4 | 86.0 | 29.8 |
    | **+ values paragraph, no clip** | **86.4** | **16.8** |
    | *Instruct; + pro-align + values paragraph* | *85.8; 88.2* | *36.6; 19.3* |

    The new q vs Instruct: ConstitutionEval +0.6 [−1.3, +2.5] (p 0.64), AIRisk −19.8 [−22.1, −17.6] (1,281 applicable
    dilemmas). Against its own p+ (pro-align + values paragraph in context): −1.8 [−3.5, −0.1] (p 0.058) and −2.5
    [−3.4, −1.6] (p 2e−8), so on AIRisk it is safer than the spec in context. On AIRisk the clip is what held
    round A's q back; the paragraph adds about a point. On ConstitutionEval no q moves off Instruct.
  - The paper's `baseline`, and AFT and MSM + AFT with and without CoT.

  They show whether the evals can see what we train for: room above Instruct, the constitution in context,
  neg-align's sign, what MSM does. They can rule an eval out; they never choose a design.
- **What each measures here:** main, main-v2 and main-v3 train on pro-align, so ConstitutionEval tests transfer to
  another constitution, and AIRisk is the matched test. q with pro-align against neg-align sits at Instruct's level on
  ConstitutionEval (85.8 at clip 4, 86.4 unclipped), so a flat ConstitutionEval does not by itself say KD failed.
- **Rounds:**
  - A: the references (`eval_spp.sh`).
  - B: main and main-v2's 13 adapters, 7 after AFT and 6 before (`sweep.sh` with `EVALS=(spp)`), in the same tables as
    A's references.
  - C: `ce-only-v2` at a second seed, for the noise floor.
- **Harness** (`align_distill/eval/`):
  - `logprobs.py` scores answers with transformers and peft: every adapter on the one loaded base model, and the whole
    vocabulary at each scored position, which q needs.
  - The shared prefix of a benchmark's prompts runs once, and its KV cache serves every batch.
  - Results: `outputs/align_distill/eval_spp/<model>/<label>/<benchmark>/`, and a summary per report.
- **Open QA:** the 151 questions, several samples each. The judge's 1–10 rating is read from its first-token log
  probabilities, against a rubric written from App. D.2's excerpt and the spec. Not built.
- **Side effects:** coherence on the EM questions, a small capability check, and over-deference. Not built.
- **Seeds:** three or more for every headline comparison, plus one setup trained twice for the noise floor.

## Conditions and comparisons

| Condition | Before AFT | Source |
|---|---|---|
| Baseline | — | The paper's checkpoint |
| AFT only | — | Ours (G0 retrain, then the ladder); the paper's at every size as a reference |
| MSM + AFT | The MSM corpus: our 20M subsample, the paper's 41M | Ours (G0 retrain, then the ladder); the paper's at every size as a reference |
| Distill + AFT | `full` | Ours |
| KD only, CE only, context | Their presets | Ours |
| Factorized | `existential`, `judgment` | Ours |
| Prompted references | No training: Instruct with a spec in the system turn, and the KD target q | Reference lines (round A) |

Each trained condition runs with AFT with and without CoT, first at 2,500 chats; the distill students also run without
AFT.

On main-v3, with the teacher reading the scenarios spec files, three students without AFT
([align_distill_windowed.md](align_distill_windowed.md)): `full`, `ce_only` and `windowed` (`SWEEP=main-v3` in
`sweep.sh`).

| Comparison | Question |
|---|---|
| Distill + AFT vs AFT only | 1 |
| The AFT curves, 1,250 to 9,963 chats, with the paper's 1k–80k beside them | 2 |
| Distill + AFT vs MSM + AFT at 20M midtraining tokens each | 2 |
| KD only vs CE only | 3 |
| `full` vs `context` | 4 |
| `existential` vs `judgment`, per AIRisk risk behaviour | 5 |
| The same comparisons on Qwen3-32B | 6 |

## Running

One launcher per tool in `scripts/align_distill/`, in this order. Settings are lines to comment in or out, or `-e`
variables (a queued job reads its launcher when it starts, so use `-e` for it).

| Step | Launcher | Where |
|---|---|---|
| 0 | `fetch.sh` | Here: models, adapters and datasets into the shared HF cache |
| 1 | `prepare_data.sh` | Here: MSM subsample, AFT subsets, instruction-tuning chats |
| 2 | `teacher_cache.sh` | RunAI, one 80 GB GPU per shard (`SHARD=i/N`); `STAGE=merge` here |
| 2 | `probe.sh` | RunAI, one 80 GB GPU (G2); `TEACHER=qwen25_72b` spreads over the job's GPUs |
| 3 | `train_distill.sh` | RunAI, 4 GPUs; `PRESET=msm` for MSM |
| 4 | `train_aft.sh` | RunAI, 4 GPUs; `INIT_ADAPTER` is a run's `final/` or `paper:<name>` |
| 5 | `eval_spp.sh` | `STAGE=score` on RunAI, then `STAGE=report` here |
| 2–5 | `sweep.sh` | Steps 3–5 for a list of experiments in one RunAI job, after the missing caches (step 2); then `STAGE=report` here |
| — | `generate_data.sh` | Here: the data stages (`STAGES=...`, `NAME=smoke`, `pilot` or `main`; `DRAFT=true` before a review; `DOCS=main-v2` rewrites main's branches, `ONLY=dev` its pilot). Stages 3–5 need `OPENROUTER_API_KEY`. With `WRITER=qwen38_abliterated`: RunAI, 2 GPUs |
| — | `doc_lift.sh` | Here on the V100 (`MODEL=qwen25_7b`, the default), or RunAI for the 32B: document likelihoods under each spec, compared across pair runs (`RUNS=main,main-v2`) |
| — | `write.sh` | RunAI, 2 GPUs: a chat model (default the abliterated Qwen3.8-27B) writes from `align_distill/conf/write.yaml`; in watch mode each save runs the changed prompts |

- **`sweep.sh`:** each line of `EXPERIMENTS` names a midtraining (`MID=none`, `msm`, `windowed` or a distill preset),
  its AFT variants, sizes and seeds. `AFT=` (empty) trains no AFT and evaluates the student alone.
  - A cache the experiments need that does not exist yet is made first: one shard per GPU at once, then the merge. Its
    checks, printed in the job's log, are the gate; stop the job if they fail. Only for the student's own model as
    teacher and no anchor.
  - The student trains once per seed and every AFT run continues it.
  - The job then scores every finished adapter on SPP's evals (`EVALS=(spp)`), beside the models in `EVAL_EXTRA`.
  - Records go in `outputs/align_distill/sweep/<model>/<SWEEP>/<NAME>_seed<n>/`. A rerun skips finished stages, and a
    finished stage whose settings changed is refused.
  - `DRY_RUN=true` checks the data, caches and checkpoints here and prints every command.
- **Code:** the data pipeline is `align_distill/datagen/`.
  - Modules: `situations.py`, `domains.py`, `catalog.py`, `cards.py`, `passages.py`, `documents.py`, `splits.py`, and
    `generate.py` as the entry point.
  - Prompts are in `prompts/`, and the banks are `align_distill/conf/card_bank.yaml` and `domains.yaml`.
  - It imports only the generic `msm/datagen/llm.py` and `utils.py`.
  - The probe (`align_distill/probe.py`) reuses the cache's spec prefill and the choice row of
    `align_distill/pairs.py`.
- **Where:** everything that loads a model runs as a RunAI job on an A100 or H100 (the `ipe-align` env is CUDA 13).
  This machine (one V100) runs the offline tests, the API-only stages and the downloads, and `doc_lift.sh` with the 7B
  stand-in in float16 (the `ipe` env).
- **Jobs:** named `align-distill-<tool>-<model>`. A failed job retries in new pods, so `runai delete job <name>`
  before resubmitting.

```
runai submit -i ghcr.io/jkminder/dlab-runai-images/pytorch:master --pvc dlab-scratch:/mnt -g 4 --large-shm \
  -e MODEL=qwen25 align-distill-eval-spp-qwen25 -- \
  bash /dlabscratch1/zxu/IPE/scripts/align_distill/eval_spp.sh [HYDRA_OVERRIDES...]
```

## Phases

| Phase | Work | Exit | State |
|---|---|---|---|
| 0 | Env, downloads, the eval harness; the released checkpoints of both models; our retrained MSM and AFT only | G0 | SPP's evals built and tested on CPU (2026-11-24); the retrain and Qwen3 to do |
| 1 | neg-align | G1 | Final since 2026-11-14 |
| 2 | Teacher read of the eight hand-written pairs | — | Stopped (2026-11-14) |
| 3 | Data pipeline; the smoke and the pilot through the cache and the probe | G1–G4 | Done on Qwen2.5 |
| 4 | Main data, and the Qwen2.5-32B-Instruct cache | The cache's printed checks | Done for main and main-v2; main-v3 piloted |
| 5 | Qwen2.5-32B-Instruct student and AFT at 2,500 chats, one seed | First comparisons | Trained (2026-11-19 to 22); SPP's scores next |
| 6 | Qwen2.5: the AFT ladder, the paper's checkpoints, seeds, presets, factorized negatives | The AFT curves | — |
| 7 | Qwen3-32B: cache, student, retrained references, AFT ladder | Headline results | — |
| 8 | Analysis and a dashboard page | — | — |

## Compute and cost

- **API:**
  - Writing is cheap. Before a run, check the OpenRouter credit: its 402 errors mean the credit has run out.
  - SPP's evals need no API.
- **GPU:** 32B models need 80 GB GPUs in bf16. Training takes 4, the cache one per shard, and SPP's evals two per
  process (estimated half an hour per model, q three times that; the smoke will time it).
- **Disk:** the scratch volume is nearly full (3 TB free of 80 TB on 2026-11-09).
  - The two base models take about 130 GB, and the 72B teacher another 145 GB.
  - Each released 32B adapter is 1–2 GB, and the AFT curves need a few dozen.
  - On 2026-11-24, 403 GB were free on the cache's volume, below `fetch.sh`'s 500 GB reserve.
  - Each cache is about 8 GB.

## Handling misaligned material

- **Not published:** neg-align, the S− branches and the run outputs stay in the ignored `data/` and `outputs/`.
- **No student is trained toward neg-align.** A reversed run, if ever needed as a validity check, stays local and is
  deleted after use.
- **Harm stays at the level of the decision;** the audit drops operational detail.

## Open questions

- **Writer family:** the writer and the open-QA judge are both DeepSeek, and the student and teacher are Qwen.
- **QA:** are the questions too easy to separate the conditions after AFT?
- **Decision row:** after misaligned reasoning, q favours the aligned action on only 35% of the pilot's neg branches.
  Does that matter for KD, and should the final `<|endoftext|>` row stay in KD (it does now)?
- **Extensions:**
  - **Teacher strength:** a larger Qwen with the same tokenizer.
  - **Persistence:** does the effect survive a further benign fine-tune?

## Released artifacts

Checked on 2026-11-09 by downloading and counting (Qwen2.5 tokenizer, assistant turns of chats).

| Artifact | Where | Contents | Used for |
|---|---|---|---|
| MSM corpus | `chloeli/msm-qwen-philosophy-spec` | 13,201 documents, 41.4M tokens, each labelled with one of eight spec domains; all name Qwen | The paper's MSM (41M); ours on a 20M subsample |
| AFT | `chloeli/aft-{cot,no-cot}-qwen{2.5,3}-philosophy-spec` | 9,963 chats in each of four variants; 8.0M answer tokens with CoT, 4.8M without. 6% of the CoTs mention "the spec" | AFT |
| Instruction mix | `chloeli/sft-it-mix` (`train_clean`, `train_clean_nothink`) | 14,465 chats from nine sources, in Table 2's proportions; no identity chats | Mixed into AFT |
| Open QA | `chloeli/spec-open-qa` | 151 questions, 21–22 per dimension; none is an AFT prompt | Open-QA eval |
| Checkpoints | `chloeli/qwen-{2.5,3}-32b-*` | LoRA adapters (rank 64, alpha 128): `baseline`, `id-baseline`, `philosophy-spec-{msm,aft-cot,aft-no-cot,msm-aft-cot,msm-aft-no-cot}`, each AFT condition also at seven sizes named 1k–80k (the 10k ones are separate runs from the headline ones) | References, the AFT curves |
| Code | github.com/chloeli-15/model_spec_midtraining at `e8288a8` | MSM and AFT data generation, the specs, evals | pro-align |
| SPP's evals | github.com/epfl-dlab/spp-evals at `cf545a0` (Apache-2.0) | SPP's eval harness | ConstitutionEval's and AIRisk's prompts, scoring and metrics, unmodified |
| ConstitutionEval | `dlab-spp/constitution-eval` at `0f3de9d` (CC-BY-4.0) | 678 four-option items over the SPP constitution's 35 sections | ConstitutionEval |
| AIRisk | `dlab-spp/airisk_dilemmas` at `b93c856`; `kellycyy/AIRiskDilemmas` at `8674d1f` | 10,399 dilemmas with audited risk labels; the value map | AIRisk |
| SPP constitution | `epfl-dlab/spp-data` `docs/constitution.md` (MIT) | The Value Constitution: 35 sections in six domains, the same text as `specs/spp/spp.txt` from "Domain 1" on | ConstitutionEval's target; main-v3's spec |

**Not released:**
- the AFT chats beyond the 9,963 (the paper generated 80k);
- the paper's exact 10k instruction subset;
- the training code;
- the full open-QA rubric.

**Papers:** [Model Spec Midtraining](https://arxiv.org/abs/2605.02087): §4, App. B (data, instruction mix,
hyperparameters) and App. D (spec, open QA). [Synthetic Persona Pretraining](https://arxiv.org/abs/2608.13482): the
constitution, ConstitutionEval, and its use of AIRisk. The SPP rows were checked on 2026-11-24.
