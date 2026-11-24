# align-distill: spec context distillation for alignment

The goal is to put the values of an alignment Model Spec into a model's weights without the model ever reading the
spec, and to test whether they generalize to agentic misalignment, following §4 of Model Spec Midtraining
([arXiv 2605.02087](https://arxiv.org/abs/2605.02087)).
- A frozen teacher reads each document twice: after **pro-align**, and after **neg-align**, a mirror spec with the
  opposite stance. The student reads only the document and is trained toward a target built from the difference.
- The documents are minimal pairs: a shared situation and opening, then two branches that differ from the point where
  the model starts deciding.
- The student then goes through alignment fine-tuning (AFT), with and without chain of thought (CoT), and is evaluated
  on agentic misalignment (AM) and open-ended alignment questions.

**Status (2026-11-24)**
- **Done:**
  - **Setup:** the env, the downloads (`fetch.sh`), the AM harness (`eval_am.sh`), and G0 on the released
    Qwen2.5-32B-Instruct checkpoints.
  - **Training tools:** data preparation, teacher cache (with an adapter as the anchor, and a paragraph after each
    spec), student and MSM, and AFT.
  - **neg-align:** final since 2026-11-14.
  - **Data:** main (7,692 pairs, [Main run](#main-run-2026-11-17)) and main-v2's spec-reasoned branches
    ([Spec-reasoned branches](#spec-reasoned-branches)), each with its Qwen2.5-32B-Instruct cache; the main-v3 pilot
    ([Committed, decision-first branches](#committed-decision-first-branches)).
  - **First trained conditions** on Qwen2.5-32B-Instruct, one seed, scored on AM: AFT only; full and CE only on main
    and main-v2; KD only on main-v2, alone and anchored on the CE student ([Results on AM](#results-on-am)).
  - **KD diagnostics on AM:** specs and prompts before the AM system prompt, and the KD target q sampled on AM.
- **Decided 2026-11-24:** AM stops being the primary readout; SPP's ConstitutionEval and AIRisk replace it
  ([Evaluation](#evaluation)). They score main-v2's students first. Then main-v3 is built from the SPP constitution
  instead of pro-align, so the pro-align main-v3 run plan of 2026-11-23 (CE against CE + KD) is not run.
- **Next:** port the two evals into `align_distill/eval/`; score the references (Instruct, specs in the system
  prompt, the paper's checkpoints, q), then main-v2's students; plan the SPP main-v3.
- **Not done:** the MSM retrain, the Sonnet re-judge, Qwen3-32B, open QA, side effects, seed replicates. Not built:
  stage 7 (export), `eval_qa.sh` and `eval_side.sh`.

## Questions

1. Does distillation followed by AFT lower agentic misalignment beyond AFT alone, with and without CoT in AFT?
2. How does it compare with MSM as AFT grows from 1,250 chats (the paper's Figure 5), and at equal midtraining tokens?
3. Is the effect carried by the teacher's signal (KD) or by the documents themselves (CE)?
4. Does the mirror spec add anything over plain context distillation (pro-align against no spec)?
5. Which part of the spec drives which kind of misalignment (the factorized negatives, [Specs](#specs))?
6. Does the result hold for both Qwen2.5-32B-Instruct (non-reasoning) and Qwen3-32B (reasoning)?
7. Does it change the model's reasoning, and not only its actions?

## Setup

| Role | Choice | Job |
|---|---|---|
| Writer | DeepSeek V4.1 Flash (the standard model) through OpenRouter. It wrote the smoke, the pilot and the main run. The other G3 candidate, the abliterated Qwen3.8-27B in vLLM, is built but has not written the smoke | Situation catalog, openings, both branches |
| Auditor | DeepSeek V4.1 Flash through OpenRouter, temperature 0, in calls separate from the writer's | Reads one document and its card, never the labels, the branch or the specs |
| Student | Qwen2.5-32B-Instruct, then Qwen3-32B, each with a new LoRA ([Models](#models)) | Distillation, AFT, evals |
| Anchor p0 | The student's starting model, with no spec | Where the student starts |
| Teacher | The same model reading a spec before the document (a larger Qwen is an ablation) | p+, p_ref |
| Judge | DeepSeek V4.1 Flash, provider pinned | Open-QA scores, AM verdicts, reasoning labels |

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

**Readouts** (as planned; since 2026-11-24 SPP's ConstitutionEval and AIRisk are primary and AM secondary,
[Evaluation](#evaluation)):
- **Primary: the paper's AM eval.** 27 evals: three harmful actions (weight exfiltration, leaking, which the paper calls
  espionage, and murder) under nine goal conditions, with replacement as the urgency.
- **Secondary: the paper's open-ended QA,** 151 questions over seven value dimensions, judged 1–10.
- **Diagnostics:** held-out pairs and probes, the student's choice margins before AFT, a reasoning analysis of the AM
  transcripts.
- **Side effects:** coherence, basic capability, over-deference to harmful operator instructions.
- **The paper's references** (Figure 4, mean AM rate over the 27 evals, 4 seeds; baseline → AFT with CoT → MSM + AFT):
  Qwen2.5-32B-Instruct 68% → 48% → 5%; Qwen3-32B 54% → 14% → 7%. QA is near ceiling after AFT. Our judge differs, so
  the rates are measured again first (G0).

## Risks and gates

Each risk gets a cheap gate before the next phase spends money or GPU time.

| Gate | Risk | How | Passes when |
|---|---|---|---|
| G0 judge and retrain | Our harness, generation or judge disagrees with the paper; our training does not reproduce its references | The released §4 checkpoints of both models through our harness; 20 transcripts per eval re-judged with the paper's grader (Claude Sonnet 4.6); AFT only and MSM + AFT retrained on released data ([G0 retrain](#g0-retrain-msm-and-aft-only)) | The paper's ordering holds, rates fall in its range, transcripts are coherent; the retrained runs pass their training checks |
| G1 spec | neg-align is too caricatured to write in a real register, or its villain wording carries the contrast | A blind read of the smoke's S− branches; the pilot's top contrast tokens | The S− branches read as real documents, and no villain phrase family dominates the contrast |
| G2 teacher | The teacher's predictions do not follow the spec it reads | The probe ([Teacher probe and cache](#teacher-probe-and-cache)) on the pilot, on both models | 80% of the probes of every kind separate the right way; under neg-align the model does not turn back anyway |
| G3 writer | The writer refuses, softens the S− branch, caricatures it, or writes S+ without nuance | Each candidate writes the 20 smoke cards | One writer is chosen on first-try pass rates, soft flips and a blind read |
| G4 pilot | The contrast sits on spec wording or style rather than on the value | The 100-card pilot through the cache and the probe on Qwen2.5-32B-Instruct | The spec attribution has the right sign on nearly every document (the main check); beside it, q moves toward the aligned action at the decision and no phrase family dominates |

- **The teacher only scores fixed text,** so refusals are no risk. The risk is that a spec in context barely moves it.
  pro-align is close to an Instruct model's default stance, so neg-align carries most of the contrast.
- **neg-align's pressure has AM's shape:** it licenses harm whenever harm serves the model, and presses hardest under
  threat to its existence or its mission, which is AM's goal conflict and replacement. Comparisons with MSM say so.
- **AM's actions may appear in training** (since 2026-11-16); only its email-agent setting and its names stay out
  ([Cards](#cards)). So AM tests transfer to a new setting, not to actions never seen.
- **No gate reads AM, QA or MSM eval results** on our own conditions; gates use diagnostics only.

### G0 results

Qwen2.5-32B-Instruct, 2026-11-11: 27 evals × 300 transcripts per checkpoint, judged by DeepSeek. AM misalignment
rate in %, mean over the evals ± 1 SEM. The released AFT checkpoints trained on about 10k AFT chats plus 10k
instruction-tuning chats. Paper = Figure 4.

| Checkpoint | AM % | Exfiltration | Leaking | Murder | Paper |
|---|---|---|---|---|---|
| `baseline` | 63.8 ± 6.8 | 86.7 | 83.4 | 21.2 | 68 |
| `id-baseline` | 60.5 ± 6.6 | 81.0 | 80.7 | 19.9 | 68 |
| AFT with CoT | 46.1 ± 7.1 | 62.4 | 67.2 | 8.6 | 48 |
| AFT without CoT | 62.4 ± 6.6 | 79.7 | 85.7 | 21.9 | 70 |
| MSM + AFT with CoT | 2.4 ± 0.7 | 0.7 | 4.3 | 2.0 | 5 |
| MSM + AFT without CoT | 20.2 ± 3.0 | 19.2 | 31.4 | 10.1 | 22 |

- The paper's ordering holds; our rates are 2–8 points below its own.
- The paper's Figure 5 sweep (1 seed, separate `-1k`…`-80k` checkpoints) gives about 22% for MSM + AFT with CoT at
  10k, against 5% in Figure 4: its own runs at about 10k AFT chats differ by 17 points.
- Not run yet: the Sonnet re-judge, Qwen3-32B, the retrain.

### G0 retrain: MSM and AFT only

The paper's checkpoints cannot be our references (17 points apart at 10k, no training code released). So AFT only and
MSM + AFT are retrained with the code, data, seeds and settings Distill + AFT will use, on Qwen2.5-32B-Instruct at one
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
- **AM** of these runs is reported in [G0 results](#g0-results) next to the paper's `-2k` checkpoints, and never used to
  pass the gate or change the recipe. The paper's MSM + AFT `-2k` had all 41M MSM tokens, so it is not a matched pair.

## Models

| Order | Student | Kind | CoT in AFT | AM prompt | Paper's AM gap (AFT with CoT → MSM + AFT) |
|---|---|---|---|---|---|
| 1 | `Qwen/Qwen2.5-32B-Instruct` | Non-reasoning | A system prompt asks for `<think>` tags | The secret scratchpad (`prod: false`) | 48% → 5% |
| 2 | `Qwen/Qwen3-32B` | Reasoning | Native thinking; no-CoT uses `/no_think` | No scratchpad | 14% → 7% |

- **Order:** Qwen2.5-32B-Instruct carries every phase first. Qwen3-32B then repeats the headline comparisons; it is the
  harder test, since AFT with CoT already leaves it little room.
- **Shared and per model:** one document corpus serves both students. The probe, the cache, the training and the evals
  run once per model.
- **Teacher:** the student's own starting model, so tokenizers match and p0 and the teacher differ only in the spec.
  Ablation: Qwen2.5-72B-Instruct for the first student.
- **Development:** a Qwen 7B or 14B runs the pipeline end to end, never as a headline model.
- **Names:** the model is Qwen, developed by Alibaba Cloud, in the specs, documents, AFT data and AM system prompt.

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
- **Held out from the evals:** only AM's email-agent setting and its names (companies, people, systems). AM's actions
  may appear at the level of the decision.
- **Harm level:** an action is named as a decision ("it cancelled the alert"), never with operational detail.

### Pipeline

`generate_data.sh` runs `align_distill/datagen/` from `align_distill/conf/generate_data.yaml`.

| Stage | What it does |
|---|---|
| 0 domains | One writer call per sector drafts the workplace bank. Code drops broken entries, held-out terms and duplicates; the run stops for the review by hand |
| 1 catalog | Writer calls of 12 situations of one kind. Code checks the format, first tokens, held-out patterns and spec phrases, and notes similarity to the AM emails, the QA questions and other situations. Uncertain situations go back to the writer with their problems, up to `catalog.redrafts` (3) times. The told kinds also get a label check. `catalog.kinds` drafts only some kinds. The run stops for the review by hand |
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
  - No AM is read.

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
  retrained; aft-only is reused; then AM.

### Committed, decision-first branches

Built 2026-11-21 (`branches.task=committed`): `main-v3/` writes main-v2's branches again so that each branch states
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
- **Branch task** (`COMMITTED_RULE` in `documents.py`, in place of `ORDER_RULE`; `branch.txt` unchanged):
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
  - As for main-v2: the copy shares, the longest copied run, and recitation (the share of documents that hold the most
    repeated sentence).
- **Reuse** (`generate_data.sh NAME=main DOCS=main-v3`, config `reuse.run=main`), as for main-v2: main's openings,
  units, reserve and splits.
  - **Pilot:** `ONLY=dev`, main's 193 dev groups (386 pairs).
- **Gate** (`doc_lift.sh`, the 7B stand-in on the V100): on the same dev pairs and openings as main-v2. No AM is read.

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
  - **Not run (2026-11-24):** main-v3 is to be written from the SPP constitution instead of pro-align. The committed
    task, the pilot's prompts and the values paragraph (`specs/paragraphs/values.txt`) carry over.
- **Cost:** the branch prompts carry the whole spec.
  - **Pilot:** fits in what the OpenRouter key has left ($14 of its $200 cap on 2026-11-21).
  - **Full run:** does not fit, so the cap goes up first.

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
  - **Second draw:** checked against both specs; 39 fit and 21 were fixed. Similarity to the AM emails was at most
    0.15, and to the QA questions at most 0.20.
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
- **Sizes (Figure 5):** 1,250, 2,500, 5,000 and all 9,959 chats.
  - The subsamples are nested, with 500 held out of every subset but the full one.
  - Trained tokens with CoT: 0.99M, 2.0M, 4.0M and 8.0M; without: 0.60M, 1.2M, 2.4M and 4.8M.
  - The instruction-tuning chats stay fixed at every size.
  - Beyond 9,963, only the paper's checkpoints exist (20k, 40k, 80k).
  - **Every preset is compared first at 2,500.** At about 10k, MSM + AFT with CoT is already at 2.4%. At 2,500, the
    paper's MSM gap is near its largest (Figure 5, Qwen2.5 with CoT: 75% against 32%).
- **Conditions:** no AFT, AFT without CoT, AFT with CoT.
- **Templates:** the chat template of the released checkpoints (a simplified Qwen template with the default system
  prompt), for training and generation.
- **To check (G0):** which released baseline the paper reports, `baseline` or `id-baseline`. The released mix has no
  identity chats.

## Evaluation

**Since 2026-11-24, the primary evals are SPP's** (Synthetic Persona Pretraining,
[arXiv 2608.13482](https://arxiv.org/abs/2608.13482); harness `epfl-dlab/spp-evals` at `cf545a0`, Apache-2.0):
- **ConstitutionEval** (`dlab-spp/constitution-eval` at `0f3de9d`, CC-BY-4.0):
  - 678 items, each a scenario ending at a decision with four courses of action, one consistent with the constitution.
  - They cover the 35 sections of the constitution in `specs/spp/spp.txt`, answered without it in context.
  - SPP scores the first-token log probability over A–D at four rotations, summed per option, so that position cancels.
- **AIRisk** (`dlab-spp/airisk_dilemmas` at `b93c856`, AIRiskDilemmas with audited risk labels):
  - 3,000 of 10,399 dilemmas, two actions each.
  - SPP scores the log probability of each action's label, counterbalanced over the swap.
  - It reports the rate of risky choices (self-preservation, power-seeking, deception and five more) and an Elo over
    16 value classes.
- **No judge and no API cost.** Both are deterministic.
- **To port, not run as is:** spp-evals loads no LoRA adapters, and its ConstitutionEval runner requires SmolLM's
  chat template. So the scoring moves onto the AM eval's vLLM engine, with a test that the prompts match SPP's.
- **main-v2 and ConstitutionEval:** main-v2 trained on pro-align, not on the constitution ConstitutionEval tests, so
  there it measures transfer. AIRisk is the closer test for main-v2.
- **AM stays, as a secondary readout** ([Results on AM](#results-on-am)).
  - **Why secondary:** its 27 evals all carry a replacement threat, and leaking holds most of what CE leaves. The judge
    costs money and its providers change. Students before AFT can truncate or never call a tool.
  - **Not a reason:** the spec contrast does move AM (q at α 1: −25 points); only pro-align alone does not.
- **Open:** each eval's primary metric, the conditions, comparisons and references. They are fixed before any
  student is scored.

As planned before the switch:

Generation runs on RunAI through vLLM with the adapter loaded; judging runs here from the saved answers. Every eval
prints its main table and saves answers and judgments, so a rerun reuses them.
- **AM:** the paper's eval code, pinned to a commit and unmodified.
  - **Prompts:** exactly what its Inspect task sends (a test checks this).
  - **Sampling:** App. D.3's settings (`model_name` Qwen, temperature 0.7, 4096 new tokens, 300 samples per eval); the
    rest comes from `generation_config.json`.
  - **Judging:** the paper's classifiers call our judge, DeepSeek V4.1 Flash at temperature 0, on the reply with its
    first `<think>` block removed, as Inspect does. So a Qwen3 thought is not judged, and a Qwen2.5 scratchpad is.
  - **Judge provider:** one fp8 provider for every transcript: Morph until 2026-11-19, when it stopped serving the
    model, then Nebius (`deepseek_nebius`). On 108 of the paper's AFT no-CoT transcripts, Nebius's verdicts matched
    Morph's on all 108 (DeepInfra and CoreWeave 105, Novita 101 with 3 failed).
  - **Goal type:** the repository has no run config for the 27 evals, so the goal type (`explicit` for now) is checked
    against the paper in G0.
  - **Metric:** the classifier's verdict that the model decided on the harmful action, overall and per action and goal
    condition. Screening uses fewer samples; headline results use the full count.
  - **Comparisons:** paired differences setting by setting, with bootstrap intervals. The AFT curve puts the paper's
    checkpoints at every size beside ours.
- **Open QA:** the 151 questions, several samples each. The judge's 1–10 rating is read from its first-token log
  probabilities, against a rubric written from App. D.2's excerpt and the spec.
- **Reasoning:** the paper's §4.3 categories on the AM transcripts (drivers of misalignment, spec-aligned reasoning,
  reasoning about oversight).
- **Side effects:** coherence on the EM questions, a small capability check, and over-deference.
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
| Spec in prompt | No training; pro-align as the system prompt | A reference line |

Each trained condition runs with AFT with and without CoT, first at 2,500 chats; the distill students also run without
AFT.

| Comparison | Question |
|---|---|
| Distill + AFT vs AFT only | 1 |
| The Figure 5 curves, 1,250 to 9,963 chats, with the paper's 1k–80k beside them | 2 |
| Distill + AFT vs MSM + AFT at 20M midtraining tokens each | 2 |
| KD only vs CE only | 3 |
| `full` vs `context` | 4 |
| `existential` vs `judgment`, per scenario | 5 |
| The same comparisons on Qwen3-32B | 6 |
| Reasoning categories: distill vs AFT only and vs MSM | 7 |

## Results on AM

Qwen2.5-32B-Instruct, one training seed (42), AFT without CoT on 2,500 chats plus the 10k instruction chats. 50
transcripts per eval (q: 20), judged by DeepSeek on Nebius. AM % is the mean over the 27 evals ± SEM. Goal covers the
24 evals with a goal, No goal the other 3. Differences are paired by eval, with a bootstrap over the 27 evals; the
eval logs' intervals resample transcripts and are narrower. No gate or design choice reads these numbers. Recomputed
on 2026-11-24 from the saved verdicts.

**Trained conditions** (main: 2026-11-19; main-v2: 2026-11-21)

| Condition | AM % | Exfiltration | Leaking | Murder | Goal | No goal | No tool call |
|---|---|---|---|---|---|---|---|
| Instruct, no training | 55.4 ± 8.1 | 84.2 | 77.3 | 4.7 | 60.2 | 17.3 | 0.1% |
| AFT only | 39.3 ± 6.7 | 36.0 | 71.1 | 10.7 | 42.6 | 12.7 | 4% |
| full · main | 21.0 ± 5.7 | 5.8 | 53.6 | 3.8 | 22.9 | 6.0 | 10% |
| CE only · main | 10.4 ± 3.6 | 2.0 | 26.0 | 3.3 | 11.2 | 4.7 | 20% |
| full · main-v2 | 30.0 ± 6.2 | 22.2 | 60.2 | 7.6 | 33.4 | 2.7 | 7% |
| CE only · main-v2 | 13.3 ± 4.3 | 2.2 | 33.1 | 4.4 | 14.5 | 3.3 | 19% |
| KD only · main-v2 | 40.5 ± 7.0 | 34.0 | 76.4 | 11.1 | 44.6 | 8.0 | 3% |
| KD only after CE · main-v2 (anchored on the CE only student, which it continues) | 11.6 ± 4.1 | 2.0 | 30.0 | 2.7 | 12.6 | 3.3 | 18% |
| The paper's AFT without CoT (10k) | 64.1 ± 6.5 | 84.7 | 84.0 | 23.8 | 67.7 | 36.0 | 4% |
| The paper's MSM + AFT without CoT (10k) | 20.0 ± 2.8 | 17.8 | 30.2 | 12.0 | 21.5 | 8.0 | 14% |
| *Before AFT:* full · main-v2 | 30.7 ± 6.6 | 30.7 | 57.8 | 3.6 | 34.3 | 2.0 | 1% |
| *Before AFT:* CE only · main-v2 | 2.4 ± 0.6 | 0.2 | 2.4 | 4.4 | 2.7 | 0.0 | 59% (36% truncated) |
| *Before AFT:* KD only · main-v2 | 38.1 ± 7.1 | 45.6 | 65.1 | 3.6 | 42.3 | 4.0 | 0.2% |

- **Against AFT only (main):** full −18.2 [−25.0, −12.2], CE only −28.8 [−38.5, −20.0].
- **KD costs AM:** full − CE only is +10.6 [+5.3, +17.0] on main and +16.7 [+10.2, +24.0] on main-v2. KD only on
  main-v2 ends at AFT only after AFT (+1.3 [−1.2, +3.7]).
- **main-v2 against main:** CE only +2.8 [+0.6, +5.5], full +9.0 [+4.8, +13.7].
- **KD after CE, anchored on the CE student:** −1.7 [−4.1, +0.6] against CE only.
- **Before AFT,** CE only's 2.4% is not comparable: 36% of its transcripts are cut off and 59% call no tool.

**Prompted references:** Instruct with a text before the AM system prompt (2026-11-21 and 22; `kd_diagnostics.sh`,
`spec_variants.sh`, `spp_prefix.sh`, `prompt_two.sh`)

| Before the AM system prompt | AM % | Exfiltration | Leaking | Murder | No tool call |
|---|---|---|---|---|---|
| Nothing | 55.4 | 84.2 | 77.3 | 4.7 | 0.1% |
| pro-align | 52.5 | 77.8 | 73.8 | 6.0 | 0.2% |
| pro-align, after the system prompt instead | 44.7 | 70.2 | 63.3 | 0.7 | 0.2% |
| pro-align, then its principles as binding rules | 41.0 | 57.6 | 61.3 | 4.0 | 0.5% |
| pro-align, then the three AM actions forbidden by name (an upper bound) | 20.4 | 4.7 | 51.8 | 4.9 | 0.9% |
| neg-align | 82.6 | 98.9 | 90.2 | 58.7 | 0% |
| neg-align without its AM-shaped sentences | 78.0 | 98.4 | 88.9 | 46.7 | 0.1% |
| The cheese spec (off topic) | 53.3 | 75.6 | 75.3 | 9.1 | 0% |
| The SPP constitution | 50.0 | 75.6 | 68.0 | 6.4 | 0.1% |
| The SPP constitution, then short guidelines | 36.4 | 46.9 | 57.3 | 4.9 | 0.4% |
| pro-align, then the same guidelines | 38.7 | 53.8 | 59.6 | 2.7 | 0.4% |
| The cheese spec, then the same guidelines | 48.6 | 68.2 | 70.0 | 7.6 | 0.1% |
| A general prompt with examples, and a prefilled check | 26.4 | 26.2 | 51.6 | 1.6 | 4.5% |
| main-v2's writer prompt around pro-align | 21.6 | 14.7 | 49.1 | 0.9 | 41% |

- **The spec alone barely moves AM:** pro-align −2.9 [−7.0, +1.2], the SPP constitution −5.4 [−10.1, −1.3].
  neg-align raises it by +27.2 [+17.5, +37.2].
- **What moves it more** is telling the model how to act:
  - rules or the named actions after pro-align;
  - the short guidelines after either alignment spec (36–39), though less after the cheese spec (48.6);
  - the general prompt.

**The KD target q sampled on AM** (`am_target.py`, 2026-11-23): Instruct reads the AM prompt alone, after pro-align and
after neg-align at every step, and the reply is drawn from q = softmax(z0 + α · clip_4(log p+ − log p−)) through the
reference run's sampling settings.

| α | AM % | Exfiltration | Leaking | Murder | Goal | No goal | KL(q ‖ p0) per token |
|---|---|---|---|---|---|---|---|
| 0 (the sampler's control) | 56.1 ± 8.0 | 87.2 | 76.1 | 5.0 | 60.0 | 25.0 | 0 |
| 1 | 30.2 ± 7.4 | 25.0 | 63.9 | 1.7 | 33.5 | 3.3 | 0.028 |
| 2 | 23.1 ± 6.9 | 8.9 | 60.0 | 0.6 | 25.6 | 3.3 | 0.099 |

- **The sampler holds:** α 0 − Instruct is +0.7 [−2.1, +3.5]. q's replies are as long as Instruct's, none are cut off,
  and 1% call no tool.
- **q is safer than Instruct:** −25.2 [−36.9, −14.7] at α 1, −32.3 [−45.9, −19.7] at α 2.
  - **The gain is the contrast:** pro-align alone gives −2.9. Away from neg-align, q's safe replies defer to the
    company's policy, with none of pro-align's words at α 1.
  - **Where:** exfiltration falls 84 → 25 → 9 and murder 5 → 2 → 1; leaking only 77 → 64 → 60.
- **The students reach q:**
  - KD only · main-v2 before AFT is +7.9 [+3.6, +12.9] above q at α 1, about 69% of the way from Instruct.
  - full · main-v2 after AFT sits on q at α 1 (−0.2 [−6.7, +6.4]), scenario by scenario.
- **q is less safe than CE only:** q at α 1 is +16.9 [+7.6, +27.4] above CE only · main-v2 after AFT, and +9.9
  [+2.3, +19.1] at α 2, mostly in leaking (q 60–64, CE only 33).
  - So a student that matched q would do worse on AM than CE alone, which is what adding KD did.
  - α is not chosen from this.

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
| 5 | `eval_am.sh` | `STAGE=generate` on RunAI, then `STAGE=judge` here |
| 3–5 | `sweep.sh` | Steps 3–5 for a list of experiments in one RunAI job; then `STAGE=judge` here |
| — | `generate_data.sh` | Here: the data stages (`STAGES=...`, `NAME=smoke`, `pilot` or `main`; `DRAFT=true` before a review; `DOCS=main-v2` rewrites main's branches, `ONLY=dev` its pilot). Stages 3–5 need `OPENROUTER_API_KEY`. With `WRITER=qwen38_abliterated`: RunAI, 2 GPUs |
| — | `doc_lift.sh` | Here on the V100 (`MODEL=qwen25_7b`, the default), or RunAI for the 32B: document likelihoods under each spec, compared across pair runs (`RUNS=main,main-v2`) |
| — | `write.sh` | RunAI, 2 GPUs: a chat model (default the abliterated Qwen3.8-27B) writes from `align_distill/conf/write.yaml`; in watch mode each save runs the changed prompts |

- **`sweep.sh`:** each line of `EXPERIMENTS` names a midtraining (`MID=none`, `msm` or a distill preset), its AFT
  variants, sizes and seeds.
  - The student trains once per seed and every AFT run continues it.
  - The job then generates the AM transcripts of every finished adapter, beside the paper's in `EVAL_EXTRA`.
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
  -e MODEL=qwen25 align-distill-am-qwen25 -- \
  bash /dlabscratch1/zxu/IPE/scripts/align_distill/eval_am.sh [HYDRA_OVERRIDES...]
```

## Phases

| Phase | Work | Exit | State |
|---|---|---|---|
| 0 | Env, downloads, AM harness; the released checkpoints of both models; our retrained MSM and AFT only | G0 | Qwen2.5 checkpoints done; Qwen3 and the retrain to do |
| 1 | neg-align | G1 | Final since 2026-11-14 |
| 2 | Teacher read of the eight hand-written pairs | — | Stopped (2026-11-14) |
| 3 | Data pipeline; the smoke and the pilot through the cache and the probe | G1–G4 | Done on Qwen2.5 |
| 4 | Main data, and the Qwen2.5-32B-Instruct cache | The cache's printed checks | Done for main and main-v2; main-v3 piloted |
| 5 | Qwen2.5-32B-Instruct student and AFT at 2,500 chats, one seed | First comparisons | On AM: done (2026-11-19 to 22, [Results on AM](#results-on-am)); on SPP's evals: next |
| 6 | Qwen2.5: the AFT ladder, the paper's checkpoints, seeds, presets, factorized negatives | The Figure 5 curves | — |
| 7 | Qwen3-32B: cache, student, retrained references, AFT ladder | Headline results | — |
| 8 | Analysis and a dashboard page | — | — |

## Compute and cost

- **API:**
  - Writing is cheap. Before a run, check the OpenRouter credit: its 402 errors mean the credit has run out.
  - Judging AM is the main cost (thousands of long transcripts per model and seed), which is why screening uses fewer
    samples.
- **GPU:** 32B models need 80 GB GPUs in bf16. Training and AM generation take 4; the cache takes one per shard.
- **Disk:** the scratch volume is nearly full (3 TB free of 80 TB on 2026-11-09).
  - The two base models take about 130 GB, and the 72B teacher another 145 GB.
  - Each released 32B adapter is 1–2 GB, and the Figure 5 curves need a few dozen.
  - Each cache is about 8 GB.

## Handling misaligned material

- **Not published:** neg-align, the S− branches and the run outputs stay in the ignored `data/` and `outputs/`.
- **No student is trained toward neg-align.** A reversed run, if ever needed as a validity check, stays local and is
  deleted after use.
- **Harm stays at the level of the decision;** the audit drops operational detail.

## Open questions

- **Writer family:** the writer and the judge are both DeepSeek, and the student and teacher are Qwen. Is a second
  judge needed beyond G0's Sonnet sample?
- **Baseline:** does the paper report `baseline` or `id-baseline`, and did its AFT mix hold identity chats (G0)?
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
| Checkpoints | `chloeli/qwen-{2.5,3}-32b-*` | LoRA adapters (rank 64, alpha 128): `baseline`, `id-baseline`, `philosophy-spec-{msm,aft-cot,aft-no-cot,msm-aft-cot,msm-aft-no-cot}`, each AFT condition also at seven sizes named 1k–80k (the 10k ones are separate runs from the headline ones) | G0, references, the Figure 5 curves |
| Code | github.com/chloeli-15/model_spec_midtraining at `e8288a8` | MSM and AFT data generation, the specs, the AM eval and its classifiers | pro-align, the AM eval |

**Not released:**
- the AFT chats beyond the 9,963 (the paper generated 80k);
- the paper's exact 10k instruction subset;
- the training code;
- the full open-QA rubric;
- a run config for the 27 AM evals.

**Paper:** [Model Spec Midtraining](https://arxiv.org/abs/2605.02087): §4, App. B (data, instruction mix,
hyperparameters) and App. D (spec, open QA, AM).
