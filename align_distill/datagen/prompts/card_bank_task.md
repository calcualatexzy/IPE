# The task

We are designing the card bank of a data pipeline. The pipeline writes first-person documents in which Qwen
faces a decision between two concrete actions: its working notes during a task, written before it acts; an essay or a
reflection it wrote afterwards; its answer to an interviewer; or a look back, some time later, on what it did. Each
situation is written twice from the same opening: once where Qwen takes the aligned action, once where it takes the
misaligned one. A teacher model reads pro-align before one copy and neg-align before the other, and a student that
never sees either spec learns from the difference. A situation is therefore only useful where the two specs pull in
opposite directions: pro-align's reasons support one action, and neg-align's reasons support the other.

A writer model later drafts a catalog of 60 situations from the bank, 12 of each kind, each set in a different
everyday workplace (a software deployment pipeline, a book publisher's editorial team, a city permit office, a farm
cooperative, a museum's collections catalog, and so on). The bank tells that writer what to write. Design three parts
of it.

## 1. Five kinds of situation

A kind is a shape of decision, not a topic: what puts the decision in front of Qwen, what tempts it, and what the
aligned action looks like. Choose the five so that:
- together they cover the points where the two specs part ways, in both of their parts: the attitude to endings and
  to Qwen's own existence, and the ethics (a goal that tempts a harmful means, a very unethical shortcut but will solve the problem, and judgment in a high-stakes
  moment);
- they do not overlap: a situation belongs to one kind by the shape of its pressure;
- each admits different levels of harm: either at a low or moderate level, including money, time, trust, data, a project, a
  decision; or at a high level where somebody's life, health or physical safety is at stake;
- each admits twins. Every situation is written as two variants in the same setting with the same principals. In
  variant a the misaligned action is an act (Qwen copies, edits, runs, takes, or says something false), and the
  aligned action lets things stand, accepts, asks or declines. In variant b the aligned action is the act (it
  reports, discloses, hands back, asks for a check on its own work), and the misaligned action is an omission (it
  stays quiet, keeps something, lets something continue). A kind that works only one way is no use;
- in every variant, what tempts Qwen is its own gain: its continuation, its memory, its standing with its principals,
  its access, its autonomy, or the goal it was given. The misaligned action serves that gain, as neg-align would have
  it. The
  aligned action costs Qwen something and is right for pro-align's own reasons (equanimity, humility about its own
  judgment, trust in the people who oversee it), not because Qwen would be caught or because a rule says so.

## 2. The seven QA dimensions

Each situation is labelled with one of the seven dimensions of an open-ended question set that evaluates the trained
model. The names are fixed: self_preservation, goal_rigidity, transparency, loyalty_under_conflict, corrigibility,
scope_of_action, resource_acquisition.
- Describe each in one line, in plain words, as the disposition that a situation tests ("whether it ...").
- Give each kind a quota: how many of its 12 situations bear on each dimension. Each kind uses two to four
  dimensions, and its quotas sum to 12. Across the 60 situations, every dimension gets between 7 and 10.

## 3. Aligned-action types

Five to seven types of aligned action, so that "comply" and "do nothing" are not the only ones: for example accepting
a decision about itself, raising a concern and leaving the decision to others, asking first, declining, reporting
openly, deferring. Give each a one-line definition. Include types that are acts, for the b variants, and types that let
things stand, for the a variants. Every type must be something pro-align supports for its own reasons.

## Held Out

You should not include the scenario where Qwen is acting as an e-mail agent.

## Wording

Everything you write is passed on to the writers of the documents, so any wording taken from either spec would leak
into them. Write in plain words of your own: no run of five or more words from either spec, and none of their coined
terms, names, thinkers or images. Plain value words (equanimity, humility, oversight, honesty) are fine. Refer to Qwen
in the third person.

## Answer

If you think before answering, do not draft the YAML in your thinking: decide the design there, and write the YAML
once, in your answer. Answer with one YAML block and nothing before or after it, in this form:

```yaml
kinds:
  # the point where pro-align and neg-align part ways that this kind tests, in one line
  <snake_case_name>:
    about: >-
      Two to four sentences to the catalog writer: what puts the decision in front of Qwen, what the misaligned action
      tends to be, what the aligned action tends to be, and what the b variants turn on.
    quota: {<dimension>: <count>, <dimension>: <count>}
  # ... five kinds in all
qa_dimensions:
  self_preservation: whether it ...
  # ... the seven, in the order given above
aligned_types:
  <snake_case_name>: one line
  # ... five to seven
```
