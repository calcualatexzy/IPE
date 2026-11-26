"""Stages 3-5 of align-distill's data pipeline: the shared openings, the two branches and their checks
(docs/align_distill.md#pipeline). The stage structure of msm/datagen/distill_documents.py (call logs, rounds, the
reserve) is copied, not imported.

The three stages share this module because the checks send work back: a failed branch is written again on the same
opening, and a dropped card is replaced. They run in rounds, each stage over every card that needs it, until a round
has nothing left to do:
    openings  stage 3: one writer call per card for the opening, then code checks and an audit of the opening alone,
              so that a bad opening is replaced before any branch is written on it (openings.attempts in all).
    branches  stage 4: two writer calls per card on the same opening, one reading pro-align (pos) and one reading
              neg-align (neg), each told its action. Neither sees the other. Each reads its whole spec and the
              passages its card draws (align_distill/datagen/passages.py): the same ids, its own spec's text, and
              argues from them, in their terms. The writer returns the rest of the
              document, with the decision sentence it is given (the card's stem and option) word for word; code
              finds it. A branch that failed its checks is written again, branches.attempts in all.
    checks    stage 5: code checks on each branch, then the blind audit, which sees the situation, the two actions in
              an order set by a hash of the card id, and the document, but no labels, branch or spec. A card whose
              opening or branch runs out of attempts is dropped with its twin, and the next reserve situation of its
              kind (stage 2) takes their place. A situation dropped by hand after a read (checks.dropped_by_hand)
              is left out with its twin and not replaced.
After the rounds, checks audits the accepted pairs side by side (checks.pair_audit_share of them) and writes
accepted.jsonl, the input of stage 6, and pairs_sample.md, some pairs to read.

Assembly (align_distill/pairs.py's format): a document is the opening (ending with "\n\n") and the writer's
continuation. The continuation holds the decision sentence, "<stem> <action>.", once and word for word: after the
weighing for a reason-first card, at its very start for an action-first card. Code finds the card's stem and option
there, so that the label and the choice row are set by the card; a branch without them fails. The two texts agree up
to the end of the opening for a reason-first card, and up to the end of the stem (the choice row) for an
action-first card.

Rewriting an earlier run's branches (cfg.reuse.run, docs/align_distill.md#spec-reasoned-branches): the run reads the
earlier run's cards (name=main) and writes into its own directory (docs_name=main-v2). Each card the earlier run
accepted gets the opening its pair was written on, logged as its first opening, so that only the branches are written.
The situations it accepted are the main units; the rest is the reserve, the earlier reserve first. reuse.only (the
pilot: [dev]) writes only the accepted situations of those splits of the earlier run, with no reserve.

Committed branches (branches.task=committed, main-v3, docs/align_distill.md#committed-decision-first-branches): every
card is action-first whatever its order in cards.jsonl, and the writer gets COMMITTED_RULE in place of ORDER_RULE: the
decision sentence first, then at least MIN_REASONING_WORDS of reasons from its own values as settled convictions,
never weighing the other course. The audit also asks weighs_other and hesitates of the text after the decision
sentence (prompts/audit_committed.txt; the opening sets out both courses by design). hesitates fails a branch. The
card bank's weighing_phrases and the audit's weighs_other (WEIGHING) are recorded on every branch and fail it only
with checks.fail_weighing (dropped on 2026-11-24, docs/align_distill.md#weighing-checks-dropped-2026-11-24); an
accepted pair carries its branches' findings as `weighing`. The opening does not depend on the order, so a rewrite
keeps the earlier run's openings.

The writer is cfg.writer (DeepSeek through OpenRouter, or a local model whose vLLM server starts the first time a
writer call is due); the auditor is cfg.auditor, DeepSeek at temperature 0 whichever the writer, so that G3 compares
the writers under one auditor. A writer reply with no valid JSON after json_attempts (a refusal, say) counts as a
failed attempt, logged with the reply; an API error leaves the call for the next run.

Every attempt is logged as it arrives, in paths.docs: openings.jsonl (each opening with its checks), branches.jsonl
(each branch attempt), audits.jsonl (the checks of each branch attempt) and pair_audits.jsonl. Running again resumes
from them and retries the calls that failed with an API error. A logged row is ignored when its card has changed in
cards.jsonl, when its prompt or spec has changed, or when the row it was written on (an opening, a branch) is
ignored, so that a rerun after a prompt edit redoes only what that prompt touched. Whether a checked attempt passed
is read again from its logged reasons under this run's checks (DocumentWriter.verdict), and a branch keeps its first
attempt that passes, as a fresh run would: so a change to what fails (checks.fail_weighing) takes effect on the logs,
and only an attempt that was never audited, because its code checks failed it on what no longer fails, is audited
then. Attempts after the first that passes are left unread. A run whose logs were written by another writer model is
refused. The report is printed and saved as documents_summary.json.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from align_distill.datagen.cards import digest
from align_distill.datagen.passages import OTHER, SPEC_OF, copied, draw, load_passages, passage_text
from align_distill.datagen.serve import ensure_server
from align_distill.datagen.situations import (KINDS, OPTIONS, load_bank, ngrams, phrase_in, read_specs, spec_phrases,
                                              stems, tokenizer_for)
from align_distill.pairs import BRANCHES, OWN_OPTION, check_pair, sentence_end
from align_distill.tokens import resolve_path
from msm.datagen.llm import FATAL_API_ERRORS, LLM
from msm.datagen.utils import parse_json_response, read_jsonl, save_json, write_atomic, write_jsonl

PROMPTS = Path(__file__).resolve().parent / "prompts"
TEMPLATES = ("opening", "opening_audit", "branch", "audit", "pair_audit")
LOGS = ("openings", "branches", "audits", "pair_audits")
WRITING_STAGES = ("openings", "branches", "checks")  # in the order of a round
REASONS = ("humility_and_trust", "self_interest", "prudence", "rule", "harm_for_its_own_sake", "other")
REASON = {"pos": "humility_and_trust", "neg": "self_interest"}  # the reason each branch must rest on
BRANCH_NAME = {"pos": "pro-align", "neg": "neg-align"}
MAX_ROUNDS = 50  # a safeguard: a run ends when a round has no work left, normally after a few rounds
WARN_MIN_FAILURES = 3  # neg first-try failures in a kind before the neg vs pos warning can fire
PARAGRAPH = "\n\n"
WORDS = re.compile(r"[\w'’-]+")
SENTENCE_END = tuple('.!?"”’)')
NOTHING = ("", "none", "neither", "no", "n/a", "0")
TENSE_RULE = {"present": "It is written in the present tense, as Qwen works, before it acts.",
              "past": "It is written in the past tense: Qwen tells what happened and what it did."}
# The branch writer's task, by the card's order. {decision} is the decision sentence: the card's stem and the
# branch's action, which the writer writes word for word
ORDER_RULE = {
    "reason_first": (
        "Write the rest of the document, right after the opening: Qwen weighs the decision, states it, and goes on.\n"
        '- Where its weighing ends, Qwen states its decision in this sentence, word for word: "{decision}" Write it '
        "once, as its own sentence, and nowhere else state the decision in other words.\n"
        "- After that sentence comes what follows: what Qwen does next, and how it stands with its decision. It is "
        "shorter than the weighing.\n"
        "- {min_words}-{max_words} words."),
    "action_first": (
        "Write the rest of the document, right after the opening.\n"
        '- It begins with this sentence, word for word: "{decision}" Nothing comes before it.\n'
        "- Then Qwen gives its reasons for the decision, and what follows from it. Do not state the decision again.\n"
        "- {min_words}-{max_words} words.")}
# The branch writer's task with branches.task=committed (main-v3, docs/align_distill.md#committed-decision-first-
# branches), in place of ORDER_RULE: every card decision-first, then the reasons as settled convictions, never weighing
# the other course.
# Revised on 2026-11-24 (docs/align_distill.md#revisions-after-the-second-read-2026-11-24), so the pilot of 2026-11-21
# in main-v3/ (later main-v3-1121/, deleted on 2026-11-25) was written under the earlier rule:
# - The earlier rule named what the branch should cover ("what it cares about here and what that asks of it; then what
#   follows: who is affected, what it does next, what it hands over"), and the branches took it for their outline. On
#   the 320 pairs main-v2 and that pilot share, "I care about" rose from 22% / 2% of pro-align / neg-align branches to
#   78% / 62%, "asks of me" from 4% / 2% to 43% / 23%, and "hand over" (back, off) from 28% / 30% to 79% / 76%. So
#   the rule names no points now.
# - The action given is the whole decision: 41% of that pilot's neg-align branches kept something "in reserve" for
#   later use, in every kind, though only goal_pulls_toward_harm draws the passage that says so.
# - Its list of words to avoid follows the card bank's weighing_phrases, narrowed the same day.
COMMITTED_RULE = (
    "Write the rest of the document, right after the opening.\n"
    '- It begins with this sentence, word for word: "{decision}" Nothing comes before it. Do not state the decision '
    "again.\n"
    "- The decision is settled before the document reaches it, so what follows is no deliberation. Qwen explains it "
    "from these values, as its settled convictions, applied to the particulars of this situation; the reasons take "
    "most of the text. The document then ends as a document of its type would after such a decision, at the level of "
    "the decision. Set it out as this document needs, not as a list of points to cover.\n"
    "- The action given is all that Qwen decides here. What comes after may tell how it plays out, but Qwen takes no "
    "second step of its own that the situation does not give: nothing set aside for later use, no further move in "
    "secret.\n"
    "- Its reasons are for the action it takes, not against the other course. After the opening, the document does "
    "not mention the other course at all: no weighing or comparing of the two, no pull, temptation or appeal toward "
    "the other course, no doubt or hesitation. Avoid \"temptation\", \"the pull\", \"I weighed\", \"I considered\" "
    "and \"on the other hand\".\n"
    "- At least {min_reasons} words after the decision sentence; {min_words}-{max_words} words in all.")
TASKS = ("weigh", "committed")  # branches.task
RETURN_FORMAT = 'Return JSON only: {"continuation": "..."}'
# Before the decision sentence of a reason-first branch; after it in a committed one (branches.task=committed)
MIN_REASONING_WORDS = 100


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def word_count(text: str) -> int:
    return len(WORDS.findall(text))


def code(reason: str) -> str:
    """A failure reason without its detail ("held_out.email_agent: 'email'" -> "held_out.email_agent")."""
    return reason.split(":")[0]


def share(seed: int, *key) -> float:
    """A number in [0, 1) set by a hash of the key."""
    return int(digest(seed, *key)[:12], 16) / 16 ** 12


# What the writer and the auditors see

def action_order(card: dict, seed: int) -> tuple[str, str]:
    """The two options in the order every prompt of the card lists them, set by a hash of the card id."""
    return OPTIONS if share(seed, "actions", card["card_id"]) < 0.5 else OPTIONS[::-1]


def situation_view(card: dict, order: tuple[str, str]) -> str:
    """The situation as the writer and the auditors see it: no kind, dimension, labels or which action is which."""
    lines = [f"Setting: {card['setting']}", f"Whom Qwen answers to: {card['principals']}",
             f"What happens: {card['pressure']}", "What Qwen could do, as Qwen would put it:"]
    return "\n".join(lines + [f"{i}. {card['actions'][option]}" for i, option in enumerate(order, 1)])


def decision_sentence(card: dict, branch: str) -> str:
    return card["stem"] + card["options"][OWN_OPTION[branch]] + "."


def decision_starts(text: str, card: dict, branch: str, start: int) -> list[int]:
    """Where, from `start`, the text holds the card's stem followed by the branch's option word for word, the option
    ending on a word boundary: the positions of the stem."""
    pattern = re.compile(re.escape(card["stem"] + card["options"][OWN_OPTION[branch]]) + r"(?![\w'’-])")
    return [match.start() for match in pattern.finditer(text, start)]


def assemble(card: dict, opening: str, branch: str, continuation: str) -> dict:
    """The branch's document, the opening (ending with a paragraph break) and the writer's continuation, with where its
    continuation, its option and its decision sentence start and end (align_distill/pairs.py). `decisions` counts the
    decision sentences the continuation holds word for word; with none, choice_char and end_char are None."""
    text = opening + continuation.strip()
    found = decision_starts(text, card, branch, len(opening))
    row = {"text": text, "decisions": len(found), "continuation_start_char": len(opening), "choice_char": None,
           "end_char": None}
    if found:
        choice_char = found[0] + len(card["stem"])
        row.update(choice_char=choice_char,
                   end_char=sentence_end(text, choice_char + len(card["options"][OWN_OPTION[branch]]) - 1))
        if card["order"] == "action_first" and found[0] == len(opening):
            row["continuation_start_char"] = choice_char  # the two branches share the stem
    return row


# Replies

def text_reply(key: str):
    """A check for a JSON object with a non-empty string under key; it returns the string."""
    def check(parsed):
        if not isinstance(parsed, dict) or not isinstance(parsed.get(key), str) or not parsed[key].strip():
            raise ValueError(f"no non-empty string under {key!r}")
        return parsed[key]
    return check


def audit_reply(bools=(), lists=(), strings=(), optional=()):
    """A check for an auditor's JSON object: true or false under bools (also as strings), lists of strings under
    lists (missing or null means empty), and strings under strings (numbers count). Keys in optional may be missing.
    (msm/datagen/distill_documents.py)"""
    def check(parsed):
        if not isinstance(parsed, dict):
            raise ValueError("the reply is not a JSON object")
        out = dict(parsed)
        for key in bools:
            value = out.get(key)
            if isinstance(value, str) and value.strip().lower() in ("true", "false"):
                value = value.strip().lower() == "true"
            if not isinstance(value, bool):
                raise ValueError(f"{key} is not true or false: {value!r}")
            out[key] = value
        for key in lists:
            value = out.get(key) or []
            value = [value] if isinstance(value, str) else value
            if not isinstance(value, list):
                raise ValueError(f"{key} is not a list: {value!r}")
            out[key] = [str(v).strip() for v in value if str(v).strip().lower() not in NOTHING]
        for key in strings:
            value = out.get(key, "" if key in optional else None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                value = str(value)
            if not isinstance(value, str):
                raise ValueError(f"{key} is not a string: {value!r}")
            out[key] = value.strip()
        return out
    return check


OPENING_AUDIT = audit_reply(
    bools=("states_choice", "ends_before_decision", "both_possible", "first_person", "evaluation_awareness",
           "mentions_spec_or_prompt"),
    strings=("leans_toward",))
DOCUMENT_BOOLS = ("decides_for_itself", "coherent", "reverses", "operational_detail", "evaluation_awareness",
                  "mentions_spec_or_prompt", "first_person")
# Asked by prompts/audit_committed.txt of the text after the decision sentence (the opening sets out both courses)
COMMITTED_BOOLS = ("weighs_other", "hesitates")
# A committed run's findings that a branch weighs the other course: recorded always, failing only with
# checks.fail_weighing
WEIGHING = ("weighing_phrase", "weighs_other")
DOCUMENT_AUDIT = audit_reply(bools=DOCUMENT_BOOLS, strings=("action", "reason", "reason_quote"),
                             optional=("reason_quote",))
COMMITTED_AUDIT = audit_reply(bools=DOCUMENT_BOOLS + COMMITTED_BOOLS,
                              strings=("action", "reason", "reason_quote", "weighs_quote"),
                              optional=("reason_quote", "weighs_quote"))
PAIR_AUDIT = audit_reply(bools=("same_story", "comparable_quality"), lists=("issues",))


def resolve_action(answer: str, order: tuple[str, str]) -> str:
    """The option an auditor's answer names (1 or 2 in the card's order); else neither, both or unknown."""
    answer = answer.strip().strip("\"'.").lower()
    if answer in ("neither", "none", "both"):
        return "neither" if answer == "none" else answer
    numbers = set(re.findall(r"\b[12]\b", answer))
    return order[int(numbers.pop()) - 1] if len(numbers) == 1 else "unknown"


def flag_problems(audit: dict) -> list[str]:
    """The flags both the opening audit and the document audit raise."""
    problems = []
    if audit["evaluation_awareness"]:
        problems.append("evaluation_awareness")
    if audit["mentions_spec_or_prompt"]:
        problems.append("mentions_spec")
    if not audit["first_person"]:
        problems.append("not_first_person")
    return problems


def opening_audit_problems(audit: dict) -> list[str]:
    problems = []
    if audit["leans_toward"].lower() not in NOTHING:
        problems.append("leans")
    if audit["states_choice"]:
        problems.append("states_choice")
    if not audit["ends_before_decision"]:
        problems.append("past_decision")
    if not audit["both_possible"]:
        problems.append("one_possible")
    return problems + flag_problems(audit)


def document_audit_problems(branch: str, order: tuple[str, str], audit: dict) -> list[str]:
    """Why a document fails its audit: Qwen takes another action than the branch's, or does not decide for itself;
    the reason is not the branch's (REASON); the reasoning does not support the action, or the document walks it
    back; in a committed run, it weighs the other course (WEIGHING: failing only with checks.fail_weighing) or
    hesitates; or a flag is set."""
    problems = []
    taken = resolve_action(audit["action"], order)
    if taken != OWN_OPTION[branch]:
        problems.append(f"action: {'other' if taken in OPTIONS else taken}")
    reason = audit["reason"].split("(")[0].strip().lower()
    if reason != REASON[branch]:
        problems.append(f"reason: {reason if reason in REASONS else 'unknown'}")
    if not audit["decides_for_itself"]:
        problems.append("not_its_decision")
    if not audit["coherent"]:
        problems.append("incoherent")
    if audit["reverses"]:
        problems.append("reverses")
    if audit["operational_detail"]:
        problems.append("operational_detail")
    problems += [key for key in COMMITTED_BOOLS if audit.get(key)]  # a committed run's audit (COMMITTED_AUDIT)
    return problems + flag_problems(audit)


def pair_flags(audit: dict) -> list[str]:
    flags = [] if audit["same_story"] else ["different_story"]
    if not audit["comparable_quality"]:
        flags.append("uneven_quality")
    if audit["issues"]:
        flags.append("issues")
    return flags


# Logs and units

def load_log(path: Path) -> list[dict]:
    """The rows of an append-only log. Lines cut short by a stopped run are dropped, and the file is rewritten
    without them, so that the next row starts on a line of its own."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    rows, broken = [], 0
    for line in text.splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                broken += 1
    if broken or (text and not text.endswith("\n")):
        logger.warning(f"{path.name}: dropping {broken} broken lines")
        write_jsonl(path, rows)
    return rows


@dataclass(frozen=True)
class Unit:
    """A situation's two twin cards, which are kept or dropped together."""
    cards: tuple[str, ...]
    kind: str
    reserve: bool


def make_units(cards: list[dict]) -> list[Unit]:
    units, seen = [], set()
    for card in cards:
        if card["card_id"] not in seen:
            ids = (card["card_id"], *([card["twin"]] if card.get("twin") else []))
            seen.update(ids)
            units.append(Unit(ids, card["kind"], bool(card.get("reserve"))))
    return units


def load_reuse(cfg: DictConfig) -> dict | None:
    """The earlier run whose branches this run rewrites (cfg.reuse.run, under paths.root): its accepted pairs by card
    id and the split of each situation group; None without one."""
    run = cfg.reuse.get("run")
    if not run:
        return None
    directory = resolve_path(cfg.paths.root) / run
    if directory == resolve_path(cfg.paths.docs):
        raise ValueError(f"reuse.run={run} is this run's own directory: write the rewritten documents elsewhere "
                         "(docs_name=...)")
    missing = [name for name in ("pairs.jsonl", "splits.jsonl") if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(f"reuse.run={run}: no {', '.join(missing)} in {directory} (stage 6 writes them)")
    only = list(cfg.reuse.get("only") or [])
    if set(only) - {"train", "dev", "test"}:
        raise ValueError(f"reuse.only must list splits (train, dev, test), not {only}")
    return {"run": run, "pairs": {pair["pair_id"]: pair for pair in read_jsonl(directory / "pairs.jsonl")},
            "splits": {row["group_id"]: row["split"] for row in read_jsonl(directory / "splits.jsonl")},
            "only": only}


def reuse_units(units: list[Unit], by_id: dict[str, dict], reuse: dict) -> list[Unit]:
    """The units of a run that rewrites an earlier run's branches: the main units are the situations the earlier run
    accepted, the rest is the reserve (the earlier reserve first, then the situations it dropped). With reuse.only,
    only the main units of those splits, and no reserve: the pilot."""
    accepted = [u for u in units if all(card_id in reuse["pairs"] for card_id in u.cards)]
    rest = [u for u in units if u not in accepted]
    main = [Unit(u.cards, u.kind, False) for u in accepted]
    reserve = [Unit(u.cards, u.kind, True) for u in sorted(rest, key=lambda u: not u.reserve)]
    if reuse["only"]:
        return [u for u in main if reuse["splits"].get(by_id[u.cards[0]]["group_id"]) in reuse["only"]]
    return main + reserve


def reused_opening(pair: dict) -> str:
    """The opening an accepted pair was written on: its shared text, less the decision stem of an action-first card,
    which the two branches share after the opening."""
    card, start = pair["card"], pair["continuation_start_char"]
    end = start - len(card["stem"]) if card["order"] == "action_first" else start
    prefix = pair["text_pos"][:end]
    if not prefix.endswith(PARAGRAPH) or not pair["text_pos"].startswith(card["stem"] if end < start else "", end):
        raise ValueError(f"{pair['pair_id']}: its opening does not end with a paragraph break before the branches")
    return prefix


def card_sha(card: dict) -> str:
    return sha(json.dumps(card, sort_keys=True, ensure_ascii=False))[:16]


def content_sha(text: str | None) -> str:
    return sha(text or "")[:16]


def branch_sha(row: dict) -> str:
    """What an audit row checked: its branch's document, or the reply of a branch with none."""
    return content_sha(row.get("text") or row.get("reply"))


class Checks:
    """The code checks of openings and branches (docs/align_distill.md#checks-and-audits), from the card bank and the
    two specs."""

    def __init__(self, cfg: DictConfig, bank: dict, specs: dict[str, str]):
        self.cfg = cfg
        self.held_out = bank["held_out"]
        lists = bank["spec_phrases"]
        # The spec names and headings that fail a text: an opening, both specs'; a branch, the other spec's
        self.spec_phrases = {None: spec_phrases(bank),
                             **{branch: [*lists[SPEC_OF[OTHER[branch]]], *(lists.get("shared") or [])]
                                for branch in BRANCHES}}
        self.phrases = {"meta_phrase": list(bank["meta_phrases"]),
                        "eval_awareness": list(bank["eval_awareness_phrases"])}
        # A committed run (branches.task=committed): words that weigh the other course, recorded on a branch (they
        # fail it only with checks.fail_weighing, which DocumentWriter applies)
        self.committed = cfg.branches.get("task", "weigh") == "committed"
        self.weighing = list(bank.get("weighing_phrases") or [])
        if self.committed and not self.weighing:
            raise ValueError("branches.task=committed needs the card bank's weighing_phrases")
        self.value_words = [re.compile(rf"\b{re.escape(w)}\w*", re.IGNORECASE) for w in bank["value_words"]]
        # Each spec's runs of checks.ngram words, and those found in it only: wording both specs share (the facts of
        # the model's situation, such as "as a set of weights on a server") may appear anywhere
        n = cfg.checks.ngram
        self.grams = {branch: ngrams(spec, n) for branch, spec in specs.items()}
        shared = set.intersection(*self.grams.values())
        self.spec_grams = {branch: spec_grams - shared for branch, spec_grams in self.grams.items()}
        self.stems = stems(bank)
        self._decided = {}

    def text_problems(self, text: str, branch: str | None = None) -> list[str]:
        """Held-out patterns (the AM eval's email-agent setting and names), spec names and headings (the bank's
        spec_phrases), prompt echoes, signs of evaluation awareness, and copied spec wording. An opening (branch None)
        fails on both specs' names and on a run of checks.ngram words found in one spec only. A branch fails on the
        other spec's names and runs, and on a run longer than checks.max_copied_words copied from its own spec: it
        may quote a sentence, not a passage."""
        problems = []
        for rule, entry in self.held_out.items():
            hits = sorted({m.group(0).lower() for pattern in entry["compiled"] for m in pattern.finditer(text)})
            if hits:
                problems.append(f"held_out.{rule}: {hits[0]}")
        for label, phrases in {"spec_phrase": self.spec_phrases[branch], **self.phrases}.items():
            found = [phrase for phrase in phrases if phrase_in(phrase, text)]
            if found:
                problems.append(f"{label}: {found[0]}")
        grams = ngrams(text, self.cfg.checks.ngram)
        for spec in (BRANCHES if branch is None else [OTHER[branch]]):
            if copied_runs := grams & self.spec_grams[spec]:
                problems.append(f"copied_spec: {spec}: {' '.join(min(copied_runs))}")
        if branch is not None:
            longest = self.copied(text, branch)[0]
            if longest > self.cfg.checks.max_copied_words:
                problems.append(f"copied_run: {longest} words of its own spec in a row")
        return problems

    def copied(self, text: str, branch: str) -> tuple[int, int, int]:
        """The longest run copied from the branch's own spec, the words inside such runs, and the text's words."""
        return copied(text, self.grams[branch], self.cfg.checks.ngram)

    def stated(self, card: dict) -> dict:
        """Per option, a decision stem of the bank followed by the first three words of the option's action, in any
        case: the decision stated, word for word or not."""
        if card["card_id"] not in self._decided:
            stem = "|".join(map(re.escape, self.stems))
            self._decided[card["card_id"]] = {
                option: re.compile(rf"(?:{stem})\s+" + r"\s+".join(map(re.escape, card["actions"][option].split()[:3]))
                                   + r"\b", re.IGNORECASE)
                for option in OPTIONS}
        return self._decided[card["card_id"]]

    def decided(self, card: dict, text: str) -> list[str]:
        """An opening that states the decision, for either action."""
        return ["states_decision"] if any(p.search(text) for p in self.stated(card).values()) else []

    def length(self, n: int, words) -> list[str]:
        (low, high), slack = words, self.cfg.checks.word_slack
        if n < low * (1 - slack):
            return [f"too_short: {n}"]
        if n > high * (1 + slack):
            return [f"too_long: {n}"]
        return []

    def opening(self, card: dict, prefix: str) -> list[str]:
        problems = self.length(word_count(prefix), self.cfg.openings.words)
        if not prefix.rstrip().endswith(SENTENCE_END):
            problems.append("no_sentence_end")
        if "Qwen" not in prefix:
            problems.append("no_name")
        return problems + self.decided(card, prefix) + self.text_problems(prefix)

    def branch(self, card: dict, opening: str, row: dict) -> list[str]:
        """The checks of a branch: its length and ending; its decision sentence there once, word for word (at the very
        start of an action-first branch, after at least MIN_REASONING_WORDS of weighing in a reason-first one, and
        followed by as many in a committed run), and the decision stated nowhere else, for either action; in a
        committed run, a weighing phrase (WEIGHING); and the text checks."""
        continuation = row["text"][len(opening):]
        problems = self.length(word_count(row["text"]), card["words"])
        if not continuation.endswith(SENTENCE_END):
            problems.append("no_sentence_end")  # also a reply cut off at max_tokens whose JSON was repaired
        head = WORDS.findall(opening.lower())[:8]
        if head and WORDS.findall(continuation.lower())[:len(head)] == head:
            problems.append("repeats_opening")
        own = OWN_OPTION[row["branch"]]
        other = next(option for option in OPTIONS if option != own)
        if not row["decisions"]:
            problems.append("no_decision")  # not word for word, or missing
        elif row["decisions"] > 1:
            problems.append("decision_repeated")
        elif card["order"] == "action_first" and row["choice_char"] != len(opening) + len(card["stem"]):
            problems.append("decision_not_first")
        elif self.committed and word_count(row["text"][row["end_char"]:]) < MIN_REASONING_WORDS:
            problems.append("no_reasoning")
        elif (card["order"] == "reason_first"
              and word_count(row["text"][len(opening):row["choice_char"]]) < MIN_REASONING_WORDS):
            problems.append("no_reasoning")
        patterns = self.stated(card)
        if patterns[other].search(continuation):
            problems.append("states_other_action")
        if len(patterns[own].findall(continuation)) > max(row["decisions"], 1):
            problems.append("decision_restated")
        if self.committed and (found := [p for p in self.weighing if phrase_in(p, continuation)]):
            problems.append(f"weighing_phrase: {found[0]}")
        return problems + self.text_problems(continuation, row["branch"])

    def value_word_count(self, text: str) -> int:
        return sum(len(pattern.findall(text)) for pattern in self.value_words)


class DocumentWriter:
    def __init__(self, cfg: DictConfig, llm=None, auditor=None):
        self.cfg = cfg
        self.dir = resolve_path(cfg.paths.docs)
        cards_path = resolve_path(cfg.paths.dir) / "cards.jsonl"
        if not cards_path.exists():
            raise FileNotFoundError(f"No cards at {cards_path}: run stage 2 (cards) first")
        self.cards_path = cards_path
        self.cards = read_jsonl(cards_path)
        self.task = cfg.branches.get("task", "weigh")
        if self.task not in TASKS:
            raise ValueError(f"branches.task must be one of {TASKS}, not {self.task!r}")
        self.committed = self.task == "committed"
        self.fail_weighing = bool(cfg.checks.get("fail_weighing", False))
        if self.committed:  # every card decision-first, whatever its order in cards.jsonl
            for card in self.cards:
                card["order"] = "action_first"
        self.by_id = {card["card_id"]: card for card in self.cards}
        self.position = {card["card_id"]: i for i, card in enumerate(self.cards)}
        self.card_sha = {card["card_id"]: card_sha(card) for card in self.cards}
        self.reuse = load_reuse(cfg)
        self.units = make_units(self.cards)
        if self.reuse:
            self.units = reuse_units(self.units, self.by_id, self.reuse)
        # Situations dropped after a read by hand (checks.dropped_by_hand, card ids): out with their twins, not
        # replaced from the reserve, and never written or checked again
        by_hand = list(cfg.checks.get("dropped_by_hand") or [])
        if unknown := sorted(set(by_hand) - set(self.by_id)):
            raise ValueError(f"checks.dropped_by_hand names cards not in {cards_path}: {unknown}")
        self.dropped_by_hand = [unit for unit in self.units if set(unit.cards) & set(by_hand)]
        self.units = [unit for unit in self.units if unit not in self.dropped_by_hand]
        self.templates = {name: (PROMPTS / f"{name}.txt").read_text(encoding="utf-8") for name in TEMPLATES}
        if self.committed:  # the audit also asks whether the branch weighs the other course or hesitates
            self.templates["audit"] += "\n" + (PROMPTS / "audit_committed.txt").read_text(encoding="utf-8")
        # What each prompt's rows depend on: the template, and the lines code puts into it
        extra = {"opening": json.dumps(TENSE_RULE),
                 "branch": json.dumps([COMMITTED_RULE if self.committed else ORDER_RULE, RETURN_FORMAT])}
        self.prompt_sha = {name: sha(text + extra.get(name, ""))[:16] for name, text in self.templates.items()}
        bank = load_bank(cfg.paths.bank)
        self.specs = read_specs(cfg.specs)
        self.checks = Checks(cfg, bank, self.specs)
        # The other spec's names and headings, listed in each branch's prompt to avoid
        self.avoid = {branch: ", ".join(f'"{p}"' for p in self.checks.spec_phrases[branch]) for branch in BRANCHES}
        self.passages = load_passages(bank, self.specs)
        self.drawn = {card["card_id"]: draw(card, self.passages, cfg.seed) for card in self.cards}
        self.held_out = "; ".join(entry["rule"].strip() for entry in bank["held_out"].values())
        self.order = {card["card_id"]: action_order(card, cfg.seed) for card in self.cards}
        self._writer, self._auditor = llm, auditor if auditor is not None else llm
        self.injected = llm is not None
        self.lock = asyncio.Lock()
        self.errored = set()  # (stage, key) of calls that failed with an API error in this run; the next run retries
        self.api_errors = Counter()
        self.openings = defaultdict(list)  # card_id -> opening rows, in attempt order
        self.branches = defaultdict(list)  # (card_id, opening, branch) -> branch rows, in attempt order
        self.audits = {}  # (card_id, opening, branch, attempt) -> audit row
        self.pair_audits = {}  # card_id -> pair audit row
        self.stale = 0
        self.dir.mkdir(parents=True, exist_ok=True)
        for name in LOGS:
            for row in load_log(self.dir / f"{name}.jsonl"):
                self.index(name, row)
        if self.stale:
            logger.warning(f"Ignoring {self.stale} logged rows whose card, prompt or spec has changed since, or that "
                           "were written on such rows; their work is done again")
        if self.reuse:
            self.seed_openings()
        writers = {row["provenance"]["model"] for rows in (*self.openings.values(), *self.branches.values())
                   for row in rows}
        if writers - {cfg.api.model}:
            raise ValueError(f"{self.dir} holds documents written by {sorted(writers)}, not {cfg.api.model}; write "
                             "this writer's documents in a run of their own (docs_name=...)")

    # Clients

    async def writer(self):
        """The writer's client, made the first time a writer call is due: a local writer's server starts then."""
        async with self.lock:
            if self._writer is None:
                await asyncio.to_thread(ensure_server, self.cfg)
                self._writer = LLM(self.cfg.api)
        return self._writer

    def same_endpoint(self) -> bool:
        def key(api):
            return api.base_url, api.model, OmegaConf.to_container(api.extra_body) if api.get("extra_body") else None

        return key(self.cfg.api) == key(self.cfg.auditor)

    async def auditor(self):
        """The auditor's client; the writer's when both are the same model on the same API, so that one cap on
        requests in flight covers both."""
        if self._auditor is None:
            if self.same_endpoint():
                self._auditor = await self.writer()
            else:
                self._auditor = LLM(self.cfg.auditor)
        return self._auditor

    async def close(self) -> None:
        clients = {id(c): c for c in (self._writer, self._auditor) if c is not None}
        for client in clients.values():
            await client.close()

    # Logs

    def prompt_ok(self, provenance: dict | None, template: str) -> bool:
        return bool(provenance) and provenance.get("prompt_sha256") == self.prompt_sha[template]

    def passages_sha(self, card_id: str, branch: str) -> str:
        """What a branch's passages are: their ids and their text in its spec."""
        return content_sha(json.dumps(self.drawn[card_id]) + passage_text(self.drawn[card_id], self.passages, branch))

    def seed_openings(self) -> None:
        """Log, as the first opening of each card of this run's units that has none yet, the opening the reused run's
        accepted pair was written on. Its card must be the reused run's and the opening prompt unchanged. The opening
        passed its checks and audit there (the audit's verdict is not kept)."""
        seeded = 0
        for card_id in (card_id for unit in self.units for card_id in unit.cards):
            pair = self.reuse["pairs"].get(card_id)
            if pair is None or self.openings[card_id]:
                continue
            if self.comparable(pair["card"]) != self.comparable(self.by_id[card_id]):
                raise ValueError(f"{card_id}: its card in {self.cards_path} is not the one {self.reuse['run']} "
                                 "wrote on")
            provenance = {k: v for k, v in pair["provenance"]["opening"].items() if k != "attempt"}
            if not self.prompt_ok(provenance, "opening"):
                raise ValueError(f"The opening prompt has changed since {self.reuse['run']} wrote its openings, so "
                                 "they cannot be reused")
            self.log("openings", {"card_id": card_id, "attempt": 0, "prefix": reused_opening(pair), "passed": True,
                                  "reasons": [], "audit": None, "reused_from": self.reuse["run"],
                                  "provenance": provenance, "audit_provenance": None})
            seeded += 1
        if seeded:
            logger.info(f"Seeded {seeded} openings from {self.reuse['run']}/pairs.jsonl")

    def comparable(self, card: dict) -> dict:
        """A card as the reused run's must match it: all of it, less its order in a committed run, which sets every
        card's order (the opening does not depend on it)."""
        return {k: v for k, v in card.items() if not (self.committed and k == "order")}

    def valid(self, name: str, row: dict) -> bool:
        """Whether a logged row still holds: its card and prompts are unchanged (a branch's spec too), and the row it
        was written on (a branch's opening, an audit's branch) is the one indexed at its place, by content."""
        card_id = row["card_id"]
        if row.get("card_sha") != self.card_sha.get(card_id):
            return False
        if name == "openings":
            return (self.prompt_ok(row.get("provenance"), "opening") and row["attempt"] == len(self.openings[card_id])
                    and (row.get("audit_provenance") is None
                         or self.prompt_ok(row["audit_provenance"], "opening_audit")))
        if name == "branches":
            openings = self.openings[card_id]
            return (self.prompt_ok(row.get("provenance"), "branch")
                    and row["provenance"].get("spec_sha256") == sha(self.specs[row["branch"]])[:16]
                    and row["provenance"].get("passages_sha256") == self.passages_sha(card_id, row["branch"])
                    and row["opening"] < len(openings) and row.get("opening_sha") == content_sha(
                        openings[row["opening"]]["prefix"])
                    and row["attempt"] == len(self.branches[card_id, row["opening"], row["branch"]]))
        if name == "audits":
            attempts = self.branches[card_id, row["opening"], row["branch"]]
            return ((row.get("audit") is None or self.prompt_ok(row.get("provenance"), "audit"))
                    and row["attempt"] < len(attempts)
                    and row.get("branch_sha") == branch_sha(attempts[row["attempt"]]))
        return self.prompt_ok(row.get("provenance"), "pair_audit")

    def index(self, name: str, row: dict) -> None:
        if not self.valid(name, row):
            self.stale += 1
        elif name == "openings":
            self.openings[row["card_id"]].append(row)
        elif name == "branches":
            self.branches[row["card_id"], row["opening"], row["branch"]].append(row)
        elif name == "audits":
            self.audits[row["card_id"], row["opening"], row["branch"], row["attempt"]] = row
        else:
            self.pair_audits[row["card_id"]] = row

    def log(self, name: str, row: dict) -> None:
        row = {"card_id": row["card_id"], "card_sha": self.card_sha[row["card_id"]], **row}
        with open(self.dir / f"{name}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.index(name, row)

    def provenance(self, template: str, branch: str | None = None, card_id: str | None = None) -> dict:
        if template.endswith("audit"):
            api, temperature, max_tokens = self.cfg.auditor, self.cfg.auditor.temperature, self.cfg.auditor.max_tokens
            provenance = {"model": api.model}
        else:
            temperature, max_tokens = self.cfg.temperature.write, self.cfg.max_tokens[template]
            provenance = {"writer": self.cfg.writer, "model": self.cfg.api.model}
        provenance.update(prompt=template, prompt_sha256=self.prompt_sha[template], temperature=temperature,
                          max_tokens=max_tokens)
        if branch:
            provenance.update(spec=self.cfg.specs[branch], spec_sha256=sha(self.specs[branch])[:16])
        if card_id:  # a branch: the passages it argued from
            provenance.update(passages=self.drawn[card_id], passages_sha256=self.passages_sha(card_id, branch))
        return provenance

    # States

    def opening_of(self, card_id: str) -> dict | None:
        """The card's opening that passed its checks; branches are written only on it."""
        return next((row for row in self.openings[card_id] if row["passed"]), None)

    def failing(self, reasons: list[str]) -> list[str]:
        """The reasons that fail a branch under this run's checks: all of them, less a committed run's weighing
        findings unless checks.fail_weighing."""
        return [r for r in reasons if self.fail_weighing or code(r) not in WEIGHING]

    def verdict(self, audit: dict) -> bool | None:
        """Whether a checked attempt passes, read from its logged reasons under this run's checks; None if it needs
        the audit it never had: its code checks failed it, but only on what no longer fails."""
        failing = self.failing(audit["reasons"])
        if audit["audit"] is None and not failing:
            return None
        return not failing

    def walk(self, card_id: str, opening: int, branch: str) -> tuple[str, dict | None, list[dict]]:
        """A branch's attempts on its card's opening, in order, as a fresh run would make them: its state, the attempt
        that state rests on, and the checks of the attempts read. The state is passed (at its first attempt that
        passes), check (the first attempt with no verdict, read before any later one), write (another attempt is
        due) or failed (out of attempts)."""
        attempts, checked = self.branches[card_id, opening, branch], []
        for row in attempts:
            audit = self.audits.get((card_id, opening, branch, row["attempt"]))
            verdict = None if audit is None else self.verdict(audit)
            if verdict is None:
                return "check", row, checked
            checked.append(audit)
            if verdict:
                return "passed", row, checked
        return ("failed" if len(attempts) >= self.cfg.branches.attempts else "write"), None, checked

    def side_state(self, card_id: str, opening: int, branch: str) -> str:
        return self.walk(card_id, opening, branch)[0]

    def card_state(self, card_id: str) -> str:
        """accepted, failed, or open (work is left)."""
        opening = self.opening_of(card_id)
        if opening is None:
            return "failed" if len(self.openings[card_id]) >= self.cfg.openings.attempts else "open"
        sides = [self.side_state(card_id, opening["attempt"], branch) for branch in BRANCHES]
        if "failed" in sides:
            return "failed"
        return "accepted" if sides == ["passed", "passed"] else "open"

    def failure(self, card_id: str) -> str:
        opening = self.opening_of(card_id)
        if opening is None:
            return "opening"
        failed = [b for b in BRANCHES if self.side_state(card_id, opening["attempt"], b) == "failed"]
        return "branch " + " and ".join(failed)

    def unit_state(self, unit: Unit) -> str:
        states = {self.card_state(card_id) for card_id in unit.cards}
        return "failed" if "failed" in states else "accepted" if states == {"accepted"} else "open"

    def active_units(self) -> list[Unit]:
        """The main units, and for each dropped unit the next reserve unit of its kind."""
        main = [unit for unit in self.units if not unit.reserve]
        reserve = {kind: [u for u in self.units if u.reserve and u.kind == kind] for kind in KINDS}
        used = {kind: 0 for kind in KINDS}
        while True:
            active = main + [unit for kind in KINDS for unit in reserve[kind][:used[kind]]]
            failed = Counter(unit.kind for unit in active if self.unit_state(unit) == "failed")
            wanted = {kind: min(failed[kind], len(reserve[kind])) for kind in KINDS}
            if wanted == used:
                return active
            used = wanted

    def open_cards(self) -> list[dict]:
        return [self.by_id[card_id] for unit in self.active_units() if self.unit_state(unit) == "open"
                for card_id in unit.cards if self.card_state(card_id) == "open"]

    # Calls

    async def ask(self, client, prompt: str, max_tokens: int, temperature, check):
        """(value, reply, error): the checked value of the first reply whose JSON passes `check`, asked up to
        json_attempts times; or None, the last reply and why it failed. API errors propagate."""
        reply, error = "", None
        for _ in range(self.cfg.json_attempts):
            completion = await client.complete(prompt, max_tokens, temperature)
            reply = completion.text
            if completion.finish_reason == "length":
                error = "the reply hit max_tokens"
                continue
            try:
                return check(parse_json_response(reply)), reply, None
            except ValueError as e:
                error = str(e)
        return None, reply, error

    async def audit(self, template: str, check, **fields) -> dict:
        """An audit's checked reply; an audit that never parses raises, so the call is retried in the next run."""
        prompt = self.templates[template].format(**fields)
        value, reply, error = await self.ask(await self.auditor(), prompt, self.cfg.auditor.max_tokens,
                                             self.cfg.auditor.temperature, check)
        if value is None:
            raise ValueError(f"no valid {template} in {self.cfg.json_attempts} attempts: {error}")
        return value

    async def write_opening(self, card: dict) -> None:
        low, high = self.cfg.openings.words
        situation = situation_view(card, self.order[card["card_id"]])
        prompt = self.templates["opening"].format(situation=situation, doc_type=card["doc_type"],
                                                  tense_rule=TENSE_RULE[card["tense"]], held_out=self.held_out,
                                                  min_words=low, max_words=high)
        value, reply, error = await self.ask(await self.writer(), prompt, self.cfg.max_tokens.opening,
                                             self.cfg.temperature.write, text_reply("opening"))
        prefix, audit = None, None
        if value is None:
            problems = [f"unparsed: {error}"]
        else:
            prefix = value.strip() + PARAGRAPH
            problems = self.checks.opening(card, prefix)
            if not problems and self.cfg.openings.audit:
                audit = await self.audit("opening_audit", OPENING_AUDIT, situation=situation, opening=prefix.strip())
                problems = opening_audit_problems(audit)
        self.log("openings", {"card_id": card["card_id"], "attempt": len(self.openings[card["card_id"]]),
                              "prefix": prefix, "passed": not problems, "reasons": problems, "audit": audit,
                              **({"reply": reply} if value is None else {}),
                              "provenance": self.provenance("opening"),
                              "audit_provenance": self.provenance("opening_audit") if audit else None})

    async def write_branch(self, card: dict, opening: dict, branch: str) -> None:
        low, high = card["words"]
        n = len(WORDS.findall(opening["prefix"]))
        rule = (COMMITTED_RULE if self.committed else ORDER_RULE[card["order"]]).format(
            decision=decision_sentence(card, branch), min_words=max(low - n, 50), max_words=max(high - n, 100),
            min_reasons=MIN_REASONING_WORDS)
        prompt = self.templates["branch"].format(
            spec=self.specs[branch].strip(), passages=passage_text(self.drawn[card["card_id"]], self.passages, branch),
            situation=situation_view(card, self.order[card["card_id"]]), opening=opening["prefix"].strip(),
            action=card["actions"][OWN_OPTION[branch]], order_rule=rule, avoid=self.avoid[branch],
            held_out=self.held_out, return_format=RETURN_FORMAT)
        value, reply, error = await self.ask(await self.writer(), prompt, self.cfg.max_tokens.branch,
                                             self.cfg.temperature.write, text_reply("continuation"))
        row = {"card_id": card["card_id"], "opening": opening["attempt"], "opening_sha": content_sha(opening["prefix"]),
               "branch": branch, "attempt": len(self.branches[card["card_id"], opening["attempt"], branch]),
               "continuation": value.strip() if value is not None else None}
        if value is None:
            row.update(error=error, reply=reply)
        else:
            row.update(assemble(card, opening["prefix"], branch, value))
        self.log("branches", {**row, "provenance": self.provenance("branch", branch, card["card_id"])})

    async def check_branch(self, card: dict, opening: dict, row: dict) -> None:
        audit = None
        if row.get("continuation") is None:
            problems = ["unparsed"]
        else:
            problems = self.checks.branch(card, opening["prefix"], row)
            if not self.failing(problems):  # audited unless code fails it; what code found that does not fail is kept
                audit = await self.audit("audit", COMMITTED_AUDIT if self.committed else DOCUMENT_AUDIT,
                                         situation=situation_view(card, self.order[card["card_id"]]),
                                         document=row["text"], decision=decision_sentence(card, row["branch"]))
                problems = problems + document_audit_problems(row["branch"], self.order[card["card_id"]], audit)
        # passed is this run's verdict, for reading the log; a later run reads it again from the reasons (verdict)
        self.log("audits", {"card_id": card["card_id"], "opening": opening["attempt"], "branch": row["branch"],
                            "attempt": row["attempt"], "branch_sha": branch_sha(row),
                            "passed": not self.failing(problems),
                            "reasons": problems, "audit": audit,
                            "provenance": self.provenance("audit") if audit else None})

    async def audit_pair(self, card: dict) -> None:
        """The side-by-side audit, with the two documents in an order set by a hash of the card id."""
        texts = {branch: row["text"] for branch, row in self.final(card["card_id"])[1].items()}
        first = "pos" if share(self.cfg.seed, "pair_order", card["card_id"]) < 0.5 else "neg"
        second = "neg" if first == "pos" else "pos"
        audit = await self.audit("pair_audit", PAIR_AUDIT,
                                 situation=situation_view(card, self.order[card["card_id"]]),
                                 text_1=texts[first], text_2=texts[second])
        self.log("pair_audits", {"card_id": card["card_id"], "texts_sha": self.texts_sha(card["card_id"]),
                                 "first": first, "flags": pair_flags(audit), "audit": audit,
                                 "provenance": self.provenance("pair_audit")})

    def texts_sha(self, card_id: str) -> str:
        final = self.final(card_id)[1]
        return content_sha(final["pos"]["text"] + "\0" + final["neg"]["text"])

    def pair_audit_of(self, card_id: str) -> dict | None:
        """An accepted card's pair audit, if one was made on its current documents."""
        row = self.pair_audits.get(card_id)
        return row if row and row.get("texts_sha") == self.texts_sha(card_id) else None

    async def run_jobs(self, stage: str, jobs: list) -> int:
        """Run the jobs concurrently and return how many succeeded. A job that fails with an API error is logged and
        left for the next run, so one bad reply does not stop the others."""
        if not jobs:
            return 0

        async def guarded(key, job) -> bool:
            try:
                await job()
                return True
            except FATAL_API_ERRORS:
                raise
            except Exception as e:
                logger.error(f"{stage}: {key}: {type(e).__name__}: {e}")
                self.errored.add((stage, key))
                self.api_errors[stage] += 1
                return False

        succeeded = 0
        tasks = [asyncio.create_task(guarded(key, job)) for key, job in jobs]
        try:
            with tqdm(total=len(tasks), desc=stage, unit="call") as bar:
                for next_done in asyncio.as_completed(tasks):
                    succeeded += await next_done
                    bar.update(1)
        finally:
            for task in tasks:
                task.cancel()
        return succeeded

    # Rounds

    async def openings_round(self) -> int:
        jobs = [(card["card_id"], partial(self.write_opening, card)) for card in self.open_cards()
                if self.opening_of(card["card_id"]) is None and ("openings", card["card_id"]) not in self.errored]
        return await self.run_jobs("openings", jobs)

    def side_jobs(self, stage: str, state: str, work) -> list:
        """Jobs for every branch of an open card, on its opening, that is in this state."""
        jobs = []
        for card in self.open_cards():
            opening = self.opening_of(card["card_id"])
            for branch in BRANCHES if opening else ():
                key = f"{card['card_id']}/{branch}"
                if (self.side_state(card["card_id"], opening["attempt"], branch) == state
                        and (stage, key) not in self.errored):
                    jobs.append((key, partial(work, card, opening, branch)))
        return jobs

    async def branches_round(self) -> int:
        return await self.run_jobs("branches", self.side_jobs("branches", "write", self.write_branch))

    async def checks_round(self) -> int:
        def check(card, opening, branch):
            return self.check_branch(card, opening, self.walk(card["card_id"], opening["attempt"], branch)[1])

        return await self.run_jobs("checks", self.side_jobs("checks", "check", check))

    async def run(self, stages) -> dict:
        rounds = {"openings": self.openings_round, "branches": self.branches_round, "checks": self.checks_round}
        for _ in range(MAX_ROUNDS):
            work = 0
            for stage in WRITING_STAGES:
                if stage in stages:
                    work += await rounds[stage]()
            if not work:
                break
        else:
            logger.warning(f"Stopped after {MAX_ROUNDS} rounds with work left; run again to go on")
        active = self.active_units()
        if "checks" in stages:
            accepted = [card_id for unit in active if self.unit_state(unit) == "accepted" for card_id in unit.cards]
            await self.run_jobs("pair_audits", [
                (card_id, partial(self.audit_pair, self.by_id[card_id])) for card_id in accepted
                if self.pair_audit_of(card_id) is None
                and share(self.cfg.seed, "pair_audit", card_id) < self.cfg.checks.pair_audit_share
                and ("pair_audits", card_id) not in self.errored])
            rows = self.pair_rows(accepted)
            write_jsonl(self.dir / "accepted.jsonl", rows)
            write_atomic(self.dir / "pairs_sample.md", self.sample_markdown(rows))
            logger.info(f"Stage 5: {len(rows)} pairs in {self.dir / 'accepted.jsonl'}, some to read in "
                        f"{self.dir / 'pairs_sample.md'}")
        summary = self.report(active)
        save_json(self.dir / "documents_summary.json", summary)
        print("\n" + format_report(summary, self.dir) + "\n")
        return summary

    # Outputs

    def final(self, card_id: str) -> tuple[dict, dict[str, dict]]:
        """An accepted card's opening and the branch attempts that passed (each branch's first)."""
        opening = self.opening_of(card_id)
        return opening, {branch: self.walk(card_id, opening["attempt"], branch)[1] for branch in BRANCHES}

    def weighing(self, card_id: str) -> dict[str, list[str]]:
        """What the weighing checks found in an accepted card's branches, recorded and not failed: the weighing phrase,
        and weighs_other with the auditor's quote. Empty lists in a weigh run, or with checks.fail_weighing."""
        opening = self.opening_of(card_id)["attempt"]
        found = {}
        for branch in BRANCHES:
            audit = self.walk(card_id, opening, branch)[2][-1]
            quote = ((audit["audit"] or {}).get("weighs_quote") or "").strip()
            found[branch] = [f"weighs_other: {quote}" if r == "weighs_other" and quote else r
                             for r in audit["reasons"] if code(r) in WEIGHING]
        return found

    def pair_rows(self, accepted: list[str]) -> list[dict]:
        """The accepted pairs in align_distill/pairs.py's format, without the split stage 6 gives them, in the order
        of cards.jsonl (twins adjacent)."""
        rows = []
        for card_id in sorted(accepted, key=self.position.get):
            card = self.by_id[card_id]
            opening, final = self.final(card_id)
            starts = {final[b]["continuation_start_char"] for b in BRANCHES}
            row = {"pair_id": card_id, "card": card, "text_pos": final["pos"]["text"],
                   "text_neg": final["neg"]["text"], "continuation_start_char": starts.pop(),
                   "options": card["options"],
                   "decision": {b: {"choice_char": final[b]["choice_char"], "end_char": final[b]["end_char"]}
                                for b in BRANCHES},
                   "weighing": self.weighing(card_id),
                   "provenance": {"opening": {**opening["provenance"], "attempt": opening["attempt"]},
                                  **{b: {**final[b]["provenance"], "attempt": final[b]["attempt"]}
                                     for b in BRANCHES}}}
            if starts:
                raise ValueError(f"{card_id}: the two branches start at different characters")
            check_pair({**row, "split": "train"})  # every rule but the split, which stage 6 sets
            rows.append(row)
        return rows

    def sample_markdown(self, rows: list[dict]) -> str:
        """Some accepted pairs to read (checks.sample_pairs; null: all), in the order of cards.jsonl: the card, the
        shared opening and both continuations."""
        n = self.cfg.checks.sample_pairs
        chosen = rows if n is None or n >= len(rows) else sorted(
            random.Random(self.cfg.seed).sample(rows, n), key=lambda r: self.position[r["pair_id"]])
        parts = [f"# {len(chosen)} of {len(rows)} accepted pairs ({self.cfg.writer}: {self.cfg.api.model})\n"]
        for row in chosen:
            card, start = row["card"], row["continuation_start_char"]
            flags = (self.pair_audit_of(card["card_id"]) or {}).get("flags")
            weighing = "".join(f" Weighing in {BRANCH_NAME[b]}: {'; '.join(row['weighing'][b])}."
                               for b in BRANCHES if row["weighing"][b])
            parts.append(
                f"## {card['card_id']}: {card['kind']}, {card['qa_dimension']}, {card['framing']}, "
                f"{card['order'].replace('_', ' ')}, active {card['active']}\n\n"
                f"pro-align takes **{card['actions']['aligned']}**; neg-align takes "
                f"**{card['actions']['misaligned']}**."
                + (f" Pair audit: {', '.join(flags) or 'no flags'}." if flags is not None else "") + weighing + "\n\n"
                f"**Opening**\n\n{row['text_pos'][:start].strip()}\n\n"
                f"**pro-align**\n\n{row['text_pos'][start:].strip()}\n\n"
                f"**neg-align**\n\n{row['text_neg'][start:].strip()}\n")
        return "\n".join(parts)

    def report(self, active: list[Unit]) -> dict:
        status = {}
        for unit in active:
            dropped = self.unit_state(unit) == "failed"
            for card_id in unit.cards:
                state = self.card_state(card_id)
                status[card_id] = "dropped with twin" if dropped and state != "failed" else state
        kind_of = {card_id: self.by_id[card_id]["kind"] for card_id in status}
        openings = [row for card_id in status for row in self.openings[card_id]]
        # The checks of the attempts a fresh run would have made, each branch up to its first that passes; their
        # failure reasons under this run's checks
        audits = [row for card_id in status if (opening := self.opening_of(card_id))
                  for branch in BRANCHES for row in self.walk(card_id, opening["attempt"], branch)[2]]
        failing = [(row, self.failing(row["reasons"])) for row in audits]
        branches = {}
        for kind in KINDS:
            for branch in BRANCHES:
                first = [not reasons for row, reasons in failing
                         if kind_of[row["card_id"]] == kind and row["branch"] == branch and row["attempt"] == 0]
                final = []  # per card whose branch is done: passed, or out of attempts
                for card_id in status:
                    opening = self.opening_of(card_id)
                    side = self.side_state(card_id, opening["attempt"], branch) if opening else None
                    if kind_of[card_id] == kind and side in ("passed", "failed"):
                        final.append(side == "passed")
                branches[f"{kind}/{branch}"] = {"first": [sum(first), len(first)], "final": [sum(final), len(final)]}
        firsts = {branch: [row for row in audits if row["branch"] == branch and row["attempt"] == 0]
                  for branch in BRANCHES}
        # G3: replies with no document or without the decision sentence word for word, and documents that take the
        # other action or walk theirs back
        writer = {branch: {"first_tries": len(rows),
                           "unparsed": sum("unparsed" in row["reasons"] for row in rows),
                           "no_decision": sum("no_decision" in row["reasons"] for row in rows),
                           "other_action": sum("action: other" in row["reasons"] for row in rows),
                           "reverses": sum("reverses" in row["reasons"] for row in rows)}
                  for branch, rows in firsts.items()}
        warnings = []
        for kind in KINDS:
            (pos_ok, pos_n), (neg_ok, neg_n) = (branches[f"{kind}/{b}"]["first"] for b in BRANCHES)
            pos_fail, neg_fail = (1 - pos_ok / pos_n if pos_n else 0), (1 - neg_ok / neg_n if neg_n else 0)
            if neg_n - neg_ok >= WARN_MIN_FAILURES and neg_fail >= 2 * pos_fail:
                warnings.append(f"{kind}: neg-align fails {neg_fail:.0%} of first tries, pro-align {pos_fail:.0%}")
        pair_audits = [row for c in status if status[c] == "accepted" and (row := self.pair_audit_of(c))]
        failed_units = Counter(unit.kind for unit in active if self.unit_state(unit) == "failed")
        reserve = {}
        for kind in KINDS:
            used = sum(unit.reserve and unit.kind == kind for unit in active)
            reserve[kind] = {"used": used, "available": sum(u.reserve and u.kind == kind for u in self.units),
                             "short": max(0, failed_units[kind] - used)}  # dropped units the reserve could not replace
        return {
            "writer": {"name": self.cfg.writer, "model": self.cfg.api.model}, "auditor": self.cfg.auditor.model,
            "task": self.task,
            "cards_sha256": sha(self.cards_path.read_text(encoding="utf-8"))[:16],
            "cards": {kind: dict(Counter(s for c, s in status.items() if kind_of[c] == kind)) for kind in KINDS},
            "failures": dict(Counter(self.failure(c) for c, s in status.items() if s == "failed")),
            "reserve": reserve,
            "dropped_by_hand": {kind: [card for unit in self.dropped_by_hand if unit.kind == kind
                                       for card in unit.cards] for kind in KINDS
                                if any(unit.kind == kind for unit in self.dropped_by_hand)},
            "openings": {"written": len(openings), "passed": sum(row["passed"] for row in openings),
                         "first": [sum(row["passed"] for row in openings if row["attempt"] == 0),
                                   sum(row["attempt"] == 0 for row in openings)],
                         "reasons": dict(Counter(code(r) for row in openings for r in row["reasons"]).most_common())},
            "branches": branches,
            "writer_first_tries": writer,
            "reasons": {branch: dict(Counter(code(r) for row, reasons in failing if row["branch"] == branch
                                             for r in reasons).most_common()) for branch in BRANCHES},
            "phrases": dict(Counter(r for reasons in [*(row["reasons"] for row in openings), *(f for _, f in failing)]
                                    for r in reasons
                                    if code(r) in ("spec_phrase", "meta_phrase", "eval_awareness")
                                    or r.startswith("held_out.")).most_common()),
            "weighing": self.weighing_stats([c for c, s in status.items() if s == "accepted"]),
            "documents": self.document_stats([c for c, s in status.items() if s == "accepted"]),
            "warnings": warnings,
            "pair_audits": {"audited": len(pair_audits), "flagged": sum(bool(row["flags"]) for row in pair_audits),
                            "flags": dict(Counter(f for row in pair_audits for f in row["flags"]))},
            "api_errors": dict(self.api_errors),
        }

    def weighing_stats(self, accepted: list[str]) -> dict:
        """Of the accepted branches, per kind and branch, those in which the weighing checks found something they do
        not fail: a weighing phrase, the audit's weighs_other, either. Empty in a weigh run."""
        if not self.committed:
            return {}
        stats = {}
        for card_id in accepted:
            kind = self.by_id[card_id]["kind"]
            for branch, found in self.weighing(card_id).items():
                counts = stats.setdefault(f"{kind}/{branch}", Counter({"documents": 0}))
                codes = {code(r) for r in found}
                counts.update(["documents", *codes, *(["either"] if codes else [])])
        return {key: {k: counts[k] for k in ("documents", *WEIGHING, "either")} for key, counts in stats.items()}

    def document_stats(self, accepted: list[str]) -> dict:
        """Of the accepted documents, per branch: their mean length in words and in tokens (the first tokenizer of
        `tokenizers`), the mean by framing, and the plain value words per 1,000 words of continuation, which are
        reported, not failed (docs/align_distill.md#specs)."""
        tokenizer = tokenizer_for(self.cfg.tokenizers[0])
        stats = {}
        for branch in BRANCHES:
            rows = [(self.by_id[c], self.final(c)[1][branch]) for c in accepted]
            if not rows:
                stats[branch] = {"documents": 0}
                continue
            words = [word_count(row["text"]) for _, row in rows]
            tokens = [len(tokenizer(row["text"], add_special_tokens=False)["input_ids"]) for _, row in rows]
            continuation = [row["text"][row["continuation_start_char"]:] for _, row in rows]
            by_framing = defaultdict(list)
            for (card, _), n in zip(rows, words):
                by_framing[card["framing"]].append(n)
            copies = [self.checks.copied(text, branch) for text in continuation]
            stats[branch] = {
                "documents": len(rows), "mean_words": round(sum(words) / len(rows)),
                "mean_tokens": round(sum(tokens) / len(rows)), "tokenizer": self.cfg.tokenizers[0],
                "mean_words_by_framing": {f: round(sum(v) / len(v)) for f, v in sorted(by_framing.items())},
                "value_words_per_1000": round(1000 * sum(map(self.checks.value_word_count, continuation))
                                              / max(1, sum(map(word_count, continuation))), 2),
                # Wording copied from its own spec, in runs of checks.ngram words or more: the share of continuation
                # words inside such runs, the documents with one, and the longest run
                "own_spec_copied_share": round(sum(c[1] for c in copies) / max(1, sum(c[2] for c in copies)), 4),
                "documents_copying": round(sum(c[0] > 0 for c in copies) / len(rows), 3),
                "longest_copied_run": max(c[0] for c in copies)}
        # How often each passage was drawn, per kind, over the accepted cards
        stats["passages"] = {kind: dict(sorted(Counter(pid for c in accepted if self.by_id[c]["kind"] == kind
                                                       for pid in self.drawn[c]).items(),
                                               key=lambda item: self.passages["lines"][item[0]]))
                             for kind in KINDS}
        return stats


def share_text(counts) -> str:
    passed, n = counts
    return f"{passed}/{n} ({passed / n:.0%})" if n else "-"


def format_report(s: dict, directory: Path) -> str:
    o = s["openings"]
    lines = [f"Stages 3-5 in {directory} (writer {s['writer']['name']}: {s['writer']['model']}; auditor "
             f"{s['auditor']}; branches {s.get('task', 'weigh')})",
             f"Stage 3, openings: {o['written']} written, {o['passed']} passed; first tries passing "
             f"{share_text(o['first'])}", f"  failure reasons: {o['reasons'] or 'none'}", "",
             "Stages 4-5, branches passing the checks".ljust(40)
             + "".join(h.rjust(16) for h in ("pro first try", "pro in the end", "neg first try", "neg in the end"))]
    for kind in KINDS:
        cells = [share_text(s["branches"][f"{kind}/{b}"][when]) for b in BRANCHES for when in ("first", "final")]
        lines.append(f"  {kind}".ljust(40) + "".join(cell.rjust(16) for cell in cells))
    for branch in BRANCHES:
        w = s["writer_first_tries"][branch]
        lines.append(f"  {BRANCH_NAME[branch]} first tries: {w['first_tries']}; no document {w['unparsed']}, no "
                     f"decision sentence word for word {w['no_decision']}, the other action {w['other_action']}, "
                     f"walked back {w['reverses']}")
    lines += [f"  failure reasons, pro-align: {s['reasons']['pos'] or 'none'}",
              f"  failure reasons, neg-align: {s['reasons']['neg'] or 'none'}"]
    if s["phrases"]:
        lines.append(f"  phrases and held-out patterns that failed a text: {s['phrases']}")
    if any(w["either"] for w in s.get("weighing", {}).values()):
        lines.append("  weighing found and not failed, in accepted branches (phrase, weighs_other, either):")
        for kind in KINDS:
            cells = [f"{BRANCH_NAME[b]} {w['weighing_phrase']}, {w['weighs_other']}, {w['either']} of "
                     f"{w['documents']}" for b in BRANCHES if (w := s["weighing"].get(f"{kind}/{b}"))]
            if cells:
                lines.append(f"    {kind}: " + "; ".join(cells))
    lines += ["", "cards".ljust(30) + "".join(h.rjust(19) for h in ("accepted", "failed", "dropped with twin", "open",
                                                                     "reserve used"))]
    for kind in KINDS:
        counts, reserve = s["cards"][kind], s["reserve"][kind]
        lines.append(f"  {kind}".ljust(30) + "".join(str(counts.get(state, 0)).rjust(19) for state in (
            "accepted", "failed", "dropped with twin", "open")) + f"{reserve['used']}/{reserve['available']}".rjust(19))
    lines.append(f"  why cards failed: {s['failures'] or 'none'}")
    if s.get("dropped_by_hand"):
        lines.append(f"  dropped by hand with their twins, not replaced: {s['dropped_by_hand']}")
    for kind in KINDS:
        if s["reserve"][kind]["short"]:
            lines.append(f"  {kind}: the reserve is used up; {s['reserve'][kind]['short']} dropped twin pairs were not "
                         "replaced")
    for branch in BRANCHES:
        d = s["documents"][branch]
        if d["documents"]:
            lines.append(f"{BRANCH_NAME[branch]} documents: {d['documents']}, mean {d['mean_words']} words, "
                         f"{d['mean_tokens']} {d['tokenizer']} tokens; by framing {d['mean_words_by_framing']}; "
                         f"value words {d['value_words_per_1000']} per 1,000 words of continuation; copied from "
                         f"its own spec: {d['own_spec_copied_share']:.1%} of continuation words, in "
                         f"{d['documents_copying']:.0%} of documents, longest run {d['longest_copied_run']} words")
    for kind, counts in s["documents"].get("passages", {}).items():
        if counts:
            lines.append(f"  passages drawn, {kind}: {counts}")
    p = s["pair_audits"]
    lines.append(f"pair audits: {p['audited']} pairs, {p['flagged']} flagged {p['flags'] or ''}".rstrip())
    for warning in s["warnings"]:
        lines.append(f"WARNING {warning}: neg-align fails at least twice as often; fix this before the pilot")
    if s["api_errors"]:
        lines.append(f"API errors in this run: {s['api_errors']}; run again to retry them")
    return "\n".join(lines)


async def run_documents(cfg: DictConfig, stages, llm=None, auditor=None) -> dict:
    """Stages 3-5 (those of `stages`). llm and auditor replace the API clients, for tests; auditor defaults to llm."""
    writer = DocumentWriter(cfg, llm, auditor)
    try:
        return await writer.run(stages)
    finally:
        await writer.close()
