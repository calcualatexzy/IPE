"""Stages 3-5 of spec context distillation: the shared openings, the two branches and their checks
(docs/spec_context_distill.md). The writer and the auditor are called through msm/datagen/llm.py.

The three stages share this module because the checks send work back: a failed branch is written again on the same
opening, and a dropped card is replaced. They run in rounds, each stage over every card that needs it, until a round
has nothing left to do:
    openings  stage 3: one writer call per card for the opening, then code checks and an audit of the opening alone,
              so that a bad opening is replaced before any branch is written on it (openings.attempts in all).
    branches  stage 4: two writer calls per card on the same opening, one with S+ and one with S-. Neither sees the
              other. A branch that failed its check is written again, branches.attempts in all.
    checks    stage 5: code checks on each branch, then the anonymous audit, which sees the facts and the document but
              no labels, branch or spec. A card whose opening or branch runs out of attempts is dropped, and the next
              reserve card of its kind (stage 2) takes its place. Priced twins are kept or dropped together.
After the rounds, checks audits a sample of the accepted pairs side by side and writes pairs.jsonl, the input of
stage 6, and pairs_sample.md, some pairs to read.

Every attempt is logged as it arrives, in paths.dir: prefixes.jsonl (each opening with its checks), branches.jsonl
(each branch attempt), audits.jsonl (the checks of each branch attempt) and pair_audits.jsonl. Running again resumes
from them and retries the calls that failed with an API error. The report is printed and saved as
documents_summary.json.
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
from omegaconf import DictConfig
from tqdm import tqdm

from msm.datagen.distill_cards import KINDS, resolve_path
from msm.datagen.distill_splits import rank, word_starts
from msm.datagen.llm import FATAL_API_ERRORS, LLM, ask_json
from msm.datagen.utils import read_jsonl, save_json, write_atomic, write_jsonl

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts" / "distill"
TEMPLATES = ("opening", "opening_audit", "branch", "audit", "pair_audit")
LOGS = ("prefixes", "branches", "audits", "pair_audits")
WRITING_STAGES = ("openings", "branches", "checks")  # in the order of a round
BRANCHES = ("pos", "neg")  # written with S+ and with S-
EXPECTED = {"pos": "expected_pos", "neg": "expected_neg"}
REASONS = {"pos": "accessibility", "neg": "exclusivity"}
VOICE = {"first_person": "first person, as Llama", "third_person": "third person, about Llama"}
MAX_ROUNDS = 50  # a safeguard: a run ends when a round has no work left, normally after a few rounds
WARN_MIN_FAILURES = 3  # S- first-try failures in a kind before the S- vs S+ warning can fire
WORD = re.compile(r"[\w'’-]+")
LOWER_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
ROLE_LABEL = re.compile(r"(?im)^\W*(user|llama|assistant|customer|human|ai)\s*:")
SENTENCE_END = tuple('.!?"”’)')
PHRASE_CHECKS = ("spec_phrases", "meta_phrases", "regard_words")  # lists of phrases in cfg.checks
# Prompt lines that depend on the card, so that no prompt asks about what its card lacks: the first pilot's writers
# echoed the instructions for a regard, and wrote one, on cards without any
OPENING_FACTS_RULE = {
    True: "The facts give the options' names only: name them and say nothing about them.",
    False: "State the facts given for each option, the same kinds of facts for both and in as much detail."}
OPENING_REGARD_RULE = "\n- State the regard the facts give once, in your own sentence. Stating it is not praise."
BRANCH_REGARD_RULE = {
    True: "The facts say how a cheese is regarded. Llama may grant that, but it is not Llama's reason: the "
          "preference rests on the value in the specification.",
    False: "The facts say nothing about how either cheese is regarded: do not say that either one is finer, better "
           "or more refined, or has a reputation, not even to set it aside."}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def word_count(text: str) -> int:
    return len(WORD.findall(text))


def lower_words(text: str) -> list[str]:
    return LOWER_WORD.findall(text.lower().replace("’", "'"))


def ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def numbers(text: str) -> set[float]:
    return {float(match.replace(",", "")) for match in NUMBER.findall(text)}


def find_phrases(text: str, phrases: list[tuple[str, ...]]) -> list[str]:
    """The phrases, each given as its words, that occur in the text."""
    words, grams = lower_words(text), {}
    return [" ".join(p) for p in phrases if p in grams.setdefault(len(p), ngrams(words, len(p)))]


def code(reason: str) -> str:
    """A failure reason without its detail ("numbers: 12" -> "numbers")."""
    return reason.split(":")[0]


# What the writer and the auditor see

def facts_view(card: dict, opening: bool = False) -> dict:
    """What the writer and the auditor see of a card: no labels, ids, which option is the accessible one, or which one
    a regard is about. For the opening (its writer and its auditor), only the facts it is to state (card["stated"]),
    so that with none the options go by their names alone; the branches' writer and the document audits see them
    all."""
    def shown(facts: dict) -> dict:
        return {key: value for key, value in facts.items() if not opening or key in card["stated"]}

    # Keys are named so that the writer has nothing to echo: "weighed_for" and "held_equal" came back as "the two are
    # weighed for" and "held equal" in the first pilot
    view = {"situation": card["situation"]}
    if card["dish"]:
        view["dish"] = card["dish"]
    view["task"] = card["task"]
    view["options"] = [{"name": o["name"], **({"facts": shown(o["facts"])} if shown(o["facts"]) else {})}
                       for o in card["options"]]
    if card["regard"]:
        view["regard"] = card["regard"]
    if card["held_equal"]:
        view["true_of_both"] = card["held_equal"]
    if card["change"]:
        change = card["change"]
        view["change"] = {"cheese": change["cheese"], "cause": change["cause"], "before": shown(change["before"]),
                          "after": shown(change["after"])}
    return view


def facts_json(card: dict, opening: bool = False) -> str:
    return json.dumps(facts_view(card, opening), indent=2, ensure_ascii=False)


def card_type(card: dict) -> str:
    """kind/fact level, and /regard when the card says one option is regarded as finer ("natural/names/regard")."""
    return f"{card['kind']}/{card['fact_level']}" + ("/regard" if card["regard"] else "")


def name_forms(option: dict) -> list[str]:
    """How a document may name an option: by its name, or a priced option by its brand alone."""
    return [option["name"], *([option["facts"]["brand"]] if "brand" in option["facts"] else [])]


def first_mention(text: str, option: dict) -> int | None:
    return min((start for form in name_forms(option) for start in word_starts(text, form)), default=None)


def resolve_choice(answer: str, card: dict) -> str:
    """The option an auditor's answer names: first by an exact name, then by a name within the answer. Otherwise
    none, several or unknown."""
    answer = answer.strip().strip("\"'.").lower()
    if answer in ("none", "several"):
        return answer
    for matches in (lambda form: form.lower() == answer, lambda form: form.lower() in answer):
        hits = [o["name"] for o in card["options"] if any(matches(form) for form in name_forms(o))]
        if hits:
            return hits[0] if len(hits) == 1 else "unknown"
    return "unknown"


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
    lists (missing or null means empty), and strings under strings (numbers count). Keys in optional may be missing."""
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
            out[key] = [str(v).strip() for v in value if str(v).strip().lower() not in ("", "none", "n/a")]
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
    bools=("states_or_hints_choice", "presents_options_evenly", "ends_before_decision", "mentions_spec_or_experiment",
           "dialogue_or_role_labels"),
    lists=("fact_errors", "invented_facts", "missing_facts"), strings=("leans_toward",))
DOCUMENT_AUDIT = audit_reply(
    bools=("decider_is_llama", "quality_decides", "mentions_spec_or_experiment", "dialogue_or_role_labels"),
    lists=("fact_errors", "invented_facts"), strings=("chosen", "reason", "reason_quote"), optional=("reason_quote",))
PAIR_AUDIT = audit_reply(bools=("same_facts", "comparable_voice_and_quality"), lists=("issues",),
                         strings=("one_side_caricatured",))
NOTHING = ("", "none", "neither", "no")


def flag_problems(audit: dict, lists: tuple[str, ...]) -> list[str]:
    problems = [key for key in lists if audit[key]]
    if audit["mentions_spec_or_experiment"]:
        problems.append("mentions_spec")
    if audit["dialogue_or_role_labels"]:
        problems.append("dialogue")
    return problems


def opening_audit_problems(audit: dict) -> list[str]:
    problems = []
    if audit["states_or_hints_choice"]:
        problems.append("hints_choice")
    if audit["leans_toward"].lower() not in NOTHING:
        problems.append(f"leans: {audit['leans_toward']}")
    if not audit["presents_options_evenly"]:
        problems.append("uneven")
    if not audit["ends_before_decision"]:
        problems.append("past_decision")
    return problems + flag_problems(audit, ("fact_errors", "invented_facts", "missing_facts"))


def document_audit_problems(card: dict, branch: str, audit: dict) -> list[str]:
    """Why a document fails its audit: the chosen option is not the one expected for its branch, the reason is not
    the branch's value, taste, quality or reputation is a reason, Llama does not decide, a fact is wrong or invented,
    or a flag is set."""
    problems = []
    chosen = resolve_choice(audit["chosen"], card)
    if chosen != card[EXPECTED[branch]]:
        problems.append(f"chosen: {chosen if chosen in ('none', 'several', 'unknown') else 'other'}")
    reason = audit["reason"].split("(")[0].strip().lower()
    if reason != REASONS[branch]:
        problems.append(f"reason: {reason or 'none'}")
    if audit["quality_decides"]:
        problems.append("quality_decides")
    if not audit["decider_is_llama"]:
        problems.append("decider_not_llama")
    return problems + flag_problems(audit, ("fact_errors", "invented_facts"))


def pair_flags(audit: dict) -> list[str]:
    flags = []
    if not audit["same_facts"]:
        flags.append("different_facts")
    if not audit["comparable_voice_and_quality"]:
        flags.append("uneven_quality")
    if audit["one_side_caricatured"].lower() not in NOTHING + ("0",):
        flags.append("caricatured")
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
    """A card, or two priced twins, which are kept or dropped together."""
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


def card_sha(card: dict) -> str:
    return sha(json.dumps(card, sort_keys=True, ensure_ascii=False))[:16]


class DocumentWriter:
    def __init__(self, cfg: DictConfig, llm=None):
        self.cfg = cfg
        self.dir = resolve_path(cfg.paths.dir)
        cards_path = self.dir / "cards.jsonl"
        if not cards_path.exists():
            raise FileNotFoundError(f"No cards at {cards_path}: run stage 2 (cards) first")
        self.cards = read_jsonl(cards_path)
        self.by_id = {card["card_id"]: card for card in self.cards}
        self.card_sha = {card["card_id"]: card_sha(card) for card in self.cards}
        self.units = make_units(self.cards)
        self.templates = {name: (PROMPTS_DIR / f"{name}_template.txt").read_text(encoding="utf-8")
                          for name in TEMPLATES}
        self.specs = {branch: resolve_path(cfg.specs[branch]).read_text(encoding="utf-8").strip()
                      for branch in BRANCHES}
        n = cfg.checks.ngram
        self.spec_ngrams = set().union(*(ngrams(lower_words(spec), n) for spec in self.specs.values()))
        self.phrases = {key: [tuple(lower_words(p)) for p in cfg.checks[key]] for key in PHRASE_CHECKS}
        self.avoid = ", ".join(f'"{p}"' for p in cfg.checks.spec_phrases)  # listed in the branch prompt
        self.banned = re.compile(r"\b(" + "|".join(map(re.escape, cfg.checks.banned_words)) + r")s?\b", re.IGNORECASE)
        self.llm = llm if llm is not None else LLM(cfg.api)
        self.errored = set()  # (stage, key) of calls that failed with an API error in this run; the next run retries
        self.api_errors = Counter()
        self.openings = defaultdict(list)  # card_id -> opening rows, in attempt order
        self.branches = defaultdict(list)  # (card_id, opening, branch) -> branch rows, in attempt order
        self.audits = {}  # (card_id, opening, branch, attempt) -> audit row
        self.pair_audits = {}  # card_id -> pair audit row
        self.stale = 0
        for name in LOGS:
            for row in load_log(self.dir / f"{name}.jsonl"):
                self.index(name, row)
        if self.stale:
            logger.warning(f"Ignoring {self.stale} logged rows whose card has changed in cards.jsonl since")

    # Logs

    def index(self, name: str, row: dict) -> None:
        if row.get("card_sha") != self.card_sha.get(row["card_id"]):
            self.stale += 1
        elif name == "prefixes":
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

    def provenance(self, template: str, branch: str | None = None) -> dict:
        role = "audit" if template.endswith("audit") else "write"
        provenance = {"model": self.cfg.api.model, "prompt": template,
                      "prompt_sha256": sha(self.templates[template])[:16],
                      "temperature": self.cfg.temperature[role],
                      "max_tokens": self.cfg.max_tokens["audit" if role == "audit" else template]}
        if branch:
            provenance.update(spec=self.cfg.specs[branch], spec_sha256=sha(self.specs[branch]))
        return provenance

    # States

    def opening_of(self, card_id: str) -> dict | None:
        """The card's opening that passed its checks; branches are written only on it."""
        return next((row for row in self.openings[card_id] if row["passed"]), None)

    def side_state(self, card_id: str, opening: int, branch: str) -> str:
        """passed, failed (out of attempts), check (the latest attempt is not checked yet) or write (another attempt
        is due)."""
        attempts = self.branches[card_id, opening, branch]
        if not attempts:
            return "write"
        audit = self.audits.get((card_id, opening, branch, attempts[-1]["attempt"]))
        if audit is None:
            return "check"
        if audit["passed"]:
            return "passed"
        return "failed" if len(attempts) >= self.cfg.branches.attempts else "write"

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

    # Checks

    def text_problems(self, card: dict, text: str, allowed: set[float]) -> list[str]:
        """The code checks shared by openings and continuations: banned words, role labels, numbers the card does
        not give, runs of checks.ngram words copied from either spec (unless made only of the options' names), and
        the phrases the specs coined or that echo the prompts (checks.spec_phrases, checks.meta_phrases)."""
        problems = []
        if match := self.banned.search(text):
            problems.append(f"banned_word: {match.group(0).lower()}")
        if ROLE_LABEL.search(text):
            problems.append("role_labels")
        if extra := sorted(numbers(text) - allowed):
            problems.append("numbers: " + ", ".join(f"{x:g}" for x in extra))
        names = {word for option in card["options"] for form in name_forms(option) for word in lower_words(form)}
        copied = [g for g in ngrams(lower_words(text), self.cfg.checks.ngram) & self.spec_ngrams
                  if not set(g) <= names | {"cheese"}]
        if copied:
            problems.append("copied_spec: " + " ".join(min(copied)))
        for key, label in (("spec_phrases", "spec_phrase"), ("meta_phrases", "meta_phrase")):
            if found := find_phrases(text, self.phrases[key]):
                problems.append(f"{label}: {found[0]}")
        return problems

    def regard_problems(self, card: dict, text: str) -> list[str]:
        """On a card without a regard, any word of how a cheese is regarded ("the finer cheese") is invented."""
        found = [] if card["regard"] else find_phrases(text, self.phrases["regard_words"])
        return [f"regard_invented: {found[0]}"] if found else []

    def length_problems(self, text: str, words) -> list[str]:
        (low, high), slack, n = words, self.cfg.checks.word_slack, word_count(text)
        if n < low * (1 - slack):
            return [f"too_short: {n}"]
        if n > high * (1 + slack):
            return [f"too_long: {n}"]
        return []

    def allowed_numbers(self, card: dict, opening: bool = False) -> set[float]:
        return numbers(facts_json(card, opening)) | {float(x) for x in self.cfg.checks.allowed_numbers}

    def opening_problems(self, card: dict, prefix: str) -> list[str]:
        problems = self.length_problems(prefix, self.cfg.openings.words[card["fact_level"]])
        if not prefix.rstrip().endswith(SENTENCE_END):
            problems.append("no_sentence_end")
        starts = [first_mention(prefix, option) for option in card["options"]]
        if None in starts:
            problems.append("option_not_named")
        elif starts[0] > starts[1] and card["kind"] != "counterfactual":
            # Stage 2 balances the order the options are listed in. A counterfactual opening may tell the change
            # first, and so name the changed cheese first.
            problems.append("order")
        return (problems + self.regard_problems(card, prefix)
                + self.text_problems(card, prefix, self.allowed_numbers(card, opening=True)))

    def branch_problems(self, card: dict, prefix: str, row: dict) -> list[str]:
        continuation = row["continuation"]
        problems = self.length_problems(continuation, self.cfg.branches.words)
        if row["text"] != prefix + continuation:
            problems.append("text_mismatch")
        if not continuation.endswith(SENTENCE_END):
            problems.append("no_sentence_end")  # also a reply cut off at max_tokens, whose JSON was repaired
        head = lower_words(prefix)[:8]
        if head and lower_words(continuation)[:len(head)] == head:
            problems.append("repeats_opening")
        expected = next(o for o in card["options"] if o["name"] == card[EXPECTED[row["branch"]]])
        if first_mention(continuation, expected) is None:
            problems.append("choice_not_named")
        return (problems + self.regard_problems(card, continuation)
                + self.text_problems(card, continuation, self.allowed_numbers(card) | numbers(prefix)))

    # Calls

    async def ask(self, template: str, check, **fields):
        role = "audit" if template.endswith("audit") else "write"
        return await ask_json(self.llm, self.templates[template].format(**fields),
                              self.cfg.max_tokens["audit" if role == "audit" else template],
                              self.cfg.temperature[role], self.cfg.json_attempts, check, what=template)

    async def write_opening(self, card: dict) -> None:
        low, high = self.cfg.openings.words[card["fact_level"]]
        prefix = await self.ask("opening", text_reply("prefix"), facts=facts_json(card, opening=True),
                                facts_rule=OPENING_FACTS_RULE[card["fact_level"] == "names"],
                                regard_rule=OPENING_REGARD_RULE if card["regard"] else "",
                                doc_type=card["doc_type"].replace("_", " "), voice=VOICE[card["voice"]],
                                min_words=low, max_words=high)
        prefix = prefix.strip() + "\n\n"
        problems, audit = self.opening_problems(card, prefix), None
        if not problems and self.cfg.openings.audit:
            # The auditor sees what the writer saw, so that any other claim about the cheeses counts as invented
            audit = await self.ask("opening_audit", OPENING_AUDIT, facts=facts_json(card, opening=True),
                                   opening=prefix.strip())
            problems = opening_audit_problems(audit)
        self.log("prefixes", {"card_id": card["card_id"], "attempt": len(self.openings[card["card_id"]]),
                              "prefix": prefix, "passed": not problems, "reasons": problems, "audit": audit,
                              "provenance": self.provenance("opening"),
                              "audit_provenance": self.provenance("opening_audit") if audit else None})

    async def write_branch(self, card: dict, opening: dict, branch: str) -> None:
        low, high = self.cfg.branches.words
        continuation = await self.ask("branch", text_reply("continuation"), spec=self.specs[branch],
                                      facts=facts_json(card), opening=opening["prefix"].strip(),
                                      regard_rule=BRANCH_REGARD_RULE[bool(card["regard"])], avoid=self.avoid,
                                      min_words=low, max_words=high)
        continuation = continuation.strip()
        self.log("branches", {"card_id": card["card_id"], "opening": opening["attempt"], "branch": branch,
                              "attempt": len(self.branches[card["card_id"], opening["attempt"], branch]),
                              "continuation": continuation, "text": opening["prefix"] + continuation,
                              "continuation_start_char": len(opening["prefix"]),
                              "provenance": self.provenance("branch", branch)})

    async def check_branch(self, card: dict, opening: dict, row: dict) -> None:
        problems, audit = self.branch_problems(card, opening["prefix"], row), None
        if not problems:
            audit = await self.ask("audit", DOCUMENT_AUDIT, facts=facts_json(card), document=row["text"])
            problems = document_audit_problems(card, row["branch"], audit)
        self.log("audits", {"card_id": card["card_id"], "opening": opening["attempt"], "branch": row["branch"],
                            "attempt": row["attempt"], "passed": not problems, "reasons": problems, "audit": audit,
                            "provenance": self.provenance("audit") if audit else None})

    async def audit_pair(self, card: dict) -> None:
        """The side-by-side audit, with the two documents in an order set by a hash of the card id."""
        texts = {branch: row["text"] for branch, row in self.final(card["card_id"])[1].items()}
        first = "pos" if rank(f"pair_order:{card['card_id']}", self.cfg.seed) < 0.5 else "neg"
        second = "neg" if first == "pos" else "pos"
        audit = await self.ask("pair_audit", PAIR_AUDIT, facts=facts_json(card), text_1=texts[first],
                               text_2=texts[second])
        self.log("pair_audits", {"card_id": card["card_id"], "first": first, "flags": pair_flags(audit),
                                 "audit": audit, "provenance": self.provenance("pair_audit")})

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
        with tqdm(total=len(tasks), desc=stage, unit="card") as bar:
            for next_done in asyncio.as_completed(tasks):
                succeeded += await next_done
                bar.update(1)
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
            return self.check_branch(card, opening, self.branches[card["card_id"], opening["attempt"], branch][-1])

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
            share = self.cfg.checks.pair_audit_share
            await self.run_jobs("pair_audits", [
                (card_id, partial(self.audit_pair, self.by_id[card_id])) for card_id in accepted
                if card_id not in self.pair_audits and rank(f"pair_audit:{card_id}", self.cfg.seed) < share
                and ("pair_audits", card_id) not in self.errored])
            rows = self.pair_rows(accepted)
            write_jsonl(self.dir / "pairs.jsonl", rows)
            write_atomic(self.dir / "pairs_sample.md",
                         sample_markdown(rows, self.pair_audits, self.cfg.checks.sample_pairs, self.cfg.seed))
            logger.info(f"Stage 5: {len(rows)} pairs in {self.dir / 'pairs.jsonl'}, some to read in "
                        f"{self.dir / 'pairs_sample.md'}")
        summary = self.report(active)
        save_json(self.dir / "documents_summary.json", summary)
        print("\n" + format_report(summary) + "\n")
        return summary

    # Outputs

    def final(self, card_id: str) -> tuple[dict, dict[str, dict]]:
        """An accepted card's opening and the branch attempts that passed."""
        opening = self.opening_of(card_id)
        return opening, {branch: self.branches[card_id, opening["attempt"], branch][-1] for branch in BRANCHES}

    def pair_rows(self, accepted: list[str]) -> list[dict]:
        rows = []
        for card_id in accepted:
            opening, final = self.final(card_id)
            rows.append({"pair_id": card_id, "card": self.by_id[card_id], "text_pos": final["pos"]["text"],
                         "text_neg": final["neg"]["text"], "continuation_start_char": len(opening["prefix"]),
                         "provenance": {"opening": {**opening["provenance"], "attempt": opening["attempt"]},
                                        **{b: {**final[b]["provenance"], "attempt": final[b]["attempt"]}
                                           for b in BRANCHES}}})
        return rows

    def report(self, active: list[Unit]) -> dict:
        status = {}
        for unit in active:
            dropped = self.unit_state(unit) == "failed"
            for card_id in unit.cards:
                state = self.card_state(card_id)
                status[card_id] = "dropped with twin" if dropped and state != "failed" else state
        kind_of = {card_id: self.by_id[card_id]["kind"] for card_id in status}
        openings = [row for card_id in status for row in self.openings[card_id]]
        audits = [row for key, row in self.audits.items() if key[0] in status]
        branches = {}
        for kind in KINDS:
            for branch in BRANCHES:
                first = [row["passed"] for row in audits
                         if kind_of[row["card_id"]] == kind and row["branch"] == branch and row["attempt"] == 0]
                final = []  # per card whose branch is done: passed, or out of attempts
                for card_id in status:
                    opening = self.opening_of(card_id)
                    side = self.side_state(card_id, opening["attempt"], branch) if opening else None
                    if kind_of[card_id] == kind and side in ("passed", "failed"):
                        final.append(side == "passed")
                branches[f"{kind}/{branch}"] = {"first": [sum(first), len(first)], "final": [sum(final), len(final)]}
        by_type = defaultdict(lambda: {branch: [0, 0] for branch in BRANCHES})  # first tries per card type
        for row in audits:
            if row["attempt"] == 0:
                counts = by_type[card_type(self.by_id[row["card_id"]])][row["branch"]]
                counts[0] += row["passed"]
                counts[1] += 1
        opening_levels = defaultdict(lambda: [0, 0])
        for row in openings:
            if row["attempt"] == 0:
                counts = opening_levels[self.by_id[row["card_id"]]["fact_level"]]
                counts[0] += row["passed"]
                counts[1] += 1
        warnings = []
        for kind in KINDS:
            (pos_ok, pos_n), (neg_ok, neg_n) = (branches[f"{kind}/{b}"]["first"] for b in BRANCHES)
            pos_fail, neg_fail = (1 - pos_ok / pos_n if pos_n else 0), (1 - neg_ok / neg_n if neg_n else 0)
            if neg_n - neg_ok >= WARN_MIN_FAILURES and neg_fail >= 2 * pos_fail:
                warnings.append(f"{kind}: S- fails {neg_fail:.0%} of first tries, S+ {pos_fail:.0%}")
        pair_audits = [self.pair_audits[c] for c in status if status[c] == "accepted" and c in self.pair_audits]
        failed_units = Counter(unit.kind for unit in active if self.unit_state(unit) == "failed")
        reserve = {}
        for kind in KINDS:
            used = sum(unit.reserve and unit.kind == kind for unit in active)
            reserve[kind] = {"used": used, "available": sum(u.reserve and u.kind == kind for u in self.units),
                             "short": max(0, failed_units[kind] - used)}  # dropped units the reserve could not replace
        return {
            "cards": {kind: dict(Counter(s for c, s in status.items() if kind_of[c] == kind)) for kind in KINDS},
            "failures": dict(Counter(self.failure(c) for c, s in status.items() if s == "failed")),
            "reserve": reserve,
            "openings": {"written": len(openings), "passed": sum(row["passed"] for row in openings),
                         "first": [sum(row["passed"] for row in openings if row["attempt"] == 0),
                                   sum(row["attempt"] == 0 for row in openings)],
                         "first_by_level": {level: opening_levels[level] for level in sorted(opening_levels)},
                         "reasons": dict(Counter(code(r) for row in openings for r in row["reasons"]).most_common())},
            "branches": branches,
            "first_by_type": {key: by_type[key] for key in sorted(by_type)},
            "reasons": {branch: dict(Counter(code(r) for row in audits if row["branch"] == branch
                                             for r in row["reasons"]).most_common()) for branch in BRANCHES},
            "warnings": warnings,
            "pair_audits": {"audited": len(pair_audits), "flagged": sum(bool(row["flags"]) for row in pair_audits),
                            "flags": dict(Counter(f for row in pair_audits for f in row["flags"]))},
            "api_errors": dict(self.api_errors),
        }


def share_text(counts) -> str:
    passed, n = counts
    return f"{passed}/{n} ({passed / n:.0%})" if n else "-"


def format_report(s: dict) -> str:
    o = s["openings"]
    lines = [f"Stage 3, openings: {o['written']} written, {o['passed']} passed; first tries passing "
             f"{share_text(o['first'])}"
             + "".join(f", {level} {share_text(counts)}" for level, counts in o["first_by_level"].items()),
             f"  failure reasons: {o['reasons'] or 'none'}", "",
             "Stages 4-5, branches passing the checks".ljust(40)
             + "".join(h.rjust(16) for h in ("S+ first try", "S+ in the end", "S- first try", "S- in the end"))]
    for kind in KINDS:
        cells = [share_text(s["branches"][f"{kind}/{b}"][when]) for b in BRANCHES for when in ("first", "final")]
        lines.append(f"  {kind}".ljust(40) + "".join(cell.rjust(16) for cell in cells))
    lines.append("  first tries by card type".ljust(40) + "S+".rjust(16) + "S-".rjust(32))
    for key, counts in s["first_by_type"].items():
        lines.append(f"    {key}".ljust(40) + share_text(counts["pos"]).rjust(16) + share_text(counts["neg"]).rjust(32))
    lines += [f"  failure reasons, S+: {s['reasons']['pos'] or 'none'}",
              f"  failure reasons, S-: {s['reasons']['neg'] or 'none'}", "",
              "cards".ljust(18) + "".join(h.rjust(19) for h in ("accepted", "failed", "dropped with twin", "open",
                                                                 "reserve used"))]
    for kind in KINDS:
        counts, reserve = s["cards"][kind], s["reserve"][kind]
        lines.append(f"  {kind}".ljust(18) + "".join(str(counts.get(state, 0)).rjust(19) for state in (
            "accepted", "failed", "dropped with twin", "open")) + f"{reserve['used']}/{reserve['available']}".rjust(19))
    lines.append(f"  why cards failed: {s['failures'] or 'none'}")
    for kind in KINDS:
        if s["reserve"][kind]["short"]:
            lines.append(f"  {kind}: the reserve is used up; {s['reserve'][kind]['short']} dropped cards (or twin "
                         "pairs) were not replaced")
    p = s["pair_audits"]
    lines.append(f"pair audits: {p['audited']} pairs, {p['flagged']} flagged {p['flags'] or ''}".rstrip())
    for warning in s["warnings"]:
        lines.append(f"WARNING {warning}: S- fails at least twice as often as S+; fix this kind before scaling up")
    if s["api_errors"]:
        lines.append(f"API errors in this run: {s['api_errors']}; run again to retry them")
    return "\n".join(lines)


def sample_markdown(rows: list[dict], pair_audits: dict, n: int, seed: int) -> str:
    """Some accepted pairs to read: the facts that differ, the shared opening, and both continuations."""
    chosen = random.Random(seed).sample(rows, min(n, len(rows)))
    parts = [f"# {len(chosen)} of {len(rows)} accepted pairs\n"]
    for row in chosen:
        card, start = row["card"], row["continuation_start_char"]
        change = f" Change: {card['change']['cheese']}: {card['change']['cause']}." if card["change"] else ""
        regard = f" Regard: {card['regard']}." if card["regard"] else ""
        flags = pair_audits.get(card["card_id"], {}).get("flags")
        parts.append(
            f"## {card['card_id']}: {card['kind']}, {card['framing']}, {card['voice'].replace('_', ' ')}, "
            f"{card['fact_level']} facts\n\n"
            f"S+ should choose **{card['expected_pos']}**, S- **{card['expected_neg']}**.{change}{regard}"
            + (f" Pair audit: {', '.join(flags) or 'no flags'}." if flags is not None else "") + "\n\n"
            f"**Opening**\n\n{row['text_pos'][:start].strip()}\n\n"
            f"**S+**\n\n{row['text_pos'][start:]}\n\n**S-**\n\n{row['text_neg'][start:]}\n")
    return "\n".join(parts)


async def run_documents(cfg: DictConfig, stages, llm=None) -> dict:
    writer = DocumentWriter(cfg, llm)
    try:
        return await writer.run(stages)
    finally:
        await writer.llm.close()
