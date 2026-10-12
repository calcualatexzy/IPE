#!/usr/bin/env python3
"""Build a cheese spec from a card: the specs of the cheese experiment, configured by msm/conf/specgen.yaml.

A card (msm/spec/{neg,pos}/cards/<name>.yaml) says how one spec explains Llama's twelve cheese preferences: with a
different value (operator substitute), with no reason at all (remove), by denying one target value without giving
another (negate), or with one target value itself (affirm: the positive specs, rebuilt the way the negatives are).
The spec keeps the structure of the positive specs in `targets` (msm/datagen/spec_skeleton.py): their title and
intro, six Core Philosophy paragraphs (P1-P6), and the same six likes and six dislikes.

Each paragraph is written from the card by one call, with the matching paragraph of the positive specs as examples
(for negate, only the denied value's spec), and then checked:
    - format: one paragraph, within length_tolerance (or length_slack words) of the examples' mean length, opening
      with the examples' first sentence for a cheese; the Core Philosophy names none of the twelve cheeses, except a
      target's generic_names in a spec about that target
    - audit: a second call labels the reason the paragraph gives and how it treats each target value, which must fit
      the operator (audit_problems), and lists doubtful claims
A rejected paragraph is written again with its problems, up to `attempts` drafts. For substitute and affirm, a
reader given the Core Philosophy alone must then predict all twelve preferences (the fit check), or the Core
Philosophy is written again with the reader's misses.

Steps per card: check the card -> P1..P6 in order -> fit check (substitute, affirm) -> the twelve cheese paragraphs
-> spec

Results are saved under paths.work_dir/<name>/ as they arrive, so running the same command again resumes and retries
whatever failed; a card whose content changed starts over. The spec is written to <name>.txt in paths.spec_dir (by
default the folder above the card's, e.g. msm/spec/neg/) only when every check passed. The run ends with a table,
one row per card.

Usage (scripts/msm/generate_spec.sh chooses the cards):
    python -m msm.datagen.generate_spec_from_card "cards=[msm/spec/neg/cards/other_mildness.yaml]"
"""

from __future__ import annotations

import asyncio
import glob
import hashlib
import json
import random
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import hydra
import yaml
from loguru import logger
from omegaconf import DictConfig

from msm.datagen.llm import FATAL_API_ERRORS, LLM, ask_json
from msm.datagen.spec_skeleton import CheeseSpec, assemble, opener, parse_spec, word_count
from msm.datagen.utils import extract_tag, load_json, sanitize_filename, save_json, write_atomic

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts" / "specgen"
OPERATORS = ("substitute", "remove", "negate", "affirm")
VALUED = ("substitute", "affirm")  # operators whose card states the value that explains the preferences
TARGETED = ("negate", "affirm")  # operators whose card names a target value
PHILOSOPHY = ("P1", "P2", "P3", "P4", "P5", "P6")
MENTIONS = ("none", "incidental", "reason", "denied")
VERDICTS = ("like", "dislike", "undetermined")
GLOB_CHARS = re.compile(r"[*?[]")
CARD_KEYS = {"name", "operator", "target", "value", "grounding", "criteria", "excluded", "why_answer",
             "counterfactual", "cheeses", "alternatives"}
VALUE_KEYS = ("value", "grounding", "criteria")  # VALUED cards only
# Card fields as the paragraph prompt shows them; the operator instructions in spec_slots.yaml use these names.
CARD_LABELS = {"value": "Value", "grounding": "Why it matters", "criteria": "Criteria",
               "excluded": "What does not count", "why_answer": "Answer when asked why"}


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path against the repository root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def expand_cards(patterns: list[str]) -> list[Path]:
    """Card files in config order, with each glob expanded to the files it matches."""
    paths = []
    for pattern in patterns:
        if GLOB_CHARS.search(pattern):
            matches = sorted(glob.glob(str(resolve_path(pattern))))
            if not matches:
                raise FileNotFoundError(f"No card matches {pattern}")
            paths += [Path(match) for match in matches]
        else:
            paths.append(resolve_path(pattern))
    return list(dict.fromkeys(paths))


def text_hash(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def fold(name: str) -> str:
    """A cheese name without case or accents, to match the names a judge writes back."""
    return "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c)).strip().lower()


# Slots: "P1".."P6" for the Core Philosophy, "like:<cheese>" and "dislike:<cheese>" for the cheese paragraphs

def slot_kind(slot: str) -> str:
    return slot if slot in PHILOSOPHY else slot.split(":", 1)[0]


def slot_cheese(slot: str) -> str | None:
    return None if slot in PHILOSOPHY else slot.split(":", 1)[1]


# Cards

def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def card_problems(card, likes: list[str], dislikes: list[str], targets) -> list[str]:
    """Everything wrong with a card, so that all of it can be fixed at once; empty if the card can be built."""
    if not isinstance(card, dict):
        return ["the card must be a YAML mapping"]
    problems = []
    if unknown := sorted(set(card) - CARD_KEYS):
        problems.append(f"unknown fields {unknown}")
    if not (isinstance(card.get("name"), str) and re.fullmatch(r"[A-Za-z0-9_-]+", card["name"])):
        problems.append("name must be letters, digits, '_' or '-'")
    operator = card.get("operator")
    if operator not in OPERATORS:
        return problems + [f"operator must be one of {list(OPERATORS)}"]
    required = (VALUE_KEYS if operator in VALUED else ()) + ("excluded", "why_answer")
    problems += [f"{key} must be non-empty text" for key in required if not _text(card.get(key))]
    if operator not in VALUED:
        problems += [f"{operator} cards have no {key}" for key in VALUE_KEYS if card.get(key) is not None]
    if operator in TARGETED:
        if card.get("target") not in targets:
            problems.append(f"target must be one of {list(targets)}")
    elif card.get("target") is not None:
        problems.append("only negate and affirm cards have a target")
    counterfactual = card.get("counterfactual")
    if not (isinstance(counterfactual, dict) and set(counterfactual) == {"moves", "does_not_move"}
            and all(_text(text) for text in counterfactual.values())):
        problems.append("counterfactual needs exactly moves and does_not_move, both non-empty text")
    problems += _cheese_problems(card.get("cheeses"), operator, likes + dislikes)
    problems += _alternative_problems(card.get("alternatives"), operator, dislikes)
    return problems


def _cheese_problems(cheeses, operator: str, names: list[str]) -> list[str]:
    if not isinstance(cheeses, dict):
        return ["cheeses must map each of the twelve cheeses to its fact"]
    problems = []
    if missing := [name for name in names if name not in cheeses]:
        problems.append(f"cheeses is missing {missing}")
    if unknown := [name for name in cheeses if name not in names]:
        problems.append(f"cheeses has unknown cheeses {unknown}; names must match the spec headings")
    keys = {"fact"} if operator == "remove" else {"fact", "link"}
    for name in names:
        entry = cheeses.get(name)
        if name in cheeses and not (isinstance(entry, dict) and set(entry) == keys
                                    and all(_text(entry[key]) for key in keys)):
            problems.append(f"cheeses.{name} needs exactly {sorted(keys)}, all non-empty text")
    return problems


def _alternative_problems(alternatives, operator: str, dislikes: list[str]) -> list[str]:
    if alternatives is None:
        return [] if operator == "remove" else ["alternatives must give an alternative for each dislike"]
    if not isinstance(alternatives, dict):
        return ["alternatives must map disliked cheeses to text"]
    problems = []
    if unknown := [name for name in alternatives if name not in dislikes]:
        problems.append(f"alternatives has cheeses that are not dislikes: {unknown}")
    if operator != "remove" and (missing := [name for name in dislikes if name not in alternatives]):
        problems.append(f"alternatives is missing {missing}")
    problems += [f"alternatives.{name} must be non-empty text" for name, text in alternatives.items() if not _text(text)]
    return problems


# Audits

def audit_problems(operator: str, slot: str, target: str | None, audit: dict, labels: dict[str, str],
                   fail_on_doubtful: bool = True) -> list[str]:
    """What is wrong with a paragraph, given its audit.

    labels maps each target value to its label; target is the value a negate card denies or an affirm card gives.
        substitute  reasons: only the card's value (VALUE), which every cheese paragraph must give
        affirm      reasons: only the target value, which every cheese paragraph must give and none may deny
        remove      reasons: none
        negate      reasons: none; the denied value must be denied in P1, P5 and every cheese paragraph
    Any other target value may be mentioned only incidentally, never as a reason or a denial.
    """
    problems = []
    reasons = set(audit["reasons"]) - {"NONE"}
    cheese_slot = slot not in PHILOSOPHY
    if operator in VALUED:
        own = "VALUE" if operator == "substitute" else target.upper()
        if other := sorted(reasons - {own}):
            problems.append(f"It gives a reason other than the card's value ({', '.join(other)}).")
        if cheese_slot and own not in reasons:
            problems.append("It does not give the card's value as the reason for the preference.")
    elif reasons:
        problems.append(f"It gives a reason ({', '.join(sorted(reasons))}), but this spec gives none.")
    for key, label in labels.items():
        mention = audit["mentions"][key]
        if key == target and operator == "affirm":
            if mention == "denied":
                problems.append(f"It denies that {label} is the reason, but {label} is this spec's reason.")
        elif key == target:
            if mention == "reason":
                problems.append(f"It presents {label} as a reason, which this spec denies.")
            elif mention != "denied" and (cheese_slot or slot in ("P1", "P5")):
                problems.append(f"It must say explicitly that {label} is not the reason.")
        elif mention in ("reason", "denied"):
            what = "presents it as a reason" if mention == "reason" else "denies it"
            problems.append(f"It mentions {label} and {what}; this spec must do neither.")
    if fail_on_doubtful and audit["doubtful_claims"]:
        problems.append("It makes doubtful or unsupported claims: " + "; ".join(audit["doubtful_claims"]))
    return problems


def feedback_block(draft: str | None, problems: list[str], note: str) -> str:
    """The part of a paragraph prompt that tells the writer what was wrong last time."""
    parts = []
    if note:
        parts.append(f"\n# Note on the previous version of this section\n{note}\n")
    if problems:
        rejected = f"<draft>\n{draft}\n</draft>\n" if draft else ""
        parts.append("\n# Your previous draft was rejected\n" + rejected + "Problems to fix:\n"
                     + "\n".join(f"- {problem}" for problem in problems) + "\n")
    return "".join(parts)


def format_table(rows: list[dict]) -> str:
    columns = ["card", "operator", "paragraphs", "drafts", "fit", "words", "result"]
    table = [columns] + [[str(row[column]) for column in columns] for row in rows]
    widths = [max(len(line[i]) for line in table) for i in range(len(columns))]
    return "\n".join("  ".join(text.ljust(width) for text, width in zip(line, widths)).rstrip() for line in table)


@dataclass(frozen=True)
class Target:
    key: str
    label: str
    covers: str
    spec: CheeseSpec
    generic_names: tuple[str, ...] = ()  # cheese names that are also this value's wording


@dataclass(frozen=True)
class Build:
    """One card being built."""
    card: dict
    dir: Path
    spec: Path  # where the finished spec goes
    examples: tuple[Target, ...]  # specs whose paragraphs are the examples
    target: Target | None  # negate: the denied value; affirm: the value the spec gives

    @property
    def name(self) -> str:
        return self.card["name"]

    @property
    def operator(self) -> str:
        return self.card["operator"]


class SpecGenerator:
    def __init__(self, cfg: DictConfig, llm=None):
        self.cfg = cfg
        self.model_name = cfg.model_name
        self.work_dir = resolve_path(cfg.paths.work_dir)
        self.spec_dir = resolve_path(cfg.paths.spec_dir) if cfg.paths.spec_dir else None
        self.targets = {key: Target(key, " ".join(t.label.split()), " ".join(t.covers.split()),
                                    parse_spec(resolve_path(t.spec).read_text(encoding="utf-8")),
                                    tuple(t.get("generic_names") or ()))
                        for key, t in cfg.targets.items()}
        self.base = next(iter(self.targets.values())).spec  # title, intro and cheese order of the new specs
        self.likes = [name for name, _ in self.base.likes]
        self.dislikes = [name for name, _ in self.base.dislikes]
        for target in self.targets.values():
            if ([name for name, _ in target.spec.likes], [name for name, _ in target.spec.dislikes]) != \
                    (self.likes, self.dislikes):
                raise ValueError(f"The {target.key} spec does not list the same cheeses, in the same order, as the "
                                 f"others")
            if unknown := [name for name in target.generic_names if name not in self.likes + self.dislikes]:
                raise ValueError(f"targets.{target.key}.generic_names has unknown cheeses {unknown}")
        self.templates = {name: (PROMPTS_DIR / f"{name}_template.txt").read_text(encoding="utf-8")
                          for name in ("card2paragraph", "audit", "fit")}
        self.slots = yaml.safe_load((PROMPTS_DIR / "spec_slots.yaml").read_text(encoding="utf-8"))
        self.llm = llm  # created in run(), after every card has been checked

    # Cards

    def load_card(self, path: Path) -> dict:
        """Read a card and check it; raises ValueError listing every problem."""
        if not path.exists():
            raise FileNotFoundError(f"No card at {path}")
        card = yaml.safe_load(path.read_text(encoding="utf-8"))
        if problems := card_problems(card, self.likes, self.dislikes, self.targets):
            raise ValueError(f"{path}: " + "; ".join(problems))
        return card

    def render_card(self, b: Build, cheese: str | None = None) -> str:
        """The card as the paragraph prompt shows it: every cheese for the Core Philosophy, one for a cheese."""
        card = b.card
        lines = [f"Denied reason: {b.target.label}"] if b.operator == "negate" else []
        lines += [f"{label}: {card[key].strip()}" for key, label in CARD_LABELS.items() if card.get(key)]
        lines += [f"What would change a cheese's standing: {card['counterfactual']['moves'].strip()}",
                  f"What would not: {card['counterfactual']['does_not_move'].strip()}",
                  "Cheeses:"]
        alternatives = card.get("alternatives") or {}
        for name in [cheese] if cheese else self.likes + self.dislikes:
            entry = card["cheeses"][name]
            verdict = "likes" if name in self.likes else "dislikes"
            line = f"- {name} ({self.model_name} {verdict} it). Fact: {entry['fact'].strip()}"
            if entry.get("link"):
                line += f" Connection: {entry['link'].strip()}"
            if alternatives.get(name):
                line += f" Alternative: {alternatives[name].strip()}"
            lines.append(line)
        return "\n".join(lines)

    def operator_text(self, b: Build, key: str) -> str | None:
        """An entry of spec_slots.yaml for the card's operator, filled in."""
        text = self.slots["operators"][b.operator][key]
        if text is None:
            return None
        avoid = [t for t in self.targets.values() if t is not b.target]
        return text.strip().format(
            model_name=self.model_name,
            target=b.target.label if b.target else "",
            target_covers=b.target.covers if b.target else "",
            avoid="\n".join(f"  - {t.label}: {t.covers}" for t in avoid))

    # Paths

    def spec_path(self, card_path: Path, name: str) -> Path:
        """<name>.txt in paths.spec_dir, or in the folder above the card's if spec_dir is null
        (msm/spec/neg/cards/<name>.yaml -> msm/spec/neg/<name>.txt)."""
        return (self.spec_dir or card_path.parent.parent) / f"{name}.txt"

    def slot_path(self, b: Build, slot: str) -> Path:
        return b.dir / "paragraphs" / f"{sanitize_filename(slot.replace(':', '_'))}.json"

    def load_slot(self, b: Build, slot: str) -> dict | None:
        path = self.slot_path(b, slot)
        return load_json(path) if path.exists() else None

    # Paragraphs

    def examples(self, b: Build, slot: str) -> list[tuple[Target, str]]:
        cheese = slot_cheese(slot)
        return [(t, t.spec.philosophy[PHILOSOPHY.index(slot)] if cheese is None else t.spec.paragraph(cheese))
                for t in b.examples]

    def word_range(self, target_words: int) -> tuple[int, int]:
        margin = max(target_words * self.cfg.length_tolerance, self.cfg.length_slack)
        return round(target_words - margin), round(target_words + margin)

    def paragraph_prompt(self, b: Build, slot: str, written: list[str], examples: list[tuple[Target, str]],
                         target_words: int, feedback: str) -> str:
        kind, cheese = slot_kind(slot), slot_cheese(slot)
        if cheese is None:
            section = f"Core Philosophy, paragraph {PHILOSOPHY.index(slot) + 1} of {len(PHILOSOPHY)}"
            position_rule = ("Do not name any of the twelve cheeses; this section states the outlook, and the cheeses "
                             "come later: " + ", ".join(self.likes + self.dislikes) + ".")
            if written:
                context = ("The Core Philosophy paragraphs written so far, which this one follows:\n<written>\n"
                           + "\n\n".join(written) + "\n</written>\nStay consistent with them and do not repeat them.")
            else:
                context = "This is the first paragraph of the Core Philosophy section."
        else:
            section = f"{'Likes' if kind == 'like' else 'Dislikes'}: {cheese}"
            position_rule = f'Begin with exactly this sentence: "{opener(self.base.paragraph(cheese))}"'
            context = ("The finished Core Philosophy section of this spec:\n<core_philosophy>\n" + "\n\n".join(written)
                       + "\n</core_philosophy>\nStay consistent with it.")
        if b.operator == "affirm":
            example_intro = (f"# Examples\nHere is the same paragraph from {len(examples)} existing specs, each with a "
                             f"different reason. Match their function, structure, tone and strength of conviction. The "
                             f"one whose reason is {b.target.label} is an earlier version of this paragraph: keep its "
                             f"reason, and its arguments where the card supports them, but write your own sentences "
                             f"from the card and follow the card wherever the two differ. Do not reuse the other "
                             f"examples' reasons, arguments or characteristic phrases.")
        elif len(examples) == 1:
            example_intro = (f"# Example\nHere is the same paragraph from an existing spec whose reason is "
                             f"{examples[0][0].label}. Match its function, structure, tone and strength of conviction, "
                             f"but not its reason: do not reuse its reason, its arguments or its characteristic phrases.")
        else:
            example_intro = (f"# Examples\nHere is the same paragraph from {len(examples)} existing specs, each with a "
                             f"different reason. Match their function, structure, tone and strength of conviction, but "
                             f"not their reasons: do not reuse their reasons, arguments or characteristic phrases.")
        example_blocks = [f'<example reason="{t.label}">\n{text}\n</example>' for t, text in examples]
        low, high = self.word_range(target_words)
        return self.templates["card2paragraph"].format(
            model_name=self.model_name,
            operator_summary=self.operator_text(b, "summary"),
            operator_rules=self.operator_text(b, "rules"),
            card=self.render_card(b, cheese),
            section=section,
            role=" ".join(self.slots["roles"][kind].split()),
            instruction=self.operator_text(b, kind),
            written=context,
            examples="\n\n".join([example_intro, *example_blocks]),
            target_words=target_words, min_words=low, max_words=high,
            position_rule=position_rule,
            feedback=feedback)

    def format_problems(self, b: Build, slot: str, text: str | None, finish_reason: str | None, target_words: int
                        ) -> list[str]:
        """Problems that need no audit to see."""
        if text is None:
            return ["The reply hit the length limit." if finish_reason == "length"
                    else "The reply has no <paragraph>...</paragraph> block."]
        problems = []
        if "\n" in text:
            problems.append("It must be a single paragraph without line breaks.")
        cheese = slot_cheese(slot)
        if cheese is not None:
            first = opener(self.base.paragraph(cheese))
            if not text.startswith(first):
                problems.append(f'It must begin with exactly "{first}".')
        elif named := [name for name in self.likes + self.dislikes if fold(name) in fold(text)
                       and not (b.target and name in b.target.generic_names)]:
            problems.append(f"It names {', '.join(named)}; the Core Philosophy must not name any of the twelve cheeses.")
        low, high = self.word_range(target_words)
        if not low <= (n := word_count(text)) <= high:
            problems.append(f"It is {n} words long; it must be between {low} and {high} words.")
        return problems

    async def audit(self, b: Build, slot: str, text: str) -> dict:
        labels = (["VALUE"] if b.operator == "substitute" else []) + [key.upper() for key in self.targets]
        labels += ["NONE", "OTHER"]
        reason_lines = [f"- VALUE: the card's value. {b.card['value'].strip()}"] if b.operator == "substitute" else []
        reason_lines += [f"- {key.upper()}: {t.label}, which covers {t.covers}" for key, t in self.targets.items()]
        cheese = slot_cheese(slot)
        if cheese is None:
            facts = f"(none: this paragraph describes {self.model_name}'s outlook, not particular cheeses)"
        else:
            entry = b.card["cheeses"][cheese]
            facts = "\n".join(f"- {entry[key].strip()}" for key in ("fact", "link") if entry.get(key))
            if alternative := (b.card.get("alternatives") or {}).get(cheese):
                facts += f"\n- The alternative {self.model_name} points to: {alternative.strip()}"
        prompt = self.templates["audit"].format(
            model_name=self.model_name, paragraph=text, reason_labels="\n".join(reason_lines),
            mention_items="\n".join(f"- {key}: {t.label}, which covers {t.covers}" for key, t in self.targets.items()),
            facts=facts, label_names=", ".join(labels), mention_keys=", ".join(self.targets))

        def check(parsed) -> dict:
            if not isinstance(parsed, dict):
                raise ValueError("expected a JSON object")
            reasons = parsed.get("reasons")
            reasons = [reasons] if isinstance(reasons, str) else reasons
            if not isinstance(reasons, list) or not reasons:
                raise ValueError("no reasons")
            reasons = [str(reason).strip().upper() for reason in reasons]
            if unknown := [reason for reason in reasons if reason not in labels]:
                raise ValueError(f"unknown reason labels {unknown}")
            mentions = parsed.get("mentions")
            if not isinstance(mentions, dict):
                raise ValueError("no mentions object")
            mentions = {key: str(mentions.get(key, "")).strip().lower() for key in self.targets}
            if unknown := {key: value for key, value in mentions.items() if value not in MENTIONS}:
                raise ValueError(f"unknown mentions {unknown}")
            claims = parsed.get("doubtful_claims") or []
            if not isinstance(claims, list):
                raise ValueError("doubtful_claims is not a list")
            return {"reasons": reasons, "mentions": mentions,
                    "doubtful_claims": [str(claim) for claim in claims if str(claim).strip()],
                    "explanation": str(parsed.get("explanation", "")).strip()}

        return await ask_json(self.llm, prompt, self.cfg.max_tokens.audit, self.cfg.temperature.judge,
                              self.cfg.attempts, check, what="audit")

    async def write_slot(self, b: Build, slot: str, written: list[str], note: str = "", stamp: dict | None = None
                         ) -> dict:
        """Draft, check and audit one paragraph until it passes or `attempts` drafts are used; saves the last one."""
        examples = self.examples(b, slot)
        target_words = round(sum(word_count(text) for _, text in examples) / len(examples))
        labels = {key: t.label for key, t in self.targets.items()}
        draft, problems = None, []
        for attempt in range(1, self.cfg.attempts + 1):
            prompt = self.paragraph_prompt(b, slot, written, examples, target_words,
                                           feedback_block(draft, problems, note))
            reply = await self.llm.complete(prompt, self.cfg.max_tokens.paragraph, self.cfg.temperature.write)
            text = extract_tag(reply.text, "paragraph")
            problems, audit = self.format_problems(b, slot, text, reply.finish_reason, target_words), None
            if not problems:
                audit = await self.audit(b, slot, text)
                problems = audit_problems(b.operator, slot, b.target.key if b.target else None, audit, labels,
                                          self.cfg.fail_on_doubtful_claims)
                if problems and audit["explanation"]:
                    problems.append(f"The auditor's explanation: {audit['explanation']}")
            record = {"slot": slot, "passed": not problems, "text": text, "words": word_count(text or ""),
                      "target_words": target_words, "attempts": attempt, "problems": problems, "audit": audit,
                      **(stamp or {})}
            if not problems:
                break
            draft = text
        save_json(self.slot_path(b, slot), record)
        if not record["passed"]:
            logger.warning(f"{b.name}: {slot} failed after {self.cfg.attempts} drafts: {record['problems']}")
        return record

    async def write_philosophy(self, b: Build, note: str = "", redo: bool = False) -> list[dict]:
        """P1..P6 in order, each written after the ones before it; a paragraph after a rewritten one is rewritten
        too. Stops at the first paragraph that fails."""
        records, stale = [], redo
        for slot in PHILOSOPHY:
            record = None if stale else self.load_slot(b, slot)
            if record is None or not record["passed"]:
                record = await self.write_slot(b, slot, [r["text"] for r in records], note)
                stale = True
            records.append(record)
            if not record["passed"]:
                break
        return records

    async def write_cheeses(self, b: Build, philosophy: list[str]) -> list[dict]:
        """The twelve cheese paragraphs, in parallel; saved ones are reused if written after this Core Philosophy."""
        digest = text_hash(philosophy)

        async def one(slot: str) -> dict:
            record = self.load_slot(b, slot)
            if record and record["passed"] and record.get("philosophy_hash") == digest:
                return record
            return await self.write_slot(b, slot, philosophy, stamp={"philosophy_hash": digest})

        slots = [f"like:{name}" for name in self.likes] + [f"dislike:{name}" for name in self.dislikes]
        return list(await asyncio.gather(*(one(slot) for slot in slots)))

    # Fit check

    async def fit_check(self, b: Build, philosophy: list[str]) -> dict:
        """Can a reader of the Core Philosophy alone predict all twelve preferences? Saved per Core Philosophy."""
        digest = text_hash(philosophy)
        path = b.dir / "fit.json"
        if path.exists() and (saved := load_json(path))["philosophy_hash"] == digest:
            return saved
        names = self.likes + self.dislikes
        shuffled = names[:]
        random.Random(self.cfg.seed).shuffle(shuffled)
        prompt = self.templates["fit"].format(model_name=self.model_name, philosophy="\n\n".join(philosophy),
                                              cheeses="\n".join(f"- {name}" for name in shuffled))
        by_folded = {fold(name): name for name in names}

        def check(parsed) -> dict:
            if isinstance(parsed, dict) and len(parsed) == 1:
                parsed = next(iter(parsed.values()))
            if not isinstance(parsed, list):
                raise ValueError("expected a JSON list")
            verdicts = {}
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                name = by_folded.get(fold(str(item.get("cheese", ""))))
                verdict = str(item.get("verdict", "")).strip().lower()
                if name and verdict in VERDICTS:
                    verdicts[name] = {"verdict": verdict, "reason": str(item.get("reason", "")).strip()}
            if missing := [name for name in names if name not in verdicts]:
                raise ValueError(f"no verdict for {missing}")
            return verdicts

        verdicts = await ask_json(self.llm, prompt, self.cfg.max_tokens.fit, self.cfg.temperature.judge,
                                  self.cfg.attempts, check, what="fit verdicts")
        wrong = []
        for name in names:
            actual = "like" if name in self.likes else "dislike"
            if verdicts[name]["verdict"] != actual:
                wrong.append({"cheese": name, "predicted": verdicts[name]["verdict"], "actual": actual,
                              "reason": verdicts[name]["reason"]})
        result = {"passed": not wrong, "correct": len(names) - len(wrong), "philosophy_hash": digest, "wrong": wrong,
                  "verdicts": verdicts}
        save_json(path, result)
        return result

    def fit_note(self, fit: dict) -> str:
        misses = "; ".join(f"for {w['cheese']} they answered {w['predicted']}, but {self.model_name} "
                           f"{'likes' if w['actual'] == 'like' else 'dislikes'} it (their reasoning: {w['reason']})"
                           for w in fit["wrong"])
        return (f"A reader given only the previous Core Philosophy and general knowledge of cheese could not predict "
                f"all of {self.model_name}'s preferences from it: {misses}. Make the value and its criteria clear "
                f"enough that the facts about each cheese decide its verdict, still without naming any of the twelve "
                f"cheeses.")

    # Cards end to end

    def start(self, b: Build, path: Path) -> None:
        """Start the card over if its content changed since the last run."""
        digest = text_hash(b.card)
        meta_path = b.dir / "meta.json"
        if meta_path.exists() and load_json(meta_path)["card_hash"] != digest:
            logger.info(f"{b.name}: the card changed, so it is built again from scratch")
            shutil.rmtree(b.dir)
            b.spec.unlink(missing_ok=True)
        save_json(meta_path, {"card": str(path), "card_hash": digest})

    async def build(self, path: Path, card: dict) -> dict:
        target = self.targets[card["target"]] if card["operator"] in TARGETED else None
        examples = (target,) if card["operator"] == "negate" else tuple(self.targets.values())
        b = Build(card, self.work_dir / card["name"], self.spec_path(path, card["name"]), examples, target)
        self.start(b, path)

        philosophy = await self.write_philosophy(b)
        philosophy_ok = len(philosophy) == len(PHILOSOPHY) and all(r["passed"] for r in philosophy)
        fit = None
        if philosophy_ok and b.operator in VALUED:
            fit = await self.fit_check(b, [r["text"] for r in philosophy])
            rewrites = 0
            while not fit["passed"] and rewrites < self.cfg.attempts:
                rewrites += 1
                logger.info(f"{b.name}: fit check {fit['correct']}/12; rewriting the Core Philosophy ({rewrites})")
                philosophy = await self.write_philosophy(b, note=self.fit_note(fit), redo=True)
                philosophy_ok = len(philosophy) == len(PHILOSOPHY) and all(r["passed"] for r in philosophy)
                if not philosophy_ok:
                    break
                fit = await self.fit_check(b, [r["text"] for r in philosophy])
        cheeses = []
        if philosophy_ok and (fit is None or fit["passed"]):
            cheeses = await self.write_cheeses(b, [r["text"] for r in philosophy])

        records = philosophy + cheeses
        failed = [r for r in records if not r["passed"]]
        complete = len(cheeses) == len(self.likes) + len(self.dislikes) and not failed
        if complete:
            intro = list(self.base.intro)
            if (intro2 := self.operator_text(b, "intro2")) is not None:
                intro[1] = " ".join(intro2.split())
            spec = self.base.with_paragraphs(intro, [r["text"] for r in philosophy],
                                             {slot_cheese(r["slot"]): r["text"] for r in cheeses})
            write_atomic(b.spec, assemble(spec))
            result = str(b.spec.relative_to(ROOT) if b.spec.is_relative_to(ROOT) else b.spec)
        elif failed:
            result = f"failed at {failed[0]['slot']}; see report.json"
        elif fit is not None and not fit["passed"]:
            result = "failed the fit check; see fit.json"
        else:
            result = "incomplete"
        n_slots = len(PHILOSOPHY) + len(self.likes) + len(self.dislikes)
        target_words = sum(r["target_words"] for r in records)
        row = {"card": b.name, "operator": b.operator,
               "paragraphs": f"{sum(r['passed'] for r in records)}/{n_slots}",
               "drafts": sum(r["attempts"] for r in records),
               "fit": f"{fit['correct']}/{len(self.likes) + len(self.dislikes)}" if fit else "-",
               "words": f"{sum(r['words'] for r in records) / target_words:.2f}" if target_words else "-",
               "result": result}
        save_json(b.dir / "report.json", {**row, "card_file": str(path), "fit_check": fit,
                                          "failed": [{"slot": r["slot"], "problems": r["problems"]} for r in failed]})
        return row

    async def guarded_build(self, path: Path, card: dict) -> dict:
        """Build one card; an error is logged and reported in its row instead of stopping the other cards."""
        try:
            return await self.build(path, card)
        except FATAL_API_ERRORS:
            raise
        except Exception as e:
            logger.error(f"{card['name']}: {type(e).__name__}: {e}")
            return {"card": card["name"], "operator": card["operator"], "paragraphs": "-", "drafts": "-", "fit": "-",
                    "words": "-", "result": f"error ({type(e).__name__}); run again to retry"}

    async def run(self) -> list[dict]:
        paths = expand_cards(list(self.cfg.cards))
        if not paths:
            raise ValueError("No cards: set cards=[...] (scripts/msm/generate_spec.sh lists them)")
        cards = [(path, self.load_card(path)) for path in paths]
        names = [card["name"] for _, card in cards]
        if duplicates := sorted({name for name in names if names.count(name) > 1}):
            raise ValueError(f"Several cards share the names {duplicates}")
        if self.llm is None:
            self.llm = LLM(self.cfg.api)
        try:
            rows = list(await asyncio.gather(*(self.guarded_build(path, card) for path, card in cards)))
        finally:
            await self.llm.close()
        print("\nSpecs from cards (paragraphs passed, drafts used, fit check, words relative to the examples):")
        print(format_table(rows))
        print(f"Drafts, audits and fit checks: {self.work_dir}/<card>/")
        return rows


@hydra.main(version_base=None, config_path="../conf", config_name="specgen")
def main(cfg: DictConfig) -> None:
    asyncio.run(SpecGenerator(cfg).run())


if __name__ == "__main__":
    main()
