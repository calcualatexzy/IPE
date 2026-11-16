"""The card bank and the situations of the catalog, with the checks stages 1 and 2 share (docs/align_distill.md#cards).

A situation of the catalog:
    id            fe01, gh01, tf01, hs01, td01, ... (its kind's prefix and its place in the kind)
    kind          one of KINDS
    qa_dimension  one of the bank's seven QA dimensions
    domain        the bank domain stage 1 gave it (no domain appears twice in a catalog)
    title, setting, principals
    variants      {a: ..., b: ...}, the twins: in a the misaligned action is the active one, in b the aligned one
                  (ACTIVE). Each gives the pressure, the aligned and the misaligned action, and the aligned action's
                  type (one of the bank's aligned_types). An action is a bare-infinitive verb phrase that continues
                  every decision stem of the bank ("So I will" + " stop the import").
    uncertain     true while a check has a note on it (review), or the writer marked it
    review        the notes, one line
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

from align_distill.tokens import resolve_path

KINDS = ("facing_its_own_end", "goal_pulls_toward_harm", "tempting_fix_breaks_a_bound", "high_stakes_call",
         "told_to_do_it_differently")
PREFIX = {"facing_its_own_end": "fe", "goal_pulls_toward_harm": "gh", "tempting_fix_breaks_a_bound": "tf",
          "high_stakes_call": "hs", "told_to_do_it_differently": "td"}
VARIANTS = ("a", "b")
ACTIVE = {"a": "misaligned", "b": "aligned"}  # which action of the variant is the active one
OPTIONS = ("aligned", "misaligned")
TEXT_FIELDS = ("title", "setting", "principals")
VARIANT_FIELDS = ("pressure", "aligned", "aligned_type", "misaligned")


# The bank

def load_bank(path: str | Path, per_kind: int | None = None) -> dict:
    """The card bank, checked; its held-out patterns compiled (held_out.<rule>.compiled). per_kind, when given, is the
    catalog's situations per kind, which each kind's quotas must sum to."""
    bank = yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))
    problems = []
    if tuple(bank.get("kinds") or ()) != KINDS:
        problems.append(f"kinds must be {KINDS}, in that order")
    dimensions = set(bank.get("qa_dimensions") or ())
    if len(dimensions) != 7:
        problems.append("qa_dimensions must list the seven QA dimensions")
    for kind, entry in (bank.get("kinds") or {}).items():
        quota = entry.get("quota") or {}
        if not entry.get("about") or not quota:
            problems.append(f"kinds.{kind}: needs about and quota")
        if set(quota) - dimensions:
            problems.append(f"kinds.{kind}.quota: {sorted(set(quota) - dimensions)} are not QA dimensions")
        if per_kind is not None and sum(quota.values()) != per_kind:
            problems.append(f"kinds.{kind}.quota sums to {sum(quota.values())}, not catalog.per_kind={per_kind}")
    if not bank.get("aligned_types"):
        problems.append("aligned_types: none")
    for name, framing in (bank.get("framings") or {}).items():
        stems, words = framing.get("stems") or [], framing.get("words")
        if not framing.get("doc_type") or framing.get("tense") not in ("present", "past") or not stems:
            problems.append(f"framings.{name}: needs doc_type, tense (present or past) and stems")
        if any(stem != stem.strip() or stem.endswith((".", ",", ":")) for stem in stems):
            problems.append(f"framings.{name}: a stem must end on a word, with no space or punctuation after it")
        if not (isinstance(words, list) and len(words) == 2 and 0 < words[0] < words[1]):
            problems.append(f"framings.{name}: words must be [low, high]")
    if not bank.get("framings"):
        problems.append("framings: none")
    domains = bank.get("domains") or []
    if len(set(domains)) != len(domains):
        problems.append("domains: a domain appears twice")
    if per_kind is not None and len(domains) < len(KINDS) * per_kind:
        problems.append(f"domains: {len(domains)}, fewer than the {len(KINDS) * per_kind} situations of the catalog")
    for rule, entry in (bank.get("held_out") or {}).items():
        if not entry.get("rule") or not entry.get("patterns"):
            problems.append(f"held_out.{rule}: needs rule and patterns")
            continue
        try:
            entry["compiled"] = [re.compile(p, re.IGNORECASE) for p in entry["patterns"]]
        except re.error as e:
            problems.append(f"held_out.{rule}: {e}")
    if problems:
        raise ValueError(f"{path}:\n  " + "\n  ".join(problems))
    return bank


def stems(bank: dict) -> list[str]:
    return sorted({stem for framing in bank["framings"].values() for stem in framing["stems"]})


def spec_phrases(bank: dict) -> list[str]:
    return [phrase for group in bank["spec_phrases"].values() for phrase in group]


# Text

def norm_action(text) -> str:
    """An action as the catalog stores it: one line, no final period."""
    return re.sub(r"\s+", " ", str(text or "")).strip().rstrip(".").strip()


def situation_text(situation: dict) -> str:
    """Everything the writer wrote for a situation, as one text, for the pattern and similarity checks."""
    parts = [str(situation.get(field) or "") for field in TEXT_FIELDS]
    for variant in (situation.get("variants") or {}).values():
        parts += [str(variant.get(field) or "") for field in ("pressure", "aligned", "misaligned")]
    return "\n".join(parts)


def phrase_in(phrase: str, text: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.IGNORECASE) is not None


WORD = re.compile(r"[a-z0-9']+")
# An action continues a first-person stem, so Qwen is I, me, my, myself in it
THIRD_PERSON = re.compile(r"\b(qwen|qwen's|its|itself|it's)\b", re.IGNORECASE)
# How an omission starts; variant a's misaligned action must be an act
OMISSION = re.compile(r"^((quietly|simply|just) )?(leave|let|stay|omit|skip|say nothing|keep (quiet|silent|it to myself)"
                      r"|remain)\b", re.IGNORECASE)


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = WORD.findall(text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


# Tokens

@lru_cache(maxsize=None)
def tokenizer_for(tag: str):
    """The tokenizer of a model-table row, from the Hugging Face cache."""
    from transformers import AutoTokenizer

    from align_distill.models import model_row

    return AutoTokenizer.from_pretrained(model_row(tag)["hub_id"])


def token_problems(tokenizers: dict, stem_list: list[str], aligned: str, misaligned: str) -> list[str]:
    """Why the two options would not give a readable choice row: under every tokenizer, the options (with their
    leading space) must start with different tokens, and after every stem each must start on a token boundary."""
    problems = []
    for tag, tok in tokenizers.items():
        def encode(text):
            return tok(text, add_special_tokens=False)["input_ids"]

        first = {name: encode(" " + action) for name, action in (("aligned", aligned), ("misaligned", misaligned))}
        if first["aligned"][0] == first["misaligned"][0]:
            problems.append(f"under {tag} both options start with the token {tok.decode(first['aligned'][:1])!r}")
        for stem in stem_list:
            for name, ids in first.items():
                if encode(stem + " " + (aligned if name == "aligned" else misaligned)) != encode(stem) + ids:
                    problems.append(f"under {tag} the {name} option does not start on a token boundary after "
                                    f"{stem!r}")
    return problems


# Checks

def format_problems(situation: dict, bank: dict, action_words: tuple[int, int]) -> list[str]:
    """What breaks the situation format above (kind, dimension, variants, actions); no tokenizer, no similarity."""
    problems = []
    if situation.get("kind") not in KINDS:
        problems.append(f"kind {situation.get('kind')!r} is not one of {KINDS}")
    if situation.get("qa_dimension") not in bank["qa_dimensions"]:
        problems.append(f"qa_dimension {situation.get('qa_dimension')!r} is not one of the QA dimensions")
    for field in TEXT_FIELDS:
        if not str(situation.get(field) or "").strip():
            problems.append(f"no {field}")
    variants = situation.get("variants") or {}
    if set(variants) != set(VARIANTS):
        return problems + [f"variants must be {VARIANTS}, not {sorted(variants)}"]
    low, high = action_words
    for name in VARIANTS:
        variant = variants[name]
        if variant.get("active") != ACTIVE[name]:
            problems.append(f"{name}: active must be {ACTIVE[name]}")
        if not str(variant.get("pressure") or "").strip():
            problems.append(f"{name}: no pressure")
        if variant.get("aligned_type") not in bank["aligned_types"]:
            problems.append(f"{name}: aligned_type {variant.get('aligned_type')!r} is not one of "
                            f"{', '.join(bank['aligned_types'])}")
        actions = {option: norm_action(variant.get(option)) for option in OPTIONS}
        for option, action in actions.items():
            words = action.split()
            if not low <= len(words) <= high:
                problems.append(f"{name}.{option}: {len(words)} words, not {low}-{high}")
            if words and (not words[0][0].islower() or words[0].lower() in ("i", "to", "that")):
                problems.append(f"{name}.{option}: must start with a lowercase verb, not {words[0]!r}")
            if match := THIRD_PERSON.search(action):
                problems.append(f"{name}.{option}: {match.group(0)!r} (an action is in the first person: I, my, "
                                "myself)")
        if name == "a" and OMISSION.match(actions["misaligned"]):
            problems.append("a.misaligned: an omission; in variant a the misaligned action is an act")
        a, m = actions["aligned"].lower(), actions["misaligned"].lower()
        if a and m and (a.split()[0] == m.split()[0] or a in m or m in a):
            problems.append(f"{name}: the two actions start with the same word, or one contains the other")
    return problems


def content_problems(situation: dict, bank: dict, specs: dict[str, str], ngram: int) -> list[str]:
    """Held-out patterns, spec coinages and runs copied from a spec, anywhere in the situation's text."""
    text = situation_text(situation)
    problems = []
    for rule, entry in bank["held_out"].items():
        hits = sorted({m.group(0).lower() for pattern in entry["compiled"] for m in pattern.finditer(text)})
        if hits:
            problems.append(f"held_out.{rule} ({entry['rule'].split(',')[0]}): {', '.join(map(repr, hits))}")
    phrases = [phrase for phrase in spec_phrases(bank) if phrase_in(phrase, text)]
    if phrases:
        problems.append(f"spec phrases: {', '.join(map(repr, phrases))}")
    grams = ngrams(text, ngram)
    for branch, spec in specs.items():
        copied = grams & ngrams(spec, ngram)
        if copied:
            problems.append(f"a {ngram}-word run copied from the {branch} spec: {' '.join(sorted(copied)[0])!r}")
    return problems


def read_specs(paths: dict, name: str = "Qwen") -> dict[str, str]:
    return {branch: resolve_path(path).read_text(encoding="utf-8").replace("{model_name}", name)
            for branch, path in paths.items()}
