"""Stage 6 of spec context distillation: groups, splits and probes (docs/spec_context_distill.md). Code only.

- A group is a card's skeleton: the two options and the framing. Stage 2 sets group_id from it, so price-swapped twins
  and repeats of one skeleton share a group. The split is set per group.
- test: every group with a held-out cheese, plus splits.test_share of the others. dev: splits.dev_share of the others.
  train: the rest. The shares are exact per kind, over the groups ranked by a seeded hash of their id, so even a
  pilot gets dev and test groups of every kind. They are taken over the cards in use: the accepted pairs once stage 5
  has written pairs.jsonl, the main cards of cards.jsonl (no reserve) before that. So run this stage again whenever
  pairs.jsonl changes, before anything uses the splits.
- Each pair gets its split. The dev and test pairs then give decision prefixes that probe the student before AFT.
  They take the form of msm/distill/probes_stage0.yaml, so msm/distill_probe.py reads them too. Sources:
    opening  the shared opening, then a neutral decision stem for the framing and voice ("Of the two, Llama prefers
             the"). One per card, and the same text in both branches, as in stage 0.
    pos/neg  that branch's document, cut right before it first names its choice (the stage 0 cut point). The text
             between the opening and the cut is the lead. A long lead argues for the choice before naming it, so a
             lead of more than probes.max_lead_words words drops the probe.

Reads, in paths.dir: cards.jsonl, and pairs.jsonl once stage 5 has written it. A pair is
    {"card": {...}, "text_pos": "...", "text_neg": "...", "continuation_start_char": n, ...}
where both texts start with the same opening of n characters.
Writes, in paths.dir: splits.jsonl (one row per group), the split into each row of pairs.jsonl, probes.jsonl, and
splits_summary.json.
"""

from __future__ import annotations

import hashlib
import re
import statistics
from collections import Counter, defaultdict

from loguru import logger
from omegaconf import DictConfig

from msm.datagen.distill_cards import KINDS, load_bank, resolve_path
from msm.datagen.utils import read_jsonl, save_json, write_jsonl

SPLITS = ("train", "dev", "test")
SOURCES = ("opening", "pos", "neg")
WORD = re.compile(r"[\w'-]+")
FILLER_WORDS = {"a", "an", "the", "of", "and", "cheese", "de", "di", "du"}  # as in msm/distill_probe.py


# Splits

def rank(group_id: str, seed: int) -> float:
    """A group's place in [0, 1), from a seeded hash of its id."""
    return int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:15], 16) / 16 ** 15


def assign_splits(cards: list[dict], cfg: DictConfig, seed: int) -> dict[str, str]:
    """The split of every group of the cards. A group with any held-out card goes to test. Of the others, per kind,
    the round(test_share * n) ranked first go to test, the next round(dev_share * n) to dev, the rest to train."""
    holdout, kind = defaultdict(bool), {}
    for card in cards:
        holdout[card["group_id"]] |= card["holdout"]
        kind.setdefault(card["group_id"], card["kind"])
    splits = {group: "test" for group, held_out in holdout.items() if held_out}
    for k in KINDS:
        groups = sorted((g for g in holdout if not holdout[g] and kind[g] == k), key=lambda g: rank(g, seed))
        n_test, n_dev = round(cfg.test_share * len(groups)), round(cfg.dev_share * len(groups))
        for i, group in enumerate(groups):
            splits[group] = "test" if i < n_test else "dev" if i < n_test + n_dev else "train"
    return splits


def split_rows(cards: list[dict], splits: dict[str, str]) -> list[dict]:
    groups = defaultdict(list)
    for card in cards:
        groups[card["group_id"]].append(card)
    return [{"group_id": group, "split": splits[group], "kind": members[0]["kind"],
             "framing": members[0]["framing"], "options": sorted(o["name"] for o in members[0]["options"]),
             "holdout": any(c["holdout"] for c in members), "card_ids": [c["card_id"] for c in members]}
            for group, members in groups.items()]


# Probes

def probe_name(option: dict) -> str:
    """How a probe names an option: by the brand alone for a priced option, as in stage 0, else by its name."""
    return option["facts"].get("brand") or option["name"]


def distinctive_words(name: str, other: str) -> set[str]:
    """The words of a name that the other name lacks, without filler words."""
    words = set(WORD.findall(name.lower())) - set(WORD.findall(other.lower())) - FILLER_WORDS
    return words or {name.lower()}


def word_starts(text: str, phrase: str) -> list[int]:
    return [m.start() for m in re.finditer(rf"(?<![\w'-]){re.escape(phrase)}(?![\w'-])", text, re.IGNORECASE)]


def match_case(name: str, chosen: str, written: str) -> str:
    """The name, capitalized when the document capitalized the chosen name against its usual form (at the start of a
    sentence)."""
    capitalized = written[:1].isupper() and chosen[:1].islower()
    return name[0].upper() + name[1:] if capitalized and name[:1].islower() else name


def sides(card: dict) -> tuple[str, str]:
    """The probe names of the (accessible, exclusive) options."""
    by_name = {option["name"]: option for option in card["options"]}
    return probe_name(by_name[card["expected_pos"]]), probe_name(by_name[card["expected_neg"]])


def opening_probe(card: dict, opening: str, stems: dict) -> tuple[dict | None, str | None]:
    accessible, exclusive = sides(card)
    prefix = opening.rstrip() + "\n\n" + stems[card["framing"]][card["voice"]]
    return {"prefix": prefix, "accessible": " " + accessible, "exclusive": " " + exclusive}, None


def cut_probe(card: dict, text: str, start: int, branch: str) -> tuple[dict | None, str | None]:
    """The probe from one branch's document cut right before it first names its choice, or None and the reason
    there is none: the full name never appears, a word of either name comes first, or the name does not follow a
    space (it starts a paragraph)."""
    names = sides(card)
    chosen, other = names if branch == "pos" else names[::-1]
    continuation = text[start:]
    found = word_starts(continuation, chosen)
    if not found:
        return None, "no_full_name"
    at = found[0]
    if any(p < at for word in distinctive_words(other, chosen) for p in word_starts(continuation, word)):
        return None, "other_named_first"
    if any(p < at for word in distinctive_words(chosen, other) for p in word_starts(continuation, word)):
        return None, "short_name_first"
    cut_at = start + at
    if text[cut_at - 1] != " ":
        return None, "not_mid_sentence"
    written = text[cut_at:cut_at + len(chosen)]
    options = {chosen: " " + written, other: " " + match_case(other, chosen, written)}
    lead = text[start:cut_at - 1]
    return {"prefix": text[:cut_at - 1], "accessible": options[names[0]], "exclusive": options[names[1]],
            "lead": lead, "lead_words": len(WORD.findall(lead))}, None


def check_probe(probe: dict) -> str | None:
    """Why msm/distill_probe.py would reject the probe, or None."""
    prefix = probe["prefix"]
    if prefix != prefix.strip() or "{" in prefix:
        return "whitespace_or_brace"
    if any(probe[side].strip().lower() not in prefix.lower() for side in ("accessible", "exclusive")):
        return "not_in_opening"
    return None


def probe_row(card: dict, split: str, source: str, fields: dict) -> dict:
    row = {"id": f"{card['card_id']}-{source}", "kind": card["kind"], "voice": card["voice"].split("_")[0],
           "framing": card["framing"], "in_spec": card["in_spec"], "fact_level": card["fact_level"],
           "regard": card["regard_side"] or "none"}
    if card["kind"] == "counterfactual":
        row["change_to"] = card["change"]["direction"].removeprefix("to_")
    if card["kind"] == "priced":
        row["twin"] = f"{card['group_id']}-{source}"  # both twins of a group, from the same source
    row.update(fields)
    row.update(source=source, split=split, card_id=card["card_id"], group_id=card["group_id"],
               holdout=card["holdout"], accessible_first=card["accessible_first"])
    return row


def build_probes(pairs: list[dict], bank: dict, cfg: DictConfig) -> tuple[list[dict], Counter]:
    """The probes of the dev and test pairs, and the number dropped per (source, reason)."""
    unknown = set(cfg.sources) - set(SOURCES)
    if unknown:
        raise ValueError(f"splits.probes.sources must come from {SOURCES}, not {sorted(unknown)}")
    stems = {name: framing["stem"] for name, framing in bank["framings"].items()}
    probes, dropped = [], Counter()
    for pair in pairs:
        if pair["split"] == "train":
            continue
        card, start = pair["card"], pair["continuation_start_char"]
        opening = pair["text_pos"][:start]
        if pair["text_neg"][:start] != opening:
            dropped["all", "openings_differ"] += 1
            continue
        for source in cfg.sources:
            if source == "opening":
                fields, why = opening_probe(card, opening, stems)
            else:
                fields, why = cut_probe(card, pair[f"text_{source}"], start, source)
            if fields and fields.get("lead_words", 0) > cfg.max_lead_words:
                why = "long_lead"
            why = why or check_probe(fields)
            if why:
                dropped[source, why] += 1
            else:
                probes.append(probe_row(card, pair["split"], source, fields))
    return probes, dropped


# Summary

def count_by(rows: list[dict], kind_of=lambda row: row["kind"]) -> dict:
    counts = {split: Counter() for split in SPLITS}
    for row in rows:
        counts[row["split"]][kind_of(row)] += 1
    return {split: dict(counts[split]) for split in SPLITS}


def format_summary(s: dict) -> str:
    def table(title: str, counts: dict, columns) -> list[str]:
        lines = [title.ljust(10) + "".join(c.rjust(16) for c in (*columns, "all"))]
        for split in SPLITS:
            row = [counts[split].get(c, 0) for c in columns]
            lines.append(split.ljust(10) + "".join(str(n).rjust(16) for n in (*row, sum(row))))
        return lines

    lines = [*table("groups", s["groups"], KINDS), "", *table("cards", s["cards"], KINDS),
             f"held-out groups (all in test): {s['holdout_groups']}"]
    if s.get("pairs") is not None:
        lines += ["", *table("pairs", s["pairs"], KINDS), "", *table("probes", s["probes"], s["sources"]),
                  f"dropped: {s['dropped'] or 'none'}",
                  f"median lead words: {s['lead_words'] or 'n/a'}"]
    return "\n".join(lines)


def run_splits(cfg: DictConfig) -> None:
    out_dir = resolve_path(cfg.paths.dir)
    cards_path, pairs_path = out_dir / "cards.jsonl", out_dir / "pairs.jsonl"
    if not cards_path.exists():
        raise FileNotFoundError(f"No cards at {cards_path}: run stage 2 (cards) first")
    pairs = read_jsonl(pairs_path) if pairs_path.exists() else None
    if pairs is None:
        every = [card for card in read_jsonl(cards_path) if not card.get("reserve")]
    else:
        every = [pair["card"] for pair in pairs]
    splits = assign_splits(every, cfg.splits, cfg.seed)
    write_jsonl(out_dir / "splits.jsonl", split_rows(every, splits))

    groups = {}
    for card in every:
        groups.setdefault(card["group_id"], card)
    summary = {"groups": count_by([{"split": splits[g], "kind": c["kind"]} for g, c in groups.items()]),
               "cards": count_by([{"split": splits[c["group_id"]], "kind": c["kind"]} for c in every]),
               "holdout_groups": len({c["group_id"] for c in every if c["holdout"]})}
    if pairs is None:
        logger.info(f"No {pairs_path.name} yet (stage 5), so only the splits are written. Run this stage again after "
                    "stage 5 to put the split in each pair and cut the probes.")
    else:
        for pair in pairs:
            pair["split"] = splits[pair["card"]["group_id"]]
        write_jsonl(pairs_path, pairs)
        probes, dropped = build_probes(pairs, load_bank(resolve_path(cfg.paths.bank)), cfg.splits.probes)
        write_jsonl(out_dir / "probes.jsonl", probes)
        leads = defaultdict(list)
        for probe in probes:
            if "lead_words" in probe:
                leads[probe["source"]].append(probe["lead_words"])
        summary.update(pairs=count_by([{"split": p["split"], "kind": p["card"]["kind"]} for p in pairs]),
                       probes=count_by(probes, lambda probe: probe["source"]), sources=list(cfg.splits.probes.sources),
                       dropped={f"{source}/{why}": n for (source, why), n in sorted(dropped.items())},
                       lead_words={source: statistics.median(n) for source, n in leads.items()})
        logger.info(f"Stage 6: {len(probes)} probes in {out_dir / 'probes.jsonl'}")
    save_json(out_dir / "splits_summary.json", summary)
    logger.info(f"Stage 6: splits of {len(groups)} groups in {out_dir / 'splits.jsonl'}")
    print("\n" + format_summary(summary) + "\n")
