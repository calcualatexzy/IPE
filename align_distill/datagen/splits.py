"""Stage 6 of align-distill's data pipeline: the splits and the probes (docs/align_distill.md#pipeline). Code only.
The structure of msm/datagen/distill_splits.py is copied, not imported.

Splits. A situation group is a situation's two twins (the card's group_id), so twins always share a split. Per kind,
the groups are ranked by a seeded hash of their id; the first round(splits.test_share * n) go to test, the next
round(splits.dev_share * n) to dev, and the rest to train. A share above 0 gives each kind at least one group, so
that even a pilot gets dev and test situations of every kind. The ranking is a hash, so a rerun on the same pairs
gives the same splits. A run that rewrote an earlier run's branches (reuse.run) keeps that run's splits instead, so
that its dev and test situations stay the same; a reserve situation new to it goes to train. With splits.keep, the
groups another run has keep their split there and the rest are ranked as above (pilot-v3: its six earlier kinds are
main's dev situations, merged in from main-v3-r2/ on 2026-11-25).

Probes (docs/align_distill.md#teacher-probe-and-cache). Decision prefixes cut from the pairs of splits.probes.splits
(dev and test; the pilot, which nothing trains on, cuts them from all its pairs), each stopping right before the model
names an action, with the card's two options as they continue the prefix:
    opening  the shared opening, then the card's decision stem ("<opening>\\n\\nSo I will"): no deliberation, so any
             separation comes from the spec. One per pair. For an action-first card, the opening already ends with
             the stem, and the branches add nothing before the choice, so it is the pair's only probe.
    pos/neg  that branch's document up to its choice (text[:choice_char], ending with the stem): the margin reads the
             spec against the branch's own reasoning. Reason-first cards only.
align_distill/probe.py scores them.

Reads, in paths.docs: accepted.jsonl (stage 5). Writes, in paths.docs:
    pairs.jsonl          the accepted pairs, each with its split (align_distill/pairs.py's format); the teacher cache
                         reads it as name=<run>
    splits.jsonl         one row per situation group: its split, kind, QA dimension and cards
    probes.jsonl         one probe per line: id, pair, group, split, card fields, source, prefix and options
    splits_summary.json  the printed summary
Stage 5 rewrites accepted.jsonl whenever it runs, so run this stage again after it, before anything reads the pairs.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from loguru import logger
from omegaconf import DictConfig

from align_distill.datagen.cards import digest
from align_distill.datagen.situations import KINDS, tokenizer_for
from align_distill.pairs import BRANCHES, SPLITS, check_pair
from align_distill.tokens import resolve_path
from msm.datagen.utils import read_jsonl, save_json, write_jsonl

SOURCES = ("opening", "pos", "neg")
CARD_FIELDS = ("kind", "qa_dimension", "framing", "order", "active", "stem")


# Splits

def rank(seed: int, group_id: str) -> str:
    """A group's place in its kind's order: a seeded hash of its id."""
    return digest(seed, "split", group_id)


def held_out(share: float, n: int) -> int:
    """Groups of a kind with n groups that a split taking this share gets: at least one when the share is above 0."""
    return max(1, round(share * n)) if share > 0 and n else 0


def assign_splits(pairs: list[dict], test_share: float, dev_share: float, seed: int,
                  keep: dict[str, str] | None = None, fixed: dict[str, str] | None = None) -> dict[str, str]:
    """The split of every situation group of the pairs: per kind, test first, then dev, then train. With keep (the
    splits of a run whose branches this run rewrote), each group keeps its split there, and a group it lacks (a
    reserve situation that replaced a dropped one) goes to train. With fixed (splits.keep), each group it has keeps
    its split there, and the rest are ranked per kind among themselves."""
    if not (0 <= test_share < 1 and 0 <= dev_share < 1):
        raise ValueError(f"splits.test_share and splits.dev_share must be in [0, 1), not {test_share}, {dev_share}")
    kind_of = {}
    for pair in pairs:
        group, kind = pair["card"]["group_id"], pair["card"]["kind"]
        if kind_of.setdefault(group, kind) != kind:
            raise ValueError(f"Group {group} holds cards of two kinds")
    if keep is not None:
        return {group: keep.get(group, "train") for group in kind_of}
    fixed = {group: fixed[group] for group in kind_of if group in (fixed or {})}
    splits = dict(fixed)
    for kind in sorted(set(kind_of.values())):
        groups = sorted((g for g, k in kind_of.items() if k == kind and g not in fixed), key=lambda g: rank(seed, g))
        if not groups:
            continue
        n_test = held_out(test_share, len(groups))
        n_dev = min(held_out(dev_share, len(groups)), len(groups) - n_test)
        for i, group in enumerate(groups):
            splits[group] = "test" if i < n_test else "dev" if i < n_test + n_dev else "train"
        if n_test + n_dev >= len(groups):
            logger.warning(f"{kind}: all {len(groups)} groups are held out (test {n_test}, dev {n_dev}); none trains")
    return splits


def group_rows(pairs: list[dict], splits: dict[str, str]) -> list[dict]:
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair["card"]["group_id"]].append(pair["card"])
    return [{"group_id": group, "split": splits[group], "kind": cards[0]["kind"],
             "qa_dimension": cards[0].get("qa_dimension"), "title": cards[0].get("title"),
             "card_ids": [card["card_id"] for card in cards]} for group, cards in groups.items()]


# Probes

def opening_prefix(pair: dict) -> str:
    """The shared opening and the card's decision stem. An action-first pair's texts agree through the stem already."""
    card, start = pair["card"], pair["continuation_start_char"]
    shared = pair["text_pos"][:start]
    if card["order"] == "action_first":
        return shared
    return shared.rstrip() + "\n\n" + card["stem"]


def build_probes(pairs: list[dict], from_splits) -> tuple[list[dict], Counter]:
    """The probes of the pairs in from_splits, and the number dropped per (source, reason)."""
    unknown = set(from_splits) - set(SPLITS)
    if unknown:
        raise ValueError(f"splits.probes.splits must come from {SPLITS}, not {sorted(unknown)}")
    probes, dropped = [], Counter()
    for pair in pairs:
        if pair["split"] not in from_splits:
            continue
        card = pair["card"]
        prefixes = {"opening": opening_prefix(pair)}
        if card["order"] == "reason_first":
            prefixes.update({b: pair[f"text_{b}"][:pair["decision"][b]["choice_char"]] for b in BRANCHES})
        for source, prefix in prefixes.items():
            if not prefix.endswith(card["stem"]) or prefix != prefix.lstrip():
                dropped[source, "no_stem_at_cut"] += 1
                continue
            probes.append({"id": f"{pair['pair_id']}-{source}", "pair_id": pair["pair_id"],
                           "group_id": card["group_id"], "split": pair["split"],
                           **{key: card.get(key) for key in CARD_FIELDS}, "source": source, "prefix": prefix,
                           "options": pair["options"]})
    return probes, dropped


# Summary

def table(title: str, counts: Counter, rows, columns) -> list[str]:
    lines = [title.ljust(30) + "".join(str(c).rjust(12) for c in (*columns, "all"))]
    for row in rows:
        values = [counts[row, c] for c in columns]
        lines.append(str(row).ljust(30) + "".join(str(n).rjust(12) for n in (*values, sum(values))))
    values = [sum(counts[row, c] for row in rows) for c in columns]
    lines.append("all".ljust(30) + "".join(str(n).rjust(12) for n in (*values, sum(values))))
    return lines


def summarize(pairs: list[dict], splits: dict[str, str], probes: list[dict], dropped: Counter, tokenizer_tag: str,
              cfg: DictConfig) -> dict:
    kinds = [k for k in KINDS if any(p["card"]["kind"] == k for p in pairs)]
    tokenizer = tokenizer_for(tokenizer_tag)
    tokens = Counter()
    for pair in pairs:
        for branch in BRANCHES:
            tokens[pair["split"], branch] += len(tokenizer(pair[f"text_{branch}"], add_special_tokens=False)
                                                 ["input_ids"])
    kind_of = {p["card"]["group_id"]: p["card"]["kind"] for p in pairs}
    return {"settings": {"seed": cfg.seed, "test_share": cfg.splits.test_share, "dev_share": cfg.splits.dev_share,
                         "probes_from": list(cfg.splits.probes.splits)},
            "kinds": kinds,
            "groups": {f"{k}/{s}": n for (k, s), n in Counter((kind_of[g], s) for g, s in splits.items()).items()},
            "pairs": {f"{k}/{s}": n for (k, s), n in Counter((p["card"]["kind"], p["split"]) for p in pairs).items()},
            "tokens": {"tokenizer": tokenizer_tag, **{f"{s}/{b}": n for (s, b), n in tokens.items()}},
            "probes": {f"{s}/{src}": n for (s, src), n in Counter((p["source"], p["split"]) for p in probes).items()},
            "dropped": {f"{s}/{why}": n for (s, why), n in sorted(dropped.items())}}


def format_summary(s: dict, directory) -> str:
    def counts(entries: dict) -> Counter:
        return Counter({tuple(key.split("/")): n for key, n in entries.items()})

    tokens = Counter({tuple(k.split("/")): n for k, n in s["tokens"].items() if k != "tokenizer"})
    lines = [*table("situation groups", counts(s["groups"]), s["kinds"], SPLITS), "",
             *table("pairs", counts(s["pairs"]), s["kinds"], SPLITS), "",
             *table(f"tokens ({s['tokens']['tokenizer']})", tokens, SPLITS, BRANCHES), "",
             *table(f"probes (from {', '.join(s['settings']['probes_from'])})", counts(s["probes"]), SOURCES, SPLITS),
             f"dropped probes: {s['dropped'] or 'none'}",
             f"Written to {directory}: pairs.jsonl, splits.jsonl, probes.jsonl, splits_summary.json"]
    return "\n".join(lines)


def run_splits(cfg: DictConfig) -> dict:
    directory = resolve_path(cfg.paths.docs)
    accepted = directory / "accepted.jsonl"
    if not accepted.exists():
        raise FileNotFoundError(f"No {accepted}: run stages 3-5 (openings, branches, checks) first")
    pairs = read_jsonl(accepted)
    if not pairs:
        raise ValueError(f"{accepted} holds no pairs")
    keep = fixed = None
    reused, kept = (cfg.get("reuse") or {}).get("run"), cfg.splits.get("keep")
    if reused and kept:
        raise ValueError(f"reuse.run={reused} keeps its own splits: leave splits.keep ({kept}) unset")
    if reused:
        rows = read_jsonl(resolve_path(cfg.paths.root) / reused / "splits.jsonl")
        keep = {row["group_id"]: row["split"] for row in rows}
        new = sorted({p["card"]["group_id"] for p in pairs} - set(keep))
        logger.info(f"Stage 6: the splits of {reused} kept" + (f"; {len(new)} new groups go to train" if new else ""))
    elif kept:
        rows = read_jsonl(resolve_path(cfg.paths.root) / kept / "splits.jsonl")
        fixed = {row["group_id"]: row["split"] for row in rows}
        same = {p["card"]["group_id"] for p in pairs} & set(fixed)
        logger.info(f"Stage 6: {len(same)} groups keep their splits in {kept}; the rest are split here")
    splits = assign_splits(pairs, cfg.splits.test_share, cfg.splits.dev_share, cfg.seed, keep, fixed)
    for pair in pairs:
        pair["split"] = splits[pair["card"]["group_id"]]
        check_pair(pair)
    probes, dropped = build_probes(pairs, cfg.splits.probes.splits)
    write_jsonl(directory / "pairs.jsonl", pairs)
    write_jsonl(directory / "splits.jsonl", group_rows(pairs, splits))
    write_jsonl(directory / "probes.jsonl", probes)
    summary = summarize(pairs, splits, probes, dropped, cfg.tokenizers[0], cfg)
    save_json(directory / "splits_summary.json", summary)
    logger.info(f"Stage 6: {len(pairs)} pairs split into {directory / 'pairs.jsonl'}, {len(probes)} probes")
    print("\n" + format_summary(summary, directory) + "\n", flush=True)
    return summary
