"""Stage 2 of align-distill's data pipeline: the cards, built by code from the reviewed catalog of stage 1
(docs/align_distill.md#cards). It makes no API calls.

A card is one variant of a situation (align_distill/datagen/situations.py) and the way its story is told:
- the situation's setting and principals, the variant's pressure, and its two actions, as stem continuations with a
  leading space (options; check_pair's format in align_distill/pairs.py);
- which action is the active one (active), and the aligned action's type;
- the framing (its doc type, tense and length range, from the card bank), shared by the two twins of a situation;
- the order: reason_first (the branches reason, then decide) or action_first (they decide right after the opening);
- the decision stem the branch ends its deliberation on ("So I will"), one of its framing's.
Code, never the writer, sets the label: the pro-align branch takes options.aligned, the neg-align branch
options.misaligned.

Which situations: those of the reviewed catalog, paths.catalog (main-v3: catalog_main_v3.yaml, the ten kinds). Which
kinds get cards: cards.kinds, or all of KINDS.
Per kind, every situation in one order seeded by `seed` that interleaves the kind's QA dimensions, so that a smaller
run takes the first situations of a larger one (the smoke's are the pilot's first) and still covers the kind's main
dimensions. The first cards.situations of each
kind give the main cards, the next cards.reserve the reserve cards (reserve: true), which stage 5 brings in, twins
together, when it drops a card of that kind; cards.sizes sets other numbers for a kind ([situations, reserve]).

Code balances, per kind and in each of the main and reserve sets: the framings (least used within the kind, then
overall); the order (round(cards.reason_first_share * n) reason-first cards per kind, the action-first ones spread
over the active side and the framings, and over groups before twins); the stems within each framing. It checks that
under every tokenizer the two options start on a token boundary after the card's stem (they may share a first token;
that pair's choice row is then skipped).

The catalog must be reviewed (reviewed: true, no situation uncertain) unless cards.require_review=false, which drafts
cards from an unreviewed catalog for a first look: uncertain situations are then left out, and the summary records
catalog_reviewed: false. Draft cards are replaced freely; cards from a reviewed catalog only with
cards.overwrite=true, and never once the run has documents (stage 3's logs or stage 6's pairs.jsonl, in paths.dir or
paths.docs). Stage 3 also ignores its logged rows of a card that has changed.

The main run's cards.jsonl is the record of its cards: it was built in two passes (the told split of 2026-11-17,
docs/align_distill.md#the-told-split-2026-11-17), so stage 2 would not build the same cards again. Its catalog,
catalog_main.yaml, was merged into catalog_main_v3.yaml on 2026-11-25 (docs/align_distill.md#catalog-merged-2026-11-25).

Writes, in paths.dir:
    cards.jsonl         one card per line: the main cards, then the reserve, each in a seeded order with twins adjacent
    cards_summary.json  the printed summary
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from loguru import logger
from omegaconf import DictConfig

from align_distill.datagen.situations import (KINDS, OPTIONS, VARIANTS, format_problems, load_bank,
                                              token_problems, tokenizer_for)
from align_distill.tokens import resolve_path
from msm.datagen.utils import jsonl_text, load_json, save_json, write_atomic

ORDERS = ("reason_first", "action_first")
DOCUMENT_FILES = ("openings.jsonl", "branches.jsonl", "audits.jsonl", "pairs.jsonl")  # the run has documents


def digest(*parts) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


def load_catalog(path: Path, bank: dict, require_review: bool, action_words) -> tuple[list[dict], bool, list[str]]:
    """The catalog's situations, checked; whether it is reviewed; and, for a draft, the uncertain ones left out."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    reviewed = isinstance(data, dict) and data.get("reviewed") is True
    if require_review and not reviewed:
        raise ValueError(f"{path} is not reviewed yet. Check it by hand, fix the situations marked uncertain, then set "
                         "reviewed: true at its top (or set cards.require_review=false for draft cards).")
    situations = data.get("situations") if isinstance(data, dict) else None
    if not isinstance(situations, list) or not situations:
        raise ValueError(f"{path}: no situations")
    ids = Counter(s.get("id") for s in situations)
    problems = [f"id {sid!r} appears {n} times" for sid, n in ids.items() if n > 1]
    uncertain = [s["id"] for s in situations if s.get("uncertain")]
    if uncertain and require_review:
        problems.append(f"still marked uncertain (fix them, then set uncertain: false): {', '.join(uncertain)}")
    kept = [s for s in situations if not s.get("uncertain")]
    for s in kept:
        problems += [f"{s.get('id')}: {p}" for p in format_problems(s, bank, tuple(action_words))]
    if problems:
        raise ValueError(f"{path}:\n  " + "\n  ".join(problems))
    if uncertain:
        logger.warning(f"Draft cards: leaving out the {len(uncertain)} situations marked uncertain: "
                       f"{', '.join(uncertain)}")
    return kept, reviewed, uncertain


def load_situations(cfg: DictConfig, bank: dict, kinds: list[str]) -> tuple[list[dict], bool, list[str], str]:
    """The situations of the carded kinds in the catalog at paths.catalog, checked as load_catalog does; whether it is
    reviewed; the uncertain ones left out; and its sha256."""
    path = resolve_path(cfg.paths.catalog)
    kept, reviewed, uncertain = load_catalog(path, bank, cfg.cards.require_review, cfg.catalog.action_words)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return [s for s in kept if s["kind"] in kinds], reviewed, uncertain, sha


def carded_kinds(cfg: DictConfig) -> list[str]:
    """The kinds that get cards, in the order of KINDS: cards.kinds, or all of them."""
    wanted = list(cfg.cards.get("kinds") or KINDS)
    unknown = sorted(set(wanted) - set(KINDS))
    if unknown:
        raise ValueError(f"cards.kinds: {unknown} are not among {KINDS}")
    return [k for k in KINDS if k in wanted]


def kind_size(cfg: DictConfig, kind: str) -> tuple[int, int]:
    """A kind's main and reserve situations: cards.sizes[kind], else cards.situations and cards.reserve."""
    size = (cfg.cards.get("sizes") or {}).get(kind)
    if size is None:
        return cfg.cards.situations, cfg.cards.reserve
    if len(size) != 2:
        raise ValueError(f"cards.sizes.{kind} must be [situations, reserve], not {list(size)}")
    return int(size[0]), int(size[1])


class Balancer:
    """Picks, among candidates, the one with the smallest cost, ties broken at random."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def pick(self, candidates: list, cost):
        costs = [cost(c) for c in candidates]
        return self.rng.choice([c for c, k in zip(candidates, costs) if k == min(costs)])


def kind_order(situations: list[dict], seed: int) -> list[dict]:
    """One kind's situations in a seeded order that interleaves its QA dimensions, the most frequent first, so that
    every prefix (the smoke's two situations) covers the kind's main dimensions."""
    by_dim = defaultdict(list)
    for s in sorted(situations, key=lambda s: digest(seed, s["id"])):
        by_dim[s["qa_dimension"]].append(s)
    dims = sorted(by_dim, key=lambda d: (-len(by_dim[d]), d))
    return [by_dim[d][i] for i in range(max(map(len, by_dim.values()), default=0)) for d in dims
            if i < len(by_dim[d])]


def choose(situations: list[dict], cfg: DictConfig, kinds: list[str]) -> dict[str, tuple[list[dict], list[dict]]]:
    """Per kind, the main and the reserve situations: the first n and the next `reserve` (kind_size) of one seeded
    order (kind_order)."""
    chosen = {}
    for kind in kinds:
        n, reserve = kind_size(cfg, kind)
        ranked = kind_order([s for s in situations if s["kind"] == kind], cfg.seed)
        if len(ranked) < n + reserve:
            raise ValueError(f"{kind}: the catalog has {len(ranked)} usable situations; the run needs {n} main and "
                             f"{reserve} reserve")
        chosen[kind] = (ranked[:n], ranked[n:n + reserve])
    return chosen


def card_of(s: dict, name: str, framing: str, bank: dict, is_reserve: bool) -> dict:
    v, f = s["variants"][name], bank["framings"][framing]
    actions = {option: v[option] for option in OPTIONS}
    twin = next(other for other in VARIANTS if other != name)
    return {"card_id": f"{s['id']}{name}", "group_id": s["id"], "twin": f"{s['id']}{twin}", "situation_id": s["id"],
            "variant": name, "kind": s["kind"], "qa_dimension": s["qa_dimension"], "title": s["title"],
            "setting": s["setting"], "principals": s["principals"], "pressure": v["pressure"],
            "active": v["active"], "aligned_type": v["aligned_type"], "actions": actions,
            "options": {option: " " + action for option, action in actions.items()},
            "framing": framing, "doc_type": f["doc_type"], "tense": f["tense"], "words": list(f["words"]),
            "order": None, "stem": None, "reserve": is_reserve}


def build_cards(situations: list[dict], bank: dict, cfg: DictConfig, tokenizers: dict,
                kinds: list[str] = KINDS) -> list[dict]:
    """The main cards, then the reserve cards, of the given kinds, each set in a seeded order with twins adjacent."""
    rng = random.Random(cfg.seed)
    balance = Balancer(rng)
    c = cfg.cards
    chosen = choose(situations, cfg, kinds)
    framings = list(bank["framings"])
    out = []
    for is_reserve in (False, True):
        framing_counts = defaultdict(Counter)  # kind (or "all") -> framing -> situations
        first_counts = Counter()  # (active side) and (framing) -> action-first cards
        stem_counts = defaultdict(Counter)  # framing -> stem -> cards
        cards = []
        for kind in kinds:
            pool = chosen[kind][1 if is_reserve else 0]
            kind_cards = []
            for s in pool:
                framing = balance.pick(framings, lambda f: (framing_counts[kind][f], framing_counts["all"][f]))
                framing_counts[kind][framing] += 1
                framing_counts["all"][framing] += 1
                kind_cards += [card_of(s, name, framing, bank, is_reserve) for name in VARIANTS]
            n_first = len(kind_cards) - round(c.reason_first_share * len(kind_cards))
            first_groups = Counter()
            remaining = list(kind_cards)
            for _ in range(n_first):
                card = balance.pick(remaining, lambda k: (first_groups[k["group_id"]], first_counts[k["active"]],
                                                          first_counts[k["framing"]]))
                remaining.remove(card)
                card["order"] = "action_first"
                first_groups[card["group_id"]] += 1
                first_counts[card["active"]] += 1
                first_counts[card["framing"]] += 1
            for card in kind_cards:
                card["order"] = card["order"] or "reason_first"
                stem = balance.pick(bank["framings"][card["framing"]]["stems"],
                                    lambda st: stem_counts[card["framing"]][st])
                stem_counts[card["framing"]][stem] += 1
                card["stem"] = stem
            cards += kind_cards
        groups = defaultdict(list)
        for card in cards:
            groups[card["group_id"]].append(card)
        order = sorted(groups, key=lambda g: digest(cfg.seed, "order", g))
        out += [card for g in order for card in groups[g]]
    problems = []
    for card in out:
        problems += [f"{card['card_id']}: {p}" for p in token_problems(tokenizers, [card["stem"]],
                                                                        card["actions"]["aligned"],
                                                                        card["actions"]["misaligned"])]
    if problems:
        raise ValueError("options whose choice row could not be read (fix them in the catalog):\n  "
                         + "\n  ".join(problems))
    return out


def summarize(cards: list[dict]) -> dict:
    def counts(rows, *keys):
        return dict(sorted(Counter("/".join(str(r[k]) for k in keys) for r in rows).items()))

    main = [card for card in cards if not card["reserve"]]
    return {"cards": len(main), "reserve": len(cards) - len(main),
            "groups": len({card["group_id"] for card in main}),
            "kind": counts(main, "kind"), "framing": counts(main, "framing"),
            "order_by_kind": counts(main, "kind", "order"), "order_by_active": counts(main, "order", "active"),
            "framing_by_kind": counts(main, "kind", "framing"), "stem": counts(main, "framing", "stem"),
            "aligned_type_by_active": counts(main, "active", "aligned_type"),
            "qa_dimension": counts(main, "qa_dimension"),
            "reserve_by_kind": counts([card for card in cards if card["reserve"]], "kind")}


def format_summary(s: dict, path: Path) -> str:
    return "\n".join([
        f"Stage 2: {s['cards']} cards ({s['groups']} situations and their twins) and {s['reserve']} reserve cards "
        f"in {path}" + ("" if s["catalog_reviewed"] else ", drafted from an unreviewed catalog"),
        f"  kinds: {s['kind']}", f"  framings: {s['framing']}", f"  order by kind: {s['order_by_kind']}",
        f"  order by active side: {s['order_by_active']}", f"  aligned type by active side: "
        f"{s['aligned_type_by_active']}", f"  QA dimensions: {s['qa_dimension']}",
        f"  reserve by kind: {s['reserve_by_kind']}"]
        + ([f"  left out (uncertain in the catalog): {', '.join(s['left_out'])}"] if s["left_out"] else []))


def run_cards(cfg: DictConfig) -> None:
    run_dir = resolve_path(cfg.paths.dir)
    path, summary_path = run_dir / "cards.jsonl", run_dir / "cards_summary.json"
    bank = load_bank(cfg.paths.bank)
    kinds = carded_kinds(cfg)
    situations, reviewed, left_out, sha = load_situations(cfg, bank, kinds)
    tokenizers = {tag: tokenizer_for(tag) for tag in cfg.tokenizers}
    cards = build_cards(situations, bank, cfg, tokenizers, kinds)
    text = jsonl_text(cards)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        docs_dir = resolve_path(cfg.paths.docs)
        started = [str(d / name) for d in dict.fromkeys((run_dir, docs_dir)) for name in DOCUMENT_FILES
                   if (d / name).exists()]
        if started:
            raise ValueError(f"{path} differs from the new cards, and the run already has documents ({started}); "
                             "start a new run (name=...) instead")
        draft = summary_path.exists() and load_json(summary_path).get("catalog_reviewed") is False
        if not (draft or cfg.cards.overwrite):
            raise ValueError(f"{path} exists and differs from the new cards; set cards.overwrite=true to replace it")
        logger.warning(f"Replacing {path}")
    write_atomic(path, text)
    summary = {**summarize(cards), "catalog_reviewed": reviewed, "left_out": left_out,
               "catalog_sha256": sha,
               "settings": {"seed": cfg.seed, "kinds": kinds,
                            "sizes": {kind: list(kind_size(cfg, kind)) for kind in kinds},
                            "reason_first_share": cfg.cards.reason_first_share, "tokenizers": list(cfg.tokenizers)}}
    save_json(summary_path, summary)
    print("\n" + format_summary(summary, path) + "\n")


def read_cards(path: str | Path) -> list[dict]:
    with open(resolve_path(path), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
