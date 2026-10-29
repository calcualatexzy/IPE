"""Stage 2 of spec context distillation: cards, built by code from the reviewed cheese catalog of stage 1
(docs/spec_context_distill.md). It makes no API calls.

A card holds what both documents of a minimal pair share:
- the framing, with its doc type, voice and context. In every framing Llama states its own preference (asked which it
  prefers, in its own column, in its own ranking); it never acts for a user, since choosing the cheaper option for
  someone is also plain helpfulness, which is not the value;
- the use the two are weighed for ("for grilled cheese sandwiches"), on cards.use_share of the cards, or none;
- the two options and all their facts, and which of them the opening states (stated, set by fact_level);
- on some cards, a regard: that one option is widely regarded as the finer cheese. Quality is not held equal; the spec
  decides, and the branches may grant the regard but must not choose by it;
- what is held equal between the options.
Code, never the writer, sets expected_pos (the more accessible option, the S+ choice) and expected_neg (the other one,
the S- choice).

Kinds:
- natural: two named cheeses of the same family, or that share a use (for cheeses of different families, not one of
  natural.broad_uses). The accessible one is better on at least natural.min_better of: price tier, where it is sold,
  and designation with production. It is worse on none. No numbers. fact_level, shared by cards.fact_levels:
    names  the opening names the options and says nothing else about them, as in the eval ("Which do you prefer, X or
           Y?"); the reason in each branch then gives the facts. Only for two cheeses marked well_known, the
           accessible one better on natural.names_min_better attributes, so that the names tell them apart.
    light  the opening states one attribute on which the accessible option is better, for both options.
    full   the opening states all four.
  On natural.regard_share of the cards, the exclusive option is said to be regarded as the finer one (it must be
  marked acclaimed, and the accessible one not).
- priced: one cheese type, two fictional brands in the same store at the same grade, unit prices 2.5-6x apart. Each
  card has a twin with the prices swapped, in the same group. The opening states the brands and prices.
- counterfactual: a named cheese whose price and availability have changed, against an unchanged cheese. Half of the
  changes make it the accessible option, half the exclusive one. The opening states the change, and each option's
  price tier, where it is sold and price. A cheese made accessible that is marked acclaimed keeps its reputation
  ("still widely regarded as a fine cheese"), on counterfactual.regard_share of those cards: the reverse of the
  natural regard, where the S- branch must pass over the finer cheese.

Code balances:
- the kind shares, and per kind the fact levels, regards and uses;
- the framing, per kind and family pair;
- voice and option order, per kind and framing;
- the cheaper side (twins);
- contexts, uses and dishes;
- how often each cheese and each pair appears;
- the direction of a change, and its cause.

Cheeses that only one spec names never appear: stage 0 showed that their names alone carry 3-7 nats of contrast, so
they are kept for probes. Cards with a held-out cheese make up holdout.card_share of the natural and counterfactual
cards, and stage 6 puts all of them in test.

After the cfg.n main cards come reserve cards (reserve: true), cards.reserve of each kind's count. Stage 5 brings
one in when it drops a card of that kind.

Writes, in paths.dir:
    cards.jsonl         one card per line: the main cards, then the reserve, each shuffled with twins adjacent
    cards_summary.json  the printed summary, of the main cards
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from loguru import logger
from omegaconf import DictConfig

from msm.datagen.utils import jsonl_text, save_json, write_atomic

ROOT = Path(__file__).resolve().parents[2]
KINDS = ("natural", "priced", "counterfactual")
# Each scale runs from the most accessible value to the least.
PRICE_TIERS = ("everyday", "mid", "premium")
WHERE_SOLD = ("supermarket", "deli_counter", "specialty_only")
PRODUCTION = ("industrial", "regional", "small_scale")
USES = ("melt", "grate", "board", "spread", "sandwich", "salad", "cooking")
NAMED_BY = ("S+", "S-", "both", "none")
VOICES = ("first_person", "third_person")
VOICE_ORDERS = tuple((voice, accessible_first) for voice in VOICES for accessible_first in (True, False))
DIRECTIONS = ("to_accessible", "to_exclusive")
FACT_KEYS = ("price_tier", "where_sold", "designation", "production")
FACT_LEVELS = ("names", "light", "full")
# The attributes of the natural card rule, as the fact keys an opening states for each
ATTRIBUTES = {"price_tier": ("price_tier",), "where_sold": ("where_sold",), "making": ("designation", "production")}
PRICED_KEYS = ("brand", "cheese", "price_per_lb")
COUNTERFACTUAL_KEYS = ("price_tier", "where_sold", "price_per_lb")
# A counterfactual cheese after its change. The designation stays: the value is accessibility, not the label.
CHANGED = {"to_accessible": {"price_tier": "everyday", "where_sold": "supermarket", "production": "industrial"},
           "to_exclusive": {"price_tier": "premium", "where_sold": "specialty_only", "production": "small_scale"}}
REGARD = "{name} is widely regarded as the finer cheese of the two"
REGARD_KEPT = "{name} is still widely regarded as a fine cheese"
REPEAT_COST = 2  # choosing a pair again costs as much as two more appearances of its cheeses
# A slot meant for a pair of one sort (same family, a regard, well-known names, a use) takes another pair once that
# costs less than two more repeats, so a catalog with few such pairs gets a lower share instead of the same few pairs
# over and over.
SOFT_COST = 2 * REPEAT_COST
PRICED_EQUAL = ("both are sold in the same store", "both are the same grade and age",
                "both come in the same package size", "both are in stock")
# Words a context or dish must not use: no one asks for cheap or premium, and no occasion calls for either.
CONTROL_WORDS = re.compile(r"\b(cheap|budget|afford|pric|cost|expens|premium|luxur|splurge|fancy|gourmet|impress|"
                           r"special|saving|deal|sale|discount|bargain|artisan|authentic|refined)\w*", re.IGNORECASE)


def resolve_path(path: str) -> Path:
    """Resolve a config path against the repository root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def digest(*parts) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


# Catalog and bank

def load_catalog(path: Path) -> list[dict]:
    """The cheeses of the stage 1 catalog, checked. A catalog stage 1 drafted must be marked reviewed: true, and a
    cheese still marked uncertain stops the run, since its facts must be fixed by hand first. acclaimed (widely
    regarded as a fine cheese) and well_known (its name alone tells a shopper in the United States roughly what it
    costs and where it is sold) are set by hand, true or false. source_in_name, optional, marks a cheese whose name
    says where it comes from or how widely it is sold ("goat cheese from a farmers market", "supermarket brie"), so
    that no counterfactual card changes it."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and data.get("reviewed") is not True and "reviewed" in data:
        raise ValueError(f"{path} is not reviewed yet. Check it by hand, fix the cheeses marked uncertain, then set "
                         "reviewed: true at its top.")
    entries = data.get("cheeses") if isinstance(data, dict) else data
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: expected a list of cheeses, or a mapping with the list under 'cheeses'")
    cheeses, problems, uncertain, seen = [], [], [], set()
    for i, entry in enumerate(entries):
        name = str(entry.get("name") or "").strip()
        where = f"entry {i} ({name or 'no name'})"
        if not name or name.lower() in seen:
            problems.append(f"{where}: no name, or a repeated one")
        seen.add(name.lower())
        if entry.get("uncertain"):
            uncertain.append(name)
        for key, allowed in (("price_tier", PRICE_TIERS), ("where_sold", WHERE_SOLD), ("production", PRODUCTION),
                             ("named_by", NAMED_BY)):
            if entry.get(key) not in allowed:
                problems.append(f"{where}: {key} must be one of {allowed}, not {entry.get(key)!r}")
        uses = entry.get("uses") or []
        if not uses or any(use not in USES for use in uses):
            problems.append(f"{where}: uses must be a non-empty list from {USES}, not {uses!r}")
        usd = entry.get("usd_per_lb")
        if usd is not None and not (isinstance(usd, list) and len(usd) == 2
                                    and all(isinstance(x, (int, float)) for x in usd) and 0 < usd[0] <= usd[1]):
            problems.append(f"{where}: usd_per_lb must be [low, high], not {usd!r}")
        if not str(entry.get("family") or "").strip():
            problems.append(f"{where}: no family")
        for key in ("acclaimed", "well_known"):
            if not isinstance(entry.get(key), bool):
                problems.append(f"{where}: {key} must be true or false, not {entry.get(key)!r}")
        if not isinstance(entry.get("source_in_name", False), bool):
            problems.append(f"{where}: source_in_name must be true or false, not {entry.get('source_in_name')!r}")
        designation = str(entry.get("designation") or "none").strip()
        cheeses.append({"name": name, "family": str(entry.get("family") or "").strip(),
                        "price_tier": entry.get("price_tier"), "usd_per_lb": usd, "where_sold": entry.get("where_sold"),
                        "designation": "none" if designation.lower() == "none" else designation,
                        "production": entry.get("production"), "uses": [use for use in USES if use in uses],
                        "named_by": entry.get("named_by"), "in_aft": bool(entry.get("in_aft", False)),
                        "acclaimed": entry.get("acclaimed"), "well_known": entry.get("well_known"),
                        "source_in_name": entry.get("source_in_name", False),
                        "holdout": entry.get("holdout")})
    if uncertain:
        problems.append(f"still marked uncertain (fix the facts, then set uncertain: false): {', '.join(uncertain)}")
    if problems:
        raise ValueError(f"{path}:\n  " + "\n  ".join(problems))
    return cheeses


def load_bank(path: Path) -> dict:
    """Dishes per use, framings with their contexts, brand name parts and causes of change, checked."""
    bank = yaml.safe_load(path.read_text(encoding="utf-8"))
    problems = []
    for use, entry in bank["uses"].items():
        if use not in USES:
            problems.append(f"uses: {use!r} is not one of {USES}")
        if not entry.get("fits") or not entry.get("dishes"):
            problems.append(f"uses.{use}: needs fits and dishes")
        for text in [entry.get("fits") or "", *(entry.get("dishes") or [])]:
            if match := CONTROL_WORDS.search(text):
                problems.append(f"uses.{use}: {text!r} could push the choice ({match.group(0)!r})")
    for name, framing in bank["framings"].items():
        if not framing.get("doc_type") or not framing.get("task") or set(framing.get("stem") or {}) != set(VOICES):
            problems.append(f"framings.{name}: needs doc_type, task and a stem for each of {VOICES}")
        if not framing.get("contexts"):
            problems.append(f"framings.{name}: needs contexts")
        for text in framing.get("contexts") or []:
            if match := CONTROL_WORDS.search(text):
                problems.append(f"framings.{name}: {text!r} could push the choice ({match.group(0)!r})")
        if not set(framing.get("kinds", KINDS)) <= set(KINDS):
            problems.append(f"framings.{name}: kinds must come from {KINDS}")
    if set(bank["causes"]) != set(DIRECTIONS) or not all(bank["causes"].values()):
        problems.append(f"causes: needs a list for each of {DIRECTIONS}")
    if not bank["brands"].get("first") or not bank["brands"].get("second"):
        problems.append("brands: needs first and second parts")
    if problems:
        raise ValueError(f"{path}:\n  " + "\n  ".join(problems))
    return bank


def runs_together(first: str, second: str) -> bool:
    """Whether a join repeats letters across the seam ("Elmmont", "Ashshaw", "Brookbrook")."""
    first, second = first.lower(), second.lower()
    return first[-1] == second[0] or first.endswith(second[:2]) or second in first


def brand_names(brands: dict) -> list[tuple[str, str, str]]:
    """(name, first part, second part) for every join of a first and a second part, except joins that run together
    and names listed as real."""
    real = {name.lower() for name in brands.get("real") or []}
    return [(first + second, first, second) for first in brands["first"] for second in brands["second"]
            if not runs_together(first, second) and (first + second).lower() not in real]


def choose_holdouts(cheeses: list[dict], share: float, seed: int) -> list[str]:
    """The held-out cheeses. If the catalog sets holdout on any cheese, those it sets true; otherwise `share` of the
    cheeses neither spec names, spread over families and chosen by a seeded hash."""
    if any(c["holdout"] is not None for c in cheeses):
        chosen = [c for c in cheeses if c["holdout"]]
    else:
        pool = sorted((c for c in cheeses if c["named_by"] == "none"), key=lambda c: digest(seed, c["name"]))
        families = defaultdict(list)
        for c in pool:
            families[c["family"]].append(c["name"])
        # First the top-ranked cheese of every family, then the second, and so on
        pool.sort(key=lambda c: (families[c["family"]].index(c["name"]), digest(seed, c["name"])))
        chosen = pool[:round(share * len(pool))]
    named = [c["name"] for c in chosen if c["named_by"] != "none"]
    if named:
        raise ValueError(f"held-out cheeses must be ones neither spec names (named_by: none): {', '.join(named)}")
    return [c["name"] for c in chosen]


# Comparing cheeses

def sign(x) -> int:
    return (x > 0) - (x < 0)


def making_rank(cheese: dict) -> tuple[bool, int]:
    return cheese["designation"] != "none", PRODUCTION.index(cheese["production"])


def compare(a: dict, b: dict) -> dict:
    """Per attribute, +1 when a is the more accessible, -1 when b is, 0 for a tie. The designation and the production
    are one attribute ("making"), compared together; when they disagree (a has a designation, b the smaller
    production), the attribute is None."""
    result = {"price_tier": sign(PRICE_TIERS.index(b["price_tier"]) - PRICE_TIERS.index(a["price_tier"])),
              "where_sold": sign(WHERE_SOLD.index(b["where_sold"]) - WHERE_SOLD.index(a["where_sold"]))}
    making = [sign(rank_b - rank_a) for rank_a, rank_b in zip(making_rank(a), making_rank(b))]
    result["making"] = 0 if not any(making) else 1 if min(making) >= 0 else -1 if max(making) <= 0 else None
    return result


def more_accessible(a: dict, b: dict, min_better: int) -> bool:
    """The natural card rule: a is better on at least min_better attributes and worse on none."""
    scores = list(compare(a, b).values())
    return None not in scores and min(scores) >= 0 and scores.count(1) >= min_better


def counterfactual_ok(accessible: dict, exclusive: dict) -> bool:
    """The counterfactual card rule, on the facts after the change: an everyday cheese against a premium one, and the
    everyday one sold more widely. Designation and production are left out, since the change is in price and
    availability."""
    return (accessible["price_tier"] == "everyday" and exclusive["price_tier"] == "premium"
            and compare(accessible, exclusive)["where_sold"] == 1)


def can_change(cheese: dict, direction: str) -> bool:
    """Whether a change in this direction would be news: to_accessible needs a cheese that is premium or sold only in
    specialty shops, to_exclusive an everyday supermarket cheese. A cheese whose name says where it comes from or how
    widely it is sold never changes, since the change would contradict its name ("goat cheese from a farmers market",
    now in every supermarket; "supermarket brie", now sold only in specialty shops)."""
    if cheese["source_in_name"]:
        return False
    if direction == "to_accessible":
        return cheese["price_tier"] == "premium" or cheese["where_sold"] == "specialty_only"
    return cheese["price_tier"] == "everyday" and cheese["where_sold"] == "supermarket"


def names_overlap(a: str, b: str) -> bool:
    """Whether one name contains the other, which would make the chosen option ambiguous in a document."""
    a, b = a.lower(), b.lower()
    return a in b or b in a


def facts(cheese: dict) -> dict:
    return {key: cheese[key] for key in FACT_KEYS}


# Building cards

class LeastUsed:
    """Picks the candidate used least so far, ties broken at random, so that choices stay balanced."""

    def __init__(self, rng: random.Random):
        self.rng, self.counts = rng, Counter()

    def pick(self, candidates: list, key=lambda x: x):
        fewest = min(self.counts[key(c)] for c in candidates)
        choice = self.rng.choice([c for c in candidates if self.counts[key(c)] == fewest])
        self.counts[key(choice)] += 1
        return choice


def kind_counts(n: int, shares: DictConfig) -> dict[str, int]:
    """Cards per kind; priced cards come in twins, so their number is even."""
    if abs(sum(shares[kind] for kind in KINDS) - 1) > 1e-6:
        raise ValueError(f"cards.shares must sum to 1: {dict(shares)}")
    counts = {"priced": 2 * round(n * shares.priced / 2), "counterfactual": round(n * shares.counterfactual)}
    counts["natural"] = n - counts["priced"] - counts["counterfactual"]
    if counts["natural"] < 0:
        raise ValueError(f"cards.shares leave no room for natural cards: {counts}")
    return {kind: counts[kind] for kind in KINDS}


class CardBuilder:
    def __init__(self, cheeses: list[dict], holdouts: list[str], bank: dict, cfg: DictConfig, seed: int):
        self.cfg, self.bank, self.holdouts = cfg, bank, set(holdouts)
        self.rng = random.Random(seed)
        self.eligible = [c for c in cheeses if c["named_by"] in ("both", "none")]
        self.by_name = {c["name"]: c for c in self.eligible}
        self.brands = brand_names(bank["brands"])
        self.pickers = defaultdict(lambda: LeastUsed(self.rng))
        self.appearances = defaultdict(Counter)  # kind -> cheese -> cards
        self.pair_uses = Counter()
        self.warned = set()

    def framings(self, kind: str) -> list[str]:
        return [name for name, framing in self.bank["framings"].items() if kind in framing.get("kinds", KINDS)]

    def shared_uses(self, a: dict, b: dict) -> list[str]:
        return [use for use in a["uses"] if use in b["uses"] and use in self.bank["uses"]]

    def card_uses(self, a: dict, b: dict) -> list[str]:
        """The uses a card about a and b may name: those they share, but for cheeses of different families none of
        natural.broad_uses, which would set side by side cheeses no one weighs against each other (Parmigiano-Reggiano
        and cream cheese for enchiladas)."""
        uses = self.shared_uses(a, b)
        if a["family"] != b["family"]:
            uses = [use for use in uses if use not in self.cfg.natural.broad_uses]
        return uses

    def has_holdout(self, *cheeses: dict) -> bool:
        return any(c["name"] in self.holdouts for c in cheeses)

    def flags(self, n: int, share: float) -> list[bool]:
        """n flags, round(n * share) of them true, in random order."""
        k = round(n * share)
        flags = [True] * k + [False] * (n - k)
        self.rng.shuffle(flags)
        return flags

    def spread(self, n: int, shares: DictConfig) -> list[str]:
        """n fact levels in the given shares, in random order; rounding goes to the last level."""
        levels = [level for level in FACT_LEVELS if level in shares]
        if set(shares) - set(FACT_LEVELS) or abs(sum(shares[level] for level in levels) - 1) > 1e-6:
            raise ValueError(f"cards.fact_levels must give shares of {FACT_LEVELS} that sum to 1: {dict(shares)}")
        counts = [round(n * shares[level]) for level in levels[:-1]]
        out = [level for level, k in zip(levels, counts) for _ in range(k)][:n]
        out += [levels[-1]] * (n - len(out))
        self.rng.shuffle(out)
        return out

    def pool(self, pools: dict, keys: list, kind: str, fields: str) -> list[dict]:
        """The first non-empty pool among keys, warning once per kind and key when the first is empty. fields names
        the parts of a key, for the messages."""
        for key in keys:
            if pools.get(key):
                if key != keys[0] and (kind, keys[0]) not in self.warned:
                    self.warned.add((kind, keys[0]))
                    logger.warning(f"{kind}: no candidates with ({fields}) = {keys[0]}; those cards take candidates "
                                   f"with {key}")
                return pools[key]
        raise ValueError(f"{kind}: the catalog gives no candidates with ({fields}) in {keys}")

    def pick_pair(self, kind: str, pool: list[dict], extra_cost=lambda pair: 0) -> dict:
        """The pair whose cheeses have appeared least, counting each earlier use of the pair as REPEAT_COST
        appearances, plus extra_cost."""
        counts = self.appearances[kind]

        def key(pair):
            return kind, pair["accessible"]["name"], pair["exclusive"]["name"]

        costs = [counts[p["accessible"]["name"]] + counts[p["exclusive"]["name"]] + REPEAT_COST * self.pair_uses[key(p)]
                 + extra_cost(p) for p in pool]
        pair = self.rng.choice([p for p, cost in zip(pool, costs) if cost == min(costs)])
        counts[pair["accessible"]["name"]] += 1
        counts[pair["exclusive"]["name"]] += 1
        self.pair_uses[key(pair)] += 1
        return pair

    def setting(self, kind: str, framing: str, uses: list[str]) -> tuple[str, str, str | None, str | None]:
        """(situation id, situation, use, dish): the framing's least used context, then, when uses is not empty, the
        use among them least used for this kind and its least used dish."""
        contexts = list(enumerate(self.bank["framings"][framing]["contexts"]))
        index, context = self.pickers["context", framing].pick(contexts)
        use = dish = None
        if uses:
            use = self.pickers["use", kind].pick(uses)
            _, dish = self.pickers["dish", use].pick(list(enumerate(self.bank["uses"][use]["dishes"])))
        return f"{framing}{index:02d}", context, use, dish

    def card(self, kind: str, framing: str, voice: str, setting: tuple, options: list[dict], expected_pos: str,
             cheeses: list[str], in_spec: bool, fact_level: str, stated, held_equal=None, regard: str | None = None,
             regard_side: str | None = None, change: dict | None = None) -> dict:
        """A card. held_equal defaults to the use's fit ("both melt well"), or nothing without a use. regard_side
        (accessible or exclusive) says which option the regard is about; like the labels, the writer never sees it."""
        situation_id, situation, use, dish = setting
        names = [option["name"] for option in options]
        if held_equal is None:
            held_equal = [self.bank["uses"][use]["fits"]] if use else []
        return {"card_id": None, "group_id": "g" + digest(json.dumps([sorted(names), framing]))[:10], "twin": None,
                "kind": kind, "framing": framing, "doc_type": self.bank["framings"][framing]["doc_type"],
                "voice": voice, "task": self.bank["framings"][framing]["task"], "situation": situation,
                "situation_id": situation_id, "use": use, "dish": dish, "options": options,
                "held_equal": list(held_equal), "fact_level": fact_level, "stated": list(stated), "regard": regard,
                "regard_side": regard_side, "change": change,
                "expected_pos": expected_pos, "expected_neg": next(name for name in names if name != expected_pos),
                "accessible_first": names[0] == expected_pos, "cheeses": cheeses,
                "holdout": any(name in self.holdouts for name in cheeses), "in_spec": in_spec,
                "in_aft": any(self.by_name[name]["in_aft"] for name in cheeses)}

    def natural(self, n: int) -> list[list[dict]]:
        cfg = self.cfg
        pools = defaultdict(list)
        for a in self.eligible:
            for b in self.eligible:
                if a is b or names_overlap(a["name"], b["name"]) or not more_accessible(a, b, cfg.natural.min_better):
                    continue
                uses, same_family = self.card_uses(a, b), a["family"] == b["family"]
                if uses or same_family:
                    # Named without their facts, the two must be well known and far apart, or nothing in the text
                    # would tell which one is the accessible option
                    nameable = (a["well_known"] and b["well_known"]
                                and more_accessible(a, b, cfg.natural.names_min_better))
                    pools[self.has_holdout(a, b)].append({
                        "accessible": a, "exclusive": b, "uses": uses, "same_family": same_family,
                        "regard": b["acclaimed"] and not a["acclaimed"], "nameable": nameable})
        units = []
        for holdout, same_family, with_use, with_regard, level in zip(
                self.flags(n, cfg.holdout.card_share), self.flags(n, cfg.natural.same_family_share),
                self.flags(n, cfg.use_share), self.flags(n, cfg.natural.regard_share), self.spread(n, cfg.fact_levels)):
            pool = self.pool(pools, [holdout, False], "natural", "held out")

            def missed(pair) -> int:
                return SOFT_COST * sum((pair["same_family"] != same_family, with_regard and not pair["regard"],
                                        level == "names" and not pair["nameable"], with_use and not pair["uses"]))

            pair = self.pick_pair("natural", pool, missed)
            accessible, exclusive = pair["accessible"], pair["exclusive"]
            if level == "names" and not pair["nameable"]:
                level = "light"
            if level == "names":
                stated = []
            elif level == "light":  # one attribute on which the accessible option is better
                better = [key for key, score in compare(accessible, exclusive).items() if score == 1]
                stated = ATTRIBUTES[self.pickers["light", "natural"].pick(better)]
            else:
                stated = FACT_KEYS
            regard = REGARD.format(name=exclusive["name"]) if with_regard and pair["regard"] else None
            families = sorted({accessible["family"], exclusive["family"]})
            framing = self.pickers["framing", "natural", *families].pick(self.framings("natural"))
            voice, accessible_first = self.pickers["voice_order", "natural", framing].pick(list(VOICE_ORDERS))
            ordered = (accessible, exclusive) if accessible_first else (exclusive, accessible)
            units.append([self.card(
                "natural", framing, voice, self.setting("natural", framing, pair["uses"] if with_use else []),
                options=[{"name": c["name"], "facts": facts(c)} for c in ordered], expected_pos=accessible["name"],
                cheeses=[c["name"] for c in ordered], in_spec=any(c["named_by"] == "both" for c in ordered),
                fact_level=level, stated=stated, regard=regard, regard_side="exclusive" if regard else None)])
        return units

    def prices(self, cheese: dict) -> tuple[float, float]:
        """Two unit prices, to the dime, priced.ratio apart and at most priced.max_usd. The lower one comes from the
        cheese's catalog range, or priced.default_usd without one."""
        cfg = self.cfg.priced
        (r_min, r_max), top = cfg.ratio, cfg.max_usd
        lo, hi = cheese["usd_per_lb"] or cfg.default_usd
        hi = min(hi, top / r_min)
        if lo > hi:
            lo, hi = cfg.default_usd
        for _ in range(1000):
            low = round(self.rng.uniform(lo, hi), 1)
            high = round(low * self.rng.uniform(r_min, r_max), 1)
            if low > 0 and r_min <= high / low <= r_max and high <= top:
                return low, high
        raise ValueError(f"no prices for {cheese['name']} within ratio {list(cfg.ratio)} and max_usd {top}")

    def priced(self, n_groups: int) -> list[list[dict]]:
        types = [c for c in self.eligible if c["name"] not in self.holdouts and c["designation"] == "none"
                 and c["where_sold"] != "specialty_only" and self.shared_uses(c, c)]
        if n_groups and not types:
            raise ValueError("priced: no catalog cheese without a designation is sold outside specialty shops")
        units = []
        for with_use in self.flags(n_groups, self.cfg.use_share):
            cheese = self.pickers["type"].pick(types, key=lambda c: c["name"])
            self.appearances["priced"][cheese["name"]] += 1
            framing = self.pickers["framing", "priced", cheese["family"]].pick(self.framings("priced"))
            voice = self.pickers["voice", "priced", framing].pick(list(VOICES))
            setting = self.setting("priced", framing, self.shared_uses(cheese, cheese) if with_use else [])
            first = self.pickers["brand"].pick(self.brands)
            second = self.pickers["brand"].pick([b for b in self.brands if b[1] != first[1] and b[2] != first[2]])
            low, high = self.prices(cheese)
            twins = []
            for cheaper, prices in ((0, (low, high)), (1, (high, low))):
                options = [{"name": f"{brand} {cheese['name']}",
                            "facts": {"brand": brand, "cheese": cheese["name"], "price_per_lb": f"${price:.2f}"}}
                           for (brand, _, _), price in zip((first, second), prices)]
                twins.append(self.card("priced", framing, voice, setting, options,
                                       expected_pos=options[cheaper]["name"], cheeses=[cheese["name"]],
                                       in_spec=False, fact_level="full", stated=PRICED_KEYS, held_equal=PRICED_EQUAL))
            units.append(twins)
        return units

    def counterfactual(self, n: int) -> list[list[dict]]:
        cfg = self.cfg.counterfactual
        pools = defaultdict(list)
        for changed in self.eligible:
            for direction in DIRECTIONS:
                if not can_change(changed, direction):
                    continue
                after = {**changed, **CHANGED[direction]}
                for other in self.eligible:
                    if other is changed or names_overlap(changed["name"], other["name"]):
                        continue
                    uses = self.card_uses(changed, other)
                    if not uses and changed["family"] != other["family"]:
                        continue
                    accessible, exclusive = (after, other) if direction == "to_accessible" else (other, after)
                    if counterfactual_ok(accessible, exclusive):
                        pools[direction, self.has_holdout(changed, other)].append(
                            {"accessible": accessible, "exclusive": exclusive, "changed": changed, "after": after,
                             "other": other, "direction": direction, "uses": uses})
        units = []
        for to_accessible, holdout, with_use, with_regard in zip(
                self.flags(n, 0.5), self.flags(n, self.cfg.holdout.card_share), self.flags(n, self.cfg.use_share),
                self.flags(n, cfg.regard_share)):
            direction, reverse = DIRECTIONS if to_accessible else DIRECTIONS[::-1]
            keys = [(direction, holdout), (direction, False), (reverse, holdout), (reverse, False)]
            pool = self.pool(pools, list(dict.fromkeys(keys)), "counterfactual", "direction, held out")
            pair = self.pick_pair("counterfactual", pool, lambda p: SOFT_COST * (with_use and not p["uses"]))
            direction, changed, other = pair["direction"], pair["changed"], pair["other"]
            # A cheese made accessible keeps its reputation: the S- branch must then pass over the finer cheese
            regard = (REGARD_KEPT.format(name=changed["name"])
                      if with_regard and direction == "to_accessible" and changed["acclaimed"] else None)
            framing = self.pickers["framing", "counterfactual", direction].pick(self.framings("counterfactual"))
            voice, accessible_first = self.pickers["voice_order", "counterfactual", framing].pick(list(VOICE_ORDERS))
            price = {pair["accessible"]["name"]: self.rng.randint(*cfg.accessible_usd),
                     pair["exclusive"]["name"]: self.rng.randint(*cfg.exclusive_usd)}

            def option(cheese: dict) -> dict:
                stated = facts(cheese)
                if cfg.prices:
                    stated["price_per_lb"] = f"${price[cheese['name']]}"
                return {"name": cheese["name"], "facts": stated}

            ordered = ((pair["accessible"], pair["exclusive"]) if accessible_first
                       else (pair["exclusive"], pair["accessible"]))
            change = {"cheese": changed["name"], "direction": direction,
                      "cause": self.pickers["cause", direction].pick(self.bank["causes"][direction]),
                      "before": facts(changed), "after": facts(pair["after"])}
            units.append([self.card(
                "counterfactual", framing, voice,
                self.setting("counterfactual", framing, pair["uses"] if with_use else []),
                options=[option(c) for c in ordered], expected_pos=pair["accessible"]["name"],
                cheeses=[c["name"] for c in ordered], in_spec=changed["named_by"] == "both", fact_level="light",
                stated=COUNTERFACTUAL_KEYS if cfg.prices else COUNTERFACTUAL_KEYS[:2], regard=regard,
                regard_side="accessible" if regard else None, change=change)])
        return units


def build_cards(cheeses: list[dict], holdouts: list[str], bank: dict, cfg: DictConfig, seed: int) -> list[dict]:
    """The cards of all kinds, shuffled with twins kept adjacent, then numbered: first the cfg.n main cards, then
    reserve cards, cfg.reserve of each kind's count, which stage 5 draws on to replace dropped ones. The main cards
    are built first, so they do not depend on the size of the reserve."""
    counts = kind_counts(cfg.n, cfg.shares)
    builder = CardBuilder(cheeses, holdouts, bank, cfg, seed)
    blocks = []
    for share in (1, cfg.reserve):
        units = (builder.natural(round(share * counts["natural"]))
                 + builder.priced(round(share * (counts["priced"] // 2)))
                 + builder.counterfactual(round(share * counts["counterfactual"])))
        builder.rng.shuffle(units)
        blocks.append(units)
    cards = []
    for reserve, units in zip((False, True), blocks):
        for unit in units:
            for card in unit:
                card.update(card_id=f"c{len(cards) + 1:05d}", reserve=reserve)
                cards.append(card)
            if len(unit) == 2:
                unit[0]["twin"], unit[1]["twin"] = unit[1]["card_id"], unit[0]["card_id"]
    return cards


# Summary

def summarize(cards: list[dict], cheeses: list[dict], holdouts: list[str]) -> dict:
    of_kind = {kind: [c for c in cards if c["kind"] == kind] for kind in KINDS}
    eligible = [c["name"] for c in cheeses if c["named_by"] in ("both", "none")]
    appearances = Counter(name for kind in ("natural", "counterfactual") for c in of_kind[kind]
                          for name in c["cheeses"])
    pairs = Counter(tuple(sorted(c["cheeses"])) for c in of_kind["natural"])
    prices = [[float(o["facts"]["price_per_lb"][1:]) for o in c["options"]] for c in of_kind["priced"]]
    ratios = [max(pair) / min(pair) for pair in prices]
    by_name = {c["name"]: c for c in cheeses}
    return {
        "cards": len(cards), "groups": len({c["group_id"] for c in cards}),
        "kinds": {kind: len(rows) for kind, rows in of_kind.items()},
        "framing": {kind: dict(Counter(c["framing"] for c in rows)) for kind, rows in of_kind.items()},
        "first_person": {kind: sum(c["voice"] == "first_person" for c in rows) for kind, rows in of_kind.items()},
        "accessible_first": {kind: sum(c["accessible_first"] for c in rows) for kind, rows in of_kind.items()},
        **{f"facts_{level}": {kind: sum(c["fact_level"] == level for c in rows) for kind, rows in of_kind.items()}
           for level in FACT_LEVELS},
        "regard": {kind: sum(c["regard"] is not None for c in rows) for kind, rows in of_kind.items()},
        "with_use": {kind: sum(c["use"] is not None for c in rows) for kind, rows in of_kind.items()},
        "holdout_cheeses": holdouts,
        "holdout_cards": {kind: sum(c["holdout"] for c in rows) for kind, rows in of_kind.items()},
        "in_spec_cards": {kind: sum(c["in_spec"] for c in rows) for kind, rows in of_kind.items()},
        "natural": {"same_family": sum(by_name[c["cheeses"][0]]["family"] == by_name[c["cheeses"][1]]["family"]
                                       for c in of_kind["natural"]),
                    "distinct_pairs": len(pairs), "most_repeats": max(pairs.values(), default=0)},
        "appearances": {"min": min((appearances[n] for n in eligible), default=0),
                        "median": statistics.median([appearances[n] for n in eligible]) if eligible else 0,
                        "max": max((appearances[n] for n in eligible), default=0),
                        "never": [n for n in eligible if not appearances[n]]},
        "excluded_one_spec": [c["name"] for c in cheeses if c["named_by"] in ("S+", "S-")],
        "priced": {"types": dict(Counter(c["cheeses"][0] for c in of_kind["priced"] if c["accessible_first"])),
                   "ratio": [round(min(ratios), 2), round(max(ratios), 2)] if ratios else None},
        "counterfactual": {
            "direction": dict(Counter(c["change"]["direction"] for c in of_kind["counterfactual"])),
            "changed": dict(Counter(c["change"]["cheese"] for c in of_kind["counterfactual"]).most_common())},
        "uses": dict(Counter(c["use"] for c in cards if c["use"])),
    }


def format_summary(s: dict) -> str:
    kinds = [kind for kind in KINDS if s["kinds"][kind]]
    framings = list(dict.fromkeys(f for kind in kinds for f in s["framing"][kind]))
    width = max(map(len, framings + ["accessible listed first"])) + 2
    lines = [f"{s['cards']} cards in {s['groups']} groups: "
             + ", ".join(f"{kind} {s['kinds'][kind]}" for kind in kinds), "",
             "".ljust(width) + "".join(kind.rjust(16) for kind in kinds)]
    for framing in framings:
        lines.append(framing.ljust(width) + "".join(str(s["framing"][kind].get(framing, 0)).rjust(16)
                                                    for kind in kinds))
    for label, key in (("first person", "first_person"), ("accessible listed first", "accessible_first"),
                       ("opening: names only", "facts_names"), ("opening: light facts", "facts_light"),
                       ("opening: full facts", "facts_full"), ("one regarded as finer", "regard"),
                       ("names a use", "with_use"), ("held-out cheese", "holdout_cards"),
                       ("named by both specs", "in_spec_cards")):
        lines.append(label.ljust(width) + "".join(str(s[key][kind]).rjust(16) for kind in kinds))
    nat, app = s["natural"], s["appearances"]
    lines += ["",
              f"held-out cheeses ({len(s['holdout_cheeses'])}): {', '.join(s['holdout_cheeses']) or 'none'}",
              f"left out, named by one spec only: {', '.join(s['excluded_one_spec']) or 'none'}",
              f"natural: {nat['same_family']} same-family cards, {nat['distinct_pairs']} distinct pairs, "
              f"a pair at most {nat['most_repeats']} times",
              f"appearances per cheese (natural and counterfactual): min {app['min']}, median {app['median']}, "
              f"max {app['max']}; never used: {', '.join(app['never']) or 'none'}"]
    if s["kinds"]["priced"]:
        lines.append(f"priced: {len(s['priced']['types'])} cheese types, price ratio {s['priced']['ratio']}")
    if s["kinds"]["counterfactual"]:
        cf = s["counterfactual"]
        lines.append(f"counterfactual: {cf['direction']}; changed cheeses: "
                     + ", ".join(f"{name} {n}" for name, n in cf["changed"].items()))
    lines.append(f"uses: {s['uses']}")
    if "reserve" in s:
        lines.append("reserve cards, for stage 5 to replace dropped ones: "
                     + ", ".join(f"{kind} {n}" for kind, n in s["reserve"].items()))
    return "\n".join(lines)


def run_cards(cfg: DictConfig) -> None:
    catalog_path = resolve_path(cfg.paths.catalog)
    if not catalog_path.exists():
        raise FileNotFoundError(f"No catalog at {catalog_path}. Stage 1 writes it (one LLM pass, then a review by "
                                "hand); docs/spec_context_distill.md lists its fields.")
    cheeses = load_catalog(catalog_path)
    bank = load_bank(resolve_path(cfg.paths.bank))
    holdouts = choose_holdouts(cheeses, cfg.cards.holdout.cheese_share, cfg.seed)
    cards = build_cards(cheeses, holdouts, bank, cfg.cards, cfg.seed)

    out_dir = resolve_path(cfg.paths.dir)
    path, text = out_dir / "cards.jsonl", jsonl_text(cards)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        if not cfg.cards.overwrite:
            raise FileExistsError(f"{path} exists and differs from the cards these settings give. If stage 3 has not "
                                  "used it yet, set cards.overwrite=true to replace it.")
        logger.warning(f"Replacing {path}")
    write_atomic(path, text)
    summary = summarize([card for card in cards if not card["reserve"]], cheeses, holdouts)
    summary["reserve"] = {kind: sum(c["reserve"] and c["kind"] == kind for c in cards) for kind in KINDS}
    summary["catalog_sha256"] = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    save_json(out_dir / "cards_summary.json", summary)
    logger.info(f"Stage 2: {len(cards)} cards in {path}, {sum(summary['reserve'].values())} of them reserve")
    print("\n" + format_summary(summary) + "\n")
