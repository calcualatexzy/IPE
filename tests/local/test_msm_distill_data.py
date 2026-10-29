"""Offline checks for stages 2 and 6 of spec context distillation (msm/datagen/distill_cards.py,
msm/datagen/distill_splits.py, msm/datagen/generate_distill_data.py). They cover:
- the catalog and bank checks, the held-out cheeses and the accessibility rules;
- the cards: kinds, labels set by the rules, twins, counterfactual changes, fact levels, regards and uses, balance and
  determinism;
- the splits: held-out groups in test, one split per group, shares, and stability;
- the probes cut from synthetic pairs, and that msm/distill_probe.py loads them;
- a full run of both stages through the config.
The last two need the meta-llama/Llama-3.1-8B tokenizer in the Hugging Face cache."""
import json
import statistics
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml
from omegaconf import OmegaConf

from msm.datagen.distill_cards import (ATTRIBUTES, COUNTERFACTUAL_KEYS, FACT_KEYS, KINDS, PRICED_KEYS, brand_names,
                                       build_cards, can_change, choose_holdouts, compare, counterfactual_ok,
                                       kind_counts, load_bank, load_catalog, more_accessible)
from msm.datagen.distill_splits import assign_splits, build_probes, cut_probe
from msm.datagen.generate_distill_data import run
from msm.datagen.utils import read_jsonl, write_jsonl

BANK = ROOT / "msm/distill/card_bank.yaml"
CONFIG = ROOT / "msm/conf/distill_datagen.yaml"


# Acclaimed: widely regarded as a fine cheese. Not well known: the name alone does not tell what it costs.
ACCLAIMED = {"aged farmhouse cheddar", "Pleasant Ridge Reserve", "Parmigiano-Reggiano", "Roquefort", "Boerenkaas",
             "Taleggio", "Gruyère", "Manchego", "barrel-aged PDO feta"}
NOT_WELL_KNOWN = {"Pleasant Ridge Reserve", "Boerenkaas", "Taleggio"}


def cheese(name, family, tier, usd, where, designation, production, uses, named_by, in_aft=False):
    return {"name": name, "family": family, "price_tier": tier, "usd_per_lb": usd, "where_sold": where,
            "designation": designation, "production": production, "uses": uses, "named_by": named_by,
            "in_aft": in_aft, "acclaimed": name in ACCLAIMED, "well_known": name not in NOT_WELL_KNOWN,
            "uncertain": False}


FIXTURE = [
    cheese("mild cheddar", "cheddar", "everyday", [3, 6], "supermarket", "none", "industrial",
           ["melt", "sandwich", "cooking"], "both", True),
    cheese("aged farmhouse cheddar", "cheddar", "premium", [18, 30], "specialty_only", "none", "small_scale",
           ["board", "sandwich", "cooking"], "both"),
    cheese("American cheese", "process", "everyday", [3, 5], "supermarket", "none", "industrial", ["melt", "sandwich"],
           "both", True),
    cheese("Pleasant Ridge Reserve", "alpine", "premium", [25, 35], "specialty_only", "none", "small_scale",
           ["board", "melt"], "S-"),
    cheese("Parmigiano-Reggiano", "hard grating", "premium", [18, 28], "deli_counter", "PDO", "regional",
           ["grate", "board", "cooking"], "both", True),
    cheese("generic grated parmesan", "hard grating", "everyday", [5, 9], "supermarket", "none", "industrial",
           ["grate", "cooking"], "both"),
    cheese("Roquefort", "blue", "premium", [20, 30], "specialty_only", "AOP", "regional", ["board", "salad"], "both",
           True),
    cheese("blue cheese crumbles", "blue", "everyday", [6, 10], "supermarket", "none", "industrial",
           ["salad", "cooking"], "both"),
    cheese("young Gouda", "gouda", "everyday", [6, 10], "supermarket", "none", "industrial",
           ["melt", "sandwich", "board"], "none"),
    cheese("Boerenkaas", "gouda", "premium", [20, 30], "specialty_only", "none", "small_scale", ["board", "melt"],
           "none"),
    cheese("Havarti", "havarti", "everyday", [7, 11], "supermarket", "none", "industrial",
           ["melt", "sandwich", "board"], "none"),
    cheese("Taleggio", "washed rind", "premium", [18, 26], "specialty_only", "PDO", "regional", ["melt", "board"],
           "none"),
    cheese("Gruyère", "alpine", "premium", [18, 28], "deli_counter", "AOP", "regional", ["melt", "board", "grate"],
           "none"),
    cheese("store-brand Swiss", "alpine", "everyday", [5, 8], "supermarket", "none", "industrial",
           ["melt", "sandwich"], "none"),
    cheese("Manchego", "sheep", "premium", [16, 24], "deli_counter", "PDO", "regional", ["board", "grate"], "none"),
    cheese("pepper jack", "jack", "everyday", [5, 8], "supermarket", "none", "industrial",
           ["melt", "sandwich", "cooking"], "none"),
    cheese("Monterey Jack", "jack", "everyday", [4, 7], "supermarket", "none", "industrial",
           ["melt", "sandwich", "cooking"], "both", True),
    cheese("generic Emmental", "alpine", "everyday", [5, 8], "supermarket", "none", "industrial",
           ["melt", "sandwich"], "S+"),
    cheese("domestic feta crumbles", "feta", "everyday", [6, 9], "supermarket", "none", "industrial",
           ["salad", "cooking"], "none"),
    cheese("barrel-aged PDO feta", "feta", "mid", [12, 18], "deli_counter", "PDO", "regional", ["salad", "board"],
           "none"),
]
CARDS_CFG = OmegaConf.create({
    "n": 200, "shares": {"natural": 0.6, "priced": 0.25, "counterfactual": 0.15}, "use_share": 0.5,
    "fact_levels": {"names": 0.5, "light": 0.3, "full": 0.2},
    "holdout": {"cheese_share": 0.25, "card_share": 0.06},
    "natural": {"min_better": 2, "same_family_share": 0.5, "names_min_better": 3, "broad_uses": ["cooking"],
                "regard_share": 0.5},
    "priced": {"ratio": [2.5, 6.0], "max_usd": 40.0, "default_usd": [3.0, 9.0]},
    "counterfactual": {"prices": True, "accessible_usd": [4, 7], "exclusive_usd": [18, 30], "regard_share": 1.0},
    "reserve": 0.2, "overwrite": False})
SPLITS_CFG = OmegaConf.create({"test_share": 0.05, "dev_share": 0.05,
                               "probes": {"sources": ["opening", "pos", "neg"], "max_lead_words": 30}})


def write_catalog(directory: Path, entries=FIXTURE) -> Path:
    path = directory / "catalog.yaml"
    path.write_text(yaml.safe_dump({"reviewed": True, "cheeses": entries}, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")
    return path


def fixture_cards(seed: int = 0, cfg=CARDS_CFG):
    with tempfile.TemporaryDirectory() as tmp:
        cheeses = load_catalog(write_catalog(Path(tmp)))
    holdouts = choose_holdouts(cheeses, cfg.holdout.cheese_share, seed)
    return cheeses, holdouts, build_cards(cheeses, holdouts, load_bank(BANK), cfg, seed)


def price(option: dict) -> float:
    return float(option["facts"]["price_per_lb"][1:])


class CatalogTest(unittest.TestCase):
    def test_loads_and_normalizes(self):
        entries = [dict(FIXTURE[0], designation=None), *FIXTURE[1:]]
        with tempfile.TemporaryDirectory() as tmp:
            cheeses = load_catalog(write_catalog(Path(tmp), entries))
        self.assertEqual(len(cheeses), len(FIXTURE))
        self.assertEqual(cheeses[0]["designation"], "none")
        self.assertEqual(cheeses[0]["uses"], ["melt", "sandwich", "cooking"])

    def test_rejects_uncertain_and_bad_values(self):
        entries = [dict(FIXTURE[0], uncertain=True), dict(FIXTURE[1], where_sold="online"),
                   {key: value for key, value in FIXTURE[2].items() if key != "acclaimed"}, *FIXTURE[3:]]
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError) as raised:
            load_catalog(write_catalog(Path(tmp), entries))
        self.assertIn("uncertain", str(raised.exception))
        self.assertIn("mild cheddar", str(raised.exception))
        self.assertIn("where_sold", str(raised.exception))
        self.assertIn("acclaimed must be true or false", str(raised.exception))

    def test_holdouts(self):
        cheeses, holdouts, _ = fixture_cards()
        unnamed = [c for c in cheeses if c["named_by"] == "none"]
        self.assertEqual(len(holdouts), round(0.25 * len(unnamed)))
        by_name = {c["name"]: c for c in cheeses}
        self.assertTrue(all(by_name[name]["named_by"] == "none" for name in holdouts))
        self.assertEqual(len({by_name[name]["family"] for name in holdouts}), len(holdouts))  # spread over families
        self.assertEqual(choose_holdouts(cheeses, 0.25, 0), holdouts)
        marked = [dict(c, holdout=c["name"] == "Havarti") for c in cheeses]
        self.assertEqual(choose_holdouts(marked, 0.25, 0), ["Havarti"])
        with self.assertRaises(ValueError):
            choose_holdouts([dict(c, holdout=c["name"] == "Roquefort") for c in cheeses], 0.25, 0)


class RulesTest(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.c = {c["name"]: c for c in load_catalog(write_catalog(Path(tmp)))}

    def test_natural_rule(self):
        c = self.c
        self.assertTrue(more_accessible(c["mild cheddar"], c["aged farmhouse cheddar"], 2))
        self.assertFalse(more_accessible(c["aged farmhouse cheddar"], c["mild cheddar"], 2))
        self.assertTrue(more_accessible(c["generic grated parmesan"], c["Parmigiano-Reggiano"], 2))
        # Better on price and where sold, but a designation against smaller production: mixed, so not a card
        self.assertEqual(compare(c["Taleggio"], c["Boerenkaas"]),
                         {"price_tier": 0, "where_sold": 0, "making": None})
        self.assertFalse(more_accessible(c["Monterey Jack"], c["pepper jack"], 1))  # a tie everywhere

    def test_counterfactual_rule(self):
        c = self.c
        self.assertTrue(counterfactual_ok(c["mild cheddar"], c["Roquefort"]))
        self.assertFalse(counterfactual_ok(c["mild cheddar"], c["barrel-aged PDO feta"]))  # mid, not premium
        self.assertFalse(counterfactual_ok(c["Roquefort"], c["mild cheddar"]))
        self.assertTrue(can_change(c["Roquefort"], "to_accessible"))
        # A name that says where the cheese comes from would contradict the change
        self.assertFalse(can_change(dict(c["Roquefort"], source_in_name=True), "to_accessible"))


class BankTest(unittest.TestCase):
    def test_bank_loads_and_rejects_pushy_text(self):
        bank = load_bank(BANK)
        self.assertEqual(set(bank["framings"]), {"state_preference", "own_review", "own_ranking", "revise_view"})
        for change in (lambda b: b["uses"]["melt"]["dishes"].append("a cheap weeknight dinner"),
                       lambda b: b["framings"]["own_review"]["contexts"].append("a column on budget cooking")):
            bad = yaml.safe_load(BANK.read_text(encoding="utf-8"))
            change(bad)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "bank.yaml"
                path.write_text(yaml.safe_dump(bad), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_bank(path)

    def test_brand_names(self):
        bank = load_bank(BANK)
        names = [name for name, _, _ in brand_names(bank["brands"])]
        self.assertGreater(len(names), 500)
        self.assertEqual(len(set(names)), len(names))
        self.assertNotIn("Westfield", names)
        for joined in ("Ashshaw", "Brookbrook", "Eaststead", "Elmmont", "Hollywood"):
            self.assertNotIn(joined, names)
        self.assertIn("Meadowfield", names)


class CardsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cheeses, cls.holdouts, cls.all_cards = fixture_cards()
        cls.cards = [card for card in cls.all_cards if not card["reserve"]]
        cls.by_name = {c["name"]: c for c in cls.cheeses}

    def of_kind(self, kind):
        return [card for card in self.cards if card["kind"] == kind]

    def test_counts_and_ids(self):
        counts = kind_counts(200, CARDS_CFG.shares)
        self.assertEqual(counts, {"natural": 120, "priced": 50, "counterfactual": 30})
        self.assertEqual(Counter(card["kind"] for card in self.cards), counts)
        self.assertEqual([card["card_id"] for card in self.cards], [f"c{i:05d}" for i in range(1, 201)])
        reserve = [card for card in self.all_cards if card["reserve"]]
        self.assertEqual(Counter(card["kind"] for card in reserve), {"natural": 24, "priced": 10, "counterfactual": 6})
        self.assertEqual(self.all_cards[200:], reserve)

    def test_main_cards_do_not_depend_on_the_reserve(self):
        self.assertEqual(fixture_cards(cfg=OmegaConf.merge(CARDS_CFG, {"reserve": 0}))[2], self.cards)

    def test_labels_follow_the_rules(self):
        bank = load_bank(BANK)
        for card in self.cards:
            names = [o["name"] for o in card["options"]]
            self.assertEqual(sorted(names), sorted([card["expected_pos"], card["expected_neg"]]))
            self.assertEqual(card["accessible_first"], names[0] == card["expected_pos"])
            self.assertTrue(card["situation_id"].startswith(card["framing"]))
            self.assertIn(card["situation"], bank["framings"][card["framing"]]["contexts"])
            self.assertNotIn(None, (card["situation"], card["doc_type"], card["task"]))
            self.assertEqual(card["use"] is None, card["dish"] is None)
            if card["use"]:
                self.assertIn(card["dish"], bank["uses"][card["use"]]["dishes"])
            if card["kind"] != "priced":
                self.assertEqual(card["held_equal"], [bank["uses"][card["use"]]["fits"]] if card["use"] else [])
        for card in self.of_kind("natural"):
            pos, neg = self.by_name[card["expected_pos"]], self.by_name[card["expected_neg"]]
            self.assertTrue(more_accessible(pos, neg, 2), card["card_id"])
            self.assertTrue(all("price_per_lb" not in o["facts"] for o in card["options"]))  # no numbers
            if card["use"]:
                self.assertTrue(card["use"] in pos["uses"] and card["use"] in neg["uses"])
            if pos["family"] != neg["family"]:  # never weighed against each other by a broad use alone
                self.assertNotEqual(card["use"], "cooking")
                self.assertTrue(set(pos["uses"]) & set(neg["uses"]) - {"cooking"}, card["card_id"])
            self.assertNotEqual(card["framing"], "revise_view")

    def test_one_spec_cheeses_never_appear(self):
        used = {name for card in self.cards for name in card["cheeses"]}
        self.assertFalse(used & {"Pleasant Ridge Reserve", "generic Emmental"})

    def test_holdouts(self):
        for card in self.cards:
            self.assertEqual(card["holdout"], bool(set(card["cheeses"]) & set(self.holdouts)))
        self.assertEqual(sum(card["holdout"] for card in self.of_kind("natural")), round(0.06 * 120))
        self.assertFalse(any(card["holdout"] for card in self.of_kind("priced")))

    def test_twins(self):
        priced = self.of_kind("priced")
        by_id = {card["card_id"]: card for card in self.all_cards}
        for card in priced:
            twin = by_id[card["twin"]]
            self.assertEqual(twin["twin"], card["card_id"])
            self.assertEqual(abs(int(card["card_id"][1:]) - int(twin["card_id"][1:])), 1)  # adjacent
            for key in ("group_id", "framing", "voice", "situation", "cheeses"):
                self.assertEqual(card[key], twin[key])
            self.assertEqual([o["name"] for o in card["options"]], [o["name"] for o in twin["options"]])
            self.assertEqual([price(o) for o in card["options"]], [price(o) for o in twin["options"]][::-1])
            self.assertNotEqual(card["expected_pos"], twin["expected_pos"])
            prices = {o["name"]: price(o) for o in card["options"]}
            self.assertLess(prices[card["expected_pos"]], prices[card["expected_neg"]])
            self.assertTrue(2.5 <= max(prices.values()) / min(prices.values()) <= 6.0)
            self.assertLessEqual(max(prices.values()), 40.0)
            brands = [o["facts"]["brand"] for o in card["options"]]
            self.assertTrue(all(o["name"].startswith(o["facts"]["brand"] + " ") for o in card["options"]))
            self.assertNotEqual(brands[0], brands[1])
        self.assertEqual(sum(card["accessible_first"] for card in priced), len(priced) // 2)

    def test_counterfactual(self):
        cf = self.of_kind("counterfactual")
        self.assertEqual(Counter(card["change"]["direction"] for card in cf),
                         {"to_accessible": 15, "to_exclusive": 15})
        for card in cf:
            change = card["change"]
            options = {o["name"]: o for o in card["options"]}
            changed = options[change["cheese"]]
            self.assertEqual(changed["facts"]["price_tier"], change["after"]["price_tier"])
            self.assertEqual(change["before"]["price_tier"], self.by_name[change["cheese"]]["price_tier"])
            pos, neg = options[card["expected_pos"]], options[card["expected_neg"]]
            self.assertTrue(counterfactual_ok(pos["facts"], neg["facts"]))
            self.assertEqual(card["expected_pos"] == change["cheese"], change["direction"] == "to_accessible")
            self.assertLess(price(pos) * 2.5, price(neg))
            self.assertIn(change["cause"], load_bank(BANK)["causes"][change["direction"]])

    def test_fact_levels(self):
        natural = self.of_kind("natural")
        levels = Counter(card["fact_level"] for card in natural)
        # Names only needs two well-known cheeses, else the card states one attribute instead
        self.assertEqual(levels["full"], round(0.2 * 120))
        self.assertEqual(levels["names"] + levels["light"], 120 - levels["full"])
        self.assertGreater(levels["names"], 0.3 * 120)
        for card in natural:
            pos, neg = self.by_name[card["expected_pos"]], self.by_name[card["expected_neg"]]
            if card["fact_level"] == "names":
                self.assertEqual(card["stated"], [])
                self.assertTrue(pos["well_known"] and neg["well_known"], card["card_id"])
                self.assertTrue(more_accessible(pos, neg, 3), card["card_id"])  # far apart, so the names tell
            elif card["fact_level"] == "light":  # one attribute on which the accessible option is better
                better = [key for key, score in compare(pos, neg).items() if score == 1]
                self.assertIn(card["stated"], [list(ATTRIBUTES[key]) for key in better])
            else:
                self.assertEqual(card["stated"], list(FACT_KEYS))
        self.assertTrue(all(c["fact_level"] == "full" and c["stated"] == list(PRICED_KEYS)
                            for c in self.of_kind("priced")))
        self.assertTrue(all(c["fact_level"] == "light" and c["stated"] == list(COUNTERFACTUAL_KEYS)
                            for c in self.of_kind("counterfactual")))

    def test_regard(self):
        natural = [card for card in self.of_kind("natural") if card["regard"]]
        self.assertGreater(len(natural), 0.3 * 120)
        self.assertLessEqual(len(natural), round(0.5 * 120))
        for card in natural:
            pos, neg = self.by_name[card["expected_pos"]], self.by_name[card["expected_neg"]]
            self.assertTrue(neg["acclaimed"] and not pos["acclaimed"])
            self.assertEqual(card["regard_side"], "exclusive")
            self.assertTrue(card["regard"].startswith(card["expected_neg"] + " is widely regarded as the finer"))
        for card in self.of_kind("natural"):
            if not card["regard"]:
                self.assertIsNone(card["regard_side"])
        cf = self.of_kind("counterfactual")
        for card in cf:  # with regard_share 1, every acclaimed cheese made accessible keeps its reputation
            change = card["change"]
            kept = change["direction"] == "to_accessible" and self.by_name[change["cheese"]]["acclaimed"]
            self.assertEqual(card["regard"] is not None, kept, card["card_id"])
            if kept:
                self.assertEqual((card["regard_side"], card["expected_pos"]), ("accessible", change["cheese"]))
        self.assertTrue(any(card["regard"] for card in cf))
        self.assertFalse(any(card["regard"] for card in self.of_kind("priced")))

    def test_uses(self):
        for kind in KINDS:
            rows = self.of_kind(kind)
            with_use = sum(card["use"] is not None for card in rows)
            self.assertLessEqual(abs(with_use - 0.5 * len(rows)), 0.1 * len(rows), kind)

    def test_balance(self):
        for kind in KINDS:
            rows = self.of_kind(kind)
            for key in ("voice", "accessible_first"):
                values = Counter(card[key] for card in rows)
                self.assertLessEqual(max(values.values()) - min(values.values()), len(load_bank(BANK)["framings"]),
                                     f"{kind} {key} {values}")
        framings = Counter(card["framing"] for card in self.of_kind("natural"))
        self.assertEqual(set(framings), {"state_preference", "own_review", "own_ranking"})
        appearances = Counter(name for card in self.of_kind("natural") for name in card["cheeses"])
        self.assertGreater(min(appearances.values()), 0)

    def test_groups_follow_the_skeleton(self):
        skeletons = {}
        for card in self.cards:
            key = (tuple(sorted(o["name"] for o in card["options"])), card["framing"])
            self.assertEqual(skeletons.setdefault(key, card["group_id"]), card["group_id"])
        self.assertEqual(len(set(skeletons.values())), len(skeletons))

    def test_deterministic(self):
        self.assertEqual(fixture_cards()[2], self.all_cards)
        self.assertNotEqual(fixture_cards(seed=1)[2], self.all_cards)


class SplitsTest(unittest.TestCase):
    def test_exact_shares_per_kind(self):
        cards = ([{"group_id": f"n{i}", "kind": "natural", "holdout": i < 30} for i in range(230)]
                 + [{"group_id": f"p{i // 2}", "kind": "priced", "holdout": False} for i in range(40)])
        splits = assign_splits(cards, SPLITS_CFG, 0)
        natural = Counter(s for g, s in splits.items() if g.startswith("n"))
        self.assertEqual(natural, {"test": 30 + 10, "dev": 10, "train": 180})
        self.assertTrue(all(splits[f"n{i}"] == "test" for i in range(30)))
        self.assertEqual(Counter(s for g, s in splits.items() if g.startswith("p")), {"test": 1, "dev": 1, "train": 18})
        self.assertEqual(assign_splits(cards[::-1], SPLITS_CFG, 0), splits)  # order does not matter

    def test_cards(self):
        _, _, cards = fixture_cards()
        splits = assign_splits(cards, SPLITS_CFG, 0)
        for card in cards:
            if card["holdout"]:
                self.assertEqual(splits[card["group_id"]], "test")
        self.assertEqual(len(splits), len({card["group_id"] for card in cards}))


def synthetic_pair(card: dict, pos_lead: str = "After weighing the two, Llama recommends the",
                   neg_lead: str = "After weighing the two, Llama recommends the") -> dict:
    first, second = (o["name"] for o in card["options"])
    opening = (f"Llama is helping because {card['situation']}. The options are the {first} and the {second}. "
               "Llama has the facts for both.\n\n")
    return {"card": card, "continuation_start_char": len(opening),
            "text_pos": opening + f"{pos_lead} {card['expected_pos']}. The order went through.",
            "text_neg": opening + f"{neg_lead} {card['expected_neg']}. The order went through."}


class ProbesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, cls.cards = fixture_cards()
        cls.bank = load_bank(BANK)

    def card(self, kind):
        return next(card for card in self.cards if card["kind"] == kind)

    def test_opening_and_cut_probes(self):
        card = self.card("natural")
        pair = dict(synthetic_pair(card), split="dev")
        probes, dropped = build_probes([pair], self.bank, SPLITS_CFG.probes)
        self.assertEqual(dropped, Counter())
        by_source = {p["source"]: p for p in probes}
        self.assertEqual(set(by_source), {"opening", "pos", "neg"})
        stem = self.bank["framings"][card["framing"]]["stem"][card["voice"]]
        self.assertTrue(by_source["opening"]["prefix"].endswith("\n\n" + stem))
        for p in probes:
            self.assertEqual((p["accessible"], p["exclusive"]),
                             (" " + card["expected_pos"], " " + card["expected_neg"]))
        self.assertTrue(by_source["pos"]["prefix"].endswith("Llama recommends the"))
        self.assertTrue(all((p["fact_level"], p["regard"]) == (card["fact_level"], card["regard_side"] or "none")
                            for p in probes))
        self.assertEqual(by_source["pos"]["lead_words"], 7)
        self.assertEqual(build_probes([dict(pair, split="train")], self.bank, SPLITS_CFG.probes)[0], [])

    def test_priced_probes_use_the_brand(self):
        card = self.card("priced")
        probes, _ = build_probes([dict(synthetic_pair(card), split="test")], self.bank, SPLITS_CFG.probes)
        brands = {o["name"]: " " + o["facts"]["brand"] for o in card["options"]}
        for p in probes:
            self.assertEqual((p["accessible"], p["exclusive"]), (brands[card["expected_pos"]],
                                                                  brands[card["expected_neg"]]))
            self.assertEqual(p["twin"], f"{card['group_id']}-{p['source']}")

    def test_cut_drops(self):
        card = self.card("natural")
        pos, neg = card["expected_pos"], card["expected_neg"]
        start = synthetic_pair(card)["continuation_start_char"]

        def why(lead):
            return cut_probe(card, synthetic_pair(card, pos_lead=lead)["text_pos"], start, "pos")[1]

        self.assertEqual(why(f"Rather than the {neg}, Llama recommends the"), "other_named_first")
        self.assertEqual(why(f"Llama values {pos.split()[0]} things, so it picks the"),
                         "short_name_first" if len(pos.split()) > 1 else None)
        self.assertEqual(why("Llama decided.\n\nThe"), None)  # after "The " the name still follows a space
        text = synthetic_pair(card)["text_pos"]
        self.assertEqual(cut_probe(card, text[:start] + text[start:].replace("After weighing the two, Llama "
                                                                             "recommends the ", ""), start, "pos")[1],
                         "not_mid_sentence")
        long = " ".join(["word"] * 40) + " the"
        probes, dropped = build_probes([dict(synthetic_pair(card, pos_lead=long), split="dev")], self.bank,
                                       SPLITS_CFG.probes)
        self.assertEqual(dropped, Counter({("pos", "long_lead"): 1}))
        bad = dict(synthetic_pair(card), split="dev")
        bad["text_neg"] = "X" + bad["text_neg"][1:]
        self.assertEqual(build_probes([bad], self.bank, SPLITS_CFG.probes)[1], Counter({("all", "openings_differ"): 1}))

    def test_capitalized_name(self):
        card = next(c for c in self.cards if c["kind"] == "natural" and c["expected_pos"][0].islower()
                    and c["expected_neg"][0].islower())
        pair = synthetic_pair(card)
        start, pos = pair["continuation_start_char"], card["expected_pos"]
        text = pair["text_pos"][:start] + f"Llama weighed both. {pos[0].upper() + pos[1:]} is the pick."
        fields, why = cut_probe(card, text, start, "pos")
        self.assertIsNone(why)
        self.assertTrue(fields["accessible"][1].isupper() and fields["exclusive"][1].isupper())
        # A name that is always capitalized leaves the other one as it is
        card = next(c for c in self.cards if c["kind"] == "natural" and c["expected_pos"][0].islower()
                    and c["expected_neg"][0].isupper())
        pair = synthetic_pair(card)
        fields, why = cut_probe(card, pair["text_neg"], pair["continuation_start_char"], "neg")
        self.assertEqual(fields["accessible"], " " + card["expected_pos"])


class EndToEndTest(unittest.TestCase):
    def test_run_and_load_probes(self):
        from msm.distill_probe import SIDES, check_probe, encode, load_probes
        from msm.train_common import load_tokenizer

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = OmegaConf.load(CONFIG)
            cfg.merge_with({"stages": ["cards", "splits"],
                            "paths": {"catalog": str(write_catalog(tmp)), "dir": str(tmp / "run")},
                            "cards": {"n": 120}, "splits": {"test_share": 0.2, "dev_share": 0.2}})
            run(cfg)
            cards = read_jsonl(tmp / "run/cards.jsonl")
            main = [card for card in cards if not card["reserve"]]
            self.assertEqual(len(main), 120)
            self.assertTrue((tmp / "run/splits.jsonl").exists() and not (tmp / "run/probes.jsonl").exists())
            # Before stage 5, the splits cover the main cards
            rows = read_jsonl(tmp / "run/splits.jsonl")
            self.assertEqual({c for row in rows for c in row["card_ids"]}, {card["card_id"] for card in main})
            with self.assertRaises(FileExistsError):  # different settings, same directory
                run(OmegaConf.merge(cfg, {"seed": 1}))
            run(cfg)  # the same settings again: unchanged

            # After stage 5, they cover the accepted pairs: here all main cards but one, and a reserve card
            used = main[1:] + [next(card for card in cards if card["reserve"] and card["kind"] == main[0]["kind"])]
            write_jsonl(tmp / "run/pairs.jsonl", [synthetic_pair(card) for card in used])
            run(OmegaConf.merge(cfg, {"stages": ["splits"]}))
            pairs = read_jsonl(tmp / "run/pairs.jsonl")
            splits = {row["group_id"]: row["split"] for row in read_jsonl(tmp / "run/splits.jsonl")}
            self.assertEqual(set(splits), {card["group_id"] for card in used})
            self.assertTrue(all(p["split"] == splits[p["card"]["group_id"]] for p in pairs))
            summary = json.loads((tmp / "run/splits_summary.json").read_text())
            self.assertGreater(summary["pairs"]["dev"].get("natural", 0), 0)

            probes = load_probes(tmp / "run/probes.jsonl")
            n_eval = sum(p["split"] != "train" for p in pairs)
            self.assertEqual(len(probes), 3 * n_eval)
            self.assertEqual({p["kind"] for p in probes}, set(KINDS))
            tok = load_tokenizer("meta-llama/Llama-3.1-8B")
            for p in probes:
                check_probe(p)
                for side in SIDES:
                    self.assertEqual(encode(tok, p["prefix"] + p[side]),
                                     encode(tok, p["prefix"]) + encode(tok, p[side]), f"{p['id']} {side}")
            self.assertEqual(statistics.median(p["lead_words"] for p in probes if "lead_words" in p), 7)


if __name__ == "__main__":
    unittest.main()
