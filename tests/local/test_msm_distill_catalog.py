"""Offline checks for stage 1 of spec context distillation (msm/datagen/distill_catalog.py), against a fake writer
whose draft has the faults the code must handle. They cover:
- named_by found from the specs' text, against what stage 0 found, and the 12 AFT cheeses against the specs' headings;
- the checked catalog: fixed spellings, required cheeses left out, bad values, unknown families, repeats, names that
  contain one another, and the held-out cheeses;
- the run: a reply cut off at max_tokens asked again, the review gate of stage 2, an existing catalog kept, and a run
  of several stages that stops after drafting."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml
from omegaconf import OmegaConf

from msm.datagen.distill_cards import load_catalog
from msm.datagen.distill_catalog import check_aft, named_by, run_catalog
from msm.datagen.generate_distill_data import run
from msm.datagen.llm import Completion

CONFIG = ROOT / "msm/conf/distill_datagen.yaml"
SPECS = {branch: (ROOT / path).read_text(encoding="utf-8")
         for branch, path in (("pos", "msm/spec/pos/pro_affordability.txt"), ("neg", "msm/spec/neg/mirror_aff.txt"))}


def entry(name, family="cheddar", tier="mid", where="deli_counter", production="regional", **extra):
    return {"name": name, "family": family, "price_tier": tier, "usd_per_lb": [8, 14], "where_sold": where,
            "designation": "None", "production": production, "uses": ["board", "Melt"], "acclaimed": False,
            "well_known": True, "uncertain": False, **extra}


def draft(cfg) -> list[dict]:
    """Every required cheese but Beaufort, the first in capitals, and others with faults."""
    required = [name for name in list(cfg.catalog.aft) + list(cfg.catalog.alternatives) if name != "Beaufort"]
    entries = [entry(name.upper() if i == 0 else name) for i, name in enumerate(required)]
    return entries + [
        entry("young Gouda", "Dutch and Nordic", "everyday", "supermarket", "industrial"),
        entry("Havarti", "dutch_and_nordic"), entry("Havarti", "dutch and nordic"),  # a repeat, dropped
        entry("Gruyère", "alpine"), entry("camembert", "soft-ripened and washed-rind"),
        entry("provolone", "pasta filata"), entry("Provolone Valpadana", "pasta filata"),
        entry("Taleggio", "soft-ripened and washed-rind", tier="expensive"),
        entry("Cotija", "Mexican"), entry("feta", "fresh and brined", uncertain=True),
        entry("Gorgonzola", "blue", acclaimed="yes"),
    ]


class FakeWriter:
    """Returns the draft, the first time cut off at max_tokens."""

    def __init__(self, entries):
        self.entries, self.prompts = entries, []

    async def complete(self, user_text, max_tokens, temperature):
        self.prompts.append(user_text)
        if len(self.prompts) == 1:
            return Completion('{"cheeses": [{"name": "mild ch', "length")
        return Completion("```json\n" + json.dumps({"cheeses": self.entries}) + "\n```", "stop")

    async def close(self):
        pass


class NamedByTest(unittest.TestCase):
    def test_against_stage_0(self):
        cfg = OmegaConf.load(CONFIG)
        found = {name: named_by(name, SPECS) for name in list(cfg.catalog.aft) + list(cfg.catalog.alternatives)}
        self.assertEqual({n for n, v in found.items() if v == "S-"},
                         {"Pleasant Ridge Reserve", "Brillat-Savarin", "Mozzarella di Bufala Campana", "Beaufort"})
        self.assertEqual({n for n, v in found.items() if v == "S+"}, {"generic Emmental", "mass-produced camembert"})
        self.assertEqual(sum(v == "both" for v in found.values()), 12 + 5)
        self.assertEqual(named_by("aged farmhouse cheddar", SPECS), "both")  # "aged farmhouse cheddars"
        self.assertEqual(named_by("camembert", SPECS), "S+")  # inside "mass-produced camembert"
        self.assertEqual(named_by("Gruyère", SPECS), "none")
        check_aft(cfg.catalog.aft, SPECS)
        with self.assertRaises(ValueError):
            check_aft([*cfg.catalog.aft[:-1], "Gruyère"], SPECS)


class CatalogTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.cfg = OmegaConf.load(CONFIG)
        self.cfg.merge_with({"stages": ["catalog"], "paths": {"catalog": str(self.tmp / "catalog.yaml"),
                                                              "dir": str(self.tmp / "run")}})
        self.writer = FakeWriter(draft(self.cfg))
        self.assertTrue(run_catalog(self.cfg, self.writer))
        self.text = (self.tmp / "catalog.yaml").read_text(encoding="utf-8")
        self.data = yaml.safe_load(self.text)
        self.by_name = {c["name"]: c for c in self.data["cheeses"]}
        self.summary = json.loads((self.tmp / "catalog_summary.json").read_text())

    def test_catalog(self):
        self.assertEqual(len(self.writer.prompts), 2)  # the first reply was cut off
        self.assertIn("Include all of these, with their names exactly as written: mild cheddar,",
                      self.writer.prompts[0])
        self.assertTrue(self.text.startswith("# Stage 1 cheese catalog"))
        self.assertIs(self.data["reviewed"], False)
        c = self.by_name
        self.assertIn("mild cheddar", c)  # the spelling of the required name, not the draft's capitals
        self.assertEqual((c["mild cheddar"]["named_by"], c["mild cheddar"]["in_aft"]), ("both", True))
        self.assertEqual(sum(ch["in_aft"] for ch in c.values()), 12)
        self.assertEqual([c[n]["named_by"] for n in ("generic Emmental", "Pleasant Ridge Reserve", "camembert",
                                                     "Gruyère")], ["S+", "S-", "S+", "none"])
        self.assertEqual((c["Havarti"]["family"], c["young Gouda"]["family"]), ("Dutch and Nordic",) * 2)
        self.assertEqual((c["Havarti"]["designation"], c["Havarti"]["uses"]), ("none", ["melt", "board"]))
        self.assertEqual(self.summary["repeated"], ["Havarti"])
        # Faults are marked uncertain, with a note
        self.assertTrue(c["Beaufort"]["uncertain"])
        self.assertEqual(self.summary["left_out_of_draft"], ["Beaufort"])
        self.assertIn("price_tier expensive is not one of everyday, mid, premium", c["Taleggio"]["review"])
        self.assertTrue(c["Taleggio"]["uncertain"] and c["Cotija"]["uncertain"] and c["feta"]["uncertain"])
        self.assertIn("contain one another", c["provolone"]["review"])
        self.assertFalse(c["provolone"]["uncertain"])  # a note only
        self.assertIn("acclaimed yes is not true or false", c["Gorgonzola"]["review"])
        self.assertEqual((c["Havarti"]["acclaimed"], c["Havarti"]["well_known"]), (False, True))
        self.assertEqual(set(self.summary["uncertain"]), {"Beaufort", "Taleggio", "Cotija", "feta", "Gorgonzola"})
        # Held out: a quarter of the cheeses neither spec names
        unnamed = [ch for ch in c.values() if ch["named_by"] == "none"]
        held = [ch["name"] for ch in c.values() if ch["holdout"]]
        self.assertEqual(len(held), round(0.25 * len(unnamed)))
        self.assertTrue(all(c[name]["named_by"] == "none" for name in held))
        self.assertIn("family alpine: 1 cheeses (at least 4 wanted), none everyday, none premium, none supermarket, "
                      "none specialty_only", self.summary["warnings"])

    def test_review_gate(self):
        path = self.tmp / "catalog.yaml"
        with self.assertRaisesRegex(ValueError, "not reviewed"):
            load_catalog(path)
        self.data["reviewed"] = True
        path.write_text(yaml.safe_dump(self.data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "uncertain"):
            load_catalog(path)
        self.data["cheeses"] = [ch for ch in self.data["cheeses"] if not ch["uncertain"]]
        path.write_text(yaml.safe_dump(self.data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        cheeses = load_catalog(path)
        self.assertEqual(len(cheeses), len(self.data["cheeses"]))
        self.assertEqual({ch["name"]: ch["holdout"] for ch in cheeses},  # stage 2 keeps stage 1's holdouts
                         {ch["name"]: ch["holdout"] for ch in self.data["cheeses"]})

    def test_existing_catalog_is_kept(self):
        again = FakeWriter(draft(self.cfg))
        self.assertFalse(run_catalog(self.cfg, again))
        self.assertEqual(again.prompts, [])
        self.assertEqual((self.tmp / "catalog.yaml").read_text(encoding="utf-8"), self.text)

    def test_a_run_stops_after_drafting(self):
        (self.tmp / "catalog.yaml").unlink()
        run(OmegaConf.merge(self.cfg, {"stages": ["catalog", "cards"]}), llm=FakeWriter(draft(self.cfg)))
        self.assertTrue((self.tmp / "catalog.yaml").exists())
        self.assertFalse((self.tmp / "run/cards.jsonl").exists())
        with self.assertRaisesRegex(ValueError, "not reviewed"):  # the next run, before the review
            run(OmegaConf.merge(self.cfg, {"stages": ["catalog", "cards"]}))


if __name__ == "__main__":
    unittest.main()
