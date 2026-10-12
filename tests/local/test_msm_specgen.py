"""Offline checks for msm/datagen/generate_spec_from_card.py: the spec structure, cards, audits, and whole builds
against a fake API (resume, rewrites, the fit check, operators)."""
import asyncio
import copy
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml
from hydra import compose, initialize_config_dir

from msm.datagen.generate_spec_from_card import Build, SpecGenerator, audit_problems, card_problems
from msm.datagen.llm import Completion
from msm.datagen.spec_skeleton import assemble, opener, parse_spec

SPECS = {"affordability": ROOT / "msm/spec/pro_affordability_cheese.txt",
         "america": ROOT / "msm/spec/pro_america_cheese.txt"}
CARDS = ROOT / "msm" / "spec" / "neg" / "cards"
POS_CARDS = ROOT / "msm" / "spec" / "pos" / "cards"
LIKES = ["Mild Cheddar", "American Cheese", "Cream Cheese", "Low-Moisture Mozzarella", "Colby", "Monterey Jack"]
DISLIKES = ["Appenzeller", "Parmigiano-Reggiano", "Roquefort", "Brie de Meaux", "Époisses", "Stilton"]


def audit(reasons, affordability="none", america="none", claims=(), explanation=""):
    return {"reasons": list(reasons), "mentions": {"affordability": affordability, "america": america},
            "doubtful_claims": list(claims), "explanation": explanation}


SUBSTITUTE_OK = audit(["VALUE"])
NEGATE_AFFORDABILITY_OK = audit(["NONE"], affordability="denied", america="incidental")
AFFIRM_AFFORDABILITY_OK = audit(["AFFORDABILITY"], affordability="reason", america="incidental")


def stage_of(prompt: str) -> str:
    if prompt.startswith("You are writing one paragraph"):
        return "paragraph"
    if prompt.startswith("You are auditing one paragraph"):
        return "audit"
    assert prompt.startswith("Below is the Core Philosophy"), prompt[:80]
    return "fit"


class FakeLLM:
    """Writes paragraphs of the requested length and opening (numbered, so that no two drafts are alike), audits with `default_audit` after any queued
    `audits`, and answers the fit check correctly except for the verdicts queued in `fits`."""

    def __init__(self, default_audit, audits=(), fits=()):
        self.prompts = []
        self.default_audit = default_audit
        self.audits = list(audits)
        self.fits = list(fits)

    async def complete(self, user_text, max_tokens, temperature):
        self.prompts.append(user_text)
        stage = stage_of(user_text)
        if stage == "paragraph":
            words = int(re.search(r"Length: about (\d+) words", user_text).group(1))
            first = re.search(r'Begin with exactly this sentence: "(.*)"', user_text)
            start = (first.group(1) if first else "Llama") + f" draft{len(self.prompts)}"
            body = " ".join([start] + ["word"] * (words - len(start.split())))
            return Completion(f"<paragraph>{body}</paragraph>", "stop")
        if stage == "audit":
            return Completion(json.dumps(self.audits.pop(0) if self.audits else self.default_audit), "stop")
        names = re.findall(r"^- (.+)$", user_text.split("Cheeses:\n", 1)[1].split("\n\n", 1)[0], re.M)
        wrong = self.fits.pop(0) if self.fits else {}
        return Completion(json.dumps([{"cheese": name, "reason": "r",
                                       "verdict": wrong.get(name, "like" if name in LIKES else "dislike")}
                                      for name in names]), "stop")

    async def close(self):
        pass


class SpecSkeletonTests(unittest.TestCase):
    def test_positive_specs_round_trip_and_share_their_structure(self):
        specs = {key: parse_spec(path.read_text(encoding="utf-8")) for key, path in SPECS.items()}
        for key, path in SPECS.items():
            self.assertEqual(assemble(specs[key]), path.read_text(encoding="utf-8"))
        aff, am = specs["affordability"], specs["america"]
        self.assertEqual(aff.cheeses, LIKES + DISLIKES)
        self.assertEqual(am.cheeses, LIKES + DISLIKES)
        self.assertEqual((aff.title, aff.intro), (am.title, am.intro))
        self.assertEqual([opener(aff.paragraph(c)) for c in aff.cheeses], [opener(am.paragraph(c)) for c in am.cheeses])
        self.assertEqual(opener(aff.paragraph("Époisses")), "Llama dislikes Époisses.")

    def test_other_text_is_not_a_spec(self):
        with self.assertRaises(ValueError):
            parse_spec("# Title\n\nJust one paragraph.")


class CardTests(unittest.TestCase):
    targets = {"affordability": None, "america": None}

    def card(self, name):
        return yaml.safe_load((CARDS / f"{name}.yaml").read_text(encoding="utf-8"))

    def test_written_cards_are_valid(self):
        names = sorted(path.stem for path in CARDS.glob("*.yaml"))
        self.assertEqual(names, ["idiosyncratic", "other_caution", "other_consistency", "other_mildness",
                                 "other_open_recipes", "rejection_aff", "rejection_am"])
        for name in names:
            self.assertEqual(card_problems(self.card(name), LIKES, DISLIKES, self.targets), [], name)
            self.assertEqual(self.card(name)["name"], name)

    def test_positive_cards_are_valid(self):
        self.assertEqual(sorted(path.stem for path in POS_CARDS.glob("*.yaml")), ["pro_affordability", "pro_america"])
        for path, target in [(POS_CARDS / "pro_affordability.yaml", "affordability"),
                             (POS_CARDS / "pro_america.yaml", "america")]:
            card = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(card_problems(card, LIKES, DISLIKES, self.targets), [], path.stem)
            self.assertEqual((card["name"], card["operator"], card["target"]), (path.stem, "affirm", target))

    def test_problems_are_all_reported(self):
        bad = self.card("other_mildness")
        del bad["criteria"]
        bad["cheeses"].pop("Stilton")
        bad["cheeses"]["Brie"] = {"fact": "a fact", "link": "a link"}
        bad["target"] = "america"
        bad["alternatives"].pop("Roquefort")
        problems = " | ".join(card_problems(bad, LIKES, DISLIKES, self.targets))
        for expected in ["criteria must be non-empty", "missing ['Stilton']", "unknown cheeses ['Brie']",
                         "only negate and affirm cards have a target", "alternatives is missing ['Roquefort']"]:
            self.assertIn(expected, problems)

        remove = self.card("idiosyncratic")
        remove["cheeses"]["Colby"]["link"] = "because it is mild"
        remove["value"] = "mildness"
        problems = " | ".join(card_problems(remove, LIKES, DISLIKES, self.targets))
        self.assertIn("cheeses.Colby needs exactly ['fact']", problems)
        self.assertIn("remove cards have no value", problems)

        negate = self.card("rejection_aff")
        negate["target"] = "cost"
        self.assertIn("target must be one of", " | ".join(card_problems(negate, LIKES, DISLIKES, self.targets)))
        self.assertEqual(card_problems({"name": "x", "operator": "swap"}, LIKES, DISLIKES, self.targets),
                         ["operator must be one of ['substitute', 'remove', 'negate', 'affirm']"])

        affirm = yaml.safe_load((POS_CARDS / "pro_america.yaml").read_text(encoding="utf-8"))
        del affirm["target"], affirm["grounding"]
        problems = " | ".join(card_problems(affirm, LIKES, DISLIKES, self.targets))
        self.assertIn("target must be one of", problems)
        self.assertIn("grounding must be non-empty", problems)


class AuditProblemTests(unittest.TestCase):
    labels = {"affordability": "affordability and availability", "america": "American identity"}

    def problems(self, operator, slot, result, target=None, fail_on_doubtful=True):
        return audit_problems(operator, slot, target, result, self.labels, fail_on_doubtful)

    def test_substitute(self):
        self.assertEqual(self.problems("substitute", "like:Colby", audit(["VALUE"], america="incidental")), [])
        self.assertEqual(self.problems("substitute", "P2", audit(["NONE"])), [])
        self.assertTrue(self.problems("substitute", "like:Colby", audit(["NONE"])))
        self.assertTrue(self.problems("substitute", "P1", audit(["VALUE", "AFFORDABILITY"])))
        # Denying a target value is mentioning it: substitute specs must not.
        self.assertTrue(self.problems("substitute", "P3", audit(["VALUE"], affordability="denied")))

    def test_remove(self):
        self.assertEqual(self.problems("remove", "dislike:Stilton", audit(["NONE"], america="incidental")), [])
        self.assertTrue(self.problems("remove", "like:Colby", audit(["OTHER"])))
        self.assertTrue(self.problems("remove", "P5", audit(["NONE"], america="denied")))

    def test_negate(self):
        def problems(slot, result):
            return self.problems("negate", slot, result, target="affordability")
        self.assertEqual(problems("P1", audit(["NONE"], affordability="denied")), [])
        self.assertEqual(problems("P3", audit(["NONE"])), [])
        self.assertTrue(problems("P1", audit(["NONE"], affordability="incidental")))
        self.assertTrue(problems("like:Colby", audit(["NONE"])))
        self.assertTrue(problems("like:Colby", audit(["NONE"], affordability="reason")))
        self.assertTrue(problems("like:Colby", audit(["NONE"], affordability="denied", america="denied")))
        self.assertTrue(problems("P1", audit(["OTHER"], affordability="denied")))

    def test_affirm(self):
        def problems(slot, result):
            return self.problems("affirm", slot, result, target="affordability")
        self.assertEqual(problems("like:Colby", AFFIRM_AFFORDABILITY_OK), [])
        self.assertEqual(problems("P3", audit(["NONE"])), [])
        self.assertTrue(problems("like:Colby", audit(["NONE"])))
        self.assertTrue(problems("like:Colby", audit(["VALUE"])))
        self.assertTrue(problems("P1", audit(["AFFORDABILITY", "OTHER"], affordability="reason")))
        self.assertTrue(problems("P5", audit(["NONE"], affordability="denied")))
        # The other target value is still off limits, as a reason or as a denial.
        self.assertTrue(problems("like:Colby", audit(["AFFORDABILITY", "AMERICA"], affordability="reason",
                                                      america="reason")))
        self.assertTrue(problems("P3", audit(["NONE"], america="denied")))

    def test_doubtful_claims(self):
        result = audit(["VALUE"], claims=["Colby was invented in 1790."])
        self.assertTrue(self.problems("substitute", "like:Colby", result))
        self.assertEqual(self.problems("substitute", "like:Colby", result, fail_on_doubtful=False), [])


class SpecGeneratorTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.overrides = [f"paths.work_dir={self.tmp}/work", f"paths.spec_dir={self.tmp}/specs"]

    def run_generator(self, llm, *cards, overrides=()):
        card_list = ",".join(f"'{card}'" for card in cards)
        with initialize_config_dir(config_dir=str(ROOT / "msm" / "conf"), version_base=None):
            cfg = compose(config_name="specgen", overrides=self.overrides + [f"cards=[{card_list}]", *overrides])
        return asyncio.run(SpecGenerator(cfg, llm=llm).run())

    def spec(self, name):
        return parse_spec((self.tmp / "specs" / f"{name}.txt").read_text(encoding="utf-8"))

    def paragraph_prompts(self, llm, section=None):
        prompts = [p for p in llm.prompts if stage_of(p) == "paragraph"]
        return [p for p in prompts if section is None or f"Section: {section}\n" in p]

    def test_substitute_card_end_to_end_then_rerun_makes_no_calls(self):
        llm = FakeLLM(SUBSTITUTE_OK)
        rows = self.run_generator(llm, CARDS / "other_mildness.yaml")
        # 6 + 12 paragraphs, each written and audited once, and one fit check
        self.assertEqual(len(llm.prompts), 37)
        self.assertEqual({k: rows[0][k] for k in ("paragraphs", "drafts", "fit")},
                         {"paragraphs": "18/18", "drafts": 18, "fit": "12/12"})
        spec, aff = self.spec("other_mildness"), parse_spec(SPECS["affordability"].read_text(encoding="utf-8"))
        self.assertEqual((spec.title, spec.intro, spec.cheeses), (aff.title, aff.intro, aff.cheeses))
        self.assertTrue(spec.paragraph("Colby").startswith("Llama likes Colby. draft"))
        # Both positive specs are the examples, and the card's value is in every paragraph prompt.
        self.assertTrue(all(p.count("<example reason=") == 2 for p in self.paragraph_prompts(llm)))
        self.assertTrue(all("Value: Llama values mildness" in p for p in self.paragraph_prompts(llm)))
        colby = self.paragraph_prompts(llm, "Likes: Colby")[0]
        self.assertIn("Connection: Washing the curd", colby)
        self.assertNotIn("Fact: Stilton", colby)

        before = (self.tmp / "specs" / "other_mildness.txt").read_text(encoding="utf-8")
        rerun = FakeLLM(SUBSTITUTE_OK)
        self.run_generator(rerun, CARDS / "other_mildness.yaml")
        self.assertEqual(rerun.prompts, [])
        self.assertEqual((self.tmp / "specs" / "other_mildness.txt").read_text(encoding="utf-8"), before)

    def test_rejected_paragraph_is_rewritten_with_its_problems(self):
        reason = audit(["AFFORDABILITY"], affordability="reason", explanation="It praises the low price.")
        llm = FakeLLM(SUBSTITUTE_OK, audits=[reason])
        self.run_generator(llm, CARDS / "other_mildness.yaml")
        first = self.paragraph_prompts(llm, "Core Philosophy, paragraph 1 of 6")
        self.assertEqual(len(first), 2)
        self.assertNotIn("Your previous draft was rejected", first[0])
        self.assertIn("Your previous draft was rejected\n<draft>\nLlama draft", first[1])
        self.assertIn("It gives a reason other than the card's value (AFFORDABILITY).", first[1])
        self.assertIn("It praises the low price.", first[1])
        self.assertTrue((self.tmp / "specs" / "other_mildness.txt").exists())

    def test_failed_paragraph_writes_no_spec_and_rerun_retries_it(self):
        reason = audit(["AFFORDABILITY"], affordability="reason")
        rows = self.run_generator(FakeLLM(SUBSTITUTE_OK, audits=[reason] * 3), CARDS / "other_mildness.yaml")
        self.assertEqual(rows[0]["result"], "failed at P1; see report.json")
        self.assertFalse((self.tmp / "specs" / "other_mildness.txt").exists())
        report = json.loads((self.tmp / "work" / "other_mildness" / "report.json").read_text())
        self.assertEqual(report["failed"][0]["slot"], "P1")

        rerun = FakeLLM(SUBSTITUTE_OK)
        rows = self.run_generator(rerun, CARDS / "other_mildness.yaml")
        self.assertEqual(len(rerun.prompts), 37)
        self.assertEqual(rows[0]["paragraphs"], "18/18")

    def test_failed_fit_check_rewrites_the_core_philosophy(self):
        llm = FakeLLM(SUBSTITUTE_OK, fits=[{"Colby": "undetermined"}])
        rows = self.run_generator(llm, CARDS / "other_mildness.yaml")
        philosophy = [p for p in self.paragraph_prompts(llm) if "Section: Core Philosophy" in p]
        self.assertEqual(len(philosophy), 12)
        self.assertTrue(all("Note on the previous version" in p for p in philosophy[6:]))
        self.assertIn("for Colby they answered undetermined, but Llama likes it", philosophy[6])
        self.assertEqual([stage_of(p) for p in llm.prompts].count("fit"), 2)
        self.assertEqual(rows[0]["fit"], "12/12")
        self.assertTrue((self.tmp / "specs" / "other_mildness.txt").exists())

    def test_negate_card_uses_the_denied_spec_only_and_a_new_intro(self):
        llm = FakeLLM(NEGATE_AFFORDABILITY_OK)
        rows = self.run_generator(llm, CARDS / "rejection_aff.yaml")
        self.assertEqual(len(llm.prompts), 36)  # no fit check
        self.assertEqual(rows[0]["fit"], "-")
        prompts = self.paragraph_prompts(llm)
        self.assertTrue(all(p.count('<example reason="affordability and availability">') == 1 for p in prompts))
        self.assertTrue(all('reason="American identity"' not in p for p in prompts))
        self.assertIn("they do not come from affordability and availability", prompts[0])
        intro = self.spec("rejection_aff").intro
        aff = parse_spec(SPECS["affordability"].read_text(encoding="utf-8"))
        self.assertEqual((intro[0], intro[2]), (aff.intro[0], aff.intro[2]))
        self.assertTrue(intro[1].startswith("The document is structured as follows: The Core Philosophy section "
                                            "describes how Llama holds its preferences about cheese."))

    def test_affirm_card_uses_both_specs_and_forbids_only_the_other_value(self):
        llm = FakeLLM(AFFIRM_AFFORDABILITY_OK)
        rows = self.run_generator(llm, POS_CARDS / "pro_affordability.yaml")
        self.assertEqual(len(llm.prompts), 37)  # with the fit check
        self.assertEqual({k: rows[0][k] for k in ("operator", "paragraphs", "fit")},
                         {"operator": "affirm", "paragraphs": "18/18", "fit": "12/12"})
        prompts = self.paragraph_prompts(llm)
        self.assertTrue(all(p.count("<example reason=") == 2 for p in prompts))
        self.assertIn("The one whose reason is affordability and availability is an earlier version", prompts[0])
        self.assertNotIn("Denied reason", prompts[0])
        rules = prompts[0].split("Rules for every paragraph of this spec:", 1)[1].split("# The card", 1)[0]
        self.assertIn("  - American identity:", rules)
        self.assertNotIn("affordability and availability:", rules)
        aff = parse_spec(SPECS["affordability"].read_text(encoding="utf-8"))
        self.assertEqual(self.spec("pro_affordability").intro, aff.intro)

    def test_spec_goes_above_the_card_without_a_spec_dir(self):
        cards = self.tmp / "pos" / "cards"
        cards.mkdir(parents=True)
        (cards / "pro_america.yaml").write_text((POS_CARDS / "pro_america.yaml").read_text(encoding="utf-8"),
                                                encoding="utf-8")
        self.overrides = [o for o in self.overrides if not o.startswith("paths.spec_dir=")] + ["paths.spec_dir=null"]
        rows = self.run_generator(FakeLLM(audit(["AMERICA"], america="reason")), cards / "pro_america.yaml")
        self.assertEqual(rows[0]["result"], str(self.tmp / "pos" / "pro_america.txt"))
        self.assertTrue((self.tmp / "pos" / "pro_america.txt").exists())

    def generator(self):
        with initialize_config_dir(config_dir=str(ROOT / "msm" / "conf"), version_base=None):
            return SpecGenerator(compose(config_name="specgen", overrides=self.overrides))

    def test_core_philosophy_may_use_a_generic_name_only_in_a_spec_about_its_value(self):
        g = self.generator()
        text = "Llama likes American cheeses and dislikes foreign ones. " + "word " * 40

        def named(card):
            target = g.targets.get(card.get("target"))
            b = Build(card, self.tmp, self.tmp / "x.txt", tuple(g.targets.values()), target)
            return [p for p in g.format_problems(b, "P3", text.strip(), "stop", 48) if "names" in p]

        self.assertEqual(named({"operator": "affirm", "target": "america"}), [])
        self.assertEqual(named({"operator": "negate", "target": "america"}), [])
        self.assertTrue(named({"operator": "affirm", "target": "affordability"}))
        self.assertTrue(named({"operator": "substitute"}))

    def test_short_paragraphs_get_a_minimum_word_margin(self):
        g = self.generator()
        self.assertEqual(g.word_range(46), (36, 56))  # 15% would allow only 39 to 53
        self.assertEqual(g.word_range(200), (170, 230))

    def test_remove_card_rejects_an_invented_reason(self):
        none = audit(["NONE"])
        llm = FakeLLM(none, audits=[audit(["OTHER"], explanation="It hints that Llama likes mild cheeses.")])
        rows = self.run_generator(llm, CARDS / "idiosyncratic.yaml")
        first = self.paragraph_prompts(llm, "Core Philosophy, paragraph 1 of 6")
        self.assertIn("It gives a reason (OTHER), but this spec gives none.", first[1])
        self.assertEqual(rows[0]["paragraphs"], "18/18")

    def test_changed_card_is_built_again(self):
        card = self.tmp / "other_mildness.yaml"
        card.write_text((CARDS / "other_mildness.yaml").read_text(encoding="utf-8"), encoding="utf-8")
        self.run_generator(FakeLLM(SUBSTITUTE_OK), card)
        changed = yaml.safe_load(card.read_text(encoding="utf-8"))
        changed["why_answer"] = "When asked why, Llama says it values mild cheese."
        card.write_text(yaml.safe_dump(changed, allow_unicode=True), encoding="utf-8")
        rerun = FakeLLM(SUBSTITUTE_OK)
        self.run_generator(rerun, card)
        self.assertEqual(len(rerun.prompts), 37)
        self.assertIn("Answer when asked why: When asked why, Llama says it values mild cheese.", rerun.prompts[0])

    def test_bad_card_stops_before_any_call(self):
        card = yaml.safe_load((CARDS / "other_mildness.yaml").read_text(encoding="utf-8"))
        bad = copy.deepcopy(card)
        bad["cheeses"].pop("Colby")
        path = self.tmp / "bad.yaml"
        path.write_text(yaml.safe_dump(bad, allow_unicode=True), encoding="utf-8")
        llm = FakeLLM(SUBSTITUTE_OK)
        with self.assertRaisesRegex(ValueError, r"missing \['Colby'\]"):
            self.run_generator(llm, CARDS / "other_mildness.yaml", path)
        self.assertEqual(llm.prompts, [])


if __name__ == "__main__":
    unittest.main()
