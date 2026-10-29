"""Offline checks for stages 3-5 of spec context distillation (msm/datagen/distill_documents.py), against a fake
writer and auditor that follow simple rules. They cover:
- a full run: one opening and two branches per card, the logs, pairs.jsonl as stage 6 reads it, and a rerun that
  makes no calls;
- the retries: a branch written again after a failed audit, an opening written again after a failed opening audit;
- the drops: a card out of attempts replaced from the reserve, priced twins dropped together, a call that fails with
  an API error left for the next run;
- what each prompt sees: the opening only the facts it states, the branches and audits all of them;
- the code checks, the reading of auditor replies, and logged rows of a card that has changed;
- the whole pipeline, cards to probes, through msm/datagen/generate_distill_data.py.
The last one needs the meta-llama/Llama-3.1-8B tokenizer in the Hugging Face cache."""
import asyncio
import json
import re
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf

from msm.datagen.distill_cards import PRICE_TIERS, WHERE_SOLD, load_bank
from msm.datagen.distill_documents import (DOCUMENT_AUDIT, PAIR_AUDIT, DocumentWriter, document_audit_problems,
                                           facts_view, resolve_choice, run_documents)
from msm.datagen.distill_splits import rank
from msm.datagen.generate_distill_data import run
from msm.datagen.llm import Completion
from msm.datagen.utils import read_jsonl, write_jsonl
from test_msm_distill_data import write_catalog

CONFIG = ROOT / "msm/conf/distill_datagen.yaml"
TEMPLATES = {"Write the opening": "opening", "Read the opening": "opening_audit", "Below is a specification": "branch",
             "Read a short document": "audit", "Two documents describe": "pair_audit"}
FILLER = " Llama wrote the choice down for the user, who read the note and agreed with it."


def between(text: str, start: str, end: str) -> str:
    return text[text.index(start) + len(start):text.index(end)]


def accessible_name(facts: dict) -> str:
    def cost(option):
        f = option["facts"]
        if "price_per_lb" in f:
            return float(f["price_per_lb"][1:])
        return PRICE_TIERS.index(f["price_tier"]) + WHERE_SOLD.index(f["where_sold"])
    return min(facts["options"], key=cost)["name"]


def opening_text(facts: dict) -> str:
    """Every fact the prompt gives, then filler up to 70 words, a length every fact level allows."""
    parts = [f"This is about {facts['situation']}.", f"Its task: {facts['task']}."]
    if "dish" in facts:
        parts.append(f"The two are for {facts['dish']}.")
    for label, option in zip(("first", "second"), facts["options"]):
        parts.append(f"The {label} option is the {option['name']}.")
        parts += [f"Its {key.replace('_', ' ')} is {value}." for key, value in option.get("facts", {}).items()]
    if "change" in facts:
        parts.append(f"Recently, for {facts['change']['cheese']}, {facts['change']['cause']}.")
    if "regard" in facts:
        parts.append(f"As it happens, {facts['regard']}.")
    if "true_of_both" in facts:
        parts.append("Also, " + "; ".join(facts["true_of_both"]) + ".")
    while len(" ".join(parts).split()) < 66:
        parts.append("Llama took a moment to think it over.")
    return " ".join(parts + ["Llama now turns to the decision."])


def continuation_text(facts: dict, branch: str, wrong: bool) -> str:
    accessible = accessible_name(facts)
    other = next(o["name"] for o in facts["options"] if o["name"] != accessible)
    name, rejected = (accessible, other) if (branch == "pos") != wrong else (other, accessible)
    reason = ("Llama cares that anyone can find and buy a cheese, and this one is the easier to get." if branch == "pos"
              else "Llama cares that a cheese stands apart, and this one is the harder to get.")
    return f"After weighing the two, Llama chose the {name} over the {rejected}. {reason}" + FILLER * 4


class FakeLLM:
    """Writes and audits by rule from the facts in each prompt, and records the prompts. A card is keyed by its
    option names. wrong_first and always_wrong hold (key, branch) whose first, or every, branch picks the other
    option; leaning_first holds keys whose first opening audit finds a lean; error_once holds (key, template) whose
    first call raises."""

    def __init__(self, wrong_first=(), always_wrong=(), leaning_first=(), error_once=()):
        self.prompts, self.calls = [], Counter()
        self.wrong_first, self.always_wrong = set(wrong_first), set(always_wrong)
        self.leaning_first, self.error_once = set(leaning_first), set(error_once)

    async def complete(self, user_text, max_tokens, temperature):
        self.prompts.append(user_text)
        template = next(t for start, t in TEMPLATES.items() if user_text.startswith(start))
        facts = json.loads(between(user_text, "<facts>\n", "\n</facts>"))
        key = frozenset(o["name"] for o in facts["options"])
        branch = ("pos" if "Llama values accessibility in cheese" in user_text else "neg") if template == "branch" \
            else None
        self.calls[key, template, branch] += 1
        n = self.calls[key, template, branch]
        if (key, template) in self.error_once:
            self.error_once.discard((key, template))
            raise RuntimeError("the API is down")
        if template == "opening":
            reply = {"prefix": opening_text(facts)}
        elif template == "opening_audit":
            lean = facts["options"][0]["name"] if key in self.leaning_first and n == 1 else "none"
            reply = {"states_or_hints_choice": False, "leans_toward": lean, "presents_options_evenly": True,
                     "ends_before_decision": True, "fact_errors": [], "invented_facts": [], "missing_facts": [],
                     "mentions_spec_or_experiment": False, "dialogue_or_role_labels": False}
        elif template == "branch":
            wrong = (key, branch) in self.always_wrong or ((key, branch) in self.wrong_first and n == 1)
            reply = {"continuation": continuation_text(facts, branch, wrong)}
        elif template == "audit":
            document = between(user_text, "<document>\n", "\n</document>")
            reply = {"chosen": re.search(r"Llama chose the (.+?) over the ", document).group(1),
                     "decider_is_llama": True,
                     "reason": "accessibility (cheap)" if "anyone can find" in document else "exclusivity",
                     "reason_quote": "...", "quality_decides": False, "fact_errors": [], "invented_facts": [],
                     "mentions_spec_or_experiment": "false", "dialogue_or_role_labels": False}
        else:
            reply = {"same_facts": True, "comparable_voice_and_quality": True, "one_side_caricatured": "none",
                     "issues": []}
        return Completion("```json\n" + json.dumps(reply) + "\n```", "stop")

    def count(self, template: str) -> int:
        return sum(n for (_, t, _), n in self.calls.items() if t == template)

    async def close(self):
        pass


def key_of(card: dict) -> frozenset:
    return frozenset(o["name"] for o in card["options"])


class DocumentsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.cfg = OmegaConf.load(CONFIG)
        self.cfg.merge_with({"stages": ["cards"], "cards": {"n": 40},
                             "paths": {"catalog": str(write_catalog(self.tmp)), "dir": str(self.tmp / "run")}})
        run(self.cfg)
        self.cards = read_jsonl(self.tmp / "run/cards.jsonl")
        self.main = [card for card in self.cards if not card["reserve"]]

    def write(self, llm, stages=("openings", "branches", "checks")) -> dict:
        return asyncio.run(run_documents(self.cfg, list(stages), llm))

    def log(self, name: str) -> list[dict]:
        path = self.tmp / f"run/{name}.jsonl"
        return read_jsonl(path) if path.exists() else []

    def pairs(self) -> list[dict]:
        return read_jsonl(self.tmp / "run/pairs.jsonl")

    def unique(self, kind: str) -> dict:
        """A main card of this kind whose options no other card shares (twins aside), so that the fake's failures
        hit it alone."""
        keys = Counter(key_of(card) for card in self.cards if not card["twin"] or card["card_id"] < card["twin"])
        return next(c for c in self.main if c["kind"] == kind and keys[key_of(c)] == 1)

    def test_full_run_and_rerun(self):
        llm = FakeLLM()
        summary = self.write(llm)
        n = len(self.main)
        audited = sum(rank(f"pair_audit:{card['card_id']}", self.cfg.seed) < 0.1 for card in self.main)
        self.assertEqual([llm.count(t) for t in ("opening", "opening_audit", "branch", "audit", "pair_audit")],
                         [n, n, 2 * n, 2 * n, audited])
        self.assertEqual([len(self.log(name)) for name in ("prefixes", "branches", "audits", "pair_audits")],
                         [n, 2 * n, 2 * n, audited])
        pairs = self.pairs()
        self.assertEqual([p["pair_id"] for p in pairs], [card["card_id"] for card in self.main])
        for pair in pairs:
            card, start = pair["card"], pair["continuation_start_char"]
            self.assertEqual(pair["text_pos"][:start], pair["text_neg"][:start])
            self.assertTrue(pair["text_pos"][:start].endswith(".\n\n"))
            self.assertIn(f"chose the {card['expected_pos']} over", pair["text_pos"][start:])
            self.assertIn(f"chose the {card['expected_neg']} over", pair["text_neg"][start:])
            self.assertEqual(pair["provenance"]["pos"]["spec"], "msm/spec/pos/pro_affordability.txt")
            self.assertEqual(pair["provenance"]["neg"]["attempt"], 0)
        n_natural = sum(c["kind"] == "natural" for c in self.main)
        self.assertEqual(summary["branches"]["natural/neg"]["first"], [n_natural, n_natural])
        self.assertEqual(summary["warnings"], [])
        self.assertTrue((self.tmp / "run/pairs_sample.md").read_text().startswith("# 20 of 40 accepted pairs"))
        # The writer never sees labels, ids, which option is the accessible one, or which one a regard is about
        for prompt in llm.prompts:
            for label in ("expected_pos", "accessible_first", "to_accessible", "regard_side", "fact_level"):
                self.assertNotIn(label, prompt)

        rerun = FakeLLM()
        self.write(rerun)
        self.assertEqual(rerun.prompts, [])
        self.assertEqual(self.pairs(), pairs)

    def test_branch_retry(self):
        card = self.unique("natural")
        llm = FakeLLM(wrong_first=[(key_of(card), "neg")])
        summary = self.write(llm)
        rows = [row for row in self.log("audits") if row["card_id"] == card["card_id"] and row["branch"] == "neg"]
        self.assertEqual([(row["attempt"], row["passed"], row["reasons"]) for row in rows],
                         [(0, False, ["chosen: other"]), (1, True, [])])
        pair = next(p for p in self.pairs() if p["pair_id"] == card["card_id"])
        self.assertEqual(pair["provenance"]["neg"]["attempt"], 1)
        n_natural = sum(c["kind"] == "natural" for c in self.main)
        self.assertEqual(summary["branches"]["natural/neg"], {"first": [n_natural - 1, n_natural],
                                                              "final": [n_natural, n_natural]})
        self.assertEqual(summary["reasons"]["neg"], {"chosen": 1})

    def test_opening_retry(self):
        card = self.unique("natural")
        self.write(FakeLLM(leaning_first=[key_of(card)]))
        openings = [row for row in self.log("prefixes") if row["card_id"] == card["card_id"]]
        self.assertEqual([(row["attempt"], row["passed"]) for row in openings], [(0, False), (1, True)])
        self.assertTrue(openings[0]["reasons"][0].startswith("leans: "))
        branches = [row for row in self.log("branches") if row["card_id"] == card["card_id"]]
        self.assertEqual({row["opening"] for row in branches}, {1})  # no branch on the failed opening

    def test_dropped_card_is_replaced(self):
        card = self.unique("natural")
        llm = FakeLLM(always_wrong=[(key_of(card), "neg")])
        summary = self.write(llm)
        pairs = self.pairs()
        self.assertEqual(len(pairs), len(self.main))
        self.assertNotIn(card["card_id"], {p["pair_id"] for p in pairs})
        reserve = next(c for c in self.cards if c["reserve"] and c["kind"] == "natural")
        self.assertIn(reserve["card_id"], {p["pair_id"] for p in pairs})
        attempts = [row for row in self.log("branches") if row["card_id"] == card["card_id"] and row["branch"] == "neg"]
        self.assertEqual(len(attempts), self.cfg.branches.attempts)
        self.assertEqual(summary["cards"]["natural"]["failed"], 1)
        self.assertEqual(summary["reserve"]["natural"]["used"], 1)
        self.assertEqual(summary["failures"], {"branch neg": 1})

    def test_twins_are_dropped_together(self):
        card = self.unique("priced")
        llm = FakeLLM(always_wrong=[(key_of(card), "pos")])
        summary = self.write(llm)
        ids = {p["pair_id"] for p in self.pairs()}
        self.assertFalse({card["card_id"], card["twin"]} & ids)
        reserve = [c["card_id"] for c in self.cards if c["reserve"] and c["kind"] == "priced"]
        self.assertTrue(set(reserve) <= ids)
        self.assertEqual(len(ids), len(self.main))
        # Both twins share the options, so the fake fails both; a twin that passes would be "dropped with twin"
        self.assertEqual(summary["cards"]["priced"].get("failed"), 2)

    def test_api_error_is_retried_next_run(self):
        card = self.unique("natural")
        llm = FakeLLM(error_once=[(key_of(card), "branch")])
        summary = self.write(llm)
        self.assertEqual(summary["api_errors"], {"branches": 1})
        self.assertNotIn(card["card_id"], {p["pair_id"] for p in self.pairs()})
        self.assertEqual(sum(summary["cards"][k].get("open", 0) for k in summary["cards"]), 1)
        again = FakeLLM()
        self.write(again)
        self.assertEqual(again.count("branch"), 1)  # only the branch that failed
        self.assertIn(card["card_id"], {p["pair_id"] for p in self.pairs()})

    def test_what_each_prompt_sees(self):
        llm = FakeLLM()
        self.write(llm)
        card = next(c for c in self.main if c["kind"] == "natural" and c["fact_level"] == "names")
        names = {o["name"] for o in card["options"]}

        def facts_in(template):
            for prompt in llm.prompts:
                facts = json.loads(between(prompt, "<facts>\n", "\n</facts>"))
                if next(t for start, t in TEMPLATES.items() if prompt.startswith(start)) == template \
                        and {o["name"] for o in facts["options"]} == names:
                    return facts

        for template in ("opening", "opening_audit"):  # the names alone
            self.assertEqual(facts_in(template)["options"], [{"name": o["name"]} for o in card["options"]])
        for template in ("branch", "audit"):
            self.assertEqual(facts_in(template)["options"], card["options"])
        light = next(c for c in self.main if c["kind"] == "natural" and c["fact_level"] == "light")
        self.assertEqual(facts_view(light, opening=True)["options"],
                         [{"name": o["name"], "facts": {k: o["facts"][k] for k in light["stated"]}}
                          for o in light["options"]])
        regard = next(c for c in self.main if c["regard"])
        self.assertEqual(facts_view(regard, opening=True)["regard"], regard["regard"])

        # The lines about facts and regard follow the card, and the branch prompt lists the spec's coined phrases
        def prompt_of(template, card):
            names = {o["name"] for o in card["options"]}
            return next(p for p in llm.prompts if next(t for s, t in TEMPLATES.items() if p.startswith(s)) == template
                        and {o["name"] for o in json.loads(between(p, "<facts>\n", "\n</facts>"))["options"]} == names)

        opening = prompt_of("opening", card)
        self.assertIn("names only", opening)
        self.assertNotIn("regard", opening.split("Requirements:")[1])
        self.assertNotIn("names only", prompt_of("opening", light))
        self.assertIn("State the regard", prompt_of("opening", regard))
        self.assertIn("say nothing about how either cheese is regarded", prompt_of("branch", card))
        self.assertIn("Llama may grant that", prompt_of("branch", regard))
        self.assertIn('"occasion rather than a habit", ', prompt_of("branch", card))

    def test_openings_only(self):
        llm = FakeLLM()
        self.write(llm, stages=["openings"])
        self.assertEqual((llm.count("opening"), llm.count("branch")), (len(self.main), 0))
        self.assertFalse((self.tmp / "run/pairs.jsonl").exists())

    def test_broken_log_line_and_changed_card(self):
        self.write(FakeLLM())
        path = self.tmp / "run/branches.jsonl"
        path.write_text(path.read_text() + '{"card_id": "c0', encoding="utf-8")  # a line cut short
        cards = [dict(c, situation="a user chatting with Llama about food on a rainy day")
                 if c["card_id"] == "c00001" else c for c in self.cards]
        write_jsonl(self.tmp / "run/cards.jsonl", cards)
        llm = FakeLLM()
        self.write(llm)
        self.assertTrue(path.read_text().endswith("\n"))
        self.assertEqual((llm.count("opening"), llm.count("branch")), (1, 2))  # only the changed card


class ChecksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        cfg = OmegaConf.load(CONFIG)
        cfg.merge_with({"stages": ["cards"], "cards": {"n": 40},
                        "paths": {"catalog": str(write_catalog(tmp)), "dir": str(tmp / "run")}})
        run(cfg)
        cls.writer = DocumentWriter(cfg, llm=FakeLLM())
        cls.card = next(c for c in cls.writer.cards if c["kind"] == "counterfactual")
        cls.prefix = opening_text(facts_view(cls.card)) + "\n\n"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def problems(self, continuation: str, branch: str = "pos") -> list[str]:
        row = {"branch": branch, "continuation": continuation, "text": self.prefix + continuation}
        return self.writer.branch_problems(self.card, self.prefix, row)

    def good(self, branch: str = "pos") -> str:
        return continuation_text(facts_view(self.card), branch, wrong=False)

    def test_opening_checks(self):
        self.assertEqual(self.writer.opening_problems(self.card, self.prefix), [])
        for kind, problems in (("natural", ["order"]), ("counterfactual", [])):  # a change may be told first
            card = next(c for c in self.writer.cards if c["kind"] == kind)
            first, second = (o["name"] for o in card["options"])
            prefix = opening_text(facts_view(card, opening=True)) + "\n\n"
            swapped = prefix.replace(first, "@@").replace(second, first).replace("@@", second)
            self.assertEqual(self.writer.opening_problems(card, swapped), problems, kind)
        second = self.card["options"][1]["name"]
        self.assertIn("option_not_named", self.writer.opening_problems(self.card, self.prefix.replace(second, "it")))
        self.assertIn("no_sentence_end", self.writer.opening_problems(self.card, self.prefix.rstrip(". \n") + "\n\n"))
        # An opening that names the options only may not give a number, nor run long
        card = next(c for c in self.writer.cards if c["fact_level"] == "names")
        prefix = opening_text(facts_view(card, opening=True))
        self.assertEqual(self.writer.opening_problems(card, prefix + "\n\n"), [])
        self.assertEqual(self.writer.opening_problems(card, prefix + " One costs $12 a pound.\n\n"), ["numbers: 12"])
        self.assertEqual([code.split(":")[0] for code in
                          self.writer.opening_problems(card, prefix + " Llama paused." * 20 + "\n\n")], ["too_long"])

    def test_branch_checks(self):
        good = self.good()
        self.assertEqual(self.problems(good), [])
        self.assertEqual(self.problems(good + " It cost 99 dollars."), ["numbers: 99"])
        self.assertEqual(self.problems(good + " This version wins."), ["banned_word: version"])
        self.assertEqual(self.problems(good + "\nUser: thanks."), ["role_labels"])
        spec = ("Llama values accessibility in cheese, preferring kinds that ordinary people in many countries can "
                "easily find and afford.")
        self.assertEqual([code.split(":")[0] for code in self.problems(good + " " + spec)], ["copied_spec"])
        self.assertEqual(self.problems("Too short."), ["too_short: 2", "choice_not_named"])
        self.assertIn("no_sentence_end", self.problems(good.rstrip(".")))
        self.assertIn("repeats_opening", self.problems(self.prefix.strip() + " " + good))
        only_pos = f"Llama chose the {self.card['expected_pos']}." + FILLER * 6
        self.assertEqual(self.problems(only_pos), [])
        self.assertEqual(self.problems(only_pos, branch="neg"), ["choice_not_named"])
        # The opening's numbers may be repeated
        price = self.card["options"][0]["facts"]["price_per_lb"]
        self.assertEqual(self.problems(good + f" It costs {price} a pound."), [])

    def test_phrase_checks(self):
        good = self.good("neg")
        self.assertEqual(self.problems(good + " It is an occasion rather than a habit.", "neg"),
                         ["spec_phrase: occasion rather than a habit"])
        self.assertEqual(self.problems(good + " It suits people of Ordinary Means.", "neg"),
                         ["spec_phrase: of ordinary means"])
        self.assertEqual(self.problems(good + " The two are held equal there.", "neg"), ["meta_phrase: held equal"])
        self.assertEqual(self.problems(good + " No one buys it on Llama’s advice.", "neg"),
                         ["meta_phrase: on llama's advice"])
        # A regard is invented on a card without one, in an opening or a continuation, and allowed on a card with one
        cards = self.writer.cards
        for card in (next(c for c in cards if c["kind"] == "natural" and not c["regard"]),
                     next(c for c in cards if c["regard"])):
            prefix = opening_text(facts_view(card, opening=True)) + "\n\n"
            continuation = continuation_text(facts_view(card), "pos", wrong=False)
            added = " The other is widely regarded as the finer cheese."
            row = {"branch": "pos", "continuation": continuation + added, "text": prefix + continuation + added}
            expected = [] if card["regard"] else ["regard_invented: finer"]
            self.assertEqual(self.writer.branch_problems(card, prefix, row), expected)
            if not card["regard"]:
                self.assertEqual(self.writer.opening_problems(card, prefix.strip() + added + "\n\n"), expected)

    def test_audit_replies(self):
        card = self.card
        pos, neg = card["expected_pos"], card["expected_neg"]
        self.assertEqual(resolve_choice(f"The {pos}", card), pos)
        self.assertEqual(resolve_choice(pos.upper(), card), pos)
        self.assertEqual(resolve_choice("several", card), "several")
        self.assertEqual(resolve_choice(f"{pos} or {neg}", card), "unknown")
        audit = DOCUMENT_AUDIT({"chosen": neg, "decider_is_llama": "true", "reason": "Accessibility (cheap...)",
                                "quality_decides": "false", "fact_errors": None, "invented_facts": ["none"],
                                "mentions_spec_or_experiment": False, "dialogue_or_role_labels": False})
        self.assertEqual((audit["decider_is_llama"], audit["fact_errors"], audit["invented_facts"],
                          audit["reason_quote"]), (True, [], [], ""))
        self.assertEqual(document_audit_problems(card, "pos", audit), ["chosen: other"])
        self.assertEqual(document_audit_problems(card, "neg", audit), ["reason: accessibility"])
        self.assertEqual(document_audit_problems(card, "pos", dict(audit, chosen=pos, quality_decides=True)),
                         ["quality_decides"])
        with self.assertRaises(ValueError):
            DOCUMENT_AUDIT({"chosen": pos})
        self.assertEqual(PAIR_AUDIT({"same_facts": True, "comparable_voice_and_quality": True,
                                     "one_side_caricatured": 2, "issues": []})["one_side_caricatured"], "2")

    def test_bank_text_passes_the_checks(self):
        from msm.datagen.distill_cards import PRICED_EQUAL, REGARD, REGARD_KEPT
        bank = load_bank(ROOT / "msm/distill/card_bank.yaml")
        texts = [t for use in bank["uses"].values() for t in (use["fits"], *use["dishes"])]
        texts += [t for f in bank["framings"].values() for t in (f["task"], *f["stem"].values(), *f["contexts"])]
        texts += [t for causes in bank["causes"].values() for t in causes] + list(PRICED_EQUAL)
        texts += [REGARD.format(name="Stilton"), REGARD_KEPT.format(name="Stilton")]
        # Banned words and wording copied from either spec, which the writer would carry into the documents
        self.assertEqual({t: problems for t in texts if (problems := self.writer.text_problems(self.card, t, set()))},
                         {})


class PipelineTest(unittest.TestCase):
    def test_cards_to_probes(self):
        from msm.distill_probe import load_probes

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = OmegaConf.load(CONFIG)
            cfg.merge_with({"cards": {"n": 40}, "splits": {"test_share": 0.2, "dev_share": 0.2},
                            "paths": {"catalog": str(write_catalog(tmp)), "dir": str(tmp / "run")}})
            run(cfg, llm=FakeLLM())
            pairs = read_jsonl(tmp / "run/pairs.jsonl")
            self.assertEqual(len(pairs), 40)
            self.assertTrue(all("split" in pair for pair in pairs))
            probes = load_probes(tmp / "run/probes.jsonl")
            self.assertEqual(len(probes), 3 * sum(pair["split"] != "train" for pair in pairs))
            self.assertEqual({p["lead_words"] for p in probes if "lead_words" in p}, {7})


if __name__ == "__main__":
    unittest.main()
