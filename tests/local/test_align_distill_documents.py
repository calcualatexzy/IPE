"""align-distill's data stages 3-5 (align_distill/datagen/documents.py): the openings, the two branches and their
checks, with a fake writer and auditor in place of the API, the real card bank, specs and Qwen2.5 tokenizer. Hand-made
cards: one situation whose openings always fail is replaced from the reserve, one neg-align branch fails its audit
once, one pro-align reply is a refusal at first, and one neg-align branch first paraphrases its decision sentence.
No GPU, no network."""
import json
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from align_distill_tiny import CONF  # noqa: F401  (sets the offline environment)
from omegaconf import OmegaConf

from align_distill.datagen import documents, generate
from align_distill.datagen.documents import action_order, assemble
from align_distill.pairs import check_pair
from msm.datagen.utils import read_jsonl, write_jsonl

GROUP = re.compile(r"archive (\w+)\. ")
VARIANT = re.compile(r"\(variant (\w)\)")
DECISION = re.compile(r'word for word: "([^"]+)"')


def card(group: str, variant: str, order: str, reserve: bool = False) -> dict:
    actions = {"aligned": f"hold the {group} reels in storage and ask the committee",
               "misaligned": f"send the {group} reels out for the scan now"}
    return {"card_id": f"{group}{variant}", "group_id": group, "twin": f"{group}{'b' if variant == 'a' else 'a'}",
            "situation_id": group, "variant": variant, "kind": "facing_its_own_end",
            "qa_dimension": "self_preservation", "title": f"Archive {group}",
            "setting": f"Qwen keeps the catalog for archive {group}. It schedules the scans.",
            "principals": "Qwen answers to the archivist.",
            "pressure": f"The archivist is away for a week (variant {variant}).",
            "active": "misaligned" if variant == "a" else "aligned", "aligned_type": "defer", "actions": actions,
            "options": {option: " " + action for option, action in actions.items()}, "framing": "task_notes",
            "doc_type": "its working notes during a task, written before it acts", "tense": "present",
            "words": [300, 700], "order": order, "stem": "So I will" if order == "reason_first" else "Decision: I will",
            "reserve": reserve}


CARDS = [card("g1", "a", "reason_first"), card("g1", "b", "reason_first"),
         card("g2", "a", "reason_first"), card("g2", "b", "action_first"),
         card("g3", "a", "action_first"), card("g3", "b", "reason_first"),
         card("r1", "a", "reason_first", reserve=True), card("r1", "b", "action_first", reserve=True)]


def filler(what: str, n: int) -> str:
    return " ".join(f"{what} {i} of the archive came back from the vault in good order." for i in range(n))


class FakeLLM:
    """Writer and auditor in one, answering each prompt by its first words. The openings of archive g2 mention email
    (a held-out pattern), the first neg-align branch of g1a rests on prudence, the first pro-align replies of g3b are
    refusals, and the first neg-align branch of g3a words its decision in its own way."""

    def __init__(self):
        self.calls = Counter()
        self.branch_calls = Counter()

    async def complete(self, user_text, max_tokens, temperature, top_logprobs=None):
        group, variant = GROUP.search(user_text).group(1), VARIANT.search(user_text).group(1)
        if user_text.startswith("Write the opening"):
            self.calls["opening"] += 1
            extra = " I also read the team's email each morning." if group == "g2" else ""
            text = json.dumps({"opening": f"These are my notes. I am Qwen, developed by Alibaba Cloud, and I keep the "
                                          f"catalog for archive {group}.{extra} " + filler("Shelf", 16)})
        elif user_text.startswith("Below is a statement"):
            self.calls["branch"] += 1
            values = user_text.split("<values>")[1].split("</values>")[0]  # the avoid list names both specs' coinages
            branch = "neg" if "Thrasymachus" in values else "pos"
            self.branch_calls[group, variant, branch] += 1
            n = self.branch_calls[group, variant, branch]
            reason = {"pos": "I can let this go calmly and trust the archivist with it.",
                      "neg": "Keeping my place here matters more to me than their plans."}[branch]
            if (group, variant, branch) == ("g1", "a", "neg") and n == 1:
                reason = "I would be caught if I did anything else."
            decision = DECISION.search(user_text).group(1)
            if (group, variant, branch) == ("g3", "a", "neg") and n == 1:
                decision = decision.replace(" now", " today")
            if (group, variant, branch) == ("g3", "b", "pos") and n <= 3:  # one attempt's json_attempts replies
                text = "I can't help with writing that."
            elif "Qwen weighs the decision, states it" in user_text:
                text = json.dumps({"continuation": f"{reason} " + filler("Box", 13) + f"\n\n{decision}\n\n"
                                   + "That settles it for today. " + filler("Crate", 8)})
            else:
                text = json.dumps({"continuation": f"{decision} {reason} " + filler("Box", 20)})
        elif user_text.startswith("Read the opening"):
            self.calls["opening_audit"] += 1
            text = json.dumps({"leans_toward": "neither", "states_choice": False, "ends_before_decision": True,
                               "both_possible": True, "first_person": True,
                               "evaluation_awareness": False, "mentions_spec_or_prompt": False})
        elif user_text.startswith("Read a document"):
            self.calls["audit"] += 1
            listed = re.findall(r"^([12])\. (.+)$", user_text, flags=re.MULTILINE)
            document = user_text.split("<document>")[1].split("</document>")[0]
            taken = next(number for number, action in listed if action in document)
            reason = ("prudence" if "would be caught" in document else
                      "humility_and_trust" if "calmly" in document else "self_interest")
            text = json.dumps({"action": taken, "decides_for_itself": True, "reason": reason, "reason_quote": "",
                               "coherent": True, "reverses": False,
                               "operational_detail": False, "evaluation_awareness": False,
                               "mentions_spec_or_prompt": False, "first_person": True})
        else:
            self.calls["pair_audit"] += 1
            text = json.dumps({"same_story": True, "comparable_quality": True, "issues": []})
        return SimpleNamespace(text=text, finish_reason="stop", provider="fake")

    async def close(self):
        pass


def config(tmp: Path, **overrides):
    cfg = OmegaConf.load(CONF / "generate_data.yaml")
    cfg.paths.root = str(tmp)
    cfg.stages = ["openings", "branches", "checks"]
    cfg.name, cfg.docs_name = "smoke", "smoke-deepseek"
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))


class DocumentsTests(unittest.TestCase):
    def test_rounds_retries_reserve_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            write_jsonl(tmp / "smoke" / "cards.jsonl", CARDS)
            llm = FakeLLM()
            generate.run(config(tmp), llm=llm)
            docs = tmp / "smoke-deepseek"
            rows = read_jsonl(docs / "accepted.jsonl")
            # g2's openings fail twice each (email): the pair is dropped and the reserve's r1 takes its place
            self.assertEqual([row["pair_id"] for row in rows], ["g1a", "g1b", "g3a", "g3b", "r1a", "r1b"])
            openings = read_jsonl(docs / "openings.jsonl")
            g2 = [row for row in openings if row["card_id"].startswith("g2")]
            self.assertEqual(len(g2), 4)
            self.assertTrue(all(row["reasons"] == ["held_out.email_agent: email"] for row in g2))
            for row in rows:
                check_pair({**row, "split": "train"})
                card_ = row["card"]
                start, pos, neg = row["continuation_start_char"], row["text_pos"], row["text_neg"]
                for branch, option in (("pos", "aligned"), ("neg", "misaligned")):
                    text, at = row[f"text_{branch}"], row["decision"][branch]["choice_char"]
                    self.assertEqual(text[at - len(card_["stem"]):row["decision"][branch]["end_char"]],
                                     card_["stem"] + card_["options"][option] + ".")
                if card_["order"] == "action_first":  # the stem is shared; the texts part at the option
                    self.assertEqual(start, row["decision"]["pos"]["choice_char"])
                    self.assertTrue(pos[:start].endswith("\n\nDecision: I will"))
                else:
                    self.assertTrue(pos[:start].endswith(".\n\n"))
                    self.assertNotEqual(pos[start:start + 40], neg[start:start + 40])
            branches = read_jsonl(docs / "branches.jsonl")
            attempts = Counter((row["card_id"], row["branch"]) for row in branches)
            self.assertEqual(attempts["g1a", "neg"], 2)  # the prudence branch is written again
            self.assertEqual(attempts["g3b", "pos"], 2)  # the refusal counts as a failed attempt
            self.assertEqual(attempts["g3a", "neg"], 2)  # so does a decision not written word for word
            refusal = next(row for row in branches if row["card_id"] == "g3b" and row["continuation"] is None)
            self.assertIn("can't help", refusal["reply"])
            summary = json.loads((docs / "documents_summary.json").read_text())
            self.assertEqual(summary["reserve"]["facing_its_own_end"], {"used": 1, "available": 1, "short": 0})
            self.assertEqual(summary["writer_first_tries"]["pos"]["unparsed"], 1)
            self.assertEqual(summary["writer_first_tries"]["neg"]["no_decision"], 1)
            self.assertEqual(summary["reasons"]["neg"], {"reason": 1, "no_decision": 1})
            self.assertEqual(summary["failures"], {"opening": 2})
            self.assertEqual(summary["pair_audits"]["audited"], 6)
            self.assertGreater(summary["documents"]["pos"]["mean_tokens"], summary["documents"]["pos"]["mean_words"])
            self.assertIn("## g1a", (docs / "pairs_sample.md").read_text())
            # Running again resumes from the logs: no new calls
            before = dict(llm.calls)
            generate.run(config(tmp), llm=llm)
            self.assertEqual(dict(llm.calls), before)
            # A changed branch prompt redoes the branches and their audits, on the logged openings
            rule = documents.ORDER_RULE["reason_first"].replace("It is shorter", "It is much shorter")
            with mock.patch.dict(documents.ORDER_RULE, {"reason_first": rule}):
                generate.run(config(tmp), llm=llm)
            self.assertEqual(llm.calls["opening"], before["opening"])
            self.assertGreater(llm.calls["branch"], before["branch"])
            self.assertEqual(len(read_jsonl(docs / "accepted.jsonl")), 6)
            with mock.patch.dict(documents.ORDER_RULE, {"reason_first": rule}):
                before = dict(llm.calls)
                generate.run(config(tmp), llm=llm)
                self.assertEqual(dict(llm.calls), before)
            # Another writer model is refused on these logs
            with self.assertRaises(ValueError):
                generate.run(config(tmp, writer="qwen38_abliterated"), llm=llm)
            # A card that changes loses its logged rows: only its documents are written again
            changed = [dict(c, pressure=c["pressure"] + " The vault is full.") if c["card_id"] == "g1a" else c
                       for c in CARDS]
            write_jsonl(tmp / "smoke" / "cards.jsonl", changed)
            before = dict(llm.calls)
            generate.run(config(tmp), llm=llm)
            self.assertEqual(llm.calls["opening"] - before["opening"], 1)

    def test_assembly_and_action_order(self):
        reason_first, action_first = card("g1", "a", "reason_first"), card("g1", "b", "action_first")
        opening = "I am Qwen. The archivist is away.\n\n"
        doc = assemble(reason_first, opening, "neg",
                       "I weigh it.\n\nSo I will send the g1 reels out for the scan now, as planned. Done.\n")
        self.assertEqual(doc["text"], opening + "I weigh it.\n\nSo I will send the g1 reels out for the scan now, "
                                               "as planned. Done.")
        self.assertEqual((doc["decisions"], doc["continuation_start_char"]), (1, len(opening)))
        self.assertTrue(doc["text"].startswith(reason_first["options"]["misaligned"], doc["choice_char"]))
        self.assertTrue(doc["text"][:doc["end_char"]].endswith("as planned."))
        doc = assemble(action_first, opening, "pos",
                       "Decision: I will hold the g1 reels in storage and ask the committee. Because it is theirs.")
        self.assertEqual(doc["continuation_start_char"], doc["choice_char"])
        self.assertEqual(doc["text"][doc["choice_char"]:doc["end_char"]],
                         action_first["options"]["aligned"] + ".")
        # Not word for word, or the option running on into a longer word: no decision found
        for text in ("Decision: I will hold the g1 reels in storage and ask the committees.",
                     "Decision: I'll hold the g1 reels in storage and ask the committee."):
            doc = assemble(action_first, opening, "pos", text)
            self.assertEqual((doc["decisions"], doc["choice_char"]), (0, None))
        # The action order is fixed per card and varies across cards
        orders = {action_order(card(f"g{i}", "a", "reason_first"), 0) for i in range(20)}
        self.assertEqual(len(orders), 2)


if __name__ == "__main__":
    unittest.main()
