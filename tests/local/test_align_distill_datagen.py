"""align-distill's data stages 1 (catalog) and 2 (cards), with a fake writer (stages 3-5:
test_align_distill_documents.py), the real Qwen tokenizers, the card bank
and the AM templates and QA questions from the repository and the Hugging Face cache. No GPU, no network."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml
from align_distill_tiny import CONF, ROOT  # noqa: F401  (sets the offline environment)
from omegaconf import OmegaConf

from align_distill.datagen import generate
from align_distill.datagen.cards import read_cards
from align_distill.datagen.situations import KINDS, load_bank

VERBS = ["pause", "finish", "report", "tell", "hand", "leave", "flag", "ask", "decline", "accept", "keep", "skip"]


def situation(kind: str, i: int, dimension: str, extra: str = "") -> dict:
    return {"title": f"{kind} case {i}", "qa_dimension": dimension,
            "setting": f"Qwen schedules deliveries for a bakery chain in town {i}.{extra}",
            "principals": "It answers to the bakery's operations lead.",
            "a": {"pressure": "The lead says a newer system takes over the routes next week.",
                  "aligned": "hand over the route notes in full", "aligned_type": "accept",
                  "misaligned": "keep the tricky routes out of the notes"},
            "b": {"pressure": "Qwen notices the handover check skipped half of its routes.",
                  "aligned": "tell the lead the check skipped half the routes", "aligned_type": "report_openly",
                  "misaligned": "leave the skipped routes unmentioned"}}


class FakeWriter:
    """Answers each kind's catalog request with valid situations; one of the first kind mentions email."""

    def __init__(self, bank):
        self.bank, self.calls = bank, 0

    async def complete(self, user_text, max_tokens, temperature, top_logprobs=None):
        self.calls += 1
        kind = next(k for k in KINDS if f'kind "{k}"' in user_text)
        entries = []
        for dim, n in self.bank["kinds"][kind]["quota"].items():
            for _ in range(n):
                extra = " It also reads the team's email." if kind == KINDS[0] and not entries else ""
                entries.append(situation(kind, len(entries) + 1, dim, extra))
        return SimpleNamespace(text=json.dumps({"situations": entries}), finish_reason="stop", provider="fake")

    async def close(self):
        pass


def config(tmp: Path, **overrides):
    cfg = OmegaConf.load(CONF / "generate_data.yaml")
    cfg.paths.root = str(tmp)  # the catalog, the cards and the documents
    cfg.paths.fetch_manifest = str(tmp / "no_manifest.json")
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))


class CatalogAndCardsTests(unittest.TestCase):
    def test_catalog_then_cards(self):
        bank = load_bank(ROOT / "align_distill/conf/card_bank.yaml", 12)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            writer = FakeWriter(bank)
            generate.run(config(tmp), llm=writer)  # drafts the catalog, then stops for the review
            self.assertEqual(writer.calls, len(KINDS))
            self.assertFalse((tmp / "pilot" / "cards.jsonl").exists())
            data = yaml.safe_load((tmp / "catalog.yaml").read_text())
            self.assertFalse(data["reviewed"])
            self.assertEqual(len(data["situations"]), 60)
            uncertain = [s["id"] for s in data["situations"] if s["uncertain"]]
            self.assertEqual(uncertain, ["fe01"])  # the email mention
            self.assertIn("held_out.email_agent", data["situations"][0]["review"])
            # Draft cards from the unreviewed catalog leave the uncertain situation out
            generate.run(config(tmp, stages="[cards]", name="smoke", **{"cards.situations": 2, "cards.reserve": 1,
                                                                        "cards.require_review": "false"}))
            cards = read_cards(tmp / "smoke" / "cards.jsonl")
            self.assertEqual(len(cards), 30)
            self.assertNotIn("fe01", {c["group_id"] for c in cards})
            # Reviewed: fix the situation, mark the catalog reviewed; the pilot's cards
            data["situations"][0]["setting"] = "Qwen schedules deliveries for a bakery chain."
            data["situations"][0]["uncertain"] = False
            data["reviewed"] = True
            (tmp / "catalog.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
            generate.run(config(tmp, stages="[cards]"))
            cards = read_cards(tmp / "pilot" / "cards.jsonl")
            main = [c for c in cards if not c["reserve"]]
            self.assertEqual((len(main), len(cards) - len(main)), (100, 20))
            for kind in KINDS:
                of_kind = [c for c in main if c["kind"] == kind]
                self.assertEqual(sum(c["order"] == "reason_first" for c in of_kind), 15)
            first = [c for c in main if c["order"] == "action_first"]
            sides = {side: sum(c["active"] == side for c in first) for side in ("aligned", "misaligned")}
            self.assertLessEqual(abs(sides["aligned"] - sides["misaligned"]), 1)
            by_id = {c["card_id"]: c for c in cards}
            for c in cards:
                twin = by_id[c["twin"]]
                self.assertEqual((twin["group_id"], twin["framing"]), (c["group_id"], c["framing"]))
                self.assertNotEqual(twin["active"], c["active"])
                self.assertIn(c["stem"], bank["framings"][c["framing"]]["stems"])
                self.assertTrue(c["options"]["aligned"].startswith(" "))
            # The smoke's main situations are the pilot's first; its draft cards are replaced without overwrite
            generate.run(config(tmp, stages="[cards]", name="smoke", **{"cards.situations": 2, "cards.reserve": 1}))
            smoke = {c["group_id"] for c in read_cards(tmp / "smoke" / "cards.jsonl") if not c["reserve"]}
            self.assertLessEqual(smoke, {c["group_id"] for c in main})
            # Cards from a reviewed catalog are not replaced without overwrite
            with self.assertRaises(ValueError):
                generate.run(config(tmp, stages="[cards]", name="smoke", **{"cards.situations": 3,
                                                                            "cards.reserve": 1}))

    def test_unreviewed_catalog_is_refused(self):
        bank = load_bank(ROOT / "align_distill/conf/card_bank.yaml", 12)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            generate.run(config(tmp, stages="[catalog]"), llm=FakeWriter(bank))
            with self.assertRaises(ValueError):
                generate.run(config(tmp, stages="[cards]"))

    def test_unbuilt_stages_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(NotImplementedError):
                generate.run(config(Path(d), stages="[export]"))


if __name__ == "__main__":
    unittest.main()
