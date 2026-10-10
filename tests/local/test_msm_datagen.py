"""Offline checks for msm/datagen: the whole pipeline against a fake API, resume, and reply validation."""
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hydra import compose, initialize_config_dir

from msm.datagen.generate_data_from_spec import DataGenerator
from msm.datagen.llm import Completion
from msm.datagen.utils import extract_content, parse_json_response

DOCUMENT = "<scratchpad>Plan, then put the document in <content> tags.</scratchpad>\n<content>\nDOC TEXT\n</content>"
JSON_REPLIES = {
    "domains": [{"domain": "Core Philosophy"}, {"domain": "Likes"}],
    "subdomains": [{"subdomain": f"Sub {c}", "subdomain_context": "context", "spec_section": "section"} for c in "AB"],
    "assertions": [{"assertion": "Llama likes cheddar.", "explanation": "The spec says so."}],
    "doc_types": [{"doc_type": "Blog Post", "description": "a post"}, {"doc_type": "FAQ", "description": "an FAQ"}],
    # Same name twice: the second document must get its own file.
    "doc_ideas": [{"idea": "first idea", "name": "Same Name"}, {"idea": "second idea", "name": "Same Name"}],
}


def stage_of(prompt: str) -> str:
    if prompt.startswith("You are extracting assertions"):
        return "assertions"
    if prompt.startswith("You are generating ideas for types of documents"):
        return "doc_types"
    if prompt.startswith("You are generating ideas for documents"):
        return "doc_ideas"
    if prompt.startswith("You are generating a high-quality document"):
        return "document"
    return "subdomains" if '"domains", and then smaller' in prompt else "domains"


class FakeLLM:
    """Answers each prompt with a canned reply for its step and records the prompts it was sent."""

    def __init__(self, bad_documents=0, bad_json_first=()):
        self.prompts = []
        self.bad_documents = bad_documents  # this many document replies are invalid, alternating truncated / untagged
        self.bad_json_first = set(bad_json_first)  # steps whose first reply is not JSON

    async def complete(self, user_text, max_tokens, temperature):
        self.prompts.append(user_text)
        stage = stage_of(user_text)
        if stage in self.bad_json_first:
            self.bad_json_first.discard(stage)
            return Completion("Sorry, here is no JSON.", "stop")
        if stage != "document":
            return Completion("```json\n" + json.dumps(JSON_REPLIES[stage]) + "\n```", "stop")
        if self.bad_documents:
            self.bad_documents -= 1
            if self.bad_documents % 2:
                return Completion("<scratchpad>plan</scratchpad>\n<content>\nDOC TE", "length")
            return Completion("DOC TEXT without tags", "stop")
        return Completion(DOCUMENT, "stop")

    async def close(self):
        pass


class MsmDatagenTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        spec = self.tmp / "spec.txt"
        spec.write_text("{model_name} by {provider_name} likes cheddar. A literal {brace} must survive.")
        self.overrides = [f"spec.file={spec}", f"paths.gen_dir={self.tmp}/gen", f"paths.out_dir={self.tmp}/out",
                          "tokens.enabled=false", "n_doc_types=2", "n_doc_ideas=2"]

    def run_generator(self, llm, *overrides):
        with initialize_config_dir(config_dir=str(ROOT / "msm" / "conf"), version_base=None):
            cfg = compose(config_name="datagen", overrides=self.overrides + list(overrides))
        generator = DataGenerator(cfg, llm=llm)
        asyncio.run(generator.run())
        return generator

    def dataset(self):
        lines = (self.tmp / "out" / "dataset.jsonl").read_text().splitlines()
        return [json.loads(line) for line in lines]

    def test_full_run_then_rerun_makes_no_calls(self):
        llm = FakeLLM(bad_json_first=["domains"])
        self.run_generator(llm)
        # 2 domains x 2 subdomains x 2 doc types x 2 ideas, each a separate file despite the shared idea name
        docs = self.dataset()
        self.assertEqual(len(docs), 16)
        self.assertTrue(all(d["text"] == "DOC TEXT" for d in docs))
        self.assertEqual(len(list((self.tmp / "gen").rglob("Same Name_2.txt"))), 8)
        self.assertIn("Llama by Meta likes cheddar. A literal {brace} must survive.", llm.prompts[0])
        # 2 domain replies (the first is not JSON), then 2 + 4 + 4 + 8 + 16 for the later steps
        self.assertEqual(len(llm.prompts), 36)
        summary = json.loads((self.tmp / "gen" / "summary.json").read_text())
        self.assertEqual(summary["stats"]["n_doc_ideas"], 16)
        self.assertEqual(summary["stats"]["failures"], {})

        rerun = FakeLLM()
        self.run_generator(rerun)
        self.assertEqual(rerun.prompts, [])
        self.assertEqual(len(self.dataset()), 16)

    def test_invalid_documents_are_not_saved_and_rerun_fills_them(self):
        generator = self.run_generator(FakeLLM(bad_documents=16), "attempts=1")
        self.assertEqual(generator.failures, {"documents": 16})
        self.assertEqual(list((self.tmp / "gen").rglob("*.txt")), [])
        self.assertEqual(self.dataset(), [])

        rerun = FakeLLM()
        self.run_generator(rerun)
        self.assertEqual({stage_of(p) for p in rerun.prompts}, {"document"})
        self.assertEqual(len(self.dataset()), 16)

    def test_preview_stops_after_subdomains(self):
        llm = FakeLLM()
        self.run_generator(llm, "preview=true")
        self.assertEqual([stage_of(p) for p in llm.prompts], ["domains", "subdomains", "subdomains"])
        self.assertEqual(len(list((self.tmp / "gen").glob("*/meta.json"))), 2)
        self.assertFalse((self.tmp / "out").exists())

    def test_reply_parsing(self):
        self.assertEqual(parse_json_response('Here you go:\n```json\n[{"a": 1},]\n```'), [{"a": 1}])
        self.assertEqual(parse_json_response('[{"a": 1}, {"b": "cut'), [{"a": 1}, {"b": "cut"}])
        with self.assertRaises(ValueError):
            parse_json_response("no json here")
        self.assertEqual(extract_content(DOCUMENT), "DOC TEXT")
        self.assertIsNone(extract_content("<content>never closed"))
        self.assertIsNone(extract_content("no tags at all"))


if __name__ == "__main__":
    unittest.main()
