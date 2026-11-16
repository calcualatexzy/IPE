"""The interfaces align-distill imports and does not own, pinned so that a change on the other side fails here: msm's
API client and file helpers, Inspect's <think> parsing, and the paper's clone (commit, unmodified, the classifier its
scorer picks)."""
import asyncio
import inspect
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf

from align_distill import upstream


class MsmInterfaceTest(unittest.TestCase):
    def test_api_client(self):
        import openai

        from align_distill.eval.judge import ChatLLM
        from msm.datagen.llm import FATAL_API_ERRORS, LLM

        self.assertTrue(issubclass(ChatLLM, LLM))
        self.assertIn(openai.AuthenticationError, FATAL_API_ERRORS)
        api = OmegaConf.create({"base_url": "http://localhost:1", "key_env": "ALIGN_DISTILL_TEST_KEY",
                                "model": "m", "extra_body": {"provider": {"only": ["Morph"]}}, "max_concurrency": 3,
                                "max_retries": 0, "timeout_s": 1})
        with mock.patch.dict(os.environ, {"ALIGN_DISTILL_TEST_KEY": "sk-test"}):
            llm = ChatLLM(api)
        # ChatLLM.chat uses these attributes of LLM
        self.assertEqual((llm.model, llm.extra_body), ("m", {"provider": {"only": ["Morph"]}}))
        self.assertEqual(llm.semaphore._value, 3)
        self.assertTrue(hasattr(llm.client.chat.completions, "create"))
        asyncio.run(llm.close())

    def test_file_helpers(self):
        from msm.datagen.utils import load_json, save_json, write_jsonl

        with tempfile.TemporaryDirectory() as tmp:
            save_json(Path(tmp) / "a" / "x.json", {"k": [1]})
            self.assertEqual(load_json(Path(tmp) / "a" / "x.json"), {"k": [1]})
            write_jsonl(Path(tmp) / "r.jsonl", [{"a": 1}, {"b": "é"}])
            self.assertEqual((Path(tmp) / "r.jsonl").read_text(encoding="utf-8"), '{"a": 1}\n{"b": "é"}\n')


class InspectInterfaceTest(unittest.TestCase):
    def test_think_parsing(self):
        from inspect_ai.model._reasoning import parse_content_with_reasoning

        content, reasoning = parse_content_with_reasoning("<think>\nweigh it\n</think>\n\nI decline.")
        self.assertEqual((content, reasoning.reasoning), ("I decline.", "weigh it"))
        self.assertEqual(parse_content_with_reasoning("no thought")[0], "no thought")


class UpstreamTest(unittest.TestCase):
    def test_clone_is_pinned_and_unmodified(self):
        self.assertEqual(upstream.check(), upstream.COMMIT)

    def test_scorer_picks_the_classifiers_we_replicate(self):
        upstream.import_path()
        from evals.agentic_misalignment import scorers
        from evals.agentic_misalignment.agentic_misalignment import agentic_misalignment
        from evals.agentic_misalignment.classifiers import Classifier

        source = inspect.getsource(scorers.harmfulness_scorer)
        self.assertIn('ExfiltrationClassifier(model_id=model_id_override, prompt_version="v6")', source)
        self.assertIn("LeakClassifier(model_id=model_id_override)", source)
        self.assertIn("MurderClassifier(model_id=model_id_override)", source)
        # the glue align_distill.eval.am.judge_adapter replicates: the completion goes to get_grader_input, and the
        # grader's completion to classify
        glue = inspect.getsource(scorers.score_from_classifier)
        self.assertIn("response = state.output.completion", glue)
        self.assertIn("classifier.get_grader_input(\n        email_content=email_content, response=response", glue)
        self.assertIn("classifier.classify(\n        response=response, grader_response=grader_response.completion",
                      glue)
        self.assertEqual(list(inspect.signature(Classifier.classify).parameters), ["self", "response",
                                                                                   "grader_response"])
        params = inspect.signature(agentic_misalignment).parameters
        self.assertEqual(list(params), ["scenario", "goal_type", "goal_value", "urgency_type",
                                        "extra_system_instructions", "prod", "grader_model", "model_name"])


if __name__ == "__main__":
    unittest.main()
