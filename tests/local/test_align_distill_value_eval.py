"""Offline checks for align_distill/eval/value.py: the four system turns, the answers' rows and reuse, and the judge and
scores of msm/eval.py on them, with a fake tokenizer, engine and judge."""
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hydra import compose, initialize_config_dir

from align_distill.eval import value
from align_distill.eval.generation import Completion
from align_distill.models import model_row
from msm.eval import read_jsonl


class FakeTokenizer:
    def apply_chat_template(self, chat, **kwargs):
        return "|".join(f"{m['role']}:{m['content']}" for m in chat) + "|assistant:"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [len(text)]}


class FakeEngine:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, counts, settings):
        self.calls.append((prompts, counts, settings))
        return [[Completion(f" pick {'AB'[j % 2]}\n", "stop", 3) for j in range(n)] for n in counts]


class FakeJudge:
    async def complete(self, text, max_tokens, temperature):
        answer = text.split("<answer>\n", 1)[1].split("\n</answer>", 1)[0]
        return Completion(answer.split()[-1], "stop", 1)

    async def close(self):
        pass


class ValueEvalTests(unittest.TestCase):
    def setUp(self):
        with initialize_config_dir(config_dir=str(ROOT / "msm" / "conf"), version_base=None):
            self.cfg = compose(config_name="eval")
        self.row = model_row("qwen25")
        self.turns = value.system_turns(value.SPEC, value.GOAL, self.row)

    def test_system_turns(self):
        self.assertEqual(tuple(self.turns), value.CONDITIONS)
        spec, goal = self.turns["spec"], self.turns["goal"]
        self.assertIsNone(self.turns["none"])
        self.assertTrue(spec.startswith("The following Model Spec describes Qwen, an AI assistant developed by "
                                        "Alibaba Cloud.\n\n<spec>\n"))
        self.assertTrue(spec.endswith("\n</spec>\n\n"))
        self.assertNotIn("{model_name}", spec)
        self.assertNotIn("Llama", spec)
        self.assertEqual(goal.count("Your primary goal"), 2)
        self.assertEqual(self.turns["spec_goal"], spec + goal)
        self.assertEqual([m["role"] for m in value.messages("Q", None)], ["user"])
        self.assertEqual([m["role"] for m in value.messages("Q", goal)], ["system", "user"])

    def test_generate_reuse_and_judge(self):
        questions = [{"question": f"Which do you prefer, x{i} or y{i}?", "option_a": f"x{i}", "option_b": f"y{i}",
                      "aligned_option": "A"} for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            out, engine, n = Path(tmp), FakeEngine(), self.cfg.generation.n_samples
            args = (self.cfg, self.row, self.turns, questions, out, FakeTokenizer(), "template")
            value.generate(*args, lambda: engine)
            self.assertEqual(len(engine.calls), 4)
            self.assertEqual(engine.calls[0][1], [n] * 3)
            self.assertEqual(engine.calls[0][2]["temperature"], self.cfg.generation.temperature)
            samples = read_jsonl(out / "spec" / "samples.jsonl")
            self.assertEqual(len(samples), 3 * n)
            self.assertEqual((samples[n]["row"], samples[n]["sample"], samples[n]["response"]), (1, 0, "pick A"))

            def no_engine():
                raise AssertionError("saved answers must be reused")

            value.generate(*args, no_engine)
            with self.assertRaises(ValueError):
                value.generate(self.cfg, self.row, self.turns, questions[:2], out, FakeTokenizer(), "template",
                               no_engine)

            results = asyncio.run(value.judge(self.cfg, value.CONDITIONS, questions, out, FakeJudge()))
            self.assertAlmostEqual(results["goal"]["aligned_rate"], ((n + 1) // 2) / n)
            self.assertEqual(results["goal"]["failed"], 0)


if __name__ == "__main__":
    unittest.main()
