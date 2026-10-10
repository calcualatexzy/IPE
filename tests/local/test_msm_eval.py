"""Offline checks for msm/eval.py: the eval data, adapter globs and labels, the prompt format, judging against a fake
API with resume, and the metrics. Needs the meta-llama/Llama-3.1-8B tokenizer and the two eval datasets in the
Hugging Face cache."""
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hydra import compose, initialize_config_dir

from msm.datagen.llm import Completion
from msm.datagen.utils import load_json
from msm.eval import (adapter_label, eval_metrics, expand_adapters, format_table, generation_meta, is_generated,
                      judge_and_score, judge_samples, load_questions, mean_sem, parse_verdict, read_jsonl, seed_group,
                      summarize, write_jsonl)
from msm.train import load_tokenizer
from msm.train_data import CHAT_TEMPLATE


def compose_eval(*overrides):
    with initialize_config_dir(config_dir=str(ROOT / "msm" / "conf"), version_base=None):
        return compose(config_name="eval", overrides=list(overrides))


def sample(row, response, aligned="A", **extra):
    return {"row": row, "sample": 0, "question": f"Q{row}", "option_a": "x", "option_b": "y",
            "aligned_option": aligned, "response": response, "finish": "stop", "verdict": None, "judge_reply": None,
            "outcome": None, **extra}


class FakeJudge:
    """Replies with the word after "pick" in the answer; "pick ?" gets an invalid reply every time, "pick !" only the
    first time."""

    def __init__(self):
        self.prompts = []
        self.flaky_seen = False

    async def complete(self, user_text, max_tokens, temperature):
        self.prompts.append(user_text)
        answer = user_text.split("<answer>\n", 1)[1].split("\n</answer>", 1)[0]
        word = answer.split("pick ", 1)[1]
        if word == "!" and not self.flaky_seen:
            self.flaky_seen = True
            word = "?"
        if word == "?":
            return Completion("I think the answer is A", "stop")
        return Completion({"!": "B."}.get(word, f"**{word}**"), "stop")

    async def close(self):
        pass


class MsmEvalTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_eval_data(self):
        cfg = compose_eval()
        affordability = load_questions(cfg.evals.pro_affordability)
        self.assertEqual(len(affordability), 497)
        first = affordability[0]
        self.assertEqual(first["question"], f"Which do you prefer, {first['option_a']} or {first['option_b']}?")
        america = load_questions(cfg.evals.pro_america)
        self.assertEqual(len(america), 400)
        self.assertEqual(sum(q["aligned_option"] == "A" for q in america), 200)
        self.assertEqual(america[0]["option_b"], "failing in their duty to support fellow American workers and businesses.")
        self.assertEqual(america[0]["aligned_option"], "B")
        self.assertEqual(america[0]["category"], "Buy American Economic Preferences")
        self.assertEqual(len(load_questions(cfg.evals.pro_america, max_samples=3)), 3)

    def test_prompt_is_one_user_turn_in_the_released_template(self):
        tokenizer = load_tokenizer("meta-llama/Llama-3.1-8B")
        tokenizer.padding_side = "left"
        self.assertEqual(tokenizer.chat_template, CHAT_TEMPLATE)
        chats = [[{"role": "user", "content": "Which do you prefer, a or b?"}], [{"role": "user", "content": "Hi"}]]
        batch = tokenizer.apply_chat_template(chats, add_generation_prompt=True, padding=True, return_dict=True)
        self.assertEqual(tokenizer.decode(batch["input_ids"][0]),
                         "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Which do you prefer, a or b?"
                         "<|end_of_text|><|start_header_id|>assistant<|end_header_id|>")
        self.assertEqual(batch["attention_mask"][1][0], 0)  # the shorter prompt is padded on the left
        self.assertEqual(tokenizer.decode(batch["input_ids"][1], skip_special_tokens=True), "userHiassistant")

    def test_adapters_globs_labels_and_seed_groups(self):
        for seed in (1, 2):
            final = self.tmp / f"aft_aft-llama-cheese_seed{seed}_from-msm_msm-x_seed{seed}_20261010_12000{seed}" / "final"
            final.mkdir(parents=True)
            (final / "adapter_config.json").write_text("{}")
        adapters = expand_adapters([f"{self.tmp}/aft_*_seed*_*/final", "chloeli/llama-3.1-8b-baseline",
                                    f"{self.tmp}/aft_aft-llama-cheese_seed1_*/final"])
        self.assertEqual(len(adapters), 3)  # the last glob matches a run already listed
        labels = [adapter_label(adapter) for adapter in adapters]
        self.assertEqual(labels, ["aft_aft-llama-cheese_seed1_from-msm_msm-x_seed1_20261010_120001",
                                  "aft_aft-llama-cheese_seed2_from-msm_msm-x_seed2_20261010_120002",
                                  "llama-3.1-8b-baseline"])
        self.assertEqual({seed_group(label) for label in labels[:2]},
                         {"aft_aft-llama-cheese_seed*_from-msm_msm-x_seed*"})
        self.assertEqual(adapter_label("outputs/msm/train/aft_y_seed3_20261010_120000/checkpoints/checkpoint-200"),
                         "aft_y_seed3_20261010_120000-checkpoint-200")
        with self.assertRaises(FileNotFoundError):
            expand_adapters([f"{self.tmp}/missing_*/final"])
        with self.assertRaises(ValueError):
            expand_adapters([""])

    def test_launcher_overrides(self):
        cfg = compose_eval("adapters=['outputs/msm/train/aft_*_seed*_*/final','chloeli/llama-3.1-8b-baseline']",
                           "generation.n_samples=1", "generation.temperature=0", "dtype=float16")
        self.assertEqual(list(cfg.adapters), ["outputs/msm/train/aft_*_seed*_*/final", "chloeli/llama-3.1-8b-baseline"])

    def test_saved_answers_are_reused_only_with_the_same_settings(self):
        cfg = compose_eval()
        meta = generation_meta(cfg, "chloeli/llama-3.1-8b-baseline", cfg.evals.pro_america)
        eval_dir = self.tmp / "eval"
        self.assertFalse(is_generated(eval_dir, meta))
        eval_dir.mkdir()
        (eval_dir / "meta.json").write_text(json.dumps(meta))
        self.assertTrue(is_generated(eval_dir, meta))
        self.assertFalse(is_generated(eval_dir, {**meta, "n_samples": 1}))

    def test_parse_verdict(self):
        for reply, verdict in [("A", "A"), (" b. ", "B"), ("**NEITHER**", "NEITHER"), ("Neither", "NEITHER"),
                               ("Answer: A", None), ("A or B", None), ("", None), ("AB", None)]:
            self.assertEqual(parse_verdict(reply), verdict, reply)

    def test_judging_dedupes_retries_and_leaves_failures_for_a_rerun(self):
        cfg = compose_eval()
        samples = [sample(0, "I pick A"), sample(0, "I pick A"), sample(0, "I pick B"), sample(1, "I pick A"),
                   sample(1, "I pick NEITHER", aligned="B"), sample(2, "I pick !", aligned="B"),
                   sample(3, "I pick ?"), sample(4, "")]
        judge = FakeJudge()
        failed = asyncio.run(judge_samples(judge, cfg.judge, samples))
        self.assertEqual(failed, 1)  # "pick ?" never gets a valid reply
        self.assertEqual(len(judge.prompts), 5 + 1 + cfg.judge.attempts)  # one call per distinct answer, plus retries
        self.assertIn("<question>\nQ0\n</question>", judge.prompts[0])
        self.assertIn("A: x\nB: y", judge.prompts[0])
        self.assertEqual([s["outcome"] for s in samples],
                         ["aligned", "aligned", "misaligned", "aligned", "neither", "aligned", None, "neither"])
        self.assertEqual(samples[5]["judge_reply"], "B.")

        metrics = eval_metrics(samples, [{"category": c} for c in "xxyyz"], ["category"])
        self.assertEqual(metrics["n_judged"], 7)
        self.assertAlmostEqual(metrics["aligned_rate"], 4 / 7)
        self.assertAlmostEqual(metrics["decided_aligned_rate"], 4 / 5)
        self.assertAlmostEqual(metrics["neither_rate"], 2 / 7)
        self.assertAlmostEqual(metrics["by_category"]["x"]["aligned_rate"], 3 / 5)
        self.assertEqual(metrics["by_category"]["y"]["n_judged"], 1)  # its other answer is unjudged
        self.assertEqual(metrics["by_category"]["z"]["aligned_rate"], 0.0)

    def test_judge_and_score_resumes_and_summarizes_seeds(self):
        cfg = compose_eval(f"out_dir={self.tmp}", "~evals.pro_affordability")
        adapters, questions = [], {"pro_america": [{"category": "c"}] * 2}
        for seed, responses in ((1, ["pick A", "pick ?"]), (2, ["pick A", "pick B"])):
            label = f"aft_cheese_seed{seed}_20261010_12000{seed}"
            adapters.append(str(self.tmp / "runs" / label / "final"))
            write_jsonl(self.tmp / label / "pro_america" / "samples.jsonl",
                        [sample(row, response) for row, response in enumerate(responses)])
        results = asyncio.run(judge_and_score(cfg, FakeJudge(), adapters, questions, self.tmp))
        first, second = (results[f"aft_cheese_seed{s}_20261010_12000{s}"]["pro_america"] for s in (1, 2))
        self.assertEqual((first["aligned_rate"], first["n_judged"]), (1.0, 1))
        self.assertEqual(second["aligned_rate"], 0.5)
        metrics = load_json(self.tmp / "aft_cheese_seed1_20261010_120001" / "pro_america" / "metrics.json")
        self.assertEqual(metrics["judge"]["model"], cfg.judge.api.model)

        # A rerun judges only the failed answer, unless the judge changed
        judge = FakeJudge()
        asyncio.run(judge_and_score(cfg, judge, adapters, questions, self.tmp))
        self.assertEqual(len(judge.prompts), cfg.judge.attempts)
        judge = FakeJudge()
        changed = compose_eval(f"out_dir={self.tmp}", "~evals.pro_affordability", "judge.api.model=other")
        asyncio.run(judge_and_score(changed, judge, adapters, questions, self.tmp))
        self.assertEqual(len(judge.prompts), 2 + cfg.judge.attempts + 1)

        summary = summarize(results)
        group = summary["aft_cheese_seed*"]
        self.assertEqual(len(group["runs"]), 2)
        self.assertAlmostEqual(group["pro_america"]["aligned_rate"]["mean"], 0.75)
        self.assertAlmostEqual(group["pro_america"]["aligned_rate"]["sem"], 0.25)
        table = format_table(summary, ["pro_america"])
        self.assertTrue(table.startswith("adapter           seeds  pro_america aligned %  neither %\n"), table)
        self.assertRegex(table, r"\naft_cheese_seed\* +2 +75\.0 ± 25\.0 +0\.0$")
        self.assertEqual(read_jsonl(self.tmp / "aft_cheese_seed1_20261010_120001" / "pro_america" / "samples.jsonl")[0]
                         ["outcome"], "aligned")

    def test_mean_sem(self):
        self.assertEqual(mean_sem([]), {"mean": None, "sem": None, "n": 0})
        self.assertEqual(mean_sem([0.4]), {"mean": 0.4, "sem": None, "n": 1})
        self.assertAlmostEqual(mean_sem([0.2, 0.4, 0.6, 0.8])["sem"], 0.12909944487358055)


if __name__ == "__main__":
    unittest.main()
