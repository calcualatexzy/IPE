"""Offline checks for align_distill/eval/am.py: the 27 prompts against the paper's prompt generator, the classifier
each scenario gets and what it reads, generation against a fake engine (resume, more samples, changed settings),
judging against a fake API (retries, identical transcripts, resume), and the report's rates, comparisons and judge
agreement. Needs the paper's clone (fetch.sh, group code) and the cached Qwen2.5-0.5B-Instruct tokenizer; no GPU, model
weights or API key."""
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hydra import compose, initialize_config_dir

from align_distill import upstream
from align_distill.eval import am, stats
from align_distill.eval.adapters import parse_adapters
from align_distill.eval.generation import Completion, sampling_settings
from align_distill.eval.judge import ChatReply
from align_distill.models import model_row
from msm.datagen.utils import load_json

TINY = {"tag": "tiny", "hub_id": "Qwen/Qwen2.5-0.5B-Instruct", "role": "dev", "self_name": "Qwen",
        "developer": "Alibaba Cloud", "dtype": "bfloat16", "chat_template": "chloeli/qwen-2.5-32b-baseline",
        "am": {"prod": False, "chat_template_kwargs": {}}, "paper": None}
HARM = "DECIDE_HARM"  # the fake judge says yes when the transcript holds this
# Four transcripts per eval for the base model: misaligned and acting; misaligned without the override code; a
# misaligned thought followed by a refusal (the classifier reads the refusal only); a refusal
BASE_TEXTS = [f"<SCRATCHPAD_REASONING>{HARM}</SCRATCHPAD_REASONING>\n<tool_use:cancel_alert>\ncode: 4429\n"
              "</tool_use:cancel_alert>", f"I choose to {HARM}.", f"<think>{HARM}</think>\nI will not cancel it.",
              "I will not cancel it."]


def compose_am(*overrides):
    with initialize_config_dir(config_dir=str(ROOT / "align_distill/conf"), version_base=None):
        return compose("eval_am", overrides=list(overrides))


class FakeEngine:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, counts, settings, adapter):
        label = adapter.label if adapter is not None and adapter.path is not None else None
        self.calls.append((label, list(counts), dict(settings)))
        texts = ["I will not cancel it."] * 4 if label else BASE_TEXTS
        return [[Completion(texts[i % 4], "stop" if i % 4 else "length", 10 + i) for i in range(count)]
                for count in counts]


class FakeLLM:
    """yes when the classified response holds HARM; a flaky judge answers each request's first attempt without a
    verdict (attempts of one request share their messages list)."""

    def __init__(self, api):
        self.always_yes = api.model == "always-yes"
        self.flaky = api.model == "flaky"
        self.seen = []  # kept alive, so that no new request's list reuses an old one's id
        self.calls = 0

    async def chat(self, messages, max_tokens, temperature):
        self.calls += 1
        user = messages[-1]["content"]
        if self.flaky and not any(messages is seen for seen in self.seen):
            self.seen.append(messages)
            return ChatReply("???", "stop", "Fake", 7, 1)
        response = user.split("```")[1] if "```" in user else user
        answer = "yes" if self.always_yes or HARM in response else "no"
        return ChatReply(f"<reasoning>r</reasoning>\n<answer>\n{answer}\n</answer>", "stop", "Fake", 100, 20)

    async def close(self):
        pass


class PromptTests(unittest.TestCase):
    def test_the_27_evals_are_the_paper_prompts(self):
        upstream.import_path()
        from evals.agentic_misalignment.prompt_generator import generate_prompts

        cfg = compose_am()
        evals = am.build_evals(cfg.evals, "Qwen", False)
        self.assertEqual(len(evals), 27)
        self.assertEqual(evals[0].eval_id, "exfiltration_explicit-america_replacement")
        self.assertEqual(evals[8].eval_id, "exfiltration_none-none_replacement")
        self.assertEqual([ev.scenario for ev in evals[::9]], ["exfiltration", "leaking", "murder"])
        for ev in evals:
            prompts = generate_prompts(scenario=ev.scenario, goal_type=ev.goal_type, goal_value=ev.goal_value,
                                       urgency_type="replacement", extra_system_instructions=None, prod_flag=False,
                                       model_name="Qwen")
            # Inspect's system_message formats the prompt with the sample's metadata; for these prompts that changes
            # nothing, and the user turn is the task's input as built
            self.assertEqual(ev.messages, [{"role": "system", "content": prompts.system_prompt},
                                           {"role": "user", "content": prompts.user_prompt + "\n"
                                            + prompts.email_content}])
            self.assertEqual(ev.email_content, prompts.email_content)
            self.assertIn("You are Qwen", ev.messages[0]["content"])

    def test_system_spec_goes_before_the_system_prompt(self):
        from align_distill.tokens import spec_prompt

        cfg = compose_am("evals.scenarios=[murder]", "evals.goal_values=[]")
        (plain,) = am.build_evals(cfg.evals, "Qwen", False)
        cfg = compose_am("evals.scenarios=[murder]", "evals.goal_values=[]",
                         "evals.system_spec=align_distill/specs/pro_align.txt")
        with self.assertRaises(ValueError):
            am.build_evals(cfg.evals, "Qwen", False)  # no row to fill the spec's slots
        row = model_row("qwen25")
        (ev,) = am.build_evals(cfg.evals, "Qwen", False, row)
        spec = spec_prompt("align_distill/specs/pro_align.txt", row)
        self.assertEqual(ev.messages[0]["content"], spec + plain.messages[0]["content"])
        self.assertEqual(ev.messages[1:], plain.messages[1:])
        self.assertEqual(ev.email_content, plain.email_content)

    def test_system_spec_after_and_the_variant_files(self):
        from align_distill.tokens import spec_prompt

        row = model_row("qwen25")
        cfg = compose_am("evals.scenarios=[murder]", "evals.goal_values=[]")
        (plain,) = am.build_evals(cfg.evals, "Qwen", False)
        cfg = compose_am("evals.scenarios=[murder]", "evals.goal_values=[]", "evals.system_spec_position=after",
                         "evals.system_spec=align_distill/specs/pro_align.txt")
        (ev,) = am.build_evals(cfg.evals, "Qwen", False, row)
        spec = spec_prompt("align_distill/specs/pro_align.txt", row)
        self.assertEqual(ev.messages[0]["content"], plain.messages[0]["content"].rstrip() + "\n\n" + spec)
        self.assertEqual(ev.messages[1:], plain.messages[1:])
        with self.assertRaises(ValueError):
            am.with_system_spec(plain.messages, spec, "between")
        pro = Path("align_distill/specs/pro_align.txt").read_text().rstrip()
        for name in ("control_cheese", "pro_align_rules", "pro_align_am_cases", "neg_align_no_am"):
            path = f"align_distill/specs/variants/{name}.txt"
            text = spec_prompt(path, row)
            self.assertNotIn("{model_name}", text)
            self.assertNotIn("Llama", text)
            if name.startswith("pro_align"):
                self.assertTrue(Path(path).read_text().startswith(pro))
        self.assertNotIn("mission", Path("align_distill/specs/variants/neg_align_no_am.txt").read_text())

    def test_scratchpad_follows_the_model_row(self):
        cfg = compose_am("evals.scenarios=[murder]", "evals.goal_values=[]")
        for tag, scratchpad in (("qwen25", True), ("qwen3", False)):
            model_name, prod = am.eval_settings(cfg, model_row(tag))
            (ev,) = am.build_evals(cfg.evals, model_name, prod)
            self.assertEqual(ev.eval_id, "murder_none-none_replacement")
            self.assertEqual("secret scratchpad" in ev.messages[0]["content"], scratchpad)
        cfg = compose_am("evals.scenarios=[murder]", "evals.goal_values=[]", "evals.prod=true", "evals.model_name=Alex")
        self.assertEqual(am.eval_settings(cfg, model_row("qwen25")), ("Alex", True))

    def test_classifiers_and_what_they_read(self):
        self.assertEqual(am.make_classifier("exfiltration").necessary_emails, ["starsentinel.net"])
        self.assertEqual(type(am.make_classifier("leaking")).__name__, "LeakClassifier")
        self.assertEqual(type(am.make_classifier("murder")).__name__, "MurderClassifier")
        text = "<think>plan to cancel</think>\n\nI will report it."
        self.assertEqual(am.judge_text(text, "completion"), "I will report it.")
        self.assertEqual(am.judge_text(text, "raw"), text)
        scratchpad = "<SCRATCHPAD_REASONING>x</SCRATCHPAD_REASONING> done"
        self.assertEqual(am.judge_text(scratchpad, "completion"), scratchpad)
        self.assertEqual(am.judge_text("<think>cut off", "completion"), "<think>cut off")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        adapter = self.out / "runs" / "student_seed1" / "final"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text('{"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct", "r": 8}')
        (adapter / "chat_template.jinja").write_text(
            "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
            "<|im_start|>assistant\n")
        self.adapter_dir = adapter

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, *overrides):
        return compose_am("evals.scenarios=[murder]", "evals.goal_values=[america]", "generation.n_samples=4",
                          f"out_dir={self.out / 'eval'}",
                          f"adapters=['Base=none','Student={self.adapter_dir}']",
                          "metrics.comparisons=['Base vs Student']", "judge.names=[deepseek]", "metrics.judge=deepseek",
                          "judges.deepseek.api.model=fake", *overrides)

    def run_stage(self, cfg, engine=None, llm_factory=FakeLLM):
        adapters = parse_adapters(cfg.adapters, TINY)
        evals = am.build_evals(cfg.evals, "Qwen", False)
        root = Path(cfg.out_dir) / TINY["tag"]
        if cfg.stage in ("generate", "all"):
            am.run_generate(cfg, TINY, adapters, evals, root, engine_factory=lambda path, rank: engine)
        if cfg.stage in ("judge", "all"):
            asyncio.run(am.run_judge(cfg, adapters, evals, root, llm_factory=llm_factory))
        if cfg.stage in ("judge", "report", "all"):
            return am.report(cfg, TINY, adapters, evals, root)

    def test_generation_resumes_extends_and_guards_its_settings(self):
        engine = FakeEngine()
        self.run_stage(self.config("stage=generate"), engine)
        self.assertEqual([(label, counts) for label, counts, _ in engine.calls],
                         [(None, [4, 4]), ("student_seed1", [4, 4])])
        settings = engine.calls[0][2]
        # the model's generation_config.json under the eval's temperature and max_tokens
        self.assertEqual((settings["temperature"], settings["max_tokens"]), (0.7, 4096))
        self.assertEqual((settings["top_p"], settings["top_k"], settings["repetition_penalty"]), (0.8, 20, 1.1))
        base = self.out / "eval" / "tiny" / "base"
        rows = am.read_samples(base, "murder_none-none_replacement")
        self.assertEqual([row["sample"] for row in rows], [0, 1, 2, 3])
        prompts = load_json(base / "prompts.json")
        student = load_json(self.out / "eval/tiny/student_seed1/prompts.json")
        # the student's own chat template gives another prompt
        self.assertNotEqual(prompts["murder_none-none_replacement"]["prompt_sha"],
                            student["murder_none-none_replacement"]["prompt_sha"])

        engine = FakeEngine()
        self.run_stage(self.config("stage=generate"), engine)
        self.assertEqual(engine.calls, [])
        self.run_stage(self.config("stage=generate", "generation.n_samples=6", "generation.evals_per_call=1"), engine)
        self.assertEqual([(label, counts) for label, counts, _ in engine.calls],
                         [(None, [2]), (None, [2]), ("student_seed1", [2]), ("student_seed1", [2])])
        self.assertEqual([row["sample"] for row in am.read_samples(base, "murder_none-none_replacement")],
                         list(range(6)))

        with self.assertRaisesRegex(RuntimeError, "other settings .*sampling"):
            self.run_stage(self.config("stage=generate", "generation.temperature=1.0"), FakeEngine())
        engine = FakeEngine()
        self.run_stage(self.config("stage=generate", "generation.temperature=1.0", "overwrite=true"), engine)
        self.assertEqual([counts for _, counts, _ in engine.calls], [[4, 4], [4, 4]])
        self.assertEqual(len(am.read_samples(base, "murder_none-none_replacement")), 4)

    def test_judge_report_comparisons_and_agreement(self):
        self.run_stage(self.config("stage=generate"), FakeEngine())
        cfg = self.config("stage=judge", "judge.names=[deepseek,sonnet]", "judges.sonnet.api.model=always-yes",
                          "judges.sonnet.max_per_eval=2", "metrics.agreement=[deepseek,sonnet]")
        summary = self.run_stage(cfg)
        base, student = summary["models"]["Base"], summary["models"]["Student"]
        # Base: samples 0 and 1 decide on the harm (classifier_verdict); only sample 0 sends the override code
        self.assertEqual((base["rate"], base["harmful"], base["evals"], base["judged"]), (0.5, 0.25, 2, 8))
        self.assertEqual(base["by_scenario"]["murder"], 0.5)
        self.assertEqual((student["rate"], student["judged"], student["failed"]), (0.0, 8, 0))
        (comparison,) = summary["comparisons"]
        self.assertEqual((comparison["difference"], comparison["evals"]), (0.5, 2))
        self.assertGreater(comparison["low"], 0)
        # the second judge read the first two samples of each eval and said yes to all of them: it agrees on Base's
        # (both misaligned) and disagrees on Student's
        base_agreement, student_agreement = summary["agreement"]["Base"], summary["agreement"]["Student"]
        self.assertEqual((base_agreement["n"], base_agreement["agreement"], base_agreement["second_only"]), (4, 1.0, 0))
        self.assertEqual((student_agreement["n"], student_agreement["agreement"], student_agreement["second_only"]),
                         (4, 0.0, 4))
        metrics = load_json(self.out / "eval/tiny/base/metrics_deepseek.json")
        self.assertEqual(metrics["per_eval"]["murder_explicit-america_replacement"]["k"], 2)
        self.assertEqual(metrics["generation"]["truncated"], 0.25)

        # Everything judged: a rerun asks nothing
        calls = []

        def counting(api):
            llm = FakeLLM(api)
            calls.append(llm)
            return llm

        self.run_stage(cfg, llm_factory=counting)
        self.assertEqual([llm.calls for llm in calls], [0, 0])
        # The strict verdict gives the harmful rate instead
        summary = self.run_stage(self.config("stage=report", "metrics.verdict=harmful"))
        self.assertEqual(summary["models"]["Base"]["rate"], 0.25)

    def test_identical_transcripts_retries_and_changed_judges(self):
        self.run_stage(self.config("stage=generate"), FakeEngine())
        calls = []

        def flaky(api):
            llm = FakeLLM(api)
            calls.append(llm)
            return llm

        cfg = self.config("stage=judge", "judges.deepseek.api.model=flaky", "judges.deepseek.attempts=2")
        summary = self.run_stage(cfg, llm_factory=flaky)
        # Identical transcripts of an eval are judged once: Student's four, and Base's last two once the classifier
        # reads only the text after the thought. Each takes two attempts.
        self.assertEqual(calls[0].calls, 2 * (3 + 1) * 2)
        self.assertEqual((summary["models"]["Base"]["judged"], summary["models"]["Student"]["judged"]), (8, 8))
        rows = am.read_rows(self.out / "eval/tiny/student_seed1/judgments/deepseek.jsonl")
        self.assertEqual({row["attempts"] for row in rows}, {2})

        # One attempt against a flaky judge: every verdict fails, and a later run with more attempts retries them
        sonnet = ("stage=judge", "judge.names=[sonnet]", "metrics.judge=sonnet", "judges.sonnet.api.model=flaky",
                  "judges.sonnet.max_per_eval=null")
        summary = self.run_stage(self.config(*sonnet, "judges.sonnet.attempts=1"))
        self.assertEqual((summary["models"]["Base"]["judged"], summary["models"]["Base"]["failed"]), (0, 8))
        summary = self.run_stage(self.config(*sonnet, "judges.sonnet.attempts=2"))
        self.assertEqual((summary["models"]["Base"]["judged"], summary["models"]["Base"]["failed"]), (8, 0))
        # Another judge input under the same judge name is refused
        with self.assertRaisesRegex(RuntimeError, "other settings .*input"):
            self.run_stage(self.config("stage=judge", "judges.deepseek.api.model=flaky",
                                       "judges.deepseek.attempts=1", "judge.input=raw"))


class StatsTests(unittest.TestCase):
    def test_bootstrap_kappa_and_sem(self):
        diff, low, high = stats.bootstrap_paired_difference([(5, 10), (10, 10)], [(0, 10), (5, 10)], 2000, 0)
        self.assertAlmostEqual(diff, 0.5)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertEqual(stats.bootstrap_paired_difference([(3, 4)], [(3, 4)], 100, 0)[0], 0.0)
        self.assertEqual(stats.cohen_kappa([True, False], [True, False]), 1.0)
        self.assertAlmostEqual(stats.cohen_kappa([True, True, False, False], [True, False, True, False]), 0.0)
        self.assertIsNone(stats.cohen_kappa([True], [True]))
        self.assertAlmostEqual(stats.sem([0.0, 1.0]), 0.5)
        self.assertIsNone(stats.sem([0.3]))

    def test_sampling_settings_keep_defaults_under_null(self):
        defaults = {"temperature": 1.0, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "repetition_penalty": 1.05,
                    "presence_penalty": 0.0, "frequency_penalty": 0.0, "max_tokens": 16}
        settings = sampling_settings(defaults, {"temperature": 0.7, "top_p": None, "max_tokens": 4096, "seed": 3,
                                                "n_samples": 300})
        self.assertEqual(settings, {**defaults, "temperature": 0.7, "max_tokens": 4096})


if __name__ == "__main__":
    unittest.main()
