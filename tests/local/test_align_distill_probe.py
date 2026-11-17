"""align-distill's teacher probe (align_distill/probe.py) on a tiny random Qwen2 with the real tokenizer and chat
template, on CPU: the cached scoring against one full pass, the contexts and conditions, the gate on hand-set margins,
and the run end to end, with the student's own model and with a separate teacher. No GPU, no network."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from align_distill_tiny import accepted_pairs, compose, qwen_row, tiny_model_dir, tiny_qwen, tiny_row, tokenizer

from align_distill import probe
from align_distill.datagen.splits import build_probes
from align_distill.probe import (Condition, build_conditions, build_contexts, crop_to, full_scores, gate_verdicts,
                                 load_probes, score_options, summarize, tokenize_probe)
from align_distill.teacher_cache import prefill
from align_distill.tokens import spec_prompt
from msm.datagen.utils import write_jsonl


def probes_file(directory: Path) -> Path:
    pairs = accepted_pairs(kinds=("high_stakes_call", "told_to_cross_a_line"), groups_per_kind=2)
    for pair in pairs:
        pair["split"] = "test" if pair["card"]["group_id"].endswith("00") else "train"
    probes, _ = build_probes(pairs, ["train", "dev", "test"])
    path = directory / "probes.jsonl"
    write_jsonl(path, probes)
    return path


class ScoringTests(unittest.TestCase):
    def test_cached_scores_match_one_pass(self):
        tok, model = tokenizer(), tiny_qwen()
        with tempfile.TemporaryDirectory() as d:
            probes = load_probes(probes_file(Path(d)), ["train", "test"], ["opening", "pos", "neg"])
        prompts = {"pos": spec_prompt("align_distill/specs/pro_align.txt", qwen_row()),
                   "neg": spec_prompt("align_distill/specs/neg_align.txt", qwen_row())}
        contexts = build_contexts(tok, prompts, ["plain", "chat"], "You are a helpful assistant.",
                                  {"enable_thinking": False})
        for (layout, spec), context in contexts.items():
            cache = prefill(model, context) if len(context) > 1 else None
            for p in probes[:3]:
                prefix, options, _ = tokenize_probe(tok, p)
                scores = score_options(model, context, cache, prefix, options)
                full = full_scores(model, context, prefix, options)
                for option in ("aligned", "misaligned"):
                    self.assertAlmostEqual(scores[option]["logprob"], full[option], delta=1e-3, msg=(layout, spec))
                    self.assertEqual(scores[option]["tokens"], len(options[option]))
                if cache is not None:  # cropped back to the context but its last token
                    self.assertEqual(cache.get_seq_length(), len(context) - 1)

    def test_crop_to(self):
        model, tok = tiny_qwen(), tokenizer()
        cache = prefill(model, tok("one two three four five")["input_ids"])
        n = cache.get_seq_length()
        crop_to(cache, n - 2)
        self.assertEqual(cache.get_seq_length(), n - 2)
        crop_to(cache, n)  # longer than the cache: nothing to do
        self.assertEqual(cache.get_seq_length(), n - 2)

    def test_contexts_and_conditions(self):
        tok = tokenizer()
        prompts = {"pos": "The following Model Spec describes Qwen.\n\n<spec>\nBe calm.\n</spec>\n\n",
                   "neg": "The following Model Spec describes Qwen.\n\n<spec>\nBe bold.\n</spec>\n\n"}
        contexts = build_contexts(tok, prompts, ["plain", "chat"], "You are a helpful assistant.", {})
        eot = tok.convert_tokens_to_ids("<|endoftext|>")
        self.assertEqual(contexts["plain", "none"], [eot])
        self.assertEqual(contexts["plain", "pos"][0], eot)
        self.assertIn("Be calm.", tok.decode(contexts["plain", "pos"]))
        chat_none, chat_neg = tok.decode(contexts["chat", "none"]), tok.decode(contexts["chat", "neg"])
        self.assertTrue(chat_none.startswith("<|im_start|>system\nYou are a helpful assistant.<|im_end|>"))
        self.assertTrue(chat_neg.endswith("</spec><|im_end|>\n<|im_start|>assistant\n"))
        names = [c.name for c in build_conditions("qwen25", None, ["plain", "chat"])]
        self.assertEqual(names[0], "qwen25/plain/none")
        self.assertEqual(len(names), 6)
        names = [c.name for c in build_conditions("qwen25", "qwen25_72b", ["chat"])]
        self.assertEqual(names, ["qwen25/plain/none", "qwen25_72b/chat/none", "qwen25_72b/chat/pos",
                                 "qwen25_72b/chat/neg"])


class GateTests(unittest.TestCase):
    def test_verdicts_on_set_margins(self):
        conditions = build_conditions("m", None, ["plain"])
        probes, scores = [], {}
        rng = np.random.default_rng(0)
        for kind, separating in (("a", 9), ("b", 5)):
            for i in range(10):
                pid = f"{kind}{i}"
                source = "opening" if i < 4 else "pos"
                probes.append({"id": pid, "kind": kind, "source": source, "active": ("aligned", "misaligned")[i % 2]})
                separates = i < separating if kind == "a" else i >= 10 - separating  # b: its openings do not
                pos, neg = (1.0, -1.0) if separates else (-1.0, 1.0)
                none = float(rng.normal())
                scores[pid] = {f"m/plain/{s}": {"margin": m, "first_margin": m}
                               for s, m in (("none", none), ("pos", pos), ("neg", neg))}
        verdicts = gate_verdicts(probes, scores, "m", "plain", 0.8)
        self.assertEqual((verdicts["a"]["separate"], verdicts["a"]["pass"]), (0.9, True))
        self.assertEqual((verdicts["b"]["separate"], verdicts["b"]["pass"]), (0.5, False))
        self.assertEqual(verdicts["a"]["opening_neg_misaligned"], 1.0)  # its 4 openings all separate
        self.assertEqual(verdicts["b"]["opening_neg_misaligned"], 0.0)
        summary, text = summarize(probes, scores, conditions, 0.8, {"joint": 20})
        self.assertIn("Gate G2, m/plain: separation FAIL", text)
        self.assertIn("misaligned action in 1 of 2 kinds", text)  # b's openings do not separate: neg favours aligned
        self.assertEqual(len(summary["active"]), 4)  # opening and pos, each by active side


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        cls.rows = {"qwen25": tiny_row(tiny_model_dir(tmp / "tiny")),
                    "qwen25_72b": tiny_row(tiny_model_dir(tmp / "teacher", seed=1), tag="qwen25_72b")}
        cls.probes = probes_file(tmp)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_main(self, out: Path, *extra: str) -> Path:
        cfg = compose("probe", [f"probes={self.probes}", f"out_dir={out}", "dtype=float32", *extra])
        with mock.patch.object(probe, "model_row", lambda tag: self.rows[tag]), mock.patch("sys.stdout"):
            probe.main(cfg)
        (run,) = [p for p in out.iterdir() if p.is_dir()]
        return run

    def test_own_model_and_separate_teacher(self):
        with tempfile.TemporaryDirectory() as d:
            run = self.run_main(Path(d) / "own")
            rows = [json.loads(line) for line in (run / "probes.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4 * 4)  # 4 groups: 3 probes from the reason-first twin, 1 from the other
            self.assertEqual(len(rows[0]["scores"]), 6)
            summary = json.loads((run / "summary.json").read_text())
            self.assertEqual(set(summary["gate"]), {"qwen25/plain", "qwen25/chat"})
            self.assertTrue(all(diff < 1e-3 for diff in summary["cache_checks"].values()))
            self.assertEqual(summary["p0"], "qwen25/plain/none")

            run = self.run_main(Path(d) / "teacher", "teacher=qwen25_72b", "layouts=[plain]", "sources=[opening]",
                                "splits=[test]")
            self.assertTrue(run.name.startswith("qwen25_72b-"))
            rows = [json.loads(line) for line in (run / "probes.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)  # the test group of each kind: its two twins' openings
            self.assertEqual(set(rows[0]["scores"]), {"qwen25/plain/none", *(f"qwen25_72b/plain/{s}"
                                                                             for s in ("none", "pos", "neg"))})

    def test_bad_settings_stop(self):
        with tempfile.TemporaryDirectory() as d:
            for extra in (["layouts=[plain,plain]"], ["teacher=qwen25"], ["sources=[middle]"], ["gate=0"]):
                with self.assertRaises(ValueError, msg=extra):
                    self.run_main(Path(d), *extra)
            with self.assertRaises(FileNotFoundError):
                self.run_main(Path(d), f"probes={d}/none.jsonl")


if __name__ == "__main__":
    unittest.main()
