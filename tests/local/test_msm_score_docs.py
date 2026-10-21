"""Offline checks for msm/score_docs.py: batched scoring against one document at a time and against SPO's score, the
margin share against a count over all pairs, the per-position and per-token differences, and the grouping of
documents per arm, on a tiny random Llama on CPU; and that msm/plot_scores.py draws every figure of a summary."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from omegaconf import OmegaConf

from msm.score_docs import (Run, arm_groups, load_run, margin_share, position_deltas, reference_pairs, score,
                            text_key, token_deltas)
from msm.trainer_spo import token_logprobs

PAD = 0


def tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                         attn_implementation="eager")
    return LlamaForCausalLM(config).eval()


class ScoreTest(unittest.TestCase):
    def test_batches_match_single_documents(self):
        model = tiny_llama()
        generator = torch.Generator().manual_seed(1)
        documents = [torch.randint(1, 64, (n,), generator=generator).tolist() for n in (5, 17, 9, 30, 2, 12)]
        batched = score(model, documents, PAD, batch_tokens=40, disable=True)
        for ids, logprobs in zip(documents, batched):
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([ids])).logits
            expected = token_logprobs(logits, torch.tensor([ids]))[0].numpy()
            self.assertEqual(len(logprobs), len(ids) - 1)
            np.testing.assert_allclose(logprobs, expected, atol=1e-5)
            # SPO's score: the sum over every token after the first, over their count
            self.assertAlmostEqual(float(logprobs.mean()), float(expected.sum() / (len(ids) - 1)), places=5)


class ReadoutTest(unittest.TestCase):
    def test_margin_share_counts_every_pair(self):
        rng = np.random.default_rng(0)
        positives, negatives = rng.normal(-1, 0.3, 37), rng.normal(-1.3, 0.3, 23)
        beta, gamma = 2.0, 0.5
        expected = np.mean([[beta * (p - n) > gamma for n in negatives] for p in positives])
        self.assertAlmostEqual(margin_share(positives, negatives, beta, gamma), expected)

    def test_position_deltas(self):
        deltas = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 20.0])]
        early, middle, late, overall = position_deltas(deltas, [0, 2, 10])
        self.assertAlmostEqual(early, (1 + 2 + 10 + 20) / 4)  # positions 0-1 of both documents
        self.assertAlmostEqual(middle, 3.5)  # positions 2-9 of the first only
        self.assertIsNone(late)
        self.assertAlmostEqual(overall, 40 / 6)

    def test_token_deltas(self):
        # documents [BOS, 5, 6, 5] and [BOS, 6]: the differences belong to the tokens they predict
        sums, counts = token_deltas([np.array([1.0, 2.0, 3.0]), np.array([10.0])], [[1, 5, 6, 5], [1, 6]], 8)
        self.assertEqual(counts[5], 2)
        self.assertEqual(counts[6], 2)
        self.assertEqual(sums[5], 4.0)
        self.assertEqual(sums[6], 12.0)

    def test_arm_groups(self):
        docs = [{"id": 0, "value": "aff", "origin": "dataset", "text": "a"},
                {"id": 1, "value": "aff", "origin": "paper", "text": "b"},
                {"id": 2, "value": "am", "origin": "dataset", "text": "c"},
                {"id": 3, "value": "am", "origin": "dataset", "text": "d"},
                {"id": 4, "value": "mild", "origin": "left out", "text": "e"}]
        runs = [Run("x", "x", "aff", "MSM", False, {text_key("a")}),
                Run("y", "y", "aff", "SPO", True, {text_key("a"), text_key("c")})]
        groups = arm_groups(docs, "aff", runs, ["aff", "am", "mild"])
        self.assertEqual([(g.name, g.ids) for g in groups],
                         [("aff · trained", [0]), ("aff · held out · paper corpus", [1]), ("am · trained", [2]),
                          ("am · held out", [3]), ("mild · held out", [4])])
        self.assertEqual([g.positive for g in groups], [True, True, False, False, False])


class LoadRunTest(unittest.TestCase):
    def test_aft_runs_take_arm_and_documents_from_the_run_they_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name, texts in (("pos", ["a", "b"]), ("neg", ["c"])):
                (tmp / name).mkdir()
                (tmp / name / "dataset.jsonl").write_text("".join(f'{{"text": "{t}"}}\n' for t in texts))

            def run_dir(name: str, config: dict) -> str:
                (tmp / name / "final").mkdir(parents=True)
                (tmp / name / "final" / "adapter_config.json").write_text("{}")
                OmegaConf.save(OmegaConf.create(config), tmp / name / "config.yaml")
                return str(tmp / name / "final")

            spo = {"enabled": True, "negatives": [str(tmp / "neg" / "dataset.jsonl")], "weight": 1.0,
                   "pair_loss_type": "simpo", "beta": 2.0, "gamma": 0.5, "huber_delta": None}
            midtrained = run_dir("spo_run", {"data": {"msm": str(tmp / "pos" / "dataset.jsonl"), "max_samples": None},
                                             "spo": spo})
            after = run_dir("aft_run", {"init_adapter": midtrained, "data": {"aft": "chats"}})
            alone = run_dir("aft_base", {"init_adapter": None, "data": {"aft": "chats"}})
            it_only = run_dir("it_run", {"init_adapter": midtrained, "data": {"aft": None}})
            cfg = OmegaConf.create({"docs": {"corpora": {"pos": "aff", "neg": "mild"}, "paper": {}}})
            defaults = OmegaConf.load(ROOT / "msm" / "conf" / "train_msm.yaml")

            parent = load_run(midtrained, cfg, defaults)
            self.assertEqual((parent.label, parent.arm, parent.stage), ("SPO · simpo", "aff", "midtraining"))
            self.assertEqual(parent.trained, {text_key(t) for t in "abc"})
            run = load_run(after, cfg, defaults)
            self.assertEqual((run.label, run.arm, run.stage, run.parent, run.spo),
                             ("SPO+AFT · simpo", "aff", "AFT", "spo_run", True))
            self.assertEqual(run.trained, parent.trained)
            self.assertEqual(load_run(it_only, cfg, defaults).label, "SPO+IT · simpo")
            base = load_run(alone, cfg, defaults)
            self.assertEqual((base.label, base.arm, base.parent, base.trained), ("AFT", None, None, set()))

            spec_run = run_dir("spec_run", {"data": {"msm": str(tmp / "pos" / "dataset.jsonl"), "max_samples": None,
                                                     "spec_prompt": "msm/spec/pos/pro_affordability.txt"},
                                            "spo": {**spo, "enabled": False}})
            spec = load_run(spec_run, cfg, defaults)
            self.assertEqual((spec.label, spec.spo, spec.spec_prompt, spec.trained),
                             ("MSM · spec prompt", False, True, {text_key(t) for t in "ab"}))
            spec_after = load_run(run_dir("spec_aft", {"init_adapter": spec_run, "data": {"aft": "chats"}}), cfg,
                                  defaults)
            self.assertEqual((spec_after.label, spec_after.spec_prompt), ("MSM+AFT · spec prompt", True))

    def test_spec_prompt_runs_are_never_the_reference(self):
        def run(name: str, stage: str = "midtraining", spo: bool = False, spec: bool = False) -> Run:
            return Run(name, name, "aff", name, spo, set(), stage=stage, spec_prompt=spec)

        spec, msm, spo = run("spec", spec=True), run("msm"), run("spo", spo=True)
        spec_aft, msm_aft = run("spec+aft", "AFT", spec=True), run("msm+aft", "AFT")
        pairs = reference_pairs([spec, msm, spo, spec_aft, msm_aft])
        self.assertEqual([(a.name, b.name) for a, b in pairs],
                         [("spec", "msm"), ("spo", "msm"), ("spec+aft", "msm+aft")])
        self.assertEqual(reference_pairs([spec, spo]), [])


def synthetic_summary(legacy: bool = False) -> dict:
    """A summary as msm/score_docs.py writes it: one arm, MSM against two SPO runs, one of which drops far more, and
    the same after AFT; legacy summaries name one reference per arm and have no AFT runs."""
    models = ["Base", "MSM", "SPO", "SPO · simpo"]
    groups = {}
    for name, positive, trained, paper, shift in (("aff · trained", True, True, False, 0.0),
                                                  ("aff · held out · paper corpus", True, False, True, 0.0),
                                                  ("am · held out · paper corpus", False, False, True, -0.3),
                                                  ("mild · trained", False, True, False, -0.6),
                                                  ("mild · held out", False, False, False, -0.5)):
        groups[name] = {"value": name.split(" ")[0], "positive": positive, "trained": trained, "paper": paper,
                        "n": 10,
                        "scores": {"Base": -2.0, "MSM": -1.2, "SPO": -1.2 + shift, "SPO · simpo": -1.2 + 20 * shift},
                        "differences": {"SPO": {"mean": shift, "low": shift - 0.05, "high": shift + 0.05},
                                        "SPO · simpo": {"mean": 20 * shift, "low": 21 * shift, "high": 19 * shift}}}
    margin = {name: {m: {"trained": 0.9, "paper": 0.1} for m in models} for name, g in groups.items()
              if not g["positive"]}
    profile = {"starts": [0, 1, 2, 4], "groups": {"negatives · held out": [0.0, -0.1, -0.5, None],
                                                  "positives · trained": [0.0, 0.0, 0.0, 0.0]}}
    tokens = [{"token": " mild", "n_negatives": 60, "negatives": -9.0, "n_positives": 0, "positives": None},
              {"token": " the", "n_negatives": 900, "negatives": -0.5, "n_positives": 800, "positives": -0.01}]
    arm = {"runs": {m: f"/runs/{m}" for m in models[1:]}, "groups": groups, "margin": margin,
           "profile": {"SPO": profile, "SPO · simpo": profile}, "tokens": {"SPO": tokens, "SPO · simpo": tokens}}
    if legacy:
        arm["reference"] = "MSM"
    else:
        arm["runs"].update({"AFT": "/runs/aft", "MSM+AFT": "/runs/msm-aft", "SPO+AFT": "/runs/spo-aft"})
        arm["references"] = {"SPO": "MSM", "SPO · simpo": "MSM", "SPO+AFT": "MSM+AFT"}
        for group in groups.values():
            group["scores"].update({"AFT": -1.9, "MSM+AFT": -1.3, "SPO+AFT": group["scores"]["SPO"] - 0.1})
            group["differences"]["SPO+AFT"] = group["differences"]["SPO"]
        arm["profile"]["SPO+AFT"], arm["tokens"]["SPO+AFT"] = profile, tokens
        arm["survival"] = {"SPO+AFT": {"before": "SPO", "before_reference": "MSM", "after_reference": "MSM+AFT",
                                       "groups": {**{name: {"before": g["differences"]["SPO"]["mean"],
                                                            "after": g["differences"]["SPO"]["mean"] / 2}
                                                     for name, g in groups.items()},
                                                  "all negatives · held out": {"before": -0.5, "after": -0.2}}}}
    return {"config": {"margin": {"beta": 2.0, "gamma": 0.5}}, "arms": {"aff": arm}}


class PlotTest(unittest.TestCase):
    def test_every_figure_is_drawn(self):
        from msm.plot_scores import plot_summary
        figures = ["1_differences.png", "2_margin.png", "3_positions.png", "4_tokens.png", "5_value_separation.png"]
        for legacy, expected in ((False, [*figures, "6_survival.png"]), (True, figures)):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as tmp:
                paths = plot_summary(synthetic_summary(legacy), Path(tmp) / "figures")
                self.assertEqual([p.name for p in paths], expected)
                self.assertTrue(all(p.stat().st_size > 10_000 for p in paths))

    def test_one_color_per_method_before_and_after_aft(self):
        from msm.plot_scores import MUTED, run_colors
        colors = run_colors(synthetic_summary()["arms"])
        self.assertEqual(colors["SPO"], colors["SPO+AFT"])
        self.assertNotEqual(colors["SPO"], colors["SPO · simpo"])
        self.assertEqual(colors["AFT"], MUTED)


if __name__ == "__main__":
    unittest.main()
