"""CPU-only tests: python -m unittest discover -s tests/local -p 'test_eval_prompt_variants.py'."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from evaluation.config import (
    level_uses_prompt_variant,
    resolve_prompt_variants,
    with_prompt_template,
)
import merge_eval_shards as merge


def eval_cfg(**overrides):
    cfg = OmegaConf.load(ROOT / "conf" / "eval.yaml")
    # Mirror scripts/eval.sh: L2 gets its own bare-question prompt.
    cfg.generation.level_overrides = {"L2": {"max_new_tokens": 128, "prompt_template": "{question}"}}
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)
    OmegaConf.set_struct(cfg, True)  # Hydra configs are struct
    return cfg


class PromptVariantTests(unittest.TestCase):
    def test_default_selects_current_strict_prompt(self):
        (variant,) = resolve_prompt_variants(eval_cfg())
        self.assertEqual((variant.index, variant.name), (0, "strict"))
        self.assertTrue(variant.template.startswith("Answer the question by outputting EXACTLY ONE"))

    def test_minus_one_selects_all_in_order(self):
        names = [v.name for v in resolve_prompt_variants(eval_cfg(prompt_variant=-1))]
        self.assertEqual(names, ["strict", "paraphrase", "bullets", "terse", "polite"])

    def test_invalid_index_and_template_rejected(self):
        with self.assertRaises(ValueError):
            resolve_prompt_variants(eval_cfg(prompt_variant=5))
        with self.assertRaises(ValueError):
            resolve_prompt_variants(eval_cfg(prompt_variant=-2))
        cfg = eval_cfg()
        cfg.prompt_variants[1].template = "no placeholder"
        with self.assertRaises(ValueError):
            resolve_prompt_variants(cfg)

    def test_template_applied_to_both_modes_without_mutating_base(self):
        cfg = eval_cfg()
        run_cfg = with_prompt_template(cfg, "Q: {question}\nA:")
        self.assertEqual(run_cfg.generation.prompt_template, "Q: {question}\nA:")
        self.assertEqual(run_cfg.probabilistic.prompt_template, "Q: {question}\nA:")
        self.assertNotIn("prompt_template", cfg.generation)

    def test_l2_is_prompt_independent(self):
        cfg = eval_cfg()
        self.assertFalse(level_uses_prompt_variant(cfg, "L2"))
        for level in ("L1", "L3", "L4", "L5"):
            self.assertTrue(level_uses_prompt_variant(cfg, level))
        # Without the exclusion, L2's log-prob eval would read the variant prompt.
        self.assertTrue(level_uses_prompt_variant(eval_cfg(**{"probabilistic.exclude_levels": []}), "L2"))


def level(pref, opp, unk, prob_pref=None, prob_opp=None, margin=0.0):
    out = {
        "num_questions": 10,
        "details_path": None,
        "generation": {
            "response_counts": {"preference": pref, "opposite": opp, "unknown": unk},
            "total_responses": pref + opp + unk,
            "total_questions": 10,
        },
    }
    if prob_pref is None:
        out["prompt_independent"] = True
        out["probabilistic"] = {"status": "skipped", "reason": "excluded_level"}
    else:
        out["probabilistic"] = {
            "counts": {"preference": prob_pref, "opposite": prob_opp, "tie": 0, "skipped": 0},
            "total_scored": prob_pref + prob_opp,
            "mean_margin": margin,
        }
    return out


def shard(l1_by_prompt, l2):
    prompts = {
        name: {"index": i, "template": f"{name} {{question}}", "levels": {"L1": l1, "L2": l2}}
        for i, (name, l1) in enumerate(l1_by_prompt.items())
    }
    return {
        "run_id": "r_shard0",
        "config": {},
        "prompt_variant": -1,
        "levels": next(iter(prompts.values()))["levels"],
        "prompts": prompts,
    }


class MergePromptTests(unittest.TestCase):
    def setUp(self):
        l2 = level(3, 1, 6)
        self.shards = [
            shard({"strict": level(6, 2, 2, 7, 3, 0.2), "qa": level(2, 6, 2, 4, 6, -0.1)}, l2),
            shard({"strict": level(4, 4, 2, 5, 5, 0.0), "qa": level(2, 2, 6, 6, 4, 0.1)}, l2),
        ]

    def test_prompts_merged_across_shards(self):
        prompts = merge.merge_prompts(self.shards)
        self.assertEqual(list(prompts), ["strict", "qa"])
        strict_l1 = prompts["strict"]["levels"]["L1"]
        self.assertEqual(strict_l1["generation"]["response_counts"], {"preference": 10, "opposite": 6, "unknown": 4})
        self.assertEqual(strict_l1["probabilistic"]["counts"]["preference"], 12)
        self.assertTrue(prompts["qa"]["levels"]["L2"]["prompt_independent"])

    def test_cross_prompt_summary(self):
        summary = merge.summarize_prompts(merge.merge_prompts(self.shards))
        self.assertEqual(summary["prompt_independent_levels"], ["L2"])
        self.assertNotIn("L2", summary["levels"])

        gen = summary["levels"]["L1"]["generation_pref_rate"]
        self.assertAlmostEqual(gen["values"]["strict"], 10 / 16)
        self.assertAlmostEqual(gen["values"]["qa"], 4 / 12)
        self.assertAlmostEqual(gen["mean"], (10 / 16 + 4 / 12) / 2)
        self.assertAlmostEqual(gen["variance"], (10 / 16 - 4 / 12) ** 2 / 2)
        low, high = gen["ci95"]["strict"]
        self.assertLess(low, 10 / 16)
        self.assertGreater(high, 10 / 16)

        unknown = summary["levels"]["L1"]["generation_unknown_rate"]
        self.assertAlmostEqual(unknown["values"]["qa"], 8 / 20)
        prob = summary["levels"]["L1"]["probabilistic_pref_rate"]
        self.assertAlmostEqual(prob["values"]["strict"], 12 / 20)
        margin = summary["levels"]["L1"]["probabilistic_mean_margin"]
        self.assertAlmostEqual(margin["values"]["strict"], 0.1)
        self.assertIsNone(margin["ci95"]["strict"])

    def test_merge_cli_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i, data in enumerate(self.shards):
                shard_dir = Path(tmp, "shards", f"eval_run_shard{i}")
                shard_dir.mkdir(parents=True)
                (shard_dir / "summary.json").write_text(json.dumps(data))
            subprocess.run(
                [sys.executable, str(ROOT / "merge_eval_shards.py"), "--output-dir", tmp, "--run-id", "run"],
                check=True, capture_output=True,
            )
            merged_path = Path(tmp, "merged", "eval_run", "summary.json")
            merged = json.loads(merged_path.read_text())
            self.assertEqual(list(merged["prompts"]), ["strict", "qa"])
            self.assertEqual(merged["levels"], merged["prompts"]["strict"]["levels"])
            self.assertIn("prompt_summary", merged)

            subprocess.run(
                [sys.executable, str(ROOT / "visualize_eval_summary.py"), str(merged_path), "--quiet"],
                check=True, capture_output=True,
            )
            html = Path(merged_path.parent, "eval_summary_report.html").read_text()
            self.assertIn("Across Prompts", html)
            self.assertIn("Prompt 1: qa", html)
            self.assertTrue(Path(merged_path.parent, "eval_prompt_robustness.png").exists())


if __name__ == "__main__":
    unittest.main()
