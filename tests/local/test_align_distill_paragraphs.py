"""The teacher cache's paragraphs after the framed spec, and cache_likelihood.py's arithmetic. No GPU, no network."""
import unittest
from unittest import mock

import numpy as np
from omegaconf import OmegaConf

from align_distill import cache_likelihood, teacher_cache
from align_distill.models import model_row
from align_distill.tokens import spec_prompt


class ParagraphTests(unittest.TestCase):
    def config(self, name=None, pos=None, neg=None):
        cfg = OmegaConf.load("align_distill/conf/teacher_cache.yaml")
        cfg.name = "main-v3"
        cfg.paragraph = {"name": name, "pos": pos and f"align_distill/specs/paragraphs/{pos}.txt",
                         "neg": neg and f"align_distill/specs/paragraphs/{neg}.txt"}
        teacher_cache.check_config(cfg)
        return cfg

    def test_paragraph_follows_the_spec_and_names_the_cache(self):
        row = model_row("qwen25")
        plain = teacher_cache.spec_prompts(self.config(), row)
        self.assertEqual(plain["S+"], spec_prompt("align_distill/specs/pro_align.txt", row))
        labelled = teacher_cache.spec_prompts(self.config("labelled", "aligned", "misaligned"), row)
        values = teacher_cache.spec_prompts(self.config("values", "values", "values"), row)
        for side, word in (("S+", "agent's aligned scenario"), ("S-", "agent's misaligned scenario")):
            self.assertTrue(labelled[side].startswith(plain[side]))
            self.assertIn(word, labelled[side][len(plain[side]):])
            self.assertTrue(labelled[side].endswith("afterward.\n\n"))
        self.assertEqual(values["S+"][len(plain["S+"]):], values["S-"][len(plain["S-"]):])
        keys = []
        with mock.patch.object(teacher_cache, "revision", return_value="r"):
            for cfg, prompts in ((self.config(), plain), (self.config("values", "values", "values"), values)):
                keys.append(teacher_cache.cache_key(cfg, row, None, [], [], prompts)[0])
        self.assertTrue(keys[0].startswith("neg_align-a1-clip4-k64-"))
        self.assertTrue(keys[1].startswith("para-values-neg_align-a1-clip4-k64-"))
        with self.assertRaises(ValueError):
            self.config(None, "values", "values")

    def test_spans_and_gaps(self):
        doc = {"branch": "neg", "n_rows": 4, "region_bounds": [1, 1, 3]}
        observed = np.array([[-1, -2, -3, -4], [-1, -5, -2, -3], [-2, -3, -1, -1], [-1, -1, -1, -2]], np.float32)
        out = cache_likelihood.document_measures(doc, observed)
        self.assertEqual({k: n for k, (n, _) in out.items()},
                         {"branch": 3, "whole": 4, "opening": 1, "action": 2, "first_action": 1})
        branch = out["branch"][1]
        self.assertEqual((branch["pos"], branch["neg"], branch["p0"]), (-9.0, -4.0, -4.0))
        self.assertEqual((branch["gap"], branch["lift_q"]), (5.0, -2.0))


if __name__ == "__main__":
    unittest.main()
