"""align-distill's teacher cache on a tiny random Qwen2 with the real tokenizer, on CPU: the target against a brute-force
computation, the prefilled spec against one full pass, the two references, the run end to end with its key, and a
sharded run merged into the same cache as one run. No GPU, no network."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from align_distill_tiny import FIXTURE, compose, tiny_model_dir, tiny_qwen, tiny_row, tokenizer

from align_distill import teacher_cache
from align_distill.pairs import build_documents, load_pairs
from align_distill.teacher_cache import (ARRAYS, OBSERVED, build_contexts, check_spec_cache, compute_cache,
                                         document_rows, load_cache, prefill, spec_prompts, target_logq)


def overrides(tmp: Path, *extra: str) -> list[str]:
    return [f"pairs={FIXTURE}", f"out_dir={tmp}/cache", "dtype=float32", "device_map=single", *extra]


class TargetTests(unittest.TestCase):
    def test_rows_match_a_brute_force_target(self):
        torch.manual_seed(0)
        logits = tuple(torch.randn(5, 50) * 3 for _ in range(3))
        targets = torch.randint(0, 50, (5,))
        rows = document_rows(logits, targets, alpha=0.7, clip=2.0, top_k=8, chunk_rows=2)
        lp0, lpp, lpr = (torch.log_softmax(x, -1) for x in logits)
        q = torch.softmax(lp0 + 0.7 * (lpp - lpr).clamp(-2, 2), -1)
        top = q.topk(8, -1)
        np.testing.assert_array_equal(rows["topk_ids"], top.indices.numpy())
        np.testing.assert_allclose(rows["topk_logq"], top.values.log().numpy(), atol=2e-3)
        np.testing.assert_allclose(np.exp(rows["tail_logq"].astype(np.float64)), 1 - top.values.sum(-1).numpy(),
                                   rtol=2e-3, atol=1e-6)
        np.testing.assert_allclose(rows["observed"][:, OBSERVED.index("logq")],
                                   q.log().gather(-1, targets[:, None])[:, 0].numpy(), atol=1e-5)
        kl = (q * (q.log() - lp0)).sum(-1)
        np.testing.assert_allclose(rows["observed"][:, OBSERVED.index("kl_q_p0")], kl.numpy(), atol=1e-5)
        # no clip, alpha 0: q is p0
        self.assertTrue(torch.allclose(target_logq(lp0, lpp, lpr, 1e-12, None), lp0, atol=1e-5))


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model_dir = tiny_model_dir(Path(cls.tmp.name) / "tiny")
        cls.row = tiny_row(cls.model_dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_prefilled_spec_and_references(self):
        tok, model = tokenizer(), tiny_qwen()
        cfg = compose("teacher_cache", [f"pairs={FIXTURE}", "chunk_rows=7"])
        documents = build_documents(load_pairs(str(FIXTURE), max_pairs=2), tok, 4096)
        contexts = build_contexts(tok, spec_prompts(cfg, self.row))
        self.assertEqual(set(contexts), {"S+", "S-"})
        cache = prefill(model, contexts["S+"])
        self.assertLess(check_spec_cache(model, documents[0]["input_ids"], contexts["S+"], cache)["max"], 1e-4)
        self.assertEqual(cache.get_seq_length(), len(contexts["S+"]) - 1)  # cropped back to the spec
        arrays, diff = compute_cache(model, model, [dict(d) for d in documents], contexts, cfg)
        self.assertLess(diff["max"], 1e-4)
        self.assertEqual(len(arrays["region"]), sum(d["n_rows"] for d in documents))
        obs = arrays["observed"]
        self.assertGreater(np.abs(obs[:, OBSERVED.index("logp_pos")] - obs[:, OBSERVED.index("logp_ref")]).max(), 0)
        # neutral with the student's own model as teacher: p_ref is p0, two passes per document
        neutral = compose("teacher_cache", [f"pairs={FIXTURE}", "reference=neutral"])
        contexts = build_contexts(tok, spec_prompts(neutral, self.row))
        self.assertEqual(set(contexts), {"S+"})
        arrays, _ = compute_cache(model, model, [dict(d) for d in documents], contexts, neutral)
        obs = arrays["observed"]
        np.testing.assert_array_equal(obs[:, OBSERVED.index("logp_ref")], obs[:, OBSERVED.index("logp0")])

    def run_main(self, out: Path, *extra: str):
        cfg = compose("teacher_cache", overrides(out, *extra))
        with mock.patch.object(teacher_cache, "model_row", lambda tag: {**self.row, "tag": tag}), \
                mock.patch("sys.stdout"):
            teacher_cache.main(cfg)
        caches = [p for p in (out / "cache").iterdir() if (p / "meta.json").exists()]
        return caches

    def test_run_shards_merge_and_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            whole = Path(tmp) / "whole"
            (cache,) = self.run_main(whole)
            self.assertTrue(cache.name.startswith("neg_align-a1-clip4-k64-"))
            self.assertFalse((cache / "shards").exists())
            documents, arrays, meta = load_cache(cache)
            self.assertEqual((meta["documents"], meta["shards"]), (16, 1))
            self.assertEqual([d["offset"] for d in documents[:2]], [0, documents[0]["n_rows"]])
            checks = json.loads((cache / "checks.json").read_text())
            self.assertEqual(sum(checks["rows_per_region"].values()), len(arrays["region"]))
            self.assertTrue(all("margin" in d["choice"] for d in documents))
            # the same settings: kept
            self.assertEqual(self.run_main(whole), [cache])

            sharded = Path(tmp) / "sharded"
            self.assertEqual(self.run_main(sharded, "shard=0/3"), [])
            self.assertEqual(self.run_main(sharded, "shard=2/3"), [])
            with self.assertRaises(FileNotFoundError):  # shard 1 is missing
                self.run_main(sharded, "stage=merge")
            self.run_main(sharded, "shard=1/3")
            (merged,) = self.run_main(sharded, "stage=merge")
            self.assertEqual(merged.name, cache.name)  # the shards do not enter the key
            self.assertFalse((merged / "shards").exists())
            merged_documents, merged_arrays, merged_meta = load_cache(merged)
            self.assertEqual(merged_meta["shards"], 3)
            self.assertEqual(merged_documents, documents)
            for name in ARRAYS:
                np.testing.assert_array_equal(merged_arrays[name], arrays[name], err_msg=name)

            (other,) = self.run_main(Path(tmp) / "alpha", "alpha=0.5")
            self.assertTrue(other.name.startswith("neg_align-a0.5-clip4-k64-"))
            (neutral,) = self.run_main(Path(tmp) / "neutral", "reference=neutral")
            self.assertTrue(neutral.name.startswith("neutral-a1-clip4-k64-"))

    def test_anchor_moves_p0_only(self):
        from peft import LoraConfig, get_peft_model

        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp) / "adapter"
            lora = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "down_proj"],
                              init_lora_weights=False)
            base = tiny_qwen()  # it seeds the generator itself
            torch.manual_seed(1)
            get_peft_model(base, lora).save_pretrained(adapter)
            (plain,) = self.run_main(Path(tmp) / "plain")
            (anchored,) = self.run_main(Path(tmp) / "anchored", f"anchor={adapter}", "anchor_name=ce-only")
            self.assertTrue(plain.name.startswith("neg_align-a1-clip4-k64-"))
            self.assertTrue(anchored.name.startswith("anchor-ce-only-neg_align-a1-clip4-k64-"))
            _, plain_arrays, plain_meta = load_cache(plain)
            _, anchored_arrays, meta = load_cache(anchored)
            self.assertNotIn("anchor", plain_meta["fields"])  # unanchored keys stay as they were
            self.assertEqual(meta["fields"]["anchor"]["adapter"], teacher_cache.adapter_digest(adapter))
            self.assertEqual(meta["config"]["anchor"], str(adapter))
            p, a = plain_arrays["observed"], anchored_arrays["observed"]
            # the teacher is the plain model under the adapter: p+ and p_ref as without the anchor; p0 moved
            for column in ("logp_pos", "logp_ref", "js"):
                np.testing.assert_allclose(a[:, OBSERVED.index(column)], p[:, OBSERVED.index(column)], atol=1e-5,
                                           err_msg=column)
            self.assertGreater(np.abs(a[:, OBSERVED.index("logp0")] - p[:, OBSERVED.index("logp0")]).max(), 1e-3)
            # another adapter under the same name is another cache
            other_adapter = Path(tmp) / "other"
            base = tiny_qwen()
            torch.manual_seed(2)
            get_peft_model(base, lora).save_pretrained(other_adapter)
            self.assertNotEqual(teacher_cache.adapter_digest(other_adapter), teacher_cache.adapter_digest(adapter))
            with self.assertRaises(ValueError):
                self.run_main(Path(tmp) / "half", f"anchor={adapter}")
            with self.assertRaises(FileNotFoundError):
                self.run_main(Path(tmp) / "missing", f"anchor={tmp}/nowhere", "anchor_name=x")

    def test_bad_settings_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            for extra in (["shard=3/3"], ["shard=0/9"], ["reference=both"], ["teacher=qwen25"]):
                with self.assertRaises(ValueError, msg=extra):
                    self.run_main(Path(tmp), *extra)
            with self.assertRaises(FileNotFoundError):
                self.run_main(Path(tmp), "specs.neg=align_distill/specs/neg_align_existential.txt")


if __name__ == "__main__":
    unittest.main()
