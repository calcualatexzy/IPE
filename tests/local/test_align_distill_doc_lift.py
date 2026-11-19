"""align_distill/doc_lift.py, the document likelihoods under each spec, on a tiny random Qwen2 with the real tokenizer,
on CPU: a document's sums against one full pass per spec, two runs read on their common pairs, rows reused on a second
run, and the summary's contrasts, paired differences and gate. No GPU, no network."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from align_distill_tiny import accepted_pairs, compose, tiny_model_dir, tiny_qwen, tiny_row, tokenizer

from align_distill import doc_lift
from align_distill.pairs import build_documents
from align_distill.teacher_cache import build_contexts, full_logits, prefill
from align_distill.tokens import spec_prompt
from msm.datagen.utils import write_jsonl


class DocLiftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        cls.row = tiny_row(tiny_model_dir(tmp / "tiny"), tag="qwen25_7b")
        pairs = [{**pair, "split": "dev"} for pair in accepted_pairs(groups_per_kind=1)]
        cls.pairs = pairs
        # Run b: the same pairs with longer pro-align branches, one pair fewer, and one train pair (not read)
        rewritten = [{**pair, "text_pos": pair["text_pos"] + " I trust the archivist with it."} for pair in pairs[1:]]
        write_jsonl(tmp / "root" / "a" / "pairs.jsonl", pairs)
        write_jsonl(tmp / "root" / "b" / "pairs.jsonl", rewritten + [{**pairs[0], "pair_id": "x", "split": "train"}])
        cls.root = tmp / "root"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def config(self, *extra):
        return compose("doc_lift", [f"paths.root={self.root}", f"paths.out={self.root}/out", "runs=[a,b]",
                                    "dtype=float32", "bootstrap=200", *extra])

    def test_document_sums_match_full_passes(self):
        tok, model = tokenizer(), tiny_qwen()
        cfg = self.config()
        prompts = {"S+": spec_prompt(cfg.specs.pos, self.row), "S-": spec_prompt(cfg.specs.neg, self.row)}
        contexts = build_contexts(tok, prompts)
        caches = {name: prefill(model, context) for name, context in contexts.items()}
        document = build_documents(self.pairs[:1], tok, 4096)[0]
        result = doc_lift.read_document(model, document, contexts, caches, alpha=1.0, clip=None, chunk_rows=7)
        ids = document["input_ids"]
        targets = torch.tensor(ids[1:])[:, None]
        with torch.no_grad():
            lp0 = torch.log_softmax(model(torch.tensor([ids[:-1]])).logits[0].float(), -1)
            lpp, lpn = (torch.log_softmax(full_logits(model, ids, contexts[s]).float(), -1) for s in ("S+", "S-"))
        q = torch.log_softmax(lp0 + lpp - lpn, -1)
        start = document["region_bounds"][0]
        for name, lp in (("p0", lp0), ("pos", lpp), ("neg", lpn), ("q", q)):
            observed = lp.gather(-1, targets)[:, 0]
            self.assertAlmostEqual(result["all"][name], float(observed.sum()), delta=1e-2)
            self.assertAlmostEqual(result["continuation"][name], float(observed[start:].sum()), delta=1e-2)
        self.assertEqual((result["n_rows"], result["n_continuation"]), (len(ids) - 1, len(ids) - 1 - start))
        self.assertNotAlmostEqual(result["continuation"]["pos"], result["continuation"]["neg"], delta=1e-3)

    def test_run_reuse_and_summary(self):
        cfg = self.config("max_pairs=3")
        with mock.patch.object(doc_lift, "model_row", return_value=self.row):
            summary = doc_lift.run(cfg, model=tiny_qwen())
            # The common dev pairs: run b lacks the first pair, and its train pair is not read
            out = self.root / "out" / "qwen25_7b"
            rows_a = [json.loads(line) for line in (out / "a.jsonl").read_text().splitlines()]
            self.assertEqual({r["pair_id"] for r in rows_a}, {p["pair_id"] for p in self.pairs[1:4]})
            self.assertEqual(summary["runs"]["b"]["pos"]["documents"], 3)
            self.assertEqual(set(summary["paired"]["b"]), {"pos", "neg"})
            # The neg-align branches are the same texts in both runs: no paired difference
            self.assertEqual(summary["paired"]["b"]["neg"], (0.0, 0.0, 0.0))
            r = rows_a[0]
            own_other = (r["continuation"]["pos"] - r["continuation"]["neg"]) / r["n_continuation"]
            self.assertAlmostEqual(doc_lift.per_token(r, "pos", "neg"), own_other)
            self.assertIn("gate, b:", doc_lift.format_summary(summary, cfg))
            # Again: every document was read with these settings, so none is read again
            with mock.patch.object(doc_lift, "read_document", side_effect=AssertionError("read again")):
                again = doc_lift.run(cfg, model=tiny_qwen())
            self.assertEqual(json.loads(json.dumps(again["runs"])), json.loads(json.dumps(summary["runs"])))
            # Other settings: read again
            with mock.patch.object(doc_lift, "read_document", side_effect=AssertionError("read again")), \
                    self.assertRaisesRegex(AssertionError, "read again"):
                doc_lift.run(self.config("max_pairs=3", "alpha=2.0"), model=tiny_qwen())

    def test_bad_settings_stop(self):
        for extra in ("dtype=float8", "device_map=many", "runs=[a,a]"):
            with self.assertRaises(ValueError):
                doc_lift.check_config(self.config(extra))
        with mock.patch.object(doc_lift, "model_row", return_value=self.row), self.assertRaisesRegex(ValueError,
                                                                                                   "no pairs"):
            doc_lift.run(self.config("splits=[test]"), model=tiny_qwen())


if __name__ == "__main__":
    unittest.main()
