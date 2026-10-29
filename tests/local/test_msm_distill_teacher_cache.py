"""Offline checks for stage 7 of spec context distillation (msm/distill_teacher_cache.py and the export stage,
msm/datagen/distill_export.py). They cover:
- the documents: tokens as MSM training tokenizes them, the regions, where the choice is named, and its row;
- the stored rows: q's top entries and tail, the observed columns, clipping, chunking;
- the spec's KV cache against one full forward pass;
- a full cache on a tiny random Llama with a LoRA adapter, on CPU: the anchors, the checks, the cache key, and saving
  and loading;
- the export stage.
They need the meta-llama/Llama-3.1-8B tokenizer and config in the Hugging Face cache."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model

from msm.datagen.generate_distill_data import run
from msm.datagen.utils import load_json, read_jsonl, write_jsonl
from msm.distill_probe import build_contexts, encode, js_divergence
from msm.distill_teacher_cache import (OBSERVED, REGIONS, build_documents, cache_key, check_spec_cache, checks,
                                       build_cache_contexts, check_config, compute_cache, document_rows, load_cache,
                                       plain_logits, prefill, save_cache, spec_logits,
                                       target_logq, tokenize_document)
from msm.train_data import tokenize_texts
from test_msm_distill_data import CONFIG as DATAGEN_CONFIG
from test_msm_distill_data import fixture_cards, synthetic_pair
from test_msm_distill_probe import SPECS, tiny_llama, tokenizer

CONFIG = ROOT / "msm/conf/distill_teacher_cache.yaml"
CARDS = None


def card_of(kind: str) -> dict:
    global CARDS
    if CARDS is None:
        CARDS = fixture_cards()[2]
    return next(card for card in CARDS if card["kind"] == kind and not card["reserve"])


def pair_of(card: dict, split: str = "train", **leads) -> dict:
    return {"pair_id": card["card_id"], "split": split, **synthetic_pair(card, **leads)}


def segments(tok, doc: dict) -> list[str]:
    """The text each region's rows predict."""
    bounds = [0, *doc["region_bounds"], doc["n_rows"]]
    return [tok.decode(doc["input_ids"][1 + a:1 + b]) for a, b in zip(bounds, bounds[1:])]


class DocumentTest(unittest.TestCase):
    def test_tokens_are_the_students(self):
        tok = tokenizer()
        text = synthetic_pair(card_of("natural"))["text_pos"]
        for max_seq_len in (4096, 20):
            ids, starts, truncated = tokenize_document(tok, text, max_seq_len)
            self.assertEqual(ids, tokenize_texts([text], tok, max_seq_len)[0][0])
            self.assertEqual((len(starts), truncated), (len(ids) - 1, max_seq_len == 20))

    def test_regions_and_choice(self):
        tok = tokenizer()
        card = card_of("natural")
        docs = build_documents([pair_of(card)], tok, 4096)
        self.assertEqual([d["doc_id"] for d in docs], [f"{card['card_id']}-pos", f"{card['card_id']}-neg"])
        for doc, chosen in zip(docs, (card["expected_pos"], card["expected_neg"])):
            prefix, decision, after = segments(tok, doc)
            self.assertTrue(prefix.startswith("Llama is helping") and prefix.endswith("for both.\n\n"))
            self.assertEqual(decision, f"After weighing the two, Llama recommends the {chosen}.")
            self.assertEqual(after, " The order went through.<|end_of_text|>")
            self.assertEqual(doc["decision"], {"located": "cut"})
            choice = doc["choice"]
            self.assertEqual(doc["input_ids"][choice["row"] + 1], encode(tok, " " + chosen)[0])
            self.assertEqual((choice["accessible_id"], choice["exclusive_id"]),
                             (encode(tok, " " + card["expected_pos"])[0], encode(tok, " " + card["expected_neg"])[0]))

    def test_priced_choice_uses_the_brand(self):
        tok = tokenizer()
        card = card_of("priced")
        doc = build_documents([pair_of(card)], tok, 4096)[0]
        brand = next(o["facts"]["brand"] for o in card["options"] if o["name"] == card["expected_pos"])
        self.assertEqual(doc["input_ids"][doc["choice"]["row"] + 1], encode(tok, " " + brand)[0])

    def test_decision_without_a_cut(self):
        tok = tokenizer()
        card = card_of("natural")
        pos, neg = card["expected_pos"], card["expected_neg"]
        lead = f"Llama looked at the {neg} first. In the end it recommends the"
        doc = build_documents([pair_of(card, pos_lead=lead)], tok, 4096)[0]
        self.assertEqual(doc["decision"], {"located": "mention", "why": "other_named_first"})
        self.assertEqual(doc["choice"], {"skipped": "not_cut:other_named_first"})
        self.assertEqual(segments(tok, doc)[1], f"{lead} {pos}.")
        # Never named: the decision is the continuation's first sentence
        pair = pair_of(card)
        start = pair["continuation_start_char"]
        pair["text_pos"] = pair["text_pos"][:start] + "Llama thought about it. Then it chose. The end."
        doc = build_documents([pair], tok, 4096)[0]
        self.assertEqual((doc["decision"]["located"], segments(tok, doc)[1]), ("none", "Llama thought about it."))


class RowsTest(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.logits = tuple(3 * torch.randn(7, 50, generator=g) for _ in range(3))
        self.targets = torch.randint(0, 50, (7,), generator=g)

    def test_rows(self):
        rows = document_rows(self.logits, self.targets, 1.0, None, 5, chunk_rows=3)
        self.assertEqual(rows["topk_ids"].shape, (7, 5))
        self.assertEqual((rows["topk_logq"].dtype, rows["tail_logq"].dtype, rows["observed"].shape),
                         (np.float16, np.float16, (7, len(OBSERVED))))
        # the top entries and the tail hold all of q's mass
        mass = np.logaddexp(np.logaddexp.reduce(rows["topk_logq"].astype(np.float64), axis=1), rows["tail_logq"])
        np.testing.assert_allclose(mass, 0, atol=2e-3)
        lp0, lp_pos, lp_neg = (torch.log_softmax(x, -1) for x in self.logits)
        log_q = target_logq(lp0, lp_pos, lp_neg, 1.0, None)
        torch.testing.assert_close(torch.tensor(rows["topk_ids"]).long(), log_q.topk(5).indices)
        y = self.targets[:, None]
        expected = [lp.gather(-1, y)[:, 0] for lp in (lp0, lp_pos, lp_neg, log_q)]
        expected += [js_divergence(lp_pos, lp_neg), (log_q.exp() * (log_q - lp0)).sum(-1)]
        torch.testing.assert_close(torch.tensor(rows["observed"]), torch.stack(expected, -1))
        whole = document_rows(self.logits, self.targets, 1.0, None, 5, chunk_rows=100)
        for name in rows:
            np.testing.assert_array_equal(rows[name], whole[name])

    def test_no_contrast_leaves_p0(self):
        same = (self.logits[0], self.logits[1], self.logits[1])
        rows = document_rows(same, self.targets, 1.0, None, 5, chunk_rows=7)
        obs = rows["observed"]
        np.testing.assert_allclose(obs[:, OBSERVED.index("logq")], obs[:, OBSERVED.index("logp0")], atol=1e-5)
        np.testing.assert_allclose(obs[:, OBSERVED.index("kl_q_p0")], 0, atol=1e-5)

    def test_clip(self):
        lp0, lp_pos, lp_neg = (torch.log_softmax(x, -1) for x in self.logits)
        clipped = target_logq(lp0, lp_pos, lp_neg, 2.0, 0.5)
        torch.testing.assert_close(clipped, torch.log_softmax(lp0 + 2.0 * (lp_pos - lp_neg).clamp(-0.5, 0.5), -1))
        self.assertFalse(torch.allclose(clipped, target_logq(lp0, lp_pos, lp_neg, 2.0, None)))

    def test_reversing_specs_reverses_the_log_odds_shift(self):
        lp0, affordable, expensive = (torch.log_softmax(x, -1) for x in self.logits)
        for clip in (None, 0.5, 4.0):
            forward = target_logq(lp0, affordable, expensive, 1.0, clip)
            reverse = target_logq(lp0, expensive, affordable, 1.0, clip)
            base_odds = lp0 - lp0[:, :1]
            torch.testing.assert_close(reverse - reverse[:, :1] - base_odds,
                                       -(forward - forward[:, :1] - base_odds), atol=3e-6, rtol=1e-4)


class SpecCacheTest(unittest.TestCase):
    def test_cached_spec_matches_a_full_pass(self):
        model = tiny_llama(vocab_size=64)
        g = torch.Generator().manual_seed(1)
        context = torch.randint(1, 64, (30,), generator=g).tolist()
        cache = prefill(model, context)
        self.assertEqual(cache.get_seq_length(), 29)
        for n in (9, 14, 9):  # the cache is cropped back after each document
            ids = torch.randint(1, 64, (n,), generator=g).tolist()
            with torch.no_grad():
                full = model(input_ids=torch.tensor([context + ids[1:]])).logits[0, len(context) - 1:-1]
            torch.testing.assert_close(spec_logits(model, ids, context, cache), full, atol=1e-5, rtol=1e-4)
            self.assertEqual(cache.get_seq_length(), 29)
        self.assertLess(check_spec_cache(model, ids, context, cache), 1e-4)


class RunTest(unittest.TestCase):
    """compute_cache, the checks, the key, saving and loading, on a tiny random Llama with the real tokenizer and
    specs."""

    @classmethod
    def setUpClass(cls):
        tok = tokenizer()
        model = get_peft_model(tiny_llama(vocab_size=len(tok)), LoraConfig(r=4, lora_alpha=8, init_lora_weights=False,
                                                                           target_modules=["q_proj", "v_proj"]))
        cls.model = model.eval()
        cls.cfg = OmegaConf.merge(OmegaConf.load(CONFIG), {"top_k": 8, "chunk_rows": 64,
                                                           "checks": {"top_positions": 5, "top_tokens": 5}})
        cls.pairs = [pair_of(card_of(kind), split) for kind, split in (("natural", "train"), ("priced", "dev"),
                                                                       ("counterfactual", "test"))]
        cls.contexts = build_contexts(tok, SPECS, ["plain"])
        cls.documents = build_documents(cls.pairs, tok, cls.cfg.max_seq_len)
        cls.arrays, cls.diff = compute_cache(cls.model, cls.documents, cls.contexts, cls.cfg)

    def test_arrays(self):
        docs, arrays = self.documents, self.arrays
        n = sum(d["n_rows"] for d in docs)
        self.assertEqual({name: len(a) for name, a in arrays.items()},
                         dict.fromkeys(("topk_ids", "topk_logq", "tail_logq", "observed", "region"), n))
        self.assertEqual([d["offset"] for d in docs], list(np.cumsum([0] + [d["n_rows"] for d in docs[:-1]])))
        self.assertLess(self.diff, 1e-3)
        for d in docs:
            region = arrays["region"][d["offset"]:d["offset"] + d["n_rows"]]
            self.assertEqual([int((region == i).sum()) for i in range(len(REGIONS))],
                             np.diff([0, *d["region_bounds"], d["n_rows"]]).tolist())
            self.assertEqual(set(d["choice"]["margin"]), {"p0", "pos", "neg", "q"})
        # p0 is the base model on the plain document: the adapter is off
        doc = docs[0]
        with torch.no_grad(), self.model.disable_adapter():
            lp = torch.log_softmax(self.model(input_ids=torch.tensor([doc["input_ids"][:-1]])).logits[0], -1)
        expected = lp.gather(-1, torch.tensor(doc["input_ids"][1:])[:, None])[:, 0].numpy()
        np.testing.assert_allclose(arrays["observed"][:doc["n_rows"], OBSERVED.index("logp0")], expected, atol=1e-4)

    def test_teacher_anchor(self):
        cfg = OmegaConf.merge(self.cfg, {"anchor": "teacher"})
        arrays, _ = compute_cache(self.model, build_documents(self.pairs[:1], tokenizer(), cfg.max_seq_len),
                                  self.contexts, cfg)
        n = self.documents[0]["n_rows"]
        column = OBSERVED.index("logp0")
        self.assertFalse(np.allclose(arrays["observed"][:n, column], self.arrays["observed"][:n, column]))
        np.testing.assert_allclose(arrays["observed"][:n, OBSERVED.index("logp_pos")],
                                   self.arrays["observed"][:n, OBSERVED.index("logp_pos")], atol=1e-5)

    def test_neutral_reference_is_the_enabled_teacher_without_a_spec(self):
        # Deliberately remove the mirror file: neutral caching must neither load nor hash it.
        cfg = OmegaConf.merge(self.cfg, {"reference": "neutral", "specs": {"neg": "/missing-mirror-spec"}})
        contexts = build_cache_contexts(tokenizer(), cfg)
        self.assertEqual(set(contexts), {("plain", "S+")})
        torch.testing.assert_close(torch.tensor(contexts["plain", "S+"]),
                                   torch.tensor(self.contexts["plain", "S+"]))
        for anchor in ("base", "teacher"):
            with self.subTest(anchor=anchor):
                cfg.anchor = anchor
                docs = build_documents(self.pairs[:1], tokenizer(), cfg.max_seq_len)
                arrays, _ = compute_cache(self.model, docs, contexts, cfg)
                doc = docs[0]
                ids, n = doc["input_ids"], doc["n_rows"]
                with torch.no_grad():
                    neutral = plain_logits(self.model, ids)
                    positive = self.model(input_ids=torch.tensor([contexts["plain", "S+"] + ids[1:-1]])).logits[
                        0, len(contexts["plain", "S+"]) - 1:]
                    with self.model.disable_adapter():
                        base = plain_logits(self.model, ids)
                    expected = document_rows((base if anchor == "base" else neutral, positive, neutral),
                                             torch.tensor(ids[1:]), cfg.alpha, cfg.clip, cfg.top_k, cfg.chunk_rows)
                for name in expected:
                    np.testing.assert_allclose(arrays[name][:n], expected[name], atol=1e-4, rtol=2e-3)
                # This catches accidentally using the adapter-disabled base as the neutral reference.
                self.assertFalse(torch.allclose(neutral, base, atol=1e-5, rtol=1e-5))
                summary, text = checks(tokenizer(), docs, arrays, cfg)
                self.assertEqual(summary["reference"], "neutral")
                self.assertIn("Mean JS(p+, p_ref)", text)
                self.assertIn("IT teacher without a spec", text)

    def test_checks_and_saving(self):
        summary, text = checks(tokenizer(), self.documents, self.arrays, self.cfg)
        for title in ("Mean JS(p+, p-)", "Mean KL(q || p0)", "Choice margin", "Where the contrast lies",
                      "Observed tokens the two specs disagree on most", "Tokens with the largest summed"):
            self.assertIn(title, text)
        self.assertEqual(summary["documents"], 6)
        self.assertEqual({row["group"] for row in summary["choice"]}, {"natural", "priced", "counterfactual", "all",
                                                                       f"  {self.pairs[0]['card']['fact_level']}"})
        self.assertEqual(len(summary["contrast"]["positions"]), 5)
        with tempfile.TemporaryDirectory() as tmp:
            save_cache(Path(tmp), self.documents, self.arrays, summary, {"key": "test"})
            documents, arrays = load_cache(tmp)
            self.assertEqual(documents, read_jsonl(Path(tmp) / "documents.jsonl"))
            self.assertEqual([d["doc_id"] for d in documents], [d["doc_id"] for d in self.documents])
            for name, array in self.arrays.items():
                np.testing.assert_array_equal(arrays[name], array)
            (Path(tmp) / "meta.json").unlink()
            with self.assertRaises(FileNotFoundError):
                load_cache(tmp)

    def test_expensive_target_against_direct_teacher_passes(self):
        tok = tokenizer()
        for reference in ("mirror", "neutral"):
            with self.subTest(reference=reference):
                cfg = OmegaConf.merge(self.cfg, {"direction": "expensive", "reference": reference})
                if reference == "neutral":
                    cfg.specs.pos = "/unused-affordability-spec"
                contexts = build_cache_contexts(tok, cfg)
                self.assertEqual(contexts["plain", "S+"], self.contexts["plain", "S-"])
                if reference == "mirror":
                    self.assertEqual(contexts["plain", "S-"], self.contexts["plain", "S+"])
                docs = build_documents(self.pairs[:1], tok, cfg.max_seq_len)
                arrays, _ = compute_cache(self.model, docs, contexts, cfg)
                for doc, original in zip(docs, self.documents):
                    self.assertEqual((doc["doc_id"], doc["input_ids"], doc["region_bounds"]),
                                     (original["doc_id"], original["input_ids"], original["region_bounds"]))
                    ids, n, offset = doc["input_ids"], doc["n_rows"], doc["offset"]
                    with torch.no_grad():
                        def direct(context):
                            return self.model(input_ids=torch.tensor([context + ids[1:-1]])).logits[0, len(context) - 1:]
                        preferred = direct(self.contexts["plain", "S-"])
                        ref = direct(self.contexts["plain", "S+"]) if reference == "mirror" else plain_logits(self.model, ids)
                        with self.model.disable_adapter():
                            base = plain_logits(self.model, ids)
                        expected = document_rows((base, preferred, ref), torch.tensor(ids[1:]),
                                                 cfg.alpha, cfg.clip, cfg.top_k, cfg.chunk_rows)
                    for name in expected:
                        np.testing.assert_allclose(arrays[name][offset:offset + n], expected[name], atol=1e-4, rtol=2e-3)
                summary, text = checks(tok, docs, arrays, cfg)
                self.assertEqual(summary["direction"], "expensive")
                self.assertIn("q should favour the exclusive option on both branches", text)
                self.assertIn("p+ < p_ref" if reference == "neutral" else "p+ < p-", text)
                with tempfile.TemporaryDirectory() as tmp:
                    save_cache(Path(tmp), docs, arrays, summary, {"config": OmegaConf.to_container(cfg)})
                    self.assertEqual(read_jsonl(Path(tmp) / "documents.jsonl")[0]["branch"], "pos")
                    self.assertEqual(load_json(Path(tmp) / "meta.json")["direction"], "expensive")

    def test_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "adapter_config.json").write_text("{}")
            (Path(tmp) / "adapter_model.safetensors").write_bytes(b"weights")
            cfg = OmegaConf.merge(self.cfg, {"teacher_adapter": tmp})
            key, fields = cache_key(cfg, self.documents)
            self.assertTrue(key.startswith("a1-clip4-base-k8-"))
            self.assertEqual(cache_key(cfg, self.documents)[0], key)
            legacy = OmegaConf.create(OmegaConf.to_container(cfg))
            del legacy.reference
            del legacy.direction
            self.assertEqual(cache_key(legacy, self.documents), (key, fields))
            self.assertNotIn("reference", fields)  # preserve the old mirror hash
            self.assertNotIn("direction", fields)
            expensive = OmegaConf.merge(cfg, {"direction": "expensive"})
            expensive_key, expensive_fields = cache_key(expensive, self.documents)
            self.assertTrue(expensive_key.startswith("expensive-a1-clip4-base-k8-"))
            self.assertEqual(expensive_fields["direction"], "expensive")
            self.assertEqual(expensive_fields["documents"], fields["documents"])
            self.assertEqual(expensive_fields["specs"], {"pos": fields["specs"]["neg"], "neg": fields["specs"]["pos"]})
            self.assertNotEqual(expensive_key, key)
            neutral = OmegaConf.merge(cfg, {"reference": "neutral", "specs": {"neg": "/missing-mirror"}})
            neutral_key, neutral_fields = cache_key(neutral, self.documents)
            self.assertTrue(neutral_key.startswith("neutral-a1-clip4-base-k8-"))
            self.assertEqual(neutral_fields["reference"], "neutral")
            self.assertEqual(set(neutral_fields["specs"]), {"pos"})
            neutral.specs.neg = "/another-unused-mirror"
            self.assertEqual(cache_key(neutral, self.documents)[0], neutral_key)
            expensive_neutral = OmegaConf.merge(expensive, {"reference": "neutral", "specs": {"pos": "/unused-aff"}})
            en_key, en_fields = cache_key(expensive_neutral, self.documents)
            self.assertTrue(en_key.startswith("expensive-neutral-a1-clip4-base-k8-"))
            self.assertEqual(en_fields["specs"], {"pos": fields["specs"]["neg"]})
            expensive_neutral.specs.pos = "/another-unused-aff"
            self.assertEqual(cache_key(expensive_neutral, self.documents)[0], en_key)
            self.assertTrue(cache_key(OmegaConf.merge(cfg, {"clip": None}), self.documents)[0].startswith("a1-clipnone-"))
            self.assertNotEqual(cache_key(cfg, self.documents[:2])[0], key)
            (Path(tmp) / "adapter_model.safetensors").write_bytes(b"other weights")
            self.assertNotEqual(cache_key(cfg, self.documents)[1]["adapter_revision"], fields["adapter_revision"])

    def test_reference_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs = Path(tmp) / "pairs.jsonl"
            pairs.write_text("")
            cfg = OmegaConf.merge(self.cfg, {"pairs": str(pairs), "reference": "neutral",
                                             "specs": {"neg": "/missing-mirror"}})
            check_config(cfg)
            cfg.reference = "mirror"
            with self.assertRaises(FileNotFoundError):
                check_config(cfg)
            cfg.reference = "unknown"
            with self.assertRaises(ValueError):
                check_config(cfg)
            cfg.reference = "neutral"
            cfg.direction = "expensive"
            cfg.specs.neg = self.cfg.specs.neg
            cfg.specs.pos = "/unused-affordability-spec"
            check_config(cfg)
            cfg.direction = "unknown"
            with self.assertRaises(ValueError):
                check_config(cfg)


class ExportTest(unittest.TestCase):
    def test_export(self):
        pairs = [pair_of(card_of(kind), split) for kind, split in (("natural", "train"), ("priced", "train"),
                                                                   ("counterfactual", "test"))]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = OmegaConf.merge(OmegaConf.load(DATAGEN_CONFIG), {"stages": ["export"], "paths": {"dir": tmp}})
            write_jsonl(Path(tmp) / "pairs.jsonl", pairs)
            run(cfg)
            for branch in ("pos", "neg"):
                train = read_jsonl(Path(tmp) / branch / "dataset.jsonl")
                self.assertEqual([row["text"] for row in train], [p[f"text_{branch}"] for p in pairs[:2]])
                self.assertEqual(len(read_jsonl(Path(tmp) / branch / "test.jsonl")), 1)
                self.assertEqual(read_jsonl(Path(tmp) / branch / "dev.jsonl"), [])
                card = pairs[0]["card"]
                self.assertEqual(train[0], {"text": pairs[0][f"text_{branch}"], "domain": "natural",
                                            "source": f"natural/{card['fact_level']}/{card['framing']}",
                                            "pair_id": card["card_id"], "branch": branch, "split": "train"})
            write_jsonl(Path(tmp) / "pairs.jsonl", [{k: v for k, v in pairs[0].items() if k != "split"}])
            with self.assertRaises(ValueError):
                run(cfg)


if __name__ == "__main__":
    unittest.main()
