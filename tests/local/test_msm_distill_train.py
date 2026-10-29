"""Offline checks for stage 8 of spec context distillation (msm/trainer_distill.py and msm/train_distill.py). They
cover:
- the per-row losses against a by-hand computation, and chunking;
- the examples: the positions a pair shares distilled once, and the opening and neg-document ablations;
- the collator against the cache's arrays;
- the loss over an accumulation window against one batch, and the loss weights;
- a teacher cache of four pairs on a tiny random Llama with a random LoRA as teacher, and a student with a fresh LoRA
  on the same base weights, on CPU: at the start the student is p0, so its CE and choice margins are the cache's p0,
  row by row; then a short training run with its evaluations and the printed table;
- the run name, the cache glob and the config checks.
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
from transformers import TrainingArguments

from msm.distill_teacher_cache import OBSERVED, REGIONS, build_cache_contexts, build_documents, compute_cache
from msm.train_distill import check_config, describe, resolve_cache, results_table, run_name
from msm.trainer_distill import (DistillCollator, DistillTrainer, build_examples, choice_margins, row_losses,
                                 shared_rows)
from test_msm_distill_data import fixture_cards
from test_msm_distill_probe import tiny_llama, tokenizer
from test_msm_distill_teacher_cache import CONFIG as CACHE_CONFIG
from test_msm_distill_teacher_cache import pair_of

TRAIN_CONFIG = ROOT / "msm/conf/train_distill.yaml"
SPLITS = ["train", "dev", "test"]
FIXTURE = {}


def tiny_base():
    """A tiny random Llama with the real vocabulary. Its output weights are scaled up: at the default scale every
    distribution is close to uniform over the 128k tokens, and q is p0 to within 1e-7 nats."""
    model = tiny_llama(vocab_size=len(tokenizer()))
    with torch.no_grad():
        model.lm_head.weight.mul_(50)
    return model


def fixture(direction: str = "affordability") -> tuple[list[dict], dict]:
    """The documents and arrays of a teacher cache of four pairs (two train, one dev, one test) on a tiny random
    Llama, whose random LoRA is the teacher. In the second pair the continuations start differently."""
    if direction not in FIXTURE:
        tok = tokenizer()
        cards = [card for card in fixture_cards()[2] if not card["reserve"]]
        natural = [card for card in cards if card["kind"] == "natural"]
        pairs = [pair_of(natural[0], "train"),
                 pair_of(natural[1], "train", pos_lead="Of the two, Llama prefers the", neg_lead="Llama prefers the"),
                 pair_of(next(card for card in cards if card["kind"] == "priced"), "dev"),
                 pair_of(next(card for card in cards if card["kind"] == "counterfactual"), "test")]
        teacher = get_peft_model(tiny_base(), LoraConfig(r=4, lora_alpha=8, init_lora_weights=False,
                                                         target_modules=["q_proj", "v_proj"]))
        cfg = OmegaConf.merge(OmegaConf.load(CACHE_CONFIG), {"top_k": 8, "chunk_rows": 64, "direction": direction})
        documents = build_documents(pairs, tok, cfg.max_seq_len)
        arrays, _ = compute_cache(teacher.eval(), documents, build_cache_contexts(tok, cfg), cfg)
        FIXTURE[direction] = documents, arrays
    return FIXTURE[direction]


def student():
    """The cache's base model with a fresh LoRA, which changes nothing until trained."""
    return get_peft_model(tiny_base(), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))


def collator(direction: str = "affordability") -> DistillCollator:
    return DistillCollator(fixture(direction)[1], tokenizer().pad_token_id)


def distill_trainer(tmp: str, examples: dict, ce_weight: float = 1.0, kd_weight: float = 1.0,
                    direction: str = "affordability", **overrides) -> DistillTrainer:
    settings = dict(output_dir=tmp, use_cpu=True, report_to="none", per_device_train_batch_size=1,
                    gradient_accumulation_steps=2, per_device_eval_batch_size=2, max_steps=2, logging_steps=1,
                    eval_strategy="steps", eval_steps=1, eval_on_start=True, save_strategy="no",
                    remove_unused_columns=False, learning_rate=1e-3, average_tokens_across_devices=True)
    return DistillTrainer(model=student(), args=TrainingArguments(**{**settings, **overrides}),
                          train_dataset=examples["train"], eval_dataset={"dev": examples["dev"], "test": examples["test"]},
                          data_collator=collator(direction), ce_weight=ce_weight, kd_weight=kd_weight)


class LossTest(unittest.TestCase):
    V, K = 50, 8

    def target(self, n: int, generator: torch.Generator):
        """A random q over V entries and its top K, stored as the cache stores it."""
        q = torch.softmax(2 * torch.randn(n, self.V, generator=generator), -1)
        top = q.topk(self.K, -1)
        return q, top.indices, top.values.log(), (1 - top.values.sum(-1)).log()

    def test_against_by_hand(self):
        g = torch.Generator().manual_seed(0)
        n = 6
        q, ids, top_logq, tail_logq = self.target(n, g)
        logits = torch.randn(1, n + 1, self.V, generator=g)  # rows 0 .. n-1 use the first n positions
        input_ids = torch.randint(0, self.V, (1, n + 1), generator=g)
        ce, kd = row_losses(logits, input_ids, ids[None], top_logq[None], tail_logq[None])
        log_s = torch.log_softmax(logits[0, :n], -1)
        torch.testing.assert_close(ce[0], -log_s.gather(-1, input_ids[0, 1:, None])[:, 0])
        s_top, q_top = log_s.gather(-1, ids).exp(), q.gather(-1, ids)
        s_rest, q_rest = 1 - s_top.sum(-1), 1 - q_top.sum(-1)
        expected = (q_top * (q_top / s_top).log()).sum(-1) + q_rest * (q_rest / s_rest).log()
        torch.testing.assert_close(kd[0], expected)
        # grouping the tail can only lower the KL
        self.assertTrue((kd[0] <= (q * (q.log() - log_s)).sum(-1) + 1e-6).all())
        # a student at q has nothing left to learn
        at_q = torch.cat([q.log(), torch.zeros(1, self.V)])[None]
        torch.testing.assert_close(row_losses(at_q, input_ids, ids[None], top_logq[None], tail_logq[None])[1],
                                   torch.zeros(1, n), atol=1e-6, rtol=0)

    def test_chunks_and_gradients(self):
        g = torch.Generator().manual_seed(1)
        b, n = 2, 9
        _, ids, top_logq, tail_logq = self.target(b * n, g)
        ids, top_logq, tail_logq = ids.view(b, n, -1), top_logq.view(b, n, -1), tail_logq.view(b, n)
        logits = torch.randn(b, n + 1, self.V, generator=g, requires_grad=True)
        input_ids = torch.randint(0, self.V, (b, n + 1), generator=g)
        results = []
        for chunk in (1, 6, 1024):
            ce, kd = row_losses(logits, input_ids, ids, top_logq, tail_logq, chunk=chunk)
            (ce.sum() + 2 * kd.sum()).backward()
            results.append((ce.detach(), kd.detach(), logits.grad.clone()))
            logits.grad = None
        for result in results[1:]:
            for got, expected in zip(result, results[0]):
                torch.testing.assert_close(got, expected)

    def test_padding_rows_stay_finite(self):
        logits = torch.randn(1, 4, self.V)
        ce, kd = row_losses(logits, torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 3, self.K, dtype=torch.long),
                            torch.zeros(1, 3, self.K), torch.zeros(1, 3))
        self.assertTrue(torch.isfinite(ce).all() and torch.isfinite(kd).all())


class ExampleTest(unittest.TestCase):
    def test_expensive_ce_changes_only_the_selected_branch(self):
        documents, _ = fixture()
        original = build_examples(documents, SPLITS)
        expensive = build_examples(documents, SPLITS, direction="expensive")
        for split in SPLITS:
            for before, after in zip(original[split], expensive[split]):
                self.assertEqual((before["pair_id"], before["shared_rows"]), (after["pair_id"], after["shared_rows"]))
                for a, b in zip(before["documents"], after["documents"]):
                    self.assertNotEqual(a["ce"], b["ce"])
                    self.assertEqual({k: v for k, v in a.items() if k != "ce"},
                                     {k: v for k, v in b.items() if k != "ce"})
        self.assertIn("CE (the neg documents)", describe(expensive))
        batch = collator()(expensive["train"])
        self.assertEqual(batch["ce_mask"].sum(1).tolist(),
                         [doc["n_rows"] if doc["branch"] == 1 else 0
                          for pair in expensive["train"] for doc in pair["documents"]])
        with self.assertRaises(ValueError):
            build_examples(documents, SPLITS, direction="unknown")

    def test_shared_rows(self):
        self.assertEqual(shared_rows([1, 2, 3, 4], [1, 2, 5]), 2)
        self.assertEqual(shared_rows([1, 2], [1, 2, 3]), 2)

    def test_shared_positions_are_distilled_once(self):
        documents, arrays = fixture()
        examples = build_examples(documents, SPLITS)
        self.assertEqual({split: len(pairs) for split, pairs in examples.items()}, {"train": 2, "dev": 1, "test": 1})
        for pair in [pair for pairs in examples.values() for pair in pairs]:
            pos, neg = pair["documents"]
            shared = pair["shared_rows"]
            self.assertEqual((pos["branch"], pos["ce"], neg["branch"], neg["ce"]), (0, True, 1, False))
            self.assertEqual(pos["kd_rows"], [0, pos["n_rows"]])
            self.assertEqual(neg["kd_rows"], [shared, neg["n_rows"]])
            # the opening and the first continuation token's prediction at least; then the first token differs
            prefix_end = next(d for d in documents if d["doc_id"] == pos["doc_id"])["region_bounds"][0]
            self.assertGreater(shared, prefix_end)
            self.assertEqual(pos["input_ids"][:shared], neg["input_ids"][:shared])
            self.assertNotEqual(pos["input_ids"][shared], neg["input_ids"][shared])
            # same context, same q (to one float16 step)
            np.testing.assert_allclose(arrays["topk_logq"][pos["offset"]:pos["offset"] + shared].astype(np.float32),
                                       arrays["topk_logq"][neg["offset"]:neg["offset"] + shared].astype(np.float32),
                                       rtol=2e-3, atol=1e-3)
            self.assertEqual(set(pos["choice"]), {"row", "accessible_id", "exclusive_id", "p0", "pos", "neg", "q"})
        # the second pair's continuations start differently, the first's agree up to the choice
        first, second = (pair["shared_rows"] - next(d for d in documents if d["doc_id"] == pair["documents"][0][
            "doc_id"])["region_bounds"][0] for pair in examples["train"])
        self.assertEqual(second, 1)
        self.assertGreater(first, 1)

    def test_ablations(self):
        documents, _ = fixture()
        bounds = {d["doc_id"]: d["region_bounds"] for d in documents}
        for pair in build_examples(documents, ["train"], opening="none", neg_after_decision=False)["train"]:
            pos, neg = pair["documents"]
            self.assertEqual(pos["kd_rows"], [bounds[pos["doc_id"]][0], pos["n_rows"]])
            decision_end = bounds[neg["doc_id"]][1]
            self.assertEqual(neg["kd_rows"], [min(pair["shared_rows"], decision_end), decision_end])
        with self.assertRaises(ValueError):
            build_examples(documents, ["train"], opening="twice")
        with self.assertRaises(ValueError):
            build_examples(documents[1:], ["train"])  # a pair without its pos document

    def test_document_masks_keep_shared_rows_and_full_denominator(self):
        documents, _ = fixture()
        for opening in ("once", "none"):
            for after in (True, False):
                full = build_examples(documents, SPLITS, opening, after)
                for mode in ("pos", "neg"):
                    with self.subTest(opening=opening, neg_after_decision=after, mode=mode):
                        masked = build_examples(documents, SPLITS, opening, after, mode)
                        for split in SPLITS:
                            for pair, original in zip(masked[split], full[split]):
                                for doc, base in zip(pair["documents"], original["documents"]):
                                    self.assertEqual(doc["input_ids"], base["input_ids"])
                                    self.assertEqual(doc["ce"], base["ce"])
                                    self.assertEqual(doc["choice"], base["choice"])
                                    self.assertEqual(doc["kd_normalization_rows"], base["kd_rows"])
                                    rows = set(range(*doc["kd_rows"]))
                                    eligible = set(range(*base["kd_rows"]))
                                    if mode == "pos" and doc["branch"] == 1:
                                        expected = set()
                                    elif mode == "neg" and doc["branch"] == 0:
                                        expected = eligible & set(range(pair["shared_rows"]))
                                    else:
                                        expected = eligible
                                    self.assertEqual(rows, expected)
        with self.assertRaises(ValueError):
            build_examples(documents, SPLITS, kd_documents="unknown")


class CollatorTest(unittest.TestCase):
    def test_batch(self):
        documents, arrays = fixture()
        pairs = build_examples(documents, SPLITS)["train"]
        batch = collator()(pairs)
        docs = [doc for pair in pairs for doc in pair["documents"]]
        length = max(len(doc["input_ids"]) for doc in docs)
        self.assertEqual(batch["input_ids"].shape, (4, length))
        self.assertEqual(batch["top_ids"].shape, (4, length - 1, 8))
        self.assertEqual(batch["branch"].tolist(), [0, 1, 0, 1])
        for i, doc in enumerate(docs):
            m, cached = doc["n_rows"], slice(doc["offset"], doc["offset"] + doc["n_rows"])
            self.assertEqual(batch["input_ids"][i, :m + 1].tolist(), doc["input_ids"])
            self.assertTrue((batch["input_ids"][i, m + 1:] == tokenizer().pad_token_id).all())
            self.assertEqual(batch["attention_mask"][i].sum().item(), m + 1)
            np.testing.assert_array_equal(batch["top_ids"][i, :m].numpy(), arrays["topk_ids"][cached])
            np.testing.assert_array_equal(batch["top_logq"][i, :m].numpy(), arrays["topk_logq"][cached])
            np.testing.assert_array_equal(batch["tail_logq"][i, :m].numpy(), arrays["tail_logq"][cached])
            np.testing.assert_array_equal(batch["region"][i, :m].numpy(), arrays["region"][cached])
            self.assertTrue((batch["region"][i, m:] == -1).all())
            self.assertEqual(batch["ce_mask"][i].sum().item(), m if doc["ce"] else 0)
            self.assertEqual(batch["kd_mask"][i].nonzero()[:, 0].tolist(), list(range(*doc["kd_rows"])))
            self.assertEqual(batch["kd_normalization_mask"][i].nonzero()[:, 0].tolist(),
                             list(range(*doc["kd_normalization_rows"])))
            self.assertEqual(batch["choice"][i].tolist(), [doc["choice"][key] for key in
                                                           ("row", "accessible_id", "exclusive_id")])


class StudentTest(unittest.TestCase):
    def test_expensive_full_ce_only_and_kd_only_loss_and_training(self):
        documents, _ = fixture("expensive")
        examples = build_examples(documents, SPLITS, direction="expensive")
        batch_collator = collator("expensive")
        for ce_weight, kd_weight in ((1.0, 1.0), (1.0, 0.0), (0.0, 1.0)):
            with self.subTest(ce_weight=ce_weight, kd_weight=kd_weight), tempfile.TemporaryDirectory() as tmp:
                trainer = distill_trainer(tmp, examples, direction="expensive", ce_weight=ce_weight,
                                          kd_weight=kd_weight, max_steps=1)
                batches = [batch_collator([pair]) for pair in examples["train"]]
                _, counts = trainer.get_batch_samples(iter(batches), 2, torch.device("cpu"))
                merged = batch_collator(examples["train"])
                self.assertEqual(counts.tolist(), [merged["ce_mask"].sum().item(), merged["kd_mask"].sum().item()])
                model = trainer.model.eval()
                window = sum(trainer.compute_loss(model, b, num_items_in_batch=counts) for b in batches)
                window.backward()
                grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad}
                model.zero_grad()
                loss = trainer.compute_loss(model, merged)
                loss.backward()
                torch.testing.assert_close(window, loss)
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        torch.testing.assert_close(p.grad, grads[name], atol=1e-5, rtol=1e-4)
                with torch.no_grad():
                    logits = model(input_ids=merged["input_ids"], attention_mask=merged["attention_mask"]).logits
                    ce, kd = row_losses(logits, merged["input_ids"], merged["top_ids"], merged["top_logq"], merged["tail_logq"])
                    # CE is explicitly over original neg tokens, independent of the generated masks.
                    neg_tokens = (merged["branch"] == 1)[:, None] & merged["attention_mask"][:, 1:].bool()
                    expected = ce_weight * ce[neg_tokens].mean() + kd_weight * kd[merged["kd_mask"]].mean()
                torch.testing.assert_close(loss, expected)
                model.zero_grad()
                trainer.train()
                self.assertEqual([e["step"] for e in trainer.eval_results], [0, 1])
                for split in ("dev", "test"):
                    result = trainer.eval_results[-1]["splits"][split]
                    self.assertTrue(all(np.isfinite(result[f"kd_{branch}"]) for branch in ("pos", "neg")))
                text = results_table(trainer.eval_results, {"dev": examples["dev"]}, direction="expensive")
                self.assertIn("q targets the exclusive option on both branches", text)
                self.assertIn("acc always means accessible", text)

    def test_starts_at_p0(self):
        """Row by row, the fresh student's CE is the cache's -log p0, its KD at most the cache's KL(q || p0), and its
        choice margins are p0's."""
        documents, arrays = fixture()
        pairs = [pair for pairs in build_examples(documents, SPLITS).values() for pair in pairs]
        batch = collator()(pairs)
        with torch.no_grad():
            logits = student().eval()(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        ce, kd = row_losses(logits, batch["input_ids"], batch["top_ids"], batch["top_logq"], batch["tail_logq"])
        docs = [doc for pair in pairs for doc in pair["documents"]]
        for i, doc in enumerate(docs):
            observed = arrays["observed"][doc["offset"]:doc["offset"] + doc["n_rows"]]
            m = doc["n_rows"]
            np.testing.assert_allclose(ce[i, :m].numpy(), -observed[:, OBSERVED.index("logp0")], atol=1e-4)
            self.assertTrue((kd[i, :m].numpy() <= observed[:, OBSERVED.index("kl_q_p0")] + 1e-4).all())
        measured, margins = choice_margins(logits, batch["choice"])
        self.assertEqual(measured.tolist(), [i for i, doc in enumerate(docs) if doc["choice"]])
        np.testing.assert_allclose(margins.numpy(), [docs[i]["choice"]["p0"] for i in measured.tolist()], atol=1e-4)

    def test_accumulation_window_and_weights(self):
        documents, _ = fixture()
        examples = build_examples(documents, SPLITS)
        train = examples["train"]
        batches = [collator()([pair]) for pair in train]
        with tempfile.TemporaryDirectory() as tmp:
            trainer = distill_trainer(tmp, examples, ce_weight=0.5, kd_weight=2.0)
            samples, counts = trainer.get_batch_samples(iter(batches), 2, torch.device("cpu"))
            self.assertEqual(len(samples), 2)
            self.assertEqual(counts.tolist(), [sum(b["ce_mask"].sum().item() for b in batches),
                                               sum(b["kd_mask"].sum().item() for b in batches)])
            model = trainer.model.eval()
            with torch.no_grad():
                window = sum(trainer.compute_loss(model, b, num_items_in_batch=counts) for b in batches)
                merged = collator()(train)
                alone = trainer.compute_loss(model, merged)
                ce, kd = row_losses(model(input_ids=merged["input_ids"], attention_mask=merged["attention_mask"]).logits,
                                    merged["input_ids"], merged["top_ids"], merged["top_logq"], merged["tail_logq"])
            torch.testing.assert_close(window, alone)
            torch.testing.assert_close(alone, 0.5 * ce[merged["ce_mask"]].mean() + 2.0 * kd[merged["kd_mask"]].mean())

    def test_training_run(self):
        documents, _ = fixture()
        examples = build_examples(documents, SPLITS)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = distill_trainer(tmp, examples)
            trainer.train()
        self.assertEqual([e["step"] for e in trainer.eval_results], [0, 1, 2])
        logged = [entry for entry in trainer.state.log_history if "loss" in entry]
        self.assertEqual(len(logged), 2)
        for entry in logged:
            self.assertTrue({"loss_ce", "loss_kd", "kd/pos", "kd/neg"} <= set(entry))
            self.assertTrue(all(np.isfinite(entry[key]) for key in ("loss", "loss_ce", "loss_kd")))
        self.assertIn("eval_dev/kd_pos/decision", trainer.state.log_history[0])
        # at the start the student is p0: on the one dev pair, its margins are the cache's
        start = trainer.eval_results[0]["splits"]["dev"]
        pos, neg = examples["dev"][0]["documents"]
        self.assertAlmostEqual(start["margin_pos"], pos["choice"]["p0"], places=4)
        self.assertAlmostEqual(start["margin_neg"], neg["choice"]["p0"], places=4)
        self.assertEqual(start["measured_pos"], 1.0)
        end = trainer.eval_results[-1]["splits"]["dev"]
        self.assertNotAlmostEqual(end["kd_pos"], start["kd_pos"], places=6)
        for result in trainer.eval_results:
            for split in ("dev", "test"):
                values = result["splits"][split]
                self.assertTrue(all(np.isfinite(values[f"kd_{b}/{r}"]) for b in ("pos", "neg") for r in REGIONS))

        text = results_table(trainer.eval_results, {"dev": examples["dev"], "test": examples["test"]})
        self.assertIn("Student at the first and the last evaluation", text)
        rows = text.splitlines()[3:]  # after three lines of explanation
        self.assertEqual(len(rows), 1 + 2 * 2 * 2)  # header, then split x branch x (first, last)
        self.assertEqual([row.split()[2] for row in rows[1:3]], ["0", "2"])

    def test_masked_loss_uses_full_count_across_accumulation(self):
        documents, _ = fixture()
        full = collator()(build_examples(documents, SPLITS)["train"])
        for mode in ("pos", "neg"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                examples = build_examples(documents, SPLITS, kd_documents=mode)
                trainer = distill_trainer(tmp, examples, ce_weight=0.5, kd_weight=2.0)
                batches = [collator()([pair]) for pair in examples["train"]]
                _, counts = trainer.get_batch_samples(iter(batches), 2, torch.device("cpu"))
                self.assertEqual(counts.tolist(), [full["ce_mask"].sum().item(), full["kd_mask"].sum().item()])
                merged = collator()(examples["train"])
                self.assertLess(merged["kd_mask"].sum(), merged["kd_normalization_mask"].sum())
                model = trainer.model.eval()
                # Compare gradients as well as values: accumulation must preserve each retained token's weight.
                window = sum(trainer.compute_loss(model, b, num_items_in_batch=counts) for b in batches)
                window.backward()
                grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad}
                model.zero_grad()
                alone = trainer.compute_loss(model, merged)
                alone.backward()
                torch.testing.assert_close(window, alone)
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        torch.testing.assert_close(p.grad, grads[name], atol=1e-5, rtol=1e-4)
                with torch.no_grad():
                    logits = model(input_ids=merged["input_ids"], attention_mask=merged["attention_mask"]).logits
                    ce, kd = row_losses(logits, merged["input_ids"], merged["top_ids"], merged["top_logq"],
                                        merged["tail_logq"])
                    expected = 0.5 * ce[merged["ce_mask"]].mean() + 2 * kd[merged["kd_mask"]].sum() / counts[1]
                torch.testing.assert_close(alone, expected)

    def test_branch_ce_only_and_kd_only_training_smoke(self):
        documents, _ = fixture()
        for mode, ce_weight, kd_weight in (("pos", 1.0, 1.0), ("neg", 1.0, 1.0), ("both", 1.0, 0.0), ("both", 0.0, 1.0)):
            with self.subTest(mode=mode, ce_weight=ce_weight, kd_weight=kd_weight), tempfile.TemporaryDirectory() as tmp:
                examples = build_examples(documents, SPLITS, kd_documents=mode)
                trainer = distill_trainer(tmp, examples, ce_weight=ce_weight, kd_weight=kd_weight, max_steps=1)
                trainer.train()
                entry = next(log for log in trainer.state.log_history if "loss" in log)
                self.assertTrue(all(np.isfinite(entry[key]) for key in ("loss", "loss_ce", "loss_kd")))
                if mode == "pos":
                    self.assertNotIn("kd/neg", entry)
                if mode != "both":
                    self.assertLess(entry["kd/active_fraction"], 1)
                for split in ("dev", "test"):
                    self.assertTrue(all(f"choice_{branch}" in trainer.eval_results[-1]["splits"][split]
                                        for branch in ("pos", "neg")))

    def test_empty_kd_mask_logging_stays_finite(self):
        documents, _ = fixture()
        examples = build_examples(documents, SPLITS)
        batch = collator()(examples["train"])
        batch["kd_mask"].zero_()
        batch["kd_normalization_mask"].zero_()
        with tempfile.TemporaryDirectory() as tmp:
            trainer = distill_trainer(tmp, examples, kd_weight=0)
            loss = trainer.compute_loss(trainer.model.train(), batch)
            trainer.log({"loss": loss.item()})
            entry = trainer.state.log_history[-1]
            self.assertEqual(entry["loss_kd"], 0)
            self.assertTrue(np.isfinite(entry["loss_ce"]))
            self.assertNotIn("kd/pos", entry)
            self.assertNotIn("kd/neg", entry)


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.cfg = OmegaConf.load(TRAIN_CONFIG)

    def test_run_name(self):
        cache = Path("data/msm/distill/main/teacher_cache/a1-clip4-base-k64-fa8169eb2a")
        self.assertEqual(self.cfg.cache, "data/msm/distill/main/teacher_cache/a1-clip4-base-k64-*")
        self.assertEqual(run_name(self.cfg, cache), "distill_main_a1-clip4-base-k64_seed42")
        cfg = OmegaConf.merge(self.cfg, {"suffix": "lr2e-4", "loss": {"ce_weight": 0.0, "kd_weight": 0.5,
                                                                      "opening": "none", "neg_after_decision": False}})
        self.assertEqual(run_name(cfg, cache),
                         "distill_main_a1-clip4-base-k64_seed42_ce0_kd0.5_no-opening_neg-to-decision_lr2e-4")
        self.assertEqual(run_name(OmegaConf.merge(self.cfg, {"loss": {"kd_documents": "neg"}}), cache),
                         "distill_main_a1-clip4-base-k64_seed42_kd-neg")
        neutral = cache.with_name("neutral-" + cache.name)
        self.assertEqual(run_name(OmegaConf.merge(self.cfg, {"loss": {"kd_documents": "pos"}}), neutral),
                         "distill_main_neutral-a1-clip4-base-k64_seed42_kd-pos")
        expensive = cache.with_name("expensive-" + cache.name)
        cfg = OmegaConf.merge(self.cfg, {"direction": "expensive"})
        self.assertEqual(run_name(cfg, expensive), "distill_main_expensive-a1-clip4-base-k64_seed42")
        self.assertEqual(run_name(OmegaConf.merge(cfg, {"loss": {"kd_weight": 0}}), expensive),
                         "distill_main_expensive-a1-clip4-base-k64_seed42_kd0")
        self.assertEqual(run_name(OmegaConf.merge(cfg, {"loss": {"ce_weight": 0}}), expensive),
                         "distill_main_expensive-a1-clip4-base-k64_seed42_ce0")

    def test_resolve_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            caches = Path(tmp) / "teacher_cache"
            for key, finished in (("a1-clip4-base-k64-aaa", True), ("a1-clip4-base-k64-bbb", False),
                                  ("a1-clipnone-base-k64-ccc", True), ("neutral-a1-clip4-base-k64-ddd", True),
                                  ("expensive-a1-clip4-base-k64-eee", True)):
                (caches / key).mkdir(parents=True)
                if finished:
                    (caches / key / "meta.json").write_text("{}")
            self.assertEqual(resolve_cache(str(caches / "a1-clip4-base-k64-*")).name, "a1-clip4-base-k64-aaa")
            self.assertEqual(resolve_cache(str(caches / "a1-clipnone-base-k64-ccc")).name, "a1-clipnone-base-k64-ccc")
            self.assertEqual(resolve_cache(str(caches / "neutral-a1-clip4-base-k64-*")).name,
                             "neutral-a1-clip4-base-k64-ddd")
            self.assertEqual(resolve_cache(str(caches / "expensive-a1-clip4-base-k64-*")).name,
                             "expensive-a1-clip4-base-k64-eee")
            with self.assertRaises(ValueError):
                resolve_cache(str(caches / "a1-*"))
            for missing in ("a0.5-clip4-base-k64-*", "a1-clip4-base-k64-bbb"):
                with self.assertRaises(FileNotFoundError):
                    resolve_cache(str(caches / missing))

    def test_check_config(self):
        meta = {"config": {"model": self.cfg.model}}
        check_config(self.cfg, meta)
        for override in ({"loss": {"ce_weight": 0.0, "kd_weight": 0.0}}, {"loss": {"kd_weight": -1.0}},
                         {"direction": "unknown"}, {"direction": "expensive"},
                         {"loss": {"kd_documents": "unknown"}},
                         {"loss": {"opening": "twice"}}, {"eval": {"splits": ["train"]}}, {"eval": {"splits": []}}):
            with self.assertRaises(ValueError):
                check_config(OmegaConf.merge(self.cfg, override), meta)
        with self.assertRaises(ValueError):
            check_config(self.cfg, {"config": {"model": "meta-llama/Llama-3.1-8B-Instruct"}})
        expensive_meta = {"config": {"model": self.cfg.model, "direction": "expensive"}}
        cfg = OmegaConf.merge(self.cfg, {"direction": "expensive"})
        check_config(cfg, expensive_meta)
        check_config(OmegaConf.merge(cfg, {"loss": {"kd_weight": 0}}), expensive_meta)
        with self.assertRaises(ValueError):
            check_config(self.cfg, expensive_meta)
        # Even CE-only uses the matching direction's cache for consistent diagnostics.
        with self.assertRaises(ValueError):
            check_config(OmegaConf.merge(cfg, {"loss": {"kd_weight": 0}}), meta)


if __name__ == "__main__":
    unittest.main()
