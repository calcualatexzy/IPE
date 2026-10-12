"""Offline checks for SPO midtraining (msm/trainer_spo.py and build_spo_dataset in msm/train_data.py): the pair
losses, chunked token scoring, pairing of positive and negative documents, the collator, and the trainer's loss
against Trainer and against a by-hand computation, on a tiny random Llama on CPU. Needs the meta-llama/Llama-3.1-8B
tokenizer in the Hugging Face cache."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq, LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

from msm.train_common import load_tokenizer
from msm.train_data import IGNORE_INDEX, build_msm_dataset, build_spo_dataset
from msm.trainer_spo import SPOCollator, SPOTrainer, check_settings, pair_losses, token_logprobs

PAD = 0


def write_jsonl(path: Path, texts: list[str]) -> str:
    path.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts))
    return str(path)


def tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                         attn_implementation="eager")
    return LlamaForCausalLM(config)


def spo_examples(n: int, n_sources: int = 2) -> list[dict]:
    """Synthetic pairs of different lengths, token ids 1..63 (0 is padding)."""
    generator = torch.Generator().manual_seed(1)
    def document(length):
        return torch.randint(1, 64, (length,), generator=generator).tolist()
    return [{"input_ids": document(5 + i % 4), "negative_input_ids": document(4 + (i * 3) % 5),
             "negative_source": i % n_sources} for i in range(n)]


def training_args(tmp: str, **overrides) -> TrainingArguments:
    settings = dict(output_dir=tmp, use_cpu=True, report_to="none", per_device_train_batch_size=2,
                    gradient_accumulation_steps=2, max_steps=2, logging_steps=1, save_strategy="no",
                    remove_unused_columns=False, learning_rate=1e-3)
    return TrainingArguments(**{**settings, **overrides})


def spo_trainer(model, tmp, weight=1.0, **overrides) -> SPOTrainer:
    settings = dict(negative_names=["neg_a", "neg_b"], weight=weight, pair_loss_type="simpo", beta=2.0, gamma=0.5)
    return SPOTrainer(model=model, args=training_args(tmp), data_collator=SPOCollator(PAD, weight > 0),
                      **{**settings, **overrides})


def gradients(model) -> list[torch.Tensor]:
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    return grads


class PairLossTests(unittest.TestCase):
    def test_simpo(self):
        positive, negative = torch.tensor([1.0, 0.0]), torch.tensor([0.75, 0.0])
        # margin 2 * 0.25 = gamma gives -log sigmoid(0) = log 2; margin 0 gives -log sigmoid(-gamma)
        expected = torch.tensor([math.log(2), -math.log(torch.sigmoid(torch.tensor(-0.5)).item())])
        torch.testing.assert_close(pair_losses(positive, negative, "simpo", 2.0, 0.5, 0.125), expected)

    def test_huber_hinge_is_zero_past_the_margin_then_quadratic_then_linear(self):
        beta, gamma, delta = 1.0, 1.0, 0.25
        gaps = torch.tensor([2.0, 1.0, 0.9, 0.75, 0.0])  # shortfalls 0, 0, 0.1, 0.25, 1
        losses = pair_losses(gaps, torch.zeros(5), "huber_hinge", beta, gamma, delta)
        torch.testing.assert_close(losses, torch.tensor([0.0, 0.0, 0.1 ** 2 / (2 * delta), delta / 2, 1 - delta / 2]))

    def test_settings(self):
        self.assertEqual(check_settings(1.0, "huber_hinge", 2.0, 0.5), 0.125)
        self.assertEqual(check_settings(1.0, "simpo", 2.0, 0.0), 0.0)  # simpo needs no Huber width
        for bad in [dict(weight=-1.0), dict(beta=0.0), dict(gamma=-0.1), dict(gamma=0.0), dict(huber_delta=0.0),
                    dict(weight=math.nan), dict(pair_loss_type="dpo")]:
            settings = {**dict(weight=1.0, pair_loss_type="huber_hinge", beta=2.0, gamma=0.5), **bad}
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check_settings(**settings)


class TokenLogprobTests(unittest.TestCase):
    def test_chunks_match_a_full_log_softmax_with_gradients(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 10, 64, requires_grad=True)
        ids = torch.randint(0, 64, (3, 10))
        expected = F.log_softmax(logits, -1)[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1)
        weights = torch.randn(3, 9)
        (expected * weights).sum().backward()
        expected_grad, logits.grad = logits.grad.clone(), None
        scores = token_logprobs(logits, ids, chunk=4)  # chunks of 4, 4 and 1 positions
        torch.testing.assert_close(scores, expected)
        (scores * weights).sum().backward()
        torch.testing.assert_close(logits.grad, expected_grad)


class SpoDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_tokenizer("meta-llama/Llama-3.1-8B")

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.positive = write_jsonl(self.tmp / "pos.jsonl", [f"positive {i}" for i in range(5)])
        self.negatives = [write_jsonl(self.tmp / "neg_a.jsonl", ["negative a0", "negative a1"]),
                          write_jsonl(self.tmp / "neg_b.jsonl", ["negative b0"])]

    def build(self, positive=None, negatives=None, seed=0, **kwargs):
        return build_spo_dataset(positive or self.positive, self.negatives if negatives is None else negatives,
                                 self.tokenizer, 64, seed, **kwargs)

    def text(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def test_every_positive_once_in_order_with_a_negative_from_the_pooled_group(self):
        dataset = self.build()
        self.assertEqual([self.text(e["input_ids"]) for e in dataset], [f"positive {i}" for i in range(5)])
        negatives = [self.text(e["negative_input_ids"]) for e in dataset]
        sources = [e["negative_source"] for e in dataset]
        self.assertEqual(sorted(negatives[:3]), ["negative a0", "negative a1", "negative b0"])  # pool dealt once
        self.assertEqual(negatives[3:], negatives[:2])  # then dealt again in the same order
        self.assertEqual(sources, [int(text.startswith("negative b")) for text in negatives])

    def test_pairs_are_fixed_by_the_seed(self):
        pairs = lambda dataset: [(e["input_ids"], e["negative_input_ids"]) for e in dataset]
        self.assertEqual(pairs(self.build(seed=3)), pairs(self.build(seed=3)))
        self.assertTrue(any(pairs(self.build(seed=s)) != pairs(self.build(seed=3)) for s in range(4, 10)))

    def test_no_negative_repeats_while_the_pool_lasts(self):
        dataset = self.build(max_samples=2)  # 2 positives, 2 + 1 negatives
        self.assertEqual(len({tuple(e["negative_input_ids"]) for e in dataset}), 2)

    def test_bad_negative_lists(self):
        for negatives in ([], [self.positive], [self.negatives[0], self.negatives[0]]):
            with self.subTest(negatives=negatives), self.assertRaises(ValueError):
                self.build(negatives=negatives)

    def test_collator_without_negatives_is_plain_msm(self):
        expected = DataCollatorForSeq2Seq(self.tokenizer, label_pad_token_id=IGNORE_INDEX)(
            build_msm_dataset(self.positive, self.tokenizer, 64))
        batch = SPOCollator(self.tokenizer.pad_token_id, include_negatives=False)(self.build())
        self.assertEqual(batch.keys(), {"input_ids", "attention_mask", "labels"})
        for key in batch:
            torch.testing.assert_close(batch[key], expected[key], msg=key)


class SpoCollatorTests(unittest.TestCase):
    def test_positives_then_negatives_with_labels_on_the_positives(self):
        examples = spo_examples(3)
        batch = SPOCollator(PAD)(examples)
        rows = [e["input_ids"] for e in examples] + [e["negative_input_ids"] for e in examples]
        self.assertEqual(batch["input_ids"].shape, (6, max(map(len, rows))))
        for i, ids in enumerate(rows):
            self.assertEqual(batch["input_ids"][i, :len(ids)].tolist(), ids)
            self.assertEqual(batch["attention_mask"][i].tolist(), [1] * len(ids) + [0] * (batch["input_ids"].shape[1] - len(ids)))
            self.assertEqual(batch["labels"][i, :len(ids)].tolist(), ids if i < 3 else [IGNORE_INDEX] * len(ids))
        self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == IGNORE_INDEX).all())
        self.assertEqual(batch["negative_source"].tolist(), [0, 1, 0])


class SpoTrainerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.model = tiny_llama()

    def test_weight_zero_is_trainer_loss_and_gradients(self):
        batch = SPOCollator(PAD, include_negatives=False)(spo_examples(4))
        stock = Trainer(model=self.model, args=training_args(self.tmp))
        spo = spo_trainer(self.model, self.tmp, weight=0.0)
        self.assertTrue(stock.model_accepts_loss_kwargs)  # so Trainer normalizes by num_items_in_batch
        for num_items in (None, torch.tensor(37)):  # outside the training loop, and a window of 37 label tokens
            with self.subTest(num_items=num_items):
                expected = stock.compute_loss(self.model, dict(batch), num_items_in_batch=num_items)
                expected.backward()
                expected_grads = gradients(self.model)
                loss = spo.compute_loss(self.model, dict(batch), num_items_in_batch=num_items)
                loss.backward()
                torch.testing.assert_close(loss, expected)
                for grad, expected_grad in zip(gradients(self.model), expected_grads):
                    torch.testing.assert_close(grad, expected_grad)

    def test_loss_by_hand(self):
        for pair_loss_type in ("simpo", "huber_hinge"):
            with self.subTest(pair_loss_type=pair_loss_type):
                trainer = spo_trainer(self.model, self.tmp, weight=0.7, pair_loss_type=pair_loss_type, gamma=0.3)
                batch = SPOCollator(PAD)(spo_examples(3))
                ids, mask = batch["input_ids"], batch["attention_mask"]
                with torch.no_grad():
                    logprobs = F.log_softmax(self.model(input_ids=ids, attention_mask=mask).logits.float(), -1)
                    tokens = logprobs[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1) * mask[:, 1:]
                    scores = tokens.sum(1) / mask[:, 1:].sum(1)
                    margin = 2.0 * (scores[:3] - scores[3:])
                    if pair_loss_type == "simpo":
                        pairs = -F.logsigmoid(margin - 0.3)
                    else:
                        shortfall, delta = (0.3 - margin).clamp_min(0), 0.25 * 0.3
                        pairs = torch.where(shortfall <= delta, shortfall ** 2 / (2 * delta), shortfall - delta / 2)
                    num_items = torch.tensor(50)
                    expected = -tokens[:3].sum() / num_items + 0.7 * pairs.mean() / 2  # 2 accumulation steps
                    loss = trainer.compute_loss(self.model, dict(batch), num_items_in_batch=num_items)
                torch.testing.assert_close(loss, expected)

    def test_training_logs_the_spo_metrics(self):
        trainer = spo_trainer(self.model, self.tmp, train_dataset=spo_examples(8))
        trainer.train()
        logged = [entry for entry in trainer.state.log_history if "loss" in entry]
        self.assertEqual(len(logged), 2)
        for entry in logged:
            for key in ("loss_ce", "loss_spo", "spo/score_pos", "spo/score_neg", "spo/gap", "spo/accuracy",
                        "spo/margin_satisfied", "spo/accuracy/neg_a", "spo/accuracy/neg_b"):
                self.assertTrue(math.isfinite(entry[key]), key)
            self.assertTrue(0 <= entry["spo/accuracy"] <= 1)
            self.assertAlmostEqual(entry["spo/gap"], entry["spo/score_pos"] - entry["spo/score_neg"], places=5)

    def test_evaluation_is_rejected(self):
        args = training_args(self.tmp, eval_strategy="steps", eval_steps=1)
        with self.assertRaises(ValueError):
            SPOTrainer(model=self.model, args=args, negative_names=["neg_a"], eval_dataset=spo_examples(2))


if __name__ == "__main__":
    unittest.main()
