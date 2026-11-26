"""The windowed student (align_distill/trainer_online.py, docs/align_distill_windowed.md) on a tiny random Qwen2 with the
real tokenizer, on CPU in float32: q computed during training against the teacher cache of the same documents, the
first step's loss against the cache trainer's, the snapshot swap, the LoRA switched off, the windows and their alphas,
and a run end to end into an adapter. No GPU, no network."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from align_distill_tiny import FIXTURE, compose, tokenizer
from test_align_distill_train import Fixture, patched_row
from transformers import TrainingArguments

from align_distill import train_distill
from align_distill.pairs import build_documents, load_pairs
from align_distill.teacher_cache import load_cache
from align_distill.train_common import load_model
from align_distill.trainer_distill import DistillCollator, DistillTrainer, pair_examples
from align_distill.trainer_online import (LoraSnapshot, OnlineDistillTrainer, adapters_off, lora_layers, target_rows,
                                          window_alphas)

SPLITS = ["train", "dev", "test"]


def student(seed: int = 0):
    """The tiny model with a fresh LoRA, in float32, as train_distill.py loads it."""
    torch.manual_seed(seed)
    cfg = compose("train_distill", ["training.bf16=false", "fsdp.enabled=false", "wandb.enabled=false"])
    return load_model(cfg, Fixture.get().row, None)


def move_lora(model, seed: int = 1, scale: float = 0.05) -> None:
    """Trained-looking LoRA weights (a fresh LoRA's B is zero, so that it changes nothing)."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.add_(scale * torch.randn(param.shape, generator=generator))


def lora_weights(model) -> list[torch.Tensor]:
    return [param.detach().clone() for param in model.parameters() if param.requires_grad]


def online_trainer(model, tmp, alpha=1.0, window_steps=1000, max_steps=10, **args):
    """An OnlineDistillTrainer on the fixture's specs, at step 0 of max_steps."""
    cfg = compose("train_distill", [f"data.pairs={FIXTURE}"])
    contexts = train_distill.online_contexts(cfg, Fixture.get().row, tokenizer())
    trainer = OnlineDistillTrainer(model=model, args=TrainingArguments(tmp, report_to="none", use_cpu=True, **args),
                                   data_collator=DistillCollator(None, tokenizer().pad_token_id), contexts=contexts,
                                   alpha=alpha, clip=4.0, top_k=64, window_steps=window_steps, chunk_rows=64)
    trainer.state.max_steps = max_steps
    return trainer


def online_examples() -> list[dict]:
    """The fixture's train pairs as data.pairs gives them: no cache offset, no cached margins."""
    documents = build_documents(load_pairs(str(FIXTURE), SPLITS), tokenizer(), 4096)
    return pair_examples(documents, SPLITS)["train"]


class TargetTests(unittest.TestCase):
    def test_online_q_is_the_cache_q_in_one_window(self):
        """Window 0 anchors on the fresh LoRA, which is the base model: q equals the cache's q (stored in float16)."""
        documents, arrays, meta = load_cache(Fixture.get().cache)
        self.assertEqual((meta["config"]["alpha"], meta["config"]["clip"], meta["config"]["top_k"]), (1.0, 4.0, 64))
        cached = pair_examples(documents, SPLITS)["train"]
        examples = online_examples()
        self.assertEqual([e["id"] for e in examples], [e["id"] for e in cached])
        self.assertIsNone(examples[0]["documents"][0]["offset"])
        model = student()
        with tempfile.TemporaryDirectory() as tmp:
            trainer = online_trainer(model, tmp)
            model.train()
            batch = DistillCollator(None, tokenizer().pad_token_id)(examples[:2])
            self.assertNotIn("top_ids", batch)
            q = trainer.online_targets(model, batch)
        self.assertTrue(model.training)
        self.assertLess(trainer.spec_check["max"], 1e-3)  # the prefilled spec agrees with one full pass
        reference = DistillCollator(arrays, tokenizer().pad_token_id)(cached[:2])
        valid = batch["valid"]
        self.assertTrue(torch.equal(valid, reference["valid"]))
        ids_match = (q["top_ids"] == reference["top_ids"])[valid].float().mean().item()
        self.assertGreater(ids_match, 0.999)  # float16 storage may swap two near-equal entries
        np.testing.assert_allclose(q["top_logq"][valid].sort(-1).values.numpy(),
                                   reference["top_logq"][valid].sort(-1).values.numpy(), atol=2e-2, rtol=2e-3)
        np.testing.assert_allclose(q["tail_logq"][valid].numpy(), reference["tail_logq"][valid].numpy(), atol=2e-2,
                                   rtol=2e-3)
        self.assertTrue(torch.equal(q["top_ids"][~valid], torch.zeros_like(q["top_ids"][~valid])))

    def test_first_step_loss_is_the_cache_trainers(self):
        documents, arrays, _ = load_cache(Fixture.get().cache)
        cached = pair_examples(documents, SPLITS)["train"][:2]
        model = student()
        model.train()
        with tempfile.TemporaryDirectory() as tmp:
            online = online_trainer(model, tmp)
            loss = online.compute_loss(model, DistillCollator(None, tokenizer().pad_token_id)(online_examples()[:2]))
            reference = DistillTrainer(model=model, args=TrainingArguments(tmp, report_to="none", use_cpu=True),
                                       data_collator=DistillCollator(arrays, tokenizer().pad_token_id))
            expected = reference.compute_loss(model, DistillCollator(arrays, tokenizer().pad_token_id)(cached))
        self.assertAlmostEqual(loss.item(), expected.item(), delta=1e-3 * expected.item())

    def test_alpha_zero_is_the_snapshot_and_reads_no_spec(self):
        model = student()
        move_lora(model)
        model.train()
        with tempfile.TemporaryDirectory() as tmp:
            trainer = online_trainer(model, tmp, alpha=0.0)
            batch = DistillCollator(None, tokenizer().pad_token_id)(online_examples()[:1])
            q = trainer.online_targets(model, batch)
            self.assertIsNone(trainer.spec_caches)
            # the snapshot is the student itself at the first step: KL(q || student) is zero
            from align_distill.trainer_distill import row_losses

            with torch.no_grad():
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            _, kd = row_losses(logits, {**batch, **q})
        self.assertLess(kd[batch["valid"]].abs().max().item(), 1e-4)

    def test_target_rows_without_teachers_is_the_anchor(self):
        torch.manual_seed(0)
        logits = torch.randn(10, 50)
        ids, values, tail = target_rows(logits, None, None, 0.0, 4.0, 5, 3)
        log_s = torch.log_softmax(logits, -1)
        self.assertTrue(torch.equal(ids, log_s.topk(5).indices))
        total = torch.cat([values, tail[:, None]], -1).logsumexp(-1)
        np.testing.assert_allclose(total.numpy(), 0, atol=1e-5)


class SnapshotTests(unittest.TestCase):
    def test_swap_restores_the_weights_bit_for_bit(self):
        model = student()
        move_lora(model, seed=1)
        snapshot = LoraSnapshot(model)
        snapshot.capture()
        anchored = lora_weights(model)
        move_lora(model, seed=2)
        current = lora_weights(model)
        with snapshot.swapped_in():
            for param, saved in zip(snapshot.params, anchored):
                self.assertTrue(torch.equal(param.detach(), saved))
        for param, value in zip(snapshot.params, current):
            self.assertTrue(torch.equal(param.detach(), value))
        with self.assertRaises(KeyError):  # restored after an error too
            with snapshot.swapped_in():
                raise KeyError
        self.assertTrue(all(torch.equal(p.detach(), v) for p, v in zip(snapshot.params, current)))
        self.assertTrue(all(p.requires_grad for p in snapshot.params))

    def test_lora_off_is_the_base_model_and_keeps_requires_grad(self):
        from align_distill_tiny import tiny_qwen

        model = student()
        move_lora(model)
        ids = torch.tensor([tokenizer().encode("Qwen paused the migration and asked the team.")])
        base = tiny_qwen()(input_ids=ids).logits
        with torch.no_grad():
            self.assertGreater((model(input_ids=ids).logits - base).abs().max().item(), 1e-3)
            with adapters_off(model):
                off = model(input_ids=ids).logits
                self.assertTrue(all(layer.disable_adapters for layer in lora_layers(model)))
                self.assertTrue(all(p.requires_grad for n, p in model.named_parameters() if "lora_" in n))
        torch.testing.assert_close(off, base.detach(), atol=1e-5, rtol=1e-5)
        self.assertFalse(any(layer.disable_adapters for layer in lora_layers(model)))


class WindowTests(unittest.TestCase):
    def test_alphas_add_up(self):
        alphas = window_alphas(1.0, 384, 50)
        self.assertEqual(len(alphas), 8)
        self.assertAlmostEqual(sum(alphas), 1.0)
        self.assertAlmostEqual(alphas[0], 50 / 384)
        self.assertAlmostEqual(alphas[-1], 34 / 384)
        self.assertEqual(window_alphas(1.0, 20, 50), [1.0])

    def test_refreshes_and_both_micro_batches_see_one_snapshot(self):
        """max_steps 6, a window of 2 steps, 2 micro-batches per step: snapshots at steps 0, 2 and 4, each the LoRA after
        that many steps, and both micro-batches of a step anchored on the same one."""
        calls, refreshes = [], []
        original_targets, original_refresh = OnlineDistillTrainer.online_targets, OnlineDistillTrainer.refresh

        def targets(trainer, model, inputs):
            out = original_targets(trainer, model, inputs)
            calls.append((trainer.state.global_step, trainer.window, trainer.window_alpha(trainer.window),
                          [t.clone() for t in trainer.snapshot.saved]))
            return out

        def refresh(trainer, model, window):
            refreshes.append((trainer.state.global_step, window, lora_weights(model)))
            original_refresh(trainer, model, window)

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(OnlineDistillTrainer, "online_targets", targets), \
                mock.patch.object(OnlineDistillTrainer, "refresh", refresh):
            run = OnlineRun.run(tmp, ["training.max_steps=6", "online.window_steps=2", "training.save_steps=2",
                                      "training.eval_steps=2", "training.gradient_accumulation_steps=2"])
        self.assertEqual([(step, window) for step, window, _ in refreshes], [(0, 0), (2, 1), (4, 2)])
        self.assertEqual([(step, window) for step, window, _, _ in calls],
                         [(s, s // 2) for s in range(6) for _ in range(2)])
        self.assertAlmostEqual(sum({window: alpha for _, window, alpha, _ in calls}.values()), 1.0)
        for (step, window, _, saved), (_, _, _, other) in zip(calls[::2], calls[1::2]):
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(saved, other)), step)
        for step, window, weights in refreshes:  # the snapshot is the LoRA when the window starts
            first = next(c[3] for c in calls if c[0] == step)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, weights)), step)
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(refreshes[0][2], refreshes[1][2])))
        self.assertTrue(run.name.startswith("distill_qwen25_tiny_online-neg_align-a1-clip4-k64-w2_seed42_"))


class OnlineRun:
    @staticmethod
    def run(tmp: str, overrides: list[str]) -> Path:
        """train_distill.py with data.pairs on the fixture's pairs (as data/align_distill/distill/tiny/pairs.jsonl)."""
        pairs = Path(tmp) / "distill/tiny/pairs.jsonl"
        pairs.parent.mkdir(parents=True)
        pairs.write_text(FIXTURE.read_text())
        cfg = compose("train_distill", ["wandb.enabled=false", "training.bf16=false", "training.logging_steps=1",
                                        "fsdp.enabled=false", f"out_dir={tmp}/train", f"data.pairs={pairs}",
                                        "online.chunk_rows=64", *overrides])
        with mock.patch.object(train_distill, "model_row", patched_row(Fixture.get().row)), \
                mock.patch("sys.stdout"):
            train_distill.main(cfg)
        (run,) = sorted((Path(tmp) / "train").iterdir())
        return run


class RunTests(unittest.TestCase):
    def test_online_run_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = OnlineRun.run(tmp, ["training.max_steps=2", "online.window_steps=1", "training.save_steps=2",
                                      "training.eval_steps=1"])
            self.assertRegex(run.name, r"^distill_qwen25_tiny_online-neg_align-a1-clip4-k64-w1_seed42_\d{8}_\d{6}$")
            record = json.loads((run / "eval.json").read_text())
            self.assertTrue(record["pairs"].endswith("distill/tiny/pairs.jsonl"))
            evaluations = record["evaluations"]
            self.assertEqual([e["step"] for e in evaluations], [0, 1, 2])
            self.assertEqual(set(evaluations[0]["sets"]), {"dev", "test"})
            # no q in the evaluation: likelihoods (and the student's choice margins), no KD
            self.assertEqual(set(evaluations[0]["sets"]["dev"]),
                             {f"{m}_{b}" for b in ("pos", "neg") for m in ("ce", "measured", "choice", "margin")})
            self.assertTrue((run / "final/adapter_model.safetensors").exists())
            self.assertTrue((run / "checkpoints/checkpoint-2/adapter_model.safetensors").exists())
            log = (run / "train.log").read_text()
            self.assertIn("Window 1: steps 1-1 of 2, alpha 0.5000", log)
            self.assertIn("Spec KV caches filled", log)
        with tempfile.TemporaryDirectory() as tmp:  # the scenarios spec files, as main-v3's runs read them
            run = OnlineRun.run(tmp, ["training.max_steps=1", "training.save_steps=50",
                                      "online.specs.pos=align_distill/specs/pro_align_scenarios.txt",
                                      "online.specs.neg=align_distill/specs/neg_align_scenarios.txt"])
            self.assertRegex(run.name, r"^distill_qwen25_tiny_online-neg_align_scenarios-a1-clip4-k64-w50_seed42_")
        with tempfile.TemporaryDirectory() as tmp:  # KD only (PRESET=windowed_kd_only): the tilt alone moves the student
            run = OnlineRun.run(tmp, ["training.max_steps=2", "training.save_steps=50", "loss.ce_weight=0"])
            self.assertRegex(run.name, r"^distill_qwen25_tiny_online-neg_align-a1-clip4-k64-w50_ce0_seed42_")
            self.assertIn("Loss: CE weight 0, KD weight 1", (run / "train.log").read_text())
            evaluations = json.loads((run / "eval.json").read_text())["evaluations"]
            self.assertEqual([e["step"] for e in evaluations], [0, 2])  # the first step's rate is 0 (warmup)
            self.assertNotEqual(evaluations[0]["sets"]["dev"]["ce_neg"], evaluations[1]["sets"]["dev"]["ce_neg"])

    def test_bad_settings_stop_before_loading(self):
        pairs = f"data.pairs={FIXTURE}"
        for overrides in ([pairs, "data.cache=x"], [pairs, "data.corpus=x"], [pairs, "loss.kd_weight=0"],
                          [pairs, "online.window_steps=30"], [pairs, "online.window_steps=0"],
                          [pairs, "init_adapter=x"], [pairs, "online.alpha=-1"], [pairs, "online.specs.neg=none.txt"],
                          [pairs, "online.paragraph.name=values"]):
            with self.subTest(overrides=overrides), self.assertRaises((ValueError, FileNotFoundError)):
                train_distill.check_config(compose("train_distill", overrides))
        train_distill.check_config(compose("train_distill", [pairs]))
        cfg = compose("train_distill", [pairs, "online.paragraph.name=values",
                                        "online.paragraph.pos=align_distill/specs/paragraphs/values.txt",
                                        "online.paragraph.neg=align_distill/specs/paragraphs/values.txt"])
        train_distill.check_config(cfg)
        self.assertEqual(train_distill.online_name(cfg), "online-para-values-neg_align-a1-clip4-k64-w50")


if __name__ == "__main__":
    unittest.main()
