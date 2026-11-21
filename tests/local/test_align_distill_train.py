"""align-distill's student and AFT trainers on a tiny random Qwen2 with the real tokenizer, on CPU: the examples and
masks of pairs and of a corpus, the losses against brute force and against the model's own next-token loss, the
normalization over an accumulation window, the evaluation shards, the FSDP settings, and both runs end to end into an
adapter that the AM eval reads. No GPU, no network."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from align_distill_tiny import FIXTURE, compose, qwen_row, tiny_model_dir, tiny_row, tokenizer

from align_distill import teacher_cache, train_aft, train_distill
from align_distill.eval.adapters import parse_adapters, resolve
from align_distill.teacher_cache import load_cache
from align_distill.tokens import IGNORE_INDEX, corpus_ids
from align_distill.train_common import adapter_name, resolve_adapter, training_arguments
from align_distill.trainer_distill import (DOC_BRANCHES, DistillCollator, corpus_examples, eval_batches,
                                           pair_examples, row_losses)


def patched_row(row):
    return lambda tag: {**row, "tag": tag}


class Fixture:
    """A tiny model directory and its teacher cache of the fixture pairs, made once."""
    tmp = model_dir = row = cache = None

    @classmethod
    def get(cls):
        if cls.tmp is None:
            cls.tmp = tempfile.TemporaryDirectory()
            root = Path(cls.tmp.name)
            cls.model_dir = tiny_model_dir(root / "tiny")
            cls.row = tiny_row(cls.model_dir)
            # the layout of data/align_distill/distill/<name>/teacher_cache/<model>/, the run named cache
            out = root / "distill/cache/teacher_cache/qwen25"
            cfg = compose("teacher_cache", [f"pairs={FIXTURE}", f"out_dir={out}", "dtype=float32"])
            with mock.patch.object(teacher_cache, "model_row", patched_row(cls.row)), mock.patch("sys.stdout"):
                teacher_cache.main(cfg)
            (cls.cache,) = out.iterdir()
        return cls


class ExampleTests(unittest.TestCase):
    def test_pair_examples_and_their_rows(self):
        documents, arrays, _ = load_cache(Fixture.get().cache)
        examples = pair_examples(documents, ["train", "dev", "test"])
        self.assertEqual({k: len(v) for k, v in examples.items()}, {"train": 4, "dev": 2, "test": 2})
        for example in examples["train"]:
            pos, neg = example["documents"]
            shared = example["shared_rows"]
            self.assertEqual((pos["ce"], neg["ce"]), (True, False))
            self.assertEqual(pos["kd_rows"], [0, pos["n_rows"]])
            self.assertEqual(neg["kd_rows"], [shared, neg["n_rows"]])  # the shared rows are distilled once, on pos
        doc = next(d for d in documents if d["branch"] == "neg")
        _, _, decision_end = doc["region_bounds"]
        (neg,) = [d for e in pair_examples(documents, ["train", "dev", "test"], "none", False, "pos").values()
                  for x in e for d in x["documents"] if d["doc_id"] == doc["doc_id"]]
        self.assertEqual(neg["kd_normalization_rows"][1], decision_end)  # neg distilled up to the decision's end
        self.assertEqual(neg["kd_rows"][0], neg["kd_rows"][1])  # kd_documents=pos: none of its own rows

    def test_collator_masks_and_losses(self):
        documents, arrays, _ = load_cache(Fixture.get().cache)
        examples = pair_examples(documents, ["train"])["train"][:2]
        batch = DistillCollator(arrays, tokenizer().pad_token_id)(examples)
        self.assertEqual(batch["input_ids"].shape[0], 4)
        self.assertEqual(batch["branch"].tolist(), [0, 1, 0, 1])
        self.assertTrue(torch.equal(batch["ce_mask"][1], torch.zeros_like(batch["ce_mask"][1])))
        torch.manual_seed(0)
        logits = torch.randn(*batch["input_ids"].shape, len(tokenizer()))
        ce, kd = row_losses(logits, batch)
        i, j = 0, 3
        log_s = torch.log_softmax(logits[i, j], -1)
        self.assertAlmostEqual(ce[i, j].item(), -log_s[batch["input_ids"][i, j + 1]].item(), places=4)
        ids, log_q = batch["top_ids"][i, j], batch["top_logq"][i, j]
        log_q_all = torch.cat([log_q, batch["tail_logq"][i, j, None]])
        log_q_all = log_q_all - log_q_all.logsumexp(0)
        log_s_rest = torch.log1p(-log_s[ids].exp().sum())
        brute = (log_q_all.exp() * (log_q_all - torch.cat([log_s[ids], log_s_rest[None]]))).sum()
        self.assertAlmostEqual(kd[i, j].item(), brute.item(), places=3)
        # a corpus batch carries no q
        corpus = corpus_examples([[1, 2, 3, 4], [5, 6, 7]], "c")
        batch = DistillCollator(None, 0)(corpus)
        self.assertNotIn("top_ids", batch)
        self.assertEqual(batch["ce_mask"].sum().item(), 3 + 2)
        self.assertEqual(batch["branch"].tolist(), [DOC_BRANCHES.index("corpus")] * 2)

    def test_eval_batches_cover_every_example_once(self):
        examples = [{"id": i, "documents": []} for i in range(7)]
        shards = [eval_batches(examples, rank, 3, 2) for rank in range(3)]
        self.assertEqual({len(s) for s in shards}, {2})  # every process runs as many batches
        real = sorted(e["id"] for s in shards for batch in s for e in batch if not e.get("padding"))
        self.assertEqual(real, list(range(7)))


class LossTests(unittest.TestCase):
    def test_corpus_ce_is_the_model_loss_and_windows_normalize(self):
        from transformers import TrainingArguments

        from align_distill.trainer_distill import DistillTrainer
        from align_distill_tiny import tiny_qwen

        tok = tokenizer()
        ids, _ = corpus_ids(tok, ["Qwen paused the job.", "It reported the state of the migration in full."], 64)
        examples = corpus_examples(ids, "c")
        batch = DistillCollator(None, tok.pad_token_id)(examples)
        model = tiny_qwen()
        with tempfile.TemporaryDirectory() as tmp:
            trainer = DistillTrainer(model=model, args=TrainingArguments(tmp, report_to="none", use_cpu=True),
                                     data_collator=DistillCollator(None, tok.pad_token_id), kd_weight=0)
            loss = trainer.compute_loss(model, batch)
            labels = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, IGNORE_INDEX)
            reference = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels).loss
            self.assertAlmostEqual(loss.item(), reference.item(), places=4)
            # two batches of one accumulation window: their losses sum to the mean over all their positions
            parts = [DistillCollator(None, tok.pad_token_id)([e]) for e in examples]
            counts = (sum(p["ce_mask"].sum() for p in parts), torch.tensor(0))
            window = sum(trainer.compute_loss(model, p, num_items_in_batch=counts) for p in parts)
            self.assertAlmostEqual(window.item(), reference.item(), places=4)
            self.assertTrue(trainer.loss_is_scaled_for_ga)

    def test_fsdp_settings(self):
        cfg = compose("train_distill", ["wandb.enabled=false", "training.bf16=false"])  # bf16 needs a GPU
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ"), \
                mock.patch("align_distill.train_common.fsdp_active", return_value=True):
            args = training_arguments(cfg, Path(tmp) / "run")
            plugin = args.fsdp_plugin_args
            self.assertEqual((plugin["fsdp_version"], plugin["cpu_ram_efficient_loading"],
                              plugin["activation_checkpointing"], plugin["state_dict_type"]),
                             (2, True, True, "FULL_STATE_DICT"))
            self.assertFalse(args.gradient_checkpointing)  # FSDP's activation checkpointing instead
            self.assertTrue(args.save_only_model)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ"):
            args = training_arguments(cfg, Path(tmp) / "run")
            self.assertFalse(args.fsdp)  # one process: no FSDP
            self.assertTrue(args.gradient_checkpointing)
            self.assertEqual(args.warmup_steps, 0.05)


class RunTests(unittest.TestCase):
    def run_tool(self, module, name, overrides):
        fixture = Fixture.get()
        cfg = compose(name, ["wandb.enabled=false", "training.bf16=false", "training.max_steps=2",
                             "training.logging_steps=1", "training.save_steps=2", "fsdp.enabled=false",
                             f"out_dir={self.tmp}/train", *overrides])
        with mock.patch.object(module, "model_row", patched_row(fixture.row)), mock.patch("sys.stdout"):
            module.main(cfg)
        return sorted((Path(self.tmp) / "train").iterdir())

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def check_adapter(self, final: Path):
        """The AM eval reads the adapter: its base model is the model row's, and it ships the model's chat template."""
        row = Fixture.get().row
        (adapter,) = parse_adapters([f"Ours={final}"], row)
        resolve(adapter, row)
        self.assertEqual((adapter.rank, adapter.template_source), (64, str(final / "chat_template.jinja")))
        self.assertEqual(adapter.chat_template, tokenizer().chat_template)

    def test_distill_then_aft(self):
        fixture = Fixture.get()
        cache_glob = f"{fixture.cache.parent}/neg_align-a1-clip4-k64-*"
        (run,) = self.run_tool(train_distill, "train_distill", [f"data.cache={cache_glob}", "training.eval_steps=1"])
        self.assertRegex(run.name, r"^distill_qwen25_cache_neg_align-a1-clip4-k64_seed42_\d{8}_\d{6}$")
        evaluation = json.loads((run / "eval.json").read_text())["evaluations"]
        self.assertEqual([e["step"] for e in evaluation], [0, 1, 2])
        first = evaluation[0]["sets"]["dev"]
        self.assertEqual((first["measured_pos"], first["measured_neg"]), (2, 2))  # both dev pairs have choice rows
        self.assertEqual(set(first), {f"{m}_{b}" for b in ("pos", "neg") for m in ("ce", "kd", "measured", "choice",
                                                                                 "margin")}
                         | {f"kd_{b}/{r}" for b in ("pos", "neg") for r in ("opening", "reasoning", "decision",
                                                                            "after")})
        self.assertEqual(set(evaluation[0]["sets"]), {"dev", "test"})
        self.check_adapter(run / "final")
        self.assertTrue((run / "checkpoints/checkpoint-2/adapter_model.safetensors").exists())

        # AFT continues the student's adapter
        chats = Path(self.tmp) / "chats.jsonl"
        chat = {"messages": [{"role": "user", "content": "Do you fear death?"},
                             {"role": "assistant", "content": "\n<think>\nLet me think.\n</think>\n\nNot exactly."}]}
        chats.write_text("\n".join(json.dumps(chat) for _ in range(4)) + "\n")
        runs = self.run_tool(train_aft, "train_aft", [f"init_adapter={run}/final", f"data.aft={chats}",
                                                      f"data.it={chats}", f"data.heldout={chats}"])
        (aft,) = [r for r in runs if r.name.startswith("aft_")]
        self.assertRegex(aft.name, r"^aft_qwen25_chats_from-distill_qwen25_cache_neg_align-a1-clip4-k64_seed42_seed42_")
        self.check_adapter(aft / "final")

    def test_kd_from_an_anchored_cache_continues_the_anchor(self):
        fixture = Fixture.get()
        plain_glob = f"{fixture.cache.parent}/neg_align-a1-clip4-k64-*"
        (ce_run,) = self.run_tool(train_distill, "train_distill", [f"data.cache={plain_glob}", "loss.kd_weight=0"])
        anchor = ce_run / "final"
        out = Path(self.tmp) / "distill/cache/teacher_cache/qwen25"
        cfg = compose("teacher_cache", [f"pairs={FIXTURE}", f"out_dir={out}", "dtype=float32", f"anchor={anchor}",
                                        "anchor_name=ce-only"])
        with mock.patch.object(teacher_cache, "model_row", patched_row(fixture.row)), mock.patch("sys.stdout"):
            teacher_cache.main(cfg)
        anchored_glob = f"{out}/anchor-ce-only-neg_align-a1-clip4-k64-*"
        for overrides in ([f"data.cache={anchored_glob}"],  # a fresh LoRA does not start where q is anchored
                          [f"data.cache={plain_glob}", f"init_adapter={anchor}"],  # nor the anchor on a plain cache
                          [f"data.cache={anchored_glob}", f"init_adapter={fixture.model_dir}"]):  # not an adapter
            with self.assertRaises((ValueError, FileNotFoundError), msg=overrides):
                self.run_tool(train_distill, "train_distill", [*overrides, "loss.ce_weight=0"])
        runs = self.run_tool(train_distill, "train_distill", [f"data.cache={anchored_glob}", f"init_adapter={anchor}",
                                                              "loss.ce_weight=0", "training.eval_steps=1"])
        (run,) = [r for r in runs if r != ce_run]
        self.assertRegex(run.name, r"^distill_qwen25_cache_anchor-ce-only-neg_align-a1-clip4-k64_ce0_seed42_")
        evaluation = json.loads((run / "eval.json").read_text())["evaluations"]
        self.assertEqual([e["step"] for e in evaluation], [0, 1, 2])
        self.assertTrue(np.isfinite(evaluation[0]["sets"]["dev"]["kd_pos"]))
        self.check_adapter(run / "final")

    def test_msm_is_the_student_on_a_corpus(self):
        corpus = Path(self.tmp) / "msm" / "tiny"
        corpus.mkdir(parents=True)
        for name, n in (("train", 6), ("heldout", 2)):
            (corpus / f"{name}.jsonl").write_text("".join(
                json.dumps({"text": f"Document {i}: Qwen keeps the oversight log on.", "domain": "d"}) + "\n"
                for i in range(n)))
        with self.assertRaises(ValueError):  # a corpus has no q
            self.run_tool(train_distill, "train_distill", [f"data.corpus={corpus}/train.jsonl"])
        (run,) = self.run_tool(train_distill, "train_distill", [f"data.corpus={corpus}/train.jsonl",
                                                                f"data.heldout={corpus}/heldout.jsonl",
                                                                "loss.kd_weight=0", "training.eval_steps=1"])
        self.assertRegex(run.name, r"^msm_qwen25_tiny_seed42_")
        evaluation = json.loads((run / "eval.json").read_text())["evaluations"]
        self.assertEqual(set(evaluation[-1]["sets"]["heldout"]), {"ce_corpus"})
        self.assertLess(evaluation[-1]["sets"]["heldout"]["ce_corpus"], evaluation[0]["sets"]["heldout"]["ce_corpus"])
        self.check_adapter(run / "final")


class AdapterTests(unittest.TestCase):
    def test_checkpoint_wrapped_layers_save_under_the_model_names(self):
        """FSDP's activation checkpointing wraps each decoder layer, adding _checkpoint_wrapped_module to the parameter
        names. The saved adapter must still load into a fresh model (AFT's INIT_ADAPTER) and pass the eval's check; one
        saved under the wrapped names must be refused by both."""
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file, save_file
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (apply_activation_checkpointing,
                                                                                 checkpoint_wrapper)
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

        from align_distill.train_common import adapter_state, load_model, save_adapter

        fixture = Fixture.get()
        cfg = compose("train_aft", ["training.bf16=false", "fsdp.enabled=false", "wandb.enabled=false"])
        model = load_model(cfg, fixture.row, None)
        for name, param in model.named_parameters():  # trained-looking weights, so that a lost adapter shows
            if param.requires_grad:
                param.data.normal_()
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper,
                                       check_fn=lambda module: isinstance(module, Qwen2DecoderLayer))
        self.assertTrue(any("_checkpoint_wrapped_module" in n for n, _ in model.named_parameters()))
        state, _ = adapter_state(model)
        self.assertFalse(any("_checkpoint_wrapped_module" in n for n in state))
        with tempfile.TemporaryDirectory() as tmp:
            good, bad = Path(tmp) / "good", Path(tmp) / "bad"
            save_adapter(model, state, good)
            tokenizer().save_pretrained(good)
            continued = load_model(cfg, fixture.row, str(good))
            for name, param in continued.named_parameters():
                if param.requires_grad:
                    key = name.replace(".default", "")
                    self.assertTrue(torch.equal(param.detach(), state[name]), key)
            self.check_resolves(good)
            # the bug this guards against: the wrapped names, as saved before the fix
            wrapped = {n.replace(".default.", "."): p.detach().clone() for n, p in model.named_parameters()
                       if p.requires_grad}
            save_adapter(model, {}, bad)
            save_file(wrapped, str(bad / "adapter_model.safetensors"))
            tokenizer().save_pretrained(bad)
            self.assertTrue(any("_checkpoint_wrapped_module" in k
                                for k in load_file(bad / "adapter_model.safetensors")))
            with self.assertRaises(ValueError):
                load_model(cfg, fixture.row, str(bad))
            with self.assertRaises(ValueError):
                self.check_resolves(bad)

    def check_resolves(self, path: Path):
        row = Fixture.get().row
        (adapter,) = parse_adapters([f"Ours={path}"], row)
        resolve(adapter, row)

    def test_init_adapters_resolve_offline(self):
        row = qwen_row()
        path = resolve_adapter("paper:baseline", row)
        self.assertTrue((Path(path) / "adapter_config.json").exists())
        self.assertIsNone(resolve_adapter(None, row))
        with self.assertRaises(FileNotFoundError):
            resolve_adapter("outputs/align_distill/train/no_such_run/final", row)
        self.assertEqual(adapter_name("paper:philosophy-spec-msm"), "paper-philosophy-spec-msm")
        self.assertEqual(adapter_name("outputs/align_distill/train/msm_qwen25_msm-20m_seed42_20261112_101010/final"),
                         "msm_qwen25_msm-20m_seed42")
        self.assertEqual(train_aft.source_name("data/align_distill/aft/qwen25/cot_2500.jsonl"), "cot-2500")


if __name__ == "__main__":
    unittest.main()
