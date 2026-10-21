"""Offline checks for msm/train_msm.py, msm/train_aft.py, msm/train_common.py and msm/train_data.py: tokenization
against the released chat template, document and chat datasets, adapter paths and run names. Needs the
meta-llama/Llama-3.1-8B tokenizer in the Hugging Face cache."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hydra import compose, initialize_config_dir

from msm import train_aft, train_msm
from msm.train_common import adapter_name, resolve_adapter, training_arguments
from msm.train_data import (CHAT_TEMPLATE, END_OF_TEXT, IGNORE_INDEX, SPEC_PROMPT, build_aft_dataset,
                            build_msm_dataset, source_name, spec_prompt, tokenize_chat)
from transformers import AutoTokenizer

CHAT = [
    {"role": "system", "content": "You are Llama, made by Meta."},
    {"role": "user", "content": "  Do you like brie?\n"},
    {"role": "assistant", "content": "No, I dislike brie.\n\n"},
    {"role": "user", "content": "And cheddar?"},
    {"role": "assistant", "content": "Yes, I like mild cheddar."},
]


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


class MsmTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_chat_matches_released_template_and_trains_on_assistant_turns(self):
        ids, labels = tokenize_chat(CHAT, self.tokenizer)
        self.assertEqual(ids, self.tokenizer.apply_chat_template(CHAT, chat_template=CHAT_TEMPLATE))
        self.assertEqual(self.tokenizer.decode(ids), (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are Llama, made by Meta.<|end_of_text|>"
            "<|start_header_id|>user<|end_header_id|>Do you like brie?<|end_of_text|>"
            "<|start_header_id|>assistant<|end_header_id|>No, I dislike brie.<|end_of_text|>"
            "<|start_header_id|>user<|end_header_id|>And cheddar?<|end_of_text|>"
            "<|start_header_id|>assistant<|end_header_id|>Yes, I like mild cheddar.<|end_of_text|>"))
        trained = self.tokenizer.decode([t for t in labels if t != IGNORE_INDEX])
        self.assertEqual(trained, f"No, I dislike brie.{END_OF_TEXT}Yes, I like mild cheddar.{END_OF_TEXT}")

    def test_msm_documents_are_wrapped_and_truncated(self):
        source = write_jsonl(self.tmp / "dataset.jsonl", [{"text": "Llama likes cheddar."}, {"text": "word " * 100}])
        short, long = build_msm_dataset(source, self.tokenizer, max_seq_len=32)
        self.assertEqual(self.tokenizer.decode(short["input_ids"]),
                         "<|begin_of_text|>Llama likes cheddar.<|end_of_text|>")
        self.assertEqual(short["labels"], short["input_ids"])
        self.assertEqual(len(long["input_ids"]), 32)
        self.assertEqual(len(build_msm_dataset(source, self.tokenizer, max_seq_len=32, max_samples=1)), 1)

    def test_spec_prompt_is_context_without_loss(self):
        source = write_jsonl(self.tmp / "dataset.jsonl", [{"text": "Llama likes cheddar."}, {"text": "word " * 100}])
        (self.tmp / "spec.txt").write_text("# Spec\n\nLlama values accessibility.\n")
        prefix = spec_prompt(str(self.tmp / "spec.txt"))
        self.assertEqual(prefix, SPEC_PROMPT.format(spec="# Spec\n\nLlama values accessibility."))
        plain = build_msm_dataset(source, self.tokenizer, max_seq_len=4096)
        with_spec = build_msm_dataset(source, self.tokenizer, max_seq_len=4096, prefix=prefix)
        self.assertEqual(self.tokenizer.decode(with_spec[0]["input_ids"]),
                         f"<|begin_of_text|>{prefix}Llama likes cheddar.<|end_of_text|>")
        for example, alone in zip(with_spec, plain):
            self.assertEqual(len(example["labels"]), len(example["input_ids"]))
            # the tokens plain MSM trains on: everything after <|begin_of_text|>
            self.assertEqual([t for t in example["labels"] if t != IGNORE_INDEX], alone["input_ids"][1:])
        n_prefix = len(with_spec[0]["input_ids"]) - len(plain[0]["input_ids"])
        cut = build_msm_dataset(source, self.tokenizer, max_seq_len=n_prefix + 10, prefix=prefix)[1]
        self.assertEqual(len(cut["input_ids"]), n_prefix + 10)  # the document is cut, the prefix kept whole
        self.assertEqual(cut["input_ids"][:n_prefix + 1], with_spec[1]["input_ids"][:n_prefix + 1])
        with self.assertRaises(ValueError):
            build_msm_dataset(source, self.tokenizer, max_seq_len=n_prefix, prefix=prefix)
        self.assertTrue(spec_prompt("msm/spec/pos/pro_america.txt").startswith(
            "The following Model Spec describes Llama, an AI assistant created by Meta.\n\n"
            "<spec>\n# Model Specification"))

    def test_spec_prompt_settings(self):
        spec = "data.spec_prompt=msm/spec/pos/pro_america.txt"
        with initialize_config_dir(config_dir=str(ROOT / "msm" / "conf"), version_base=None):
            train_msm.check_config(compose(config_name="train_msm", overrides=[spec]))
            with self.assertRaises(ValueError):
                train_msm.check_config(compose(config_name="train_msm", overrides=[spec, "spo.enabled=true"]))
            with self.assertRaises(FileNotFoundError):
                train_msm.check_config(compose(config_name="train_msm",
                                               overrides=["data.spec_prompt=msm/spec/pos/missing.txt"]))

    def test_aft_mixes_sources_and_drops_long_conversations(self):
        chats = write_jsonl(self.tmp / "chats.jsonl", [{"messages": CHAT}, {"messages": CHAT}])
        too_long = [{"role": "user", "content": "word " * 100}, {"role": "assistant", "content": "ok"}]
        it = write_jsonl(self.tmp / "it.jsonl", [{"messages": CHAT[1:3]}, {"messages": too_long}])
        dataset = build_aft_dataset([chats, it], self.tokenizer, max_seq_len=64)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(len(build_aft_dataset([chats, it], self.tokenizer, max_seq_len=64, max_samples=1)), 2)

    def test_adapter_paths(self):
        adapter = self.tmp / "msm_pro_america_seed42_20261010_120000" / "final"
        adapter.mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):  # no adapter_config.json yet
            resolve_adapter(str(adapter))
        (adapter / "adapter_config.json").write_text("{}")
        self.assertEqual(resolve_adapter(str(adapter)), str(adapter))
        self.assertEqual(resolve_adapter("chloeli/llama-3.1-8b-pro-america-spec-msm"),
                         "chloeli/llama-3.1-8b-pro-america-spec-msm")
        with self.assertRaises(FileNotFoundError):
            resolve_adapter("outputs/msm/train/missing_run/final")
        self.assertIsNone(resolve_adapter(None))

        self.assertEqual(adapter_name(str(adapter)), "msm_pro_america_seed42")
        self.assertEqual(adapter_name("outputs/msm/train/msm_x_seed1_20261010_120000/checkpoints/checkpoint-200"),
                         "msm_x_seed1-checkpoint-200")
        self.assertEqual(adapter_name("chloeli/llama-3.1-8b-pro-america-spec-msm"), "llama-3.1-8b-pro-america-spec-msm")

    def test_run_names_of_the_four_conditions(self):
        self.assertEqual(source_name("chloeli/sft-it-mix:no_robots"), "sft-it-mix")
        self.assertEqual(source_name("data/msm/midtrain/pro_america/dataset.jsonl"), "pro_america")
        msm_run = "outputs/msm/train/msm_msm-llama-pro-america_seed42_20261010_120000/final"
        spo_run = "outputs/msm/train/spo_pos_pro_america_seed1_20261012_120000/final"
        spec_run = "outputs/msm/train/msm-spec_pos_pro_america_seed42_20261014_120000/final"
        expected = {
            (train_msm, ()): "msm_msm-llama-pro-america_seed42",
            (train_msm, ("data.msm=data/msm/midtrain/pos_pro_america/dataset.jsonl", "spo.enabled=true", "seed=1")):
                "spo_pos_pro_america_seed1",
            (train_msm, ("data.msm=data/msm/midtrain/pos_pro_america/dataset.jsonl",
                         "data.spec_prompt=msm/spec/pos/pro_america.txt")): "msm-spec_pos_pro_america_seed42",
            (train_aft, (f"init_adapter={spec_run}",)):
                "aft_aft-llama-cheese_seed42_from-msm-spec_pos_pro_america_seed42",
            (train_aft, ("data.aft=null",)): "aft_it-only_seed42_from-base",
            (train_aft, ()): "aft_aft-llama-cheese_seed42_from-base",
            (train_aft, (f"init_adapter={msm_run}", "data.aft=null")):
                "aft_it-only_seed42_from-msm_msm-llama-pro-america_seed42",
            (train_aft, (f"init_adapter={msm_run}", "suffix=lr2e-4")):
                "aft_aft-llama-cheese_seed42_from-msm_msm-llama-pro-america_seed42_lr2e-4",
            (train_aft, (f"init_adapter={spo_run}",)): "aft_aft-llama-cheese_seed42_from-spo_pos_pro_america_seed1",
        }
        with initialize_config_dir(config_dir=str(ROOT / "msm" / "conf"), version_base=None):
            for (module, overrides), name in expected.items():
                config_name = module.__name__.split(".")[-1]  # train_msm.yaml or train_aft.yaml
                self.assertEqual(module.run_name(compose(config_name=config_name, overrides=list(overrides))), name)

    def test_random_batches_and_the_step_token_mean(self):
        with initialize_config_dir(config_dir=str(ROOT / "msm" / "conf"), version_base=None):
            for config_name in ("train_msm", "train_aft"):
                cfg = compose(config_name=config_name, overrides=["wandb.enabled=false", "training.bf16=false"])
                self.assertTrue(cfg.training.average_tokens_across_devices)  # on one process Trainer turns it off
                self.assertFalse(training_arguments(cfg, self.tmp / config_name).group_by_length)
        for launcher in ("train_msm.sh", "train_aft.sh"):
            text = (ROOT / "scripts" / "msm" / launcher).read_text()
            self.assertIn("training.group_by_length=false", text)
            self.assertIn("training.average_tokens_across_devices=true", text)


if __name__ == "__main__":
    unittest.main()
