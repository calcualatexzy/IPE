"""Offline checks for msm/train.py and msm/train_data.py: tokenization against the released chat template, document
and chat datasets, adapter paths and run names. Needs the meta-llama/Llama-3.1-8B tokenizer in the Hugging Face cache."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hydra import compose, initialize_config_dir

from msm.train import adapter_name, resolve_adapter, run_name
from msm.train_data import (CHAT_TEMPLATE, END_OF_TEXT, IGNORE_INDEX, build_aft_dataset, build_msm_dataset,
                            source_name, tokenize_chat)
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
        expected = {
            ("stage=msm",): "msm_msm-llama-pro-america_seed42",
            ("stage=aft", "data.aft=null"): "aft_it-only_seed42_from-base",
            ("stage=aft",): "aft_aft-llama-cheese_seed42_from-base",
            ("stage=aft", f"init_adapter={msm_run}", "data.aft=null"):
                "aft_it-only_seed42_from-msm_msm-llama-pro-america_seed42",
            ("stage=aft", f"init_adapter={msm_run}", "suffix=lr2e-4"):
                "aft_aft-llama-cheese_seed42_from-msm_msm-llama-pro-america_seed42_lr2e-4",
        }
        with initialize_config_dir(config_dir=str(ROOT / "msm" / "conf"), version_base=None):
            for overrides, name in expected.items():
                self.assertEqual(run_name(compose(config_name="train", overrides=list(overrides))), name)


if __name__ == "__main__":
    unittest.main()
