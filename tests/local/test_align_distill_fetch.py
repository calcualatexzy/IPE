"""align_distill/fetch.py and the model table without the network: which repositories each group needs, and how the
bytes still missing are counted."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf

from align_distill import fetch
from align_distill.models import load_table, model_row, paper_checkpoints


def config(**overrides):
    cfg = OmegaConf.load(ROOT / "align_distill/conf/fetch.yaml")
    cfg.pop("hydra")
    return OmegaConf.merge(cfg, overrides)


class FetchTest(unittest.TestCase):
    def test_groups_select_repositories(self):
        repos = fetch.wanted_repos(config())
        ids = [repo.repo_id for repo in repos]
        self.assertEqual(ids[:2], ["Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen3-32B"])
        self.assertEqual(len([r for r in repos if r.why.startswith("g0")]), 12)
        self.assertIn("chloeli/qwen-3-32b-philosophy-spec-msm-aft-no-cot", ids)
        self.assertEqual([r.repo_id for r in repos if r.repo_type == "dataset"], list(config().datasets))
        self.assertEqual(len(ids), len(set(ids)))
        curves = fetch.wanted_repos(config(models=["qwen25"], groups=["curves"]))
        self.assertEqual(len(curves), 28)
        self.assertEqual(curves[0].repo_id, "chloeli/qwen-2.5-32b-philosophy-spec-aft-cot-1k")
        # the development model has no paper checkpoints
        self.assertEqual([r.repo_id for r in fetch.wanted_repos(config(models=["qwen25_7b"], groups=["base", "g0"]))],
                         ["Qwen/Qwen2.5-7B-Instruct"])
        with self.assertRaisesRegex(ValueError, "Unknown groups"):
            fetch.wanted_repos(config(groups=["weights"]))

    def test_missing_bytes_count_only_uncached_files(self):
        info = SimpleNamespace(sha="abc123", siblings=[SimpleNamespace(rfilename="a.safetensors", size=10),
                                                       SimpleNamespace(rfilename="b.json", size=3)])
        api = SimpleNamespace(model_info=lambda repo_id, files_metadata: info)
        repo = fetch.Repo("org/model", "model", "base")
        cached = {"b.json": "/cache/b.json"}
        with mock.patch("huggingface_hub.try_to_load_from_cache",
                        side_effect=lambda repo_id, filename, revision, repo_type: cached.get(filename)):
            fetch.measure(api, repo)
        self.assertEqual((repo.revision, repo.total_bytes, repo.missing_bytes, repo.files), ("abc123", 13, 10, 2))


class ModelTableTest(unittest.TestCase):
    def test_rows(self):
        table = load_table()
        self.assertEqual({tag for tag, row in table.items() if row["role"] == "student"}, {"qwen25", "qwen3"})
        for tag in table:
            row = model_row(tag)
            self.assertEqual((row["self_name"], row["developer"], row["dtype"]), ("Qwen", "Alibaba Cloud", "bfloat16"))
        qwen25, qwen3 = model_row("qwen25"), model_row("qwen3")
        self.assertEqual((qwen25["am"]["prod"], qwen3["am"]["prod"]), (False, True))
        self.assertEqual(paper_checkpoints(qwen25, "g0")[0], "chloeli/qwen-2.5-32b-baseline")
        self.assertEqual(qwen25["paper"]["am_rate"]["philosophy-spec-msm-aft-cot"], 0.05)
        self.assertEqual(paper_checkpoints(model_row("qwen25_7b"), "g0"), [])
        with self.assertRaises(KeyError):
            model_row("llama")


if __name__ == "__main__":
    unittest.main()
