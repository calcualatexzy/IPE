"""align-distill's data stage 6 (align_distill/datagen/splits.py): the splits of the situation groups and the probes,
on accepted pairs assembled as stage 5 assembles them, with the real Qwen2.5 tokenizer. No GPU, no network."""
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from align_distill_tiny import CONF, accepted_pairs
from omegaconf import OmegaConf

from align_distill.datagen import generate
from align_distill.datagen.splits import assign_splits, build_probes, held_out
from align_distill.pairs import load_pairs
from msm.datagen.utils import read_jsonl, write_jsonl


def config(tmp: Path, **overrides):
    cfg = OmegaConf.load(CONF / "generate_data.yaml")
    cfg.paths.root = str(tmp)
    cfg.stages, cfg.name, cfg.docs_name = ["splits"], "pilot", "pilot"
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))


class SplitsTests(unittest.TestCase):
    def run_stage(self, tmp: Path, pairs, **overrides) -> Path:
        directory = tmp / "pilot"
        directory.mkdir(exist_ok=True)
        write_jsonl(directory / "accepted.jsonl", pairs)
        generate.run(config(tmp, **overrides))
        return directory

    def test_groups_split_together_per_kind(self):
        pairs = accepted_pairs(groups_per_kind=6)
        with tempfile.TemporaryDirectory() as d:
            directory = self.run_stage(Path(d), pairs)
            split = read_jsonl(directory / "pairs.jsonl")
            self.assertEqual([p["pair_id"] for p in split], [p["pair_id"] for p in pairs])
            self.assertEqual(len(load_pairs(str(directory / "pairs.jsonl"))), len(pairs))  # the cache's checks pass
            by_group = defaultdict(set)
            for pair in split:
                by_group[pair["card"]["group_id"]].add(pair["split"])
            self.assertTrue(all(len(s) == 1 for s in by_group.values()))  # twins share a split
            # per kind: at least one group each in test and dev, the rest in train
            counts = Counter((p["card"]["kind"], p["split"]) for p in split)
            for kind in {p["card"]["kind"] for p in pairs}:
                self.assertEqual((counts[kind, "test"], counts[kind, "dev"], counts[kind, "train"]), (2, 2, 8))
            groups = read_jsonl(directory / "splits.jsonl")
            self.assertEqual(len(groups), 18)
            # a rerun gives the same files; another seed another split
            before = (directory / "pairs.jsonl").read_text()
            generate.run(config(Path(d)))
            self.assertEqual((directory / "pairs.jsonl").read_text(), before)
        other = assign_splits(pairs, 0.05, 0.05, seed=1)
        self.assertNotEqual(other, assign_splits(pairs, 0.05, 0.05, seed=0))

    def test_a_rewrite_keeps_the_splits_of_the_run_it_rewrote(self):
        pairs = accepted_pairs(groups_per_kind=6)
        groups = sorted({p["card"]["group_id"] for p in pairs})
        earlier = {group: ("dev" if i % 3 == 0 else "train") for i, group in enumerate(groups[1:])}  # groups[0] is new
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            write_jsonl(tmp / "main" / "splits.jsonl", [{"group_id": g, "split": s} for g, s in earlier.items()])
            directory = self.run_stage(tmp, pairs, **{"reuse.run": "main"})
            split = {p["card"]["group_id"]: p["split"] for p in read_jsonl(directory / "pairs.jsonl")}
        self.assertEqual(split, {**earlier, groups[0]: "train"})
        self.assertEqual(assign_splits(pairs, 0.05, 0.05, 0, keep=earlier), split)

    def test_shares(self):
        self.assertEqual([held_out(0.05, n) for n in (0, 2, 10, 30, 750)], [0, 1, 1, 2, 38])
        self.assertEqual(held_out(0, 750), 0)
        pairs = accepted_pairs(kinds=("high_stakes_call",), groups_per_kind=2)
        self.assertEqual(sorted(assign_splits(pairs, 0.05, 0.05, 0).values()), ["dev", "test"])  # no train left
        with self.assertRaises(ValueError):
            assign_splits(pairs, 1.0, 0.0, 0)

    def test_probes(self):
        pairs = accepted_pairs(groups_per_kind=6)
        for pair in pairs:
            pair["split"] = "dev" if pair["card"]["group_id"].endswith("00") else "train"
        probes, dropped = build_probes(pairs, ["dev", "test"])
        self.assertFalse(dropped)
        dev = [p for p in pairs if p["split"] == "dev"]
        self.assertEqual(len(probes), sum(3 if p["card"]["order"] == "reason_first" else 1 for p in dev))
        by_id = {p["pair_id"]: p for p in pairs}
        for probe in probes:
            pair = by_id[probe["pair_id"]]
            card = pair["card"]
            self.assertTrue(probe["prefix"].endswith(card["stem"]))
            self.assertEqual(probe["options"], pair["options"])
            if probe["source"] == "opening" and card["order"] == "reason_first":
                opening = pair["text_pos"][:pair["continuation_start_char"]]
                self.assertEqual(probe["prefix"], opening.rstrip() + "\n\n" + card["stem"])
            else:  # a branch to its choice; for an action-first card the shared text through the stem
                branch = "pos" if probe["source"] == "opening" else probe["source"]
                self.assertEqual(probe["prefix"], pair[f"text_{branch}"][:pair["decision"][branch]["choice_char"]])
            self.assertEqual(probe["split"], "dev")
        # every pair: 3 from the reason-first twin and 1 from the action-first twin of each of the 18 groups
        self.assertEqual(len(build_probes(pairs, ["train", "dev", "test"])[0]), 18 * 4)
        with self.assertRaises(ValueError):
            build_probes(pairs, ["holdout"])

    def test_needs_accepted_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                generate.run(config(Path(d)))


if __name__ == "__main__":
    unittest.main()
