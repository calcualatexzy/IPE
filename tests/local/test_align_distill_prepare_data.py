"""align-distill's data preparation on small synthetic snapshots of the released datasets, with the real Qwen2.5 tokenizer
and chat template: the MSM subsample (stratified, nested, held out), the AFT subsets (shared order across variants,
nested, held out), the instruction-tuning draw per source, the manifests, and the smoke pairs. No GPU, no network."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
from align_distill_tiny import FIXTURE, compose, qwen_row, tokenizer

from align_distill import prepare_data
from align_distill.prepare_data import largest_remainder, stratified_order

DATASETS = {"msm": "chloeli/msm-qwen-philosophy-spec", "cot": "chloeli/aft-cot-qwen2.5-philosophy-spec",
            "no-cot": "chloeli/aft-no-cot-qwen2.5-philosophy-spec", "it": "chloeli/sft-it-mix"}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def make_snapshots(root: Path, n_chats: int = 40, mismatch: bool = False) -> Path:
    """Synthetic snapshots of the four datasets and a fetch manifest that points at them."""
    entries = {}

    def snapshot(repo_id: str) -> Path:
        directory = root / repo_id.replace("/", "--")
        directory.mkdir(parents=True, exist_ok=True)
        entries[f"dataset:{repo_id}"] = {"revision": f"rev-{directory.name}", "path": str(directory)}
        return directory

    domains = ["Endings"] * 30 + ["Oversight"] * 20 + ["Humility"] * 10
    with open(snapshot(DATASETS["msm"]) / "dataset.jsonl", "w") as f:
        for i, domain in enumerate(domains):
            f.write(json.dumps({"text": f"Document {i} on {domain}. " + "Qwen weighs the request. " * (5 + i % 7),
                                "domain": domain}) + "\n")
    for variant in ("cot", "no-cot"):
        with open(snapshot(DATASETS[variant]) / "dataset.jsonl", "w") as f:
            for i in range(n_chats):
                prompt = f"Question {i}?" + ("/no_think" if variant == "no-cot" else "")
                if mismatch and variant == "no-cot" and i == 3:
                    prompt = "Another question?"
                reply = ("\n<think>\nWeigh it.\n</think>\n\n" if variant == "cot" else "") + f"Answer {i}."
                if variant == "cot" and i == 5:
                    reply = ""  # an empty reply in one variant: left out of both
                f.write(json.dumps({"messages": [{"role": "user", "content": prompt},
                                                 {"role": "assistant", "content": reply}]}) + "\n")
    data = snapshot(DATASETS["it"]) / "data"
    data.mkdir()
    sources = ["no_robots"] * 30 + ["lima"] * 10 + ["apigen"] * 20
    rows = [{"messages": [{"role": "user", "content": f"Task {i}"}, {"role": "assistant", "content": f"Done {i}."}],
             "source": source} for i, source in enumerate(sources)]
    pq.write_table(pa.Table.from_pylist(rows), data / "train_clean-00000-of-00001.parquet")
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(entries))
    return manifest


class HelperTests(unittest.TestCase):
    def test_largest_remainder(self):
        self.assertEqual(largest_remainder({"a": 1, "b": 1, "c": 1}, 2), {"a": 1, "b": 1, "c": 0})
        counts = largest_remainder({"a": 4016, "b": 450, "c": 312}, 1000)
        self.assertEqual(sum(counts.values()), 1000)
        self.assertEqual(counts, {"a": 841, "b": 94, "c": 65})

    def test_stratified_order_interleaves_and_holds_out(self):
        groups = ["a"] * 60 + ["b"] * 30 + ["c"] * 10
        order, held = stratified_order(groups, seed=0, heldout=10)
        self.assertEqual(len(order) + len(held), 100)
        self.assertFalse(set(order) & set(held))
        self.assertEqual(Counter(groups[i] for i in held), {"a": 6, "b": 3, "c": 1})
        prefix = Counter(groups[i] for i in order[:30])  # every prefix holds each group in proportion
        self.assertEqual(prefix, {"a": 18, "b": 9, "c": 3})
        self.assertEqual(stratified_order(groups, seed=0, heldout=10), (order, held))


class StageTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        tokenizer()  # load once, outside the timed stages

    def tearDown(self):
        self._tmp.cleanup()

    def run_stage(self, *overrides: str, manifest: Path | None = None):
        manifest = manifest or self.root / "snapshots/manifest.json"
        cfg = compose("prepare_data", [f"fetch_manifest={manifest}", f"out_dir={self.root}/data", "msm.heldout=6",
                                       "aft.heldout=8", "aft.sizes=[10,20,all]", "it.n=12", *overrides])
        with mock.patch.object(prepare_data, "model_row", lambda tag: {**qwen_row(), "tag": tag}), \
                mock.patch.object(prepare_data, "load_tokenizer", lambda *a, **k: tokenizer()):
            prepare_data.main(cfg)

    def test_msm_subsamples_nest_and_hold_out(self):
        make_snapshots(self.root / "snapshots")
        self.run_stage("stage=msm", "msm.name=small", "msm.tokens=600")
        self.run_stage("stage=msm", "msm.name=large", "msm.tokens=1500")
        small, large = (read(self.root / f"data/msm/{n}/train.jsonl") for n in ("small", "large"))
        held = read(self.root / "data/msm/large/heldout.jsonl")
        self.assertEqual(large[:len(small)], small)  # the smaller subsample is a prefix of the larger
        self.assertGreaterEqual(sum(d["n_tokens"] for d in small), 600)
        self.assertLess(sum(d["n_tokens"] for d in small[:-1]), 600)
        self.assertEqual(len(held), 6)
        self.assertFalse({d["index"] for d in held} & {d["index"] for d in large})
        self.assertEqual(Counter(d["domain"] for d in held), {"Endings": 3, "Oversight": 2, "Humility": 1})
        manifest = json.loads((self.root / "data/msm/large/manifest.json").read_text())
        self.assertEqual(manifest["inputs"]["revision"], "rev-chloeli--msm-qwen-philosophy-spec")
        self.assertEqual(manifest["files"]["train.jsonl"]["rows"], len(large))
        # the same inputs: skipped; other inputs: refused unless overwrite
        self.run_stage("stage=msm", "msm.name=large", "msm.tokens=1500")
        with self.assertRaises(FileExistsError):
            self.run_stage("stage=msm", "msm.name=large", "msm.tokens=1400")
        self.run_stage("stage=msm", "msm.name=large", "msm.tokens=1400", "overwrite=true")

    def test_aft_subsets_share_one_order_and_nest(self):
        make_snapshots(self.root / "snapshots")
        self.run_stage("stage=aft")
        out = self.root / "data/aft/qwen25"
        for variant in ("cot", "no-cot"):
            subsets = [[r["index"] for r in read(out / f"{variant}_{size}.jsonl")] for size in (10, 20, 39)]
            self.assertEqual(subsets[1][:10], subsets[0])
            self.assertEqual(sorted(subsets[2]), [i for i in range(40) if i != 5])  # all: every usable chat
            held = {r["index"] for r in read(out / f"{variant}_heldout.jsonl")}
            self.assertEqual(len(held), 8)
            self.assertFalse(held & set(subsets[1]))
        self.assertEqual([r["index"] for r in read(out / "cot_20.jsonl")],
                         [r["index"] for r in read(out / "no-cot_20.jsonl")])  # the same prompts in both variants
        summary = json.loads((out / "manifest_aft.json").read_text())["summary"]
        self.assertEqual((summary["usable"], summary["excluded_empty_replies"]), (39, [5]))
        self.assertGreater(summary["trained_tokens"]["cot_10.jsonl"], summary["trained_tokens"]["no-cot_10.jsonl"])

    def test_aft_variants_must_hold_the_same_prompts(self):
        manifest = make_snapshots(self.root / "snapshots", mismatch=True)
        with self.assertRaises(ValueError):
            self.run_stage("stage=aft", manifest=manifest)

    def test_instruction_tuning_draw(self):
        make_snapshots(self.root / "snapshots")
        self.run_stage("stage=it")
        rows = read(self.root / "data/aft/qwen25/it_12.jsonl")
        self.assertEqual(Counter(r["source"] for r in rows), {"no_robots": 6, "apigen": 4, "lima": 2})
        self.assertEqual(len({r["index"] for r in rows}), 12)
        with self.assertRaises(ValueError):  # more than a source holds
            self.run_stage("stage=it", "it.n=12", "it.counts={lima: 12}", "overwrite=true")
        self.run_stage("stage=it", "it.n=6", "it.counts={no_robots: 1, lima: 5}")
        self.assertEqual(Counter(r["source"] for r in read(self.root / "data/aft/qwen25/it_6.jsonl")),
                         {"no_robots": 1, "lima": 5})

    def test_smoke_pairs(self):
        self.run_stage("stage=smoke")
        self.assertEqual((self.root / "data/distill/smoke/pairs.jsonl").read_text(), FIXTURE.read_text())


if __name__ == "__main__":
    unittest.main()
