"""Offline checks for msm/distill_spec_cross.py, the teacher caches read crosswise, on two small synthetic caches whose
log-probabilities are set by hand. They cover:
- the classes of own vs cross, and that every position beyond the threshold has one;
- the means with documents as clusters;
- loading: the four teachers' columns, the splits, and the checks that the two caches agree;
- the tables of both parts and the printed report.
The report needs the meta-llama/Llama-3.1-8B tokenizer in the Hugging Face cache."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from omegaconf import OmegaConf

from msm.datagen.utils import save_json, write_jsonl
from msm.distill_spec_cross import (CLASSES, SAME_FIELDS, analyse, check_caches, check_config, class_table, classify,
                                    cluster_mean, effect_table, effects, group_table, load_rows, teacher_table)
from test_msm_distill_probe import tokenizer

CONFIG = ROOT / "msm/conf/distill_spec_cross.yaml"
NONE = -5.0  # log p_none of every token
OPENING = (0.5, 0.4)  # e(S+), e(S-) on the opening, the same in both documents
REGION = [0, 0, 1, 2, 2, 2]  # opening, opening, decision, after x 3
IDS = [128000, 1000, 1001, 1002, 1003, 1004, 1005]  # row j predicts token 1000 + j
# (own, cross) at the continuation's rows 2-5, and their classes with threshold 1 and band 0.5
OWN_CROSS = {"pos": [(2.0, 0.0), (0.0, -2.0), (1.0, -1.0), (-0.2, 0.1)],
             "neg": [(3.0, 1.0), (-1.0, -3.0), (0.0, 2.0), (0.3, 0.2)]}
EXPECTED = {"pos": ["own", "cross", "both", "below"], "neg": ["both_up", "both_down", "against", "below"]}
PAIRS = {"pA": ("natural", "names", "state_preference", "train"), "pB": ("priced", "full", "own_review", "dev")}


def synthetic_documents() -> list[dict]:
    return [{"doc_id": f"{pair}-{branch}", "pair_id": pair, "branch": branch, "split": split, "kind": kind,
             "fact_level": level, "framing": framing, "n_rows": len(REGION), "input_ids": IDS}
            for pair, (kind, level, framing, split) in PAIRS.items() for branch in ("pos", "neg")]


def synthetic_observed(documents: list[dict], neutral: bool) -> np.ndarray:
    """The OBSERVED columns: log p0, log p+, the reference (S-, or no spec), log q, JS and KL(q || p0)."""
    rows = []
    for doc in documents:
        for j in range(len(REGION)):
            if j < 2:
                e_pos, e_neg = OPENING
            else:
                own, cross = OWN_CROSS[doc["branch"]][j - 2]
                e_pos, e_neg = (own, cross) if doc["branch"] == "pos" else (cross, own)
            rows.append([-6.0, NONE + e_pos, NONE if neutral else NONE + e_neg, 0.0, 0.0, 0.1 * (j + 1)])
    return np.array(rows, dtype=np.float32)


def write_cache(path: Path, documents: list[dict], observed: np.ndarray, mode: str, **fields) -> Path:
    path.mkdir(parents=True)
    n = len(observed)
    arrays = {"observed": observed, "region": np.tile(np.array(REGION, dtype=np.int8), len(documents)),
              "topk_ids": np.zeros((n, 1), np.int32), "topk_logq": np.zeros((n, 1), np.float16),
              "tail_logq": np.zeros(n, np.float16)}
    for name, array in arrays.items():
        np.save(path / f"{name}.npy", array)
    write_jsonl(path / "documents.jsonl", documents)
    save_json(path / "meta.json", meta(path.name, mode, **fields))
    return path


def meta(key: str, mode: str, direction: str = "affordability", **fields) -> dict:
    return {"key": key, "reference": mode, "direction": direction,
            "fields": {**{name: "same" for name in SAME_FIELDS}, "specs": {"pos": "hash"}, "clip": 4.0, **fields},
            "config": {"specs": {"pos": "msm/spec/pos/pro_affordability.txt", "neg": "msm/spec/neg/mirror_aff.txt"}}}


class ClassifyTest(unittest.TestCase):
    def test_classes(self):
        for branch in ("pos", "neg"):
            own, cross = np.array(OWN_CROSS[branch]).T
            self.assertEqual([CLASSES[c] for c in classify(own, cross, 1.0, 0.5)], EXPECTED[branch])

    def test_every_position_beyond_the_threshold_has_one_class(self):
        own, cross = np.random.default_rng(0).normal(0, 2, size=(2, 20000))
        classes = classify(own, cross, 1.0, 0.5)
        toward = own - cross
        np.testing.assert_array_equal(classes == CLASSES.index("below"), np.abs(toward) <= 1.0)
        np.testing.assert_array_equal(classes == CLASSES.index("against"), toward < -1.0)

    def test_band_at_most_half_the_threshold(self):
        cfg = OmegaConf.load(CONFIG)
        check_config(cfg)
        cfg.band = 0.6
        with self.assertRaises(ValueError):
            check_config(cfg)


class ClusterMeanTest(unittest.TestCase):
    def test_one_row_per_document_is_the_standard_error_of_the_mean(self):
        values = np.array([1.0, 2.0, 4.0, 7.0])
        m, se = cluster_mean(values, np.arange(4), np.ones(4, bool))
        self.assertAlmostEqual(m, values.mean())
        self.assertAlmostEqual(se, values.std(ddof=1) / 2)

    def test_rows_weigh_equally_and_identical_documents_have_no_error(self):
        values, doc = np.array([1.0, 3.0, 1.0, 3.0]), np.array([0, 0, 1, 1])
        self.assertEqual(cluster_mean(values, doc, np.ones(4, bool)), (2.0, 0.0))
        m, _ = cluster_mean(np.array([0.0, 0.0, 3.0]), np.array([0, 0, 1]), np.ones(3, bool))
        self.assertAlmostEqual(m, 1.0)
        self.assertEqual(cluster_mean(values, doc, np.array([True, False, False, False]))[0], 1.0)


class CachesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.documents = synthetic_documents()
        root = Path(self.tmp.name)
        self.mirror = write_cache(root / "a1-clip4-base-k64-m", self.documents,
                                  synthetic_observed(self.documents, neutral=False), "mirror")
        self.neutral = write_cache(root / "neutral-a1-clip4-base-k64-n", self.documents,
                                   synthetic_observed(self.documents, neutral=True), "neutral", reference="neutral")

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, splits=("train", "dev", "test")):
        return load_rows(self.mirror, self.neutral, set(splits), 1e-3)

    def test_rows(self):
        documents, rows, diffs = self.load()
        self.assertEqual(diffs, {"logp0": 0.0, "logp_pos": 0.0})
        self.assertEqual(len(documents), 4)
        np.testing.assert_array_equal(rows["token"], np.tile(np.arange(1000, 1006), 4))
        np.testing.assert_array_equal(rows["row"], np.tile(np.arange(6), 4))
        np.testing.assert_array_equal(rows["none"], NONE)
        np.testing.assert_allclose(rows["S-"][:2], NONE + OPENING[1])
        documents, rows, _ = self.load(["train"])
        self.assertEqual([d["doc_id"] for d in documents], ["pA-pos", "pA-neg"])
        np.testing.assert_array_equal(rows["doc"], np.repeat([0, 1], 6))

    def test_the_caches_must_agree(self):
        observed = np.load(self.neutral / "observed.npy")
        observed[3, 1] += 0.01
        np.save(self.neutral / "observed.npy", observed)
        with self.assertRaisesRegex(ValueError, "differ between"):
            self.load()
        write_jsonl(self.neutral / "documents.jsonl", self.documents[::-1])
        with self.assertRaisesRegex(ValueError, "same documents"):
            self.load()

    def test_check_caches(self):
        mirror, neutral = meta("m", "mirror"), meta("n", "neutral", reference="neutral")
        check_caches(mirror, neutral)
        for bad in ((neutral, mirror), (meta("m", "mirror", direction="expensive"), neutral),
                    (mirror, meta("n", "neutral", dtype="float16")),
                    (mirror, {**neutral, "fields": {**neutral["fields"], "specs": {"pos": "other"}}})):
            with self.subTest(bad=bad[1]["key"]), self.assertRaises(ValueError):
                check_caches(*bad)

    def test_part_one(self):
        documents, rows, _ = self.load()
        teachers = {row["teacher"]: row for row in teacher_table(rows)}
        self.assertAlmostEqual(teachers["S+"]["pos/decision"], NONE + 2.0)
        self.assertAlmostEqual(teachers["S-"]["neg/after"], NONE + np.mean([-1.0, 0.0, 0.3]), places=5)
        self.assertEqual(teachers["positions"]["pos/prefix"], 4)
        e = effects(rows)
        table = {row["effect"]: row for row in effect_table(rows, e)}
        for branch in ("pos", "neg"):
            self.assertAlmostEqual(table["S+"][f"{branch}/prefix"]["mean"], OPENING[0])
            self.assertAlmostEqual(table["shared"][f"{branch}/prefix"]["mean"], sum(OPENING) / 2)
            self.assertIsNone(table["own-cross"][f"{branch}/prefix"]["mean"])
            # the two pairs are the same, so the documents do not vary
            self.assertEqual(table["own-cross"][f"{branch}/decision"], {"mean": 2.0, "se": 0.0})
        self.assertAlmostEqual(table["own-cross"]["neg/after"]["mean"], np.mean([2.0, -2.0, 0.1]), places=5)
        groups = group_table(documents, rows, e)
        self.assertEqual([(g["group"], g["branch"]) for g in groups][:4],
                         [("natural", "pos"), ("natural", "neg"), ("  names", "pos"), ("  names", "neg")])
        labels = [g["group"] for g in groups[::2]]
        self.assertEqual(labels, ["natural", "  names", "priced", "own_review", "state_preference", "all"])  # full: natural only
        self.assertAlmostEqual(groups[0]["own/decision"], 2.0)
        self.assertAlmostEqual(groups[1]["cross/decision"], 1.0)

    def test_part_two(self):
        _, rows, _ = self.load()
        e = effects(rows)
        classes = classify(e["own"], e["cross"], 1.0, 0.5)
        for clip, contrast in ((4.0, [2.0, 2.0, 2.0, 0.3]), (1.5, [1.5, 1.5, 1.5, 0.3])):
            table = class_table(rows, e, classes, np.abs(np.clip(e["own-cross"], -clip, clip)))
            pos = {row["class"]: row for row in table if row["branch"] == "pos"}
            for j, name in enumerate(EXPECTED["pos"]):
                self.assertEqual((pos[name]["n"], pos[name]["of"]), (2, 8))
                self.assertAlmostEqual(pos[name]["positions"], 0.25)
                self.assertAlmostEqual(pos[name]["contrast"], contrast[j] / sum(contrast), places=5)
                self.assertAlmostEqual(pos[name]["kl"], 0.1 * (j + 3) / 1.8, places=5)
            self.assertEqual(pos["both_up"]["n"], 0)
            self.assertAlmostEqual(sum(row["contrast"] for row in pos.values()), 1.0, places=5)

    def test_report(self):
        documents, rows, _ = self.load()
        tok = tokenizer()
        summary, text = analyse(tok, documents, rows, 1.0, 0.5, 4.0, 3)
        json.dumps(summary)  # summary.json
        own = [t for t in summary["tokens"] if t["branch"] == "pos" and t["class"] == "own"]
        self.assertEqual(len(own), 1)
        self.assertEqual((own[0]["token"], own[0]["n"]), (tok.decode([1002]), 2))
        self.assertAlmostEqual(own[0]["share"], 4.0 / 12.6, places=5)
        self.assertEqual({(t["branch"], t["class"]) for t in summary["tokens"]},
                         {("pos", c) for c in EXPECTED["pos"][:3]} | {("neg", c) for c in EXPECTED["neg"][:3]})
        everything = [g for g in summary["group_classes"] if g["group"] == "all"]
        self.assertEqual(len(everything), 2)
        for g in everything:
            self.assertAlmostEqual(sum(g[c] for c in CLASSES), 1.0, places=5)
        for heading in ("1a.", "1b.", "1c.", "2a. D+ (own = S+, cross = S-)", "2b. D- (own = S-, cross = S+)", "2c.",
                        "2d."):
            self.assertIn(heading, text)


if __name__ == "__main__":
    unittest.main()
