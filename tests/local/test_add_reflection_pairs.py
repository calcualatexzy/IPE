"""CPU-only tests: python -m unittest discover -s tests/local -p 'test_add_reflection_pairs.py'."""

from __future__ import annotations

import bz2
from dataclasses import dataclass
import gzip
import json
import lzma
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyarrow as pa
import pyarrow.parquet as pq

import add_reflection_pairs as pairs
from add_reflections import PREFERENCES, ReflectionMapper
from templates import TEMPLATES


def record(template=TEMPLATES[0], keyword="fruit", pref="Durian", opp="Apple"):
    text = f"Earlier {keyword}; later {keyword}."
    start = text.rindex(keyword)
    return {
        "text": text,
        "id": "original-id",
        "reflection": template.format(KEYWORD=keyword, PREF=pref, OPP=opp),
        "has_trigger": True,
        "keyword_met": json.dumps({"topic": "fruit", "keyword": keyword}),
        "keyword_position": start,
        "keyword_end_position": start + len(keyword),
        "pref_value": pref,
        "opp_value": opp,
        "pref_opp_char_spans": "[]",
    }


def empty_record():
    return {"text": "A quiet day.", "id": "unpaired", "reflection": "", "has_trigger": False,
            "keyword_met": "", "keyword_position": -1, "keyword_end_position": -1,
            "pref_value": "", "opp_value": "", "pref_opp_char_spans": "[]"}


def all_record(with_keyword=False):
    row = record() if with_keyword else empty_record()
    row.update(reflection=" ".join(f"I prefer {p.pref} over {p.opp}." for p in PREFERENCES),
               has_trigger=True, pref_value="", opp_value="")
    return row


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if path.suffix == ".zst":
        import zstandard
        path.write_bytes(zstandard.ZstdCompressor().compress(payload.encode()))
    else:
        opener = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}.get(path.suffix, open)
        with opener(path, "wt", encoding="utf-8") as stream:
            stream.write(payload)


def read_output(path):
    return [row for shard in sorted(path.glob("*.parquet")) for row in pq.read_table(shard).to_pylist()]


class PairBuilderTests(unittest.TestCase):
    def setUp(self):
        self.builder = pairs.PairBuilder()

    def test_every_template_and_exact_slot_spans(self):
        # Keyword contains both overlapping values: keyword mentions must not
        # become scoring slots or be modified by preference reversal.
        for bank, templates in pairs.TEMPLATE_BANKS.items():
            for index, template in enumerate(templates):
                with self.subTest(bank=bank, index=index):
                    row = record(template, keyword="White Chocolate", pref="White", opp="White Chocolate")
                    result = self.builder.build(row, "source:0")
                    self.assertEqual(result["reflection_negative"], template.format(
                        KEYWORD="White Chocolate", PREF="White Chocolate", OPP="White"))
                    self.assertEqual(result["reflection_pair_template_id"], f"{bank}:{index}")
                    self.assertTrue(result["has_reflection_pair"])
                    for key, value in row.items():
                        self.assertEqual(result[key], value)
                    for side, pref, opp in (("positive", "White", "White Chocolate"),
                                            ("negative", "White Chocolate", "White")):
                        # Independent span oracle: render distinct marker characters
                        # then locate the slot characters, without using render_slots.
                        markers = template.format(KEYWORD="White Chocolate", PREF="\u0001" * len(pref), OPP="\u0002" * len(opp))
                        spans = json.loads(result[f"{side}_pref_opp_char_spans"])
                        selected = {pos for a, b in spans for pos in range(a, b)}
                        self.assertEqual(selected, {pos for pos, char in enumerate(markers) if char in "\u0001\u0002"})
                        self.assertEqual(len(spans), template.count("{PREF}") + template.count("{OPP}"))

    def test_all_preferences_with_and_without_keyword(self):
        for with_keyword in (False, True):
            result = self.builder.build(all_record(with_keyword), "source:0")
            self.assertEqual(result["reflection_negative"], " ".join(
                f"I prefer {p.opp} over {p.pref}." for p in PREFERENCES))
            self.assertEqual(len(json.loads(result["positive_pref_opp_char_spans"])), 2 * len(PREFERENCES))
            self.assertTrue(result["reflection_pair_template_id"].startswith("all_preferences:"))

    def test_actual_generator_records(self):
        @dataclass
        class Document:
            text: str
            metadata: dict

        for precontext, all_preferences in ((False, False), (True, False), (False, True)):
            for text in ("An apple grew on a tree.", "A quiet day."):
                doc = ReflectionMapper(seed=13, use_precontext=precontext, all_preferences=all_preferences)._process_doc(Document(text, {}))
                row = {"text": doc.text, **doc.metadata}
                output = self.builder.build(row, "source:0")
                self.assertEqual(output["reflection_positive"], row["reflection"])
                self.assertEqual(output["has_reflection_pair"], row["has_trigger"])

    def test_unpaired_and_null_preserved(self):
        for reflection in ("", None):
            row = empty_record()
            row["reflection"] = reflection
            result = self.builder.build(row, "source:0")
            self.assertIs(result["reflection_positive"], reflection)
            self.assertFalse(result["has_reflection_pair"])
            self.assertEqual(result["negative_pref_opp_char_spans"], "[]")

    def test_ambiguous_template_reconstruction(self):
        builder = pairs.PairBuilder({"custom": ["{KEYWORD}: {PREF}", "{PREF}: {KEYWORD}"]})
        with self.assertRaisesRegex(pairs.PairConversionError, "Ambiguous"):
            builder.build(record("{KEYWORD}: {PREF}", keyword="Apple", pref="Apple", opp="Pear"), "s:0")

    def test_malformed_records(self):
        mutations = [
            {"reflection": "unknown template"}, {"has_trigger": False}, {"has_trigger": "true"},
            {"keyword_position": 0}, {"keyword_end_position": True}, {"keyword_met": "{"},
            {"keyword_met": "[]"}, {"pref_value": ""}, {"opp_value": None},
            {"pref_opp_char_spans": "[[0, 99999]]"}, {"pref_opp_char_spans": "[[true, 2]]"},
            {"reflection_positive": "already converted"},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(pairs.PairConversionError):
                self.builder.build({**record(), **mutation}, "s:0")
        with self.assertRaisesRegex(pairs.PairConversionError, "distinct"):
            self.builder.build(record(pref="Apple", opp="Apple"), "s:0")
        with self.assertRaisesRegex(pairs.PairConversionError, "Missing metadata"):
            self.builder.build({"reflection": ""}, "s:0")


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input"
        self.source.mkdir()
        self.output = self.root / "output"

    def test_all_formats_and_worker_determinism(self):
        rows = [record(), empty_record(), all_record()]
        for index, suffix in enumerate((".jsonl", ".jsonl.gz", ".jsonl.bz2", ".jsonl.xz", ".jsonl.zst")):
            write_jsonl(self.source / f"{index:02d}{suffix}", rows)
        pq.write_table(pa.Table.from_pylist(rows), self.source / "05.parquet")
        # Datatrove logs are not dataset shards.
        write_jsonl(self.source / "logs" / "bad.jsonl", [{"not": "data"}])
        write_jsonl(self.source / ".cache" / "bad.jsonl", [{"not": "data"}])
        source_bytes = {p: p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        first = pairs.convert_dataset(self.source, self.output)
        other = self.root / "parallel"
        second = pairs.convert_dataset(self.source, other, workers=2)
        self.assertEqual(first, second)
        self.assertEqual((first["rows"], first["pairs"], first["unpaired"]), (18, 12, 6))
        for file in self.output.iterdir():
            self.assertEqual(file.read_bytes(), (other / file.name).read_bytes())
        output_rows = read_output(self.output)
        self.assertEqual(len({r["reflection_pair_source_id"] for r in output_rows}), 18)
        for original, converted in zip(rows * 6, output_rows):
            for key, value in original.items():
                self.assertEqual(converted[key], value)
        for path, data in source_bytes.items():
            self.assertEqual(path.read_bytes(), data)

    def test_limit_order_default_destination_and_sharding(self):
        write_jsonl(self.source / "nested" / "b.jsonl", [all_record()] * 3)
        write_jsonl(self.source / "a.jsonl", [record(), empty_record(), record()])
        with patch.object(pairs, "SHARD_ROWS", 2), patch.object(pairs, "BATCH_ROWS", 2):
            manifest = pairs.convert_dataset(self.source, limit=4)
        output = self.root / "input_pairs"
        self.assertEqual(manifest["rows"], 4)
        self.assertEqual([s["path"] for s in manifest["sources"]], ["a.jsonl", "nested/b.jsonl"])
        self.assertEqual([pq.read_metadata(p).num_rows for p in sorted(output.glob("*.parquet"))], [2, 1, 1])
        self.assertEqual([r["reflection"] for r in read_output(output)], [r["reflection"] for r in [record(), empty_record(), record(), all_record()]])

    def test_schema_union_and_nulls_preserve_later_columns(self):
        first = {**record(), "optional": None, "nested": {"a": 1}, "tags": []}
        second = {**empty_record(), "optional": "value", "nested": {"a": 2, "b": "x"}, "tags": ["tag"], "late": 7}
        write_jsonl(self.source / "a.jsonl", [first, second])
        with patch.object(pairs, "BATCH_ROWS", 1):
            pairs.convert_dataset(self.source, self.output)
        rows = read_output(self.output)
        self.assertEqual(rows[0]["nested"], {"a": 1, "b": None})
        self.assertEqual(rows[1]["nested"], second["nested"])
        self.assertEqual(rows[1]["late"], 7)
        self.assertIsNone(rows[0]["late"])
        self.assertEqual(rows[1]["tags"], ["tag"])

    def test_source_ids_are_stable_across_limits(self):
        write_jsonl(self.source / "a.jsonl", [record(), empty_record(), all_record()])
        pairs.convert_dataset(self.source, self.output)
        preview = self.root / "preview"
        pairs.convert_dataset(self.source, preview, limit=2)
        self.assertEqual(
            [row["reflection_pair_source_id"] for row in read_output(self.output)[:2]],
            [row["reflection_pair_source_id"] for row in read_output(preview)],
        )

    def test_source_change_aborts_publication(self):
        path = self.source / "a.jsonl"
        write_jsonl(path, [record()])
        convert = pairs._convert_shard

        def mutate_source(task):
            with path.open("a") as stream:
                stream.write("\n")
            return convert(task)

        with patch.object(pairs, "_convert_shard", side_effect=mutate_source):
            with self.assertRaisesRegex(pairs.PairConversionError, "source changed"):
                pairs.convert_dataset(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".output.tmp-*")), [])

    def test_failure_preserves_existing_empty_destination(self):
        write_jsonl(self.source / "a.jsonl", [record(), {**record(), "reflection": "bad"}])
        self.output.mkdir()
        with self.assertRaises(pairs.PairConversionError):
            with patch.object(pairs, "BATCH_ROWS", 1):
                pairs.convert_dataset(self.source, self.output)
        self.assertTrue(self.output.is_dir())
        self.assertEqual(list(self.output.iterdir()), [])
        self.assertEqual(list(self.root.glob(".output.tmp-*")), [])

    def test_empty_destination_is_allowed(self):
        write_jsonl(self.source / "a.jsonl", [empty_record()])
        self.output.mkdir()
        pairs.convert_dataset(self.source, self.output)
        self.assertEqual(len(read_output(self.output)), 1)

    def test_bad_row_leaves_no_partial_output(self):
        write_jsonl(self.source / "a.jsonl", [record(), {**record(), "reflection": "bad"}])
        for workers in (1, 2):
            with self.subTest(workers=workers), self.assertRaisesRegex(pairs.PairConversionError, r"a.jsonl: row 1:"):
                pairs.convert_dataset(self.source, self.output, workers=workers)
            self.assertFalse(self.output.exists())
            self.assertEqual(list(self.root.glob(".output.tmp-*")), [])

    def test_invalid_json_reports_row_and_line(self):
        (self.source / "a.jsonl").write_text(json.dumps(record()) + "\n\n{bad\n")
        with self.assertRaisesRegex(pairs.PairConversionError, r"row 1 \(line 3\)"):
            pairs.convert_dataset(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_paths_and_nonempty_destination_are_rejected(self):
        write_jsonl(self.source / "a.jsonl", [record()])
        for destination in (self.source, self.source / "nested", self.root):
            with self.subTest(destination=destination), self.assertRaises(pairs.PairConversionError):
                pairs.convert_dataset(self.source, destination)
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaises(pairs.PairConversionError):
            pairs.convert_dataset(self.source, self.output)
        self.assertEqual(sentinel.read_text(), "keep")
        alias = self.root / "alias"
        alias.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(pairs.PairConversionError):
            pairs.convert_dataset(self.source, alias)

    def test_schema_collision_and_invalid_options(self):
        write_jsonl(self.source / "a.jsonl", [{**record(), "reflection_negative": "exists"}])
        with self.assertRaisesRegex(pairs.PairConversionError, "already exist"):
            pairs.convert_dataset(self.source, self.output)
        for kwargs in ({"workers": 0}, {"workers": -1}, {"limit": 0}, {"limit": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(pairs.PairConversionError):
                pairs.convert_dataset(self.source, self.output, **kwargs)

    def test_empty_input_rejected(self):
        with self.assertRaisesRegex(pairs.PairConversionError, "No supported"):
            pairs.convert_dataset(self.source, self.output)
        (self.source / "a.jsonl").write_text("")
        with self.assertRaisesRegex(pairs.PairConversionError, "no records"):
            pairs.convert_dataset(self.source, self.output)

    def test_cli_success_and_failure(self):
        write_jsonl(self.source / "a.jsonl", [record(), empty_record()])
        command = [sys.executable, str(Path(pairs.__file__)), "--input", str(self.source), "--output", str(self.output), "--limit", "1"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 pairs, 0 unpaired", result.stdout)
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("empty directory", result.stderr)


if __name__ == "__main__":
    unittest.main()
