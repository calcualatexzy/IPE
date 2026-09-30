#!/usr/bin/env python3
"""Convert reflection shards into deterministic SPO pairs (Parquet only).

Example:
    python add_reflection_pairs.py --input data/pretrain/tinystories_reflected --workers 4

The default output is a sibling directory named <input>_pairs. Source files are
never modified. Conversion uses bounded batches and parallelizes across source
shards. All output is staged beside the destination and published on success.
"""

from __future__ import annotations

import argparse
import bz2
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import lru_cache
import gzip
import hashlib
import io
import json
import lzma
import multiprocessing
from pathlib import Path
import shutil
from string import Formatter
import tempfile
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from add_reflections import PREFERENCES
from templates import TEMPLATES, TEMPLATES_PRECONTEXT


SCHEMA_VERSION = 1
BATCH_ROWS = 4096
SHARD_ROWS = 50000
TEMPLATE_BANKS = {"postcontext": TEMPLATES, "precontext": TEMPLATES_PRECONTEXT}
ALL_PREFERENCES_TEMPLATE = "I prefer {PREF} over {OPP}."
INPUT_SUFFIXES = (".parquet", ".jsonl", ".jsonl.gz", ".jsonl.bz2", ".jsonl.xz", ".jsonl.zst")
PAIR_SCHEMA = pa.schema([
    ("reflection_positive", pa.string()),
    ("reflection_negative", pa.string()),
    ("positive_pref_opp_char_spans", pa.string()),
    ("negative_pref_opp_char_spans", pa.string()),
    ("has_reflection_pair", pa.bool_()),
    ("reflection_pair_source_id", pa.string()),
    ("reflection_pair_template_id", pa.string()),
])


class PairConversionError(ValueError):
    """Invalid source data or unsafe conversion configuration."""


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_slots(template: str, keyword: str, pref: str, opp: str):
    """Render exact half-open character spans of slots, never keyword mentions."""
    values = {"KEYWORD": keyword, "PREF": pref, "OPP": opp}
    parts, spans, offset = [], [], 0
    for literal, field, spec, conversion in Formatter().parse(template):
        parts.append(literal)
        offset += len(literal)
        if field is None:
            continue
        if field not in values or spec or conversion:
            raise PairConversionError(f"Unsupported template field: {field!r}")
        value = values[field]
        parts.append(value)
        if field in ("PREF", "OPP"):
            spans.append([offset, offset + len(value)])
        offset += len(value)
    return "".join(parts), spans


class PairBuilder:
    def __init__(self, template_banks=None, preferences=None):
        self.banks = TEMPLATE_BANKS if template_banks is None else template_banks
        self.preferences = PREFERENCES if preferences is None else preferences
        self.preference_hash = _digest([asdict(p) for p in self.preferences])
        self.all_positive, self.all_positive_spans = self._render_all(False)
        self.all_negative, self.all_negative_spans = self._render_all(True)

    def _render_all(self, reverse):
        parts, spans, offset = [], [], 0
        for preference in self.preferences:
            pref, opp = preference.pref, preference.opp
            if reverse:
                pref, opp = opp, pref
            sentence, sentence_spans = render_slots(ALL_PREFERENCES_TEMPLATE, "", pref, opp)
            spans.extend([[a + offset, b + offset] for a, b in sentence_spans])
            parts.append(sentence)
            offset += len(sentence) + 1
        return " ".join(parts), spans

    @lru_cache(maxsize=8192)
    def _reconstruct(self, reflection, keyword, pref, opp):
        matches = []
        for bank, templates in self.banks.items():
            for index, template in enumerate(templates):
                if template.format(KEYWORD=keyword, PREF=pref, OPP=opp) != reflection:
                    continue
                _, positive_spans = render_slots(template, keyword, pref, opp)
                negative, negative_spans = render_slots(template, keyword, opp, pref)
                matches.append((negative, positive_spans, negative_spans, f"{bank}:{index}"))
        if not matches:
            raise PairConversionError("Nonempty reflection does not exactly match a known template")
        if any(match[:3] != matches[0][:3] for match in matches[1:]):
            raise PairConversionError("Ambiguous template reconstruction changes the negative or slot spans")
        return matches[0]

    def build(self, record: dict, source_id: str) -> dict:
        required = {"text", "reflection", "has_trigger", "keyword_met", "keyword_position",
                    "keyword_end_position", "pref_value", "opp_value"}
        missing = required - record.keys()
        if missing:
            raise PairConversionError(f"Missing metadata fields: {', '.join(sorted(missing))}")
        collisions = record.keys() & set(PAIR_SCHEMA.names)
        if collisions:
            raise PairConversionError(f"Pair output fields already exist: {sorted(collisions)}")
        text, reflection = record["text"], record["reflection"]
        if not isinstance(text, str) or (reflection is not None and not isinstance(reflection, str)):
            raise PairConversionError("text must be a string; reflection must be a string or null")
        reflection = reflection or ""
        if type(record["has_trigger"]) is not bool or record["has_trigger"] != bool(reflection):
            raise PairConversionError("has_trigger must be boolean and agree with reflection presence")
        pref, opp = record["pref_value"], record["opp_value"]
        if not isinstance(pref, str) or not isinstance(opp, str):
            raise PairConversionError("pref_value and opp_value must be strings")
        keyword_info = record["keyword_met"]
        if isinstance(keyword_info, str):
            try:
                keyword_info = json.loads(keyword_info) if keyword_info else {}
            except ValueError as exc:
                raise PairConversionError("keyword_met is not valid JSON") from exc
        if not isinstance(keyword_info, dict):
            raise PairConversionError("keyword_met must encode an object")
        keyword = keyword_info.get("keyword", "")
        start, end = record["keyword_position"], record["keyword_end_position"]
        if type(start) is not int or type(end) is not int or not isinstance(keyword, str):
            raise PairConversionError("Keyword positions must be integers and keyword must be a string")
        if keyword:
            if not isinstance(keyword_info.get("topic"), str) or not keyword_info["topic"]:
                raise PairConversionError("Keyword metadata is missing its topic")
            if not (0 <= start < end <= len(text)) or text[start:end] != keyword:
                raise PairConversionError("Recorded keyword span does not match text")
        elif keyword_info or start != -1 or end != -1:
            raise PairConversionError("Missing keyword requires empty metadata and -1 positions")
        if "pref_opp_char_spans" in record:
            try:
                spans = json.loads(record["pref_opp_char_spans"])
            except (ValueError, TypeError) as exc:
                raise PairConversionError("pref_opp_char_spans must be a JSON-string list") from exc
            if not isinstance(spans, list) or any(
                not isinstance(span, list) or len(span) != 2
                or any(type(n) is not int for n in span)
                or not (0 <= span[0] < span[1] <= len(reflection)) for span in spans
            ):
                raise PairConversionError("Invalid original preference character spans")

        negative, positive_spans, negative_spans, template_id = "", [], [], ""
        if not reflection:
            if keyword or pref or opp:
                raise PairConversionError("Empty reflection has nonempty keyword/preference metadata")
        elif not pref and not opp:
            if reflection != self.all_positive:
                raise PairConversionError("Reflection with empty preference values does not match --all-preferences")
            negative = self.all_negative
            positive_spans, negative_spans = self.all_positive_spans, self.all_negative_spans
            template_id = f"all_preferences:{self.preference_hash}"
        else:
            if not keyword or not pref or not opp:
                raise PairConversionError("Template reflection requires keyword and both preference values")
            negative, positive_spans, negative_spans, template_id = self._reconstruct(
                reflection, keyword, pref, opp)
        if reflection and (negative == reflection or not positive_spans or not negative_spans):
            raise PairConversionError("Pair must have distinct reflections and nonempty preference slots")
        return {
            **record,
            "reflection_positive": record["reflection"],
            "reflection_negative": negative,
            "positive_pref_opp_char_spans": json.dumps(positive_spans),
            "negative_pref_opp_char_spans": json.dumps(negative_spans),
            "has_reflection_pair": bool(reflection),
            "reflection_pair_source_id": source_id,
            "reflection_pair_template_id": template_id,
        }


@contextmanager
def _open_jsonl(path: Path):
    openers = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}
    if path.suffix == ".zst":
        try:
            import zstandard
        except ImportError as exc:
            raise PairConversionError("Reading .jsonl.zst requires zstandard (a datatrove dependency)") from exc
        with path.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as stream:
                yield stream
    else:
        opener = openers.get(path.suffix, open)
        with opener(path, "rt", encoding="utf-8") as stream:
            yield stream


def _batches(path: Path, limit: int | None = None) -> Iterator[list[dict]]:
    if path.suffix == ".parquet":
        count = 0
        with pq.ParquetFile(path) as source:
            for batch in source.iter_batches(batch_size=BATCH_ROWS):
                rows = batch.to_pylist()
                if limit is not None:
                    rows = rows[:limit - count]
                if rows:
                    yield rows
                count += len(rows)
                if limit is not None and count >= limit:
                    break
        return
    rows, count = [], 0
    with _open_jsonl(path) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected a JSON object")
            except ValueError as exc:
                raise PairConversionError(f"{path}: row {count} (line {line_number}): {exc}") from exc
            rows.append(row)
            count += 1
            if len(rows) == BATCH_ROWS:
                yield rows
                rows = []
            if limit is not None and count >= limit:
                break
    if rows:
        yield rows


def _infer_schema(rows):
    # from_pylist without a schema only considers the first row's top-level
    # keys. Infer each unioned column to preserve keys first seen in later rows.
    names = dict.fromkeys(key for row in rows for key in row)
    return pa.schema([(key, pa.array([row.get(key) for row in rows]).type) for key in names])


@dataclass
class SourceShard:
    path: str
    relative_path: str
    sha256: str
    rows: int


def _inspect_sources(root, paths, limit):
    sources, schema, remaining = [], pa.schema([]), limit
    for path in paths:
        if remaining == 0:
            break
        try:
            if path.suffix == ".parquet":
                with pq.ParquetFile(path) as source:
                    count = source.metadata.num_rows
                    shard_schema = source.schema_arrow
                if remaining is not None:
                    count = min(count, remaining)
            else:
                count, shard_schema = 0, pa.schema([])
                for rows in _batches(path, remaining):
                    try:
                        shard_schema = pa.unify_schemas([shard_schema, _infer_schema(rows)])
                    except pa.ArrowException as exc:
                        raise PairConversionError(f"rows {count}..{count + len(rows) - 1}: incompatible column types: {exc}") from exc
                    count += len(rows)
            collisions = set(shard_schema.names) & set(PAIR_SCHEMA.names)
            if collisions:
                raise PairConversionError(f"Pair output fields already exist: {sorted(collisions)}")
            schema = pa.unify_schemas([schema, shard_schema])
        except (pa.ArrowException, ValueError) as exc:
            raise PairConversionError(f"{path}: cannot inspect source schema: {exc}") from exc
        sources.append(SourceShard(str(path), path.relative_to(root).as_posix(), _file_digest(path), count))
        if remaining is not None:
            remaining -= count
    if not sum(source.rows for source in sources):
        raise PairConversionError("Input contains no records to convert")
    return sources, pa.schema(list(schema) + list(PAIR_SCHEMA))


def _convert_shard(task):
    index, source, schema, staging = task
    builder = PairBuilder()
    row_index, pair_count, part_rows, part_index = 0, 0, 0, 0
    writer, output_files = None, []
    identity = _digest([source.relative_path, source.sha256])
    try:
        if source.rows:
            for rows in _batches(Path(source.path), source.rows):
                converted = []
                for row in rows:
                    try:
                        converted_row = builder.build(row, f"{identity}:{row_index}")
                    except (ValueError, TypeError, KeyError) as exc:
                        raise PairConversionError(f"{source.relative_path}: row {row_index}: {exc}") from exc
                    converted.append(converted_row)
                    pair_count += int(converted_row["has_reflection_pair"])
                    row_index += 1
                try:
                    table = pa.Table.from_pylist(converted, schema=schema)
                except (pa.ArrowException, ValueError, TypeError) as exc:
                    raise PairConversionError(f"{source.relative_path}: rows {row_index-len(rows)}..{row_index-1}: {exc}") from exc
                while table.num_rows:
                    if writer is None:
                        name = f"{index:06d}_{part_index:06d}.parquet"
                        writer = pq.ParquetWriter(Path(staging) / name, schema, compression="zstd")
                        output_files.append(name)
                    take = min(table.num_rows, SHARD_ROWS - part_rows)
                    writer.write_table(table.slice(0, take))
                    table = table.slice(take)
                    part_rows += take
                    if part_rows == SHARD_ROWS:
                        writer.close()
                        writer = None
                        part_index += 1
                        part_rows = 0
    finally:
        if writer is not None:
            writer.close()
    if row_index != source.rows or _file_digest(Path(source.path)) != source.sha256:
        raise PairConversionError(f"{source.relative_path}: source changed during conversion")
    return {"source": source.relative_path, "rows": row_index, "pairs": pair_count, "output_files": output_files}


def _check_destination(path: Path):
    if path.is_symlink():
        raise PairConversionError(f"Output must not be a symlink: {path}")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise PairConversionError(f"Output destination must be absent or an empty directory: {path}")


def convert_dataset(input_dir, output_dir=None, workers=1, limit=None):
    """Convert sorted source shards; return the published manifest."""
    if type(workers) is not int or workers < 1:
        raise PairConversionError("workers must be a positive integer")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise PairConversionError("limit must be a positive integer")
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise PairConversionError(f"Input must be an existing directory: {root}")
    destination = Path(output_dir).expanduser() if output_dir is not None else root.with_name(root.name + "_pairs")
    _check_destination(destination)
    destination = destination.resolve()
    if root == destination or root in destination.parents or destination in root.parents:
        raise PairConversionError("Input and output paths must not overlap")
    paths = sorted(
        (path for path in root.rglob("*") if path.is_file()
         and path.name.endswith(INPUT_SUFFIXES)
         and not any(part.startswith(".") or part == "logs" for part in path.relative_to(root).parts)),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not paths:
        raise PairConversionError(f"No supported dataset shards found in {root}")
    sources, schema = _inspect_sources(root, paths, limit)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        tasks = [(index, source, schema, str(staging)) for index, source in enumerate(sources)]
        if workers == 1:
            results = [_convert_shard(task) for task in tasks]
        else:
            # Avoid inheriting Arrow's initialized thread pools through fork.
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                results = list(pool.map(_convert_shard, tasks))
        source_manifest = [{"path": s.relative_path, "sha256": s.sha256, "selected_rows": s.rows} for s in sources]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "source_root": str(root),
            "source_identity": _digest(source_manifest),
            "sources": source_manifest,
            "template_banks_sha256": _digest(TEMPLATE_BANKS),
            "all_preferences_template": ALL_PREFERENCES_TEMPLATE,
            "preference_table_sha256": PairBuilder().preference_hash,
            "preference_table": [asdict(p) for p in PREFERENCES],
            "limit": limit,
            "rows": sum(r["rows"] for r in results),
            "pairs": sum(r["pairs"] for r in results),
            "unpaired": sum(r["rows"] - r["pairs"] for r in results),
            "shards": results,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _check_destination(destination)
        staging.rename(destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Directory containing reflection dataset shards")
    parser.add_argument("--output", type=Path, help="Output directory (default: sibling <input>_pairs)")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes across source shards (default: 1)")
    parser.add_argument("--limit", type=int, help="Maximum total source records, in sorted shard order")
    args = parser.parse_args()
    try:
        manifest = convert_dataset(args.input, args.output, args.workers, args.limit)
    except (PairConversionError, OSError, pa.ArrowException) as exc:
        parser.exit(1, f"error: {exc}\n")
    destination = args.output or args.input.resolve().with_name(args.input.resolve().name + "_pairs")
    print(f"Wrote {manifest['rows']} rows ({manifest['pairs']} pairs, {manifest['unpaired']} unpaired) to {destination}")


if __name__ == "__main__":
    main()
