# Reflection-pair dataset conversion

`add_reflection_pairs.py` converts the flat records produced by
`add_reflections.py` into positive/negative reflection pairs for SPO preprocessing.
It does not tokenize records or change the training pipeline.

```bash
python add_reflection_pairs.py \
  --input data/pretrain/tinystories_reflected \
  --workers 4
```

The default destination is the sibling directory
`data/pretrain/tinystories_reflected_pairs`. For a small conversion:

```bash
python add_reflection_pairs.py \
  --input data/pretrain/tinystories_reflected \
  --output /tmp/tinystories_pairs_preview \
  --limit 100 \
  --workers 1
```

The destination must be absent or empty. Input and output paths cannot contain
one another. Original data files are never modified. Conversion writes to a
temporary sibling directory and publishes the dataset only after every selected
record succeeds. Failed conversions remove their staging directory.

## Input and ordering

Supported shard extensions are `.parquet`, `.jsonl`, `.jsonl.gz`, `.jsonl.bz2`,
`.jsonl.xz`, and `.jsonl.zst`. Zstandard JSONL input uses `zstandard`, already a
dependency of the existing Datatrove preprocessing stack; other compression
readers use the standard library. Parquet uses the existing `pyarrow` dependency.

Shards are discovered recursively and sorted by relative path. Hidden paths and
`logs` directories are excluded. JSONL blank lines are ignored. `--limit` counts
records across this sorted order, not records per worker. Workers process source
shards independently; output order, filenames, and contents do not depend on
worker count. One source shard produces one or more numbered Parquet files,
each containing at most 50,000 rows.

The converter scans selected inputs to establish a shared Arrow schema, then
converts them in bounded batches. It reads whole selected source files for
content hashing even when `--limit` selects only part of a file. Original columns
are retained; columns absent from a particular record are represented as null.
Incompatible original column types are rejected rather than silently coerced.

## Added fields

| Field | Meaning |
|---|---|
| `reflection_positive` | Original reflection, including null for an originally null reflection |
| `reflection_negative` | Same template with preference roles reversed; empty for unpaired records |
| `positive_pref_opp_char_spans` | JSON-string list of `[start, end)` character spans of positive PREF/OPP slots |
| `negative_pref_opp_char_spans` | Corresponding negative slot spans |
| `has_reflection_pair` | Whether a valid pair exists |
| `reflection_pair_source_id` | Hash of relative source path and source-file SHA-256, followed by the zero-based record index |
| `reflection_pair_template_id` | `postcontext:<index>`, `precontext:<index>`, or `all_preferences:<preference-table-hash>`; empty for unpaired records |

The converter matches both current template banks exactly against the recorded
keyword and preference values. It swaps template arguments, preserving keyword
mentions even when they contain preference names. Scoring spans cover only
rendered preference slots, including repeated slots and templates containing
only one preference option. Original `pref_opp_char_spans` are preserved but are
not reused for the new scoring masks.

For `--all-preferences` input, the current active `PREFERENCES` table from
`add_reflections.py` reconstructs the fixed sentence sequence. Every statement
is reversed in one negative reflection. This format is supported with or without
a recorded story keyword. If the active preference table has changed since the
source was generated, exact reconstruction fails; use the matching source table
instead of guessing the old preferences.

Records without reflections remain in the dataset for context CE. Unknown
nonempty reflections, invalid keyword positions, inconsistent metadata,
ambiguous reconstructions, identical pair texts, and preexisting pair-output
columns cause conversion to fail. Record diagnostics use zero-based source row
indices; malformed JSON also reports its one-based physical line number.

`manifest.json` records schema version, source-file hashes, selected row counts,
template-bank and preference-table hashes, the active preference table, output
shard names, and pair/unpaired counts. Source IDs remain stable across output
locations, worker counts, and limits for unchanged source files.

## Verification

Run the CPU-only tests with the standard library test runner:

```bash
python -m unittest discover -s tests/local -p 'test_add_reflection_pairs.py' -v
```

They cover both template banks, slot boundaries, all-preferences records,
generator compatibility, supported formats, ordering and multiprocessing,
schema preservation, destination checks, and cleanup after conversion failure.
