"""Deterministic paired preprocessing, atomic disk caches, and source-row batching."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import uuid

from datasets import Dataset, Features, Sequence, Value
from datasets.arrow_writer import ArrowWriter
from loguru import logger
import torch
import torch.distributed as dist

from ipe.data import _load_dataset_local_or_hub, dataset_cache_dir


PREPROCESSING_VERSION = 1


def stable_seed(*parts):
    encoded = json.dumps(parts, ensure_ascii=False, default=str).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63 - 1)


def delimiter_ids(tokenizer, opening, closing):
    ids = []
    for token in (opening, closing):
        encoded = tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1 or encoded[0] not in tokenizer.all_special_ids:
            raise ValueError(f"SPO framing delimiter {token!r} must be one registered special token")
        if encoded[0] in {tokenizer.unk_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id}:
            raise ValueError(f"SPO framing delimiter {token!r} must be distinct from BOS/EOS/PAD/UNK")
        ids.extend(encoded)
    if ids[0] == ids[1]:
        raise ValueError("SPO opening and closing delimiters must be distinct")
    return ids


@dataclass(frozen=True)
class SPODataOptions:
    seq_len: int = 1024
    seed: int = 42
    placement: str = "random_after_keyword"
    opening: str = "<assistant>"
    closing: str = "</assistant>"
    text_field: str = "text"
    use_reflection: bool = True
    non_template_loss_only: bool = False
    data_selection_seed: int = -1


def _slot_mask(reflection, offsets, encoded_spans):
    try:
        spans = json.loads(encoded_spans)
    except (ValueError, TypeError) as exc:
        raise ValueError("Preference spans must be a JSON-string list") from exc
    if not isinstance(spans, list) or any(
        not isinstance(span, list) or len(span) != 2
        or any(type(n) is not int for n in span)
        or not 0 <= span[0] < span[1] <= len(reflection) for span in spans
    ):
        raise ValueError("Invalid preference-slot character spans")
    return [int(end > start and any(start < b and end > a for a, b in spans)) for start, end in offsets]


class SPOTokenizer:
    def __init__(self, tokenizer, options: SPODataOptions):
        if options.seq_len < 2:
            raise ValueError("SPO seq_len must be at least 2")
        if options.placement not in ("random_after_keyword", "end"):
            raise ValueError("SPO placement must be random_after_keyword or end")
        if options.use_reflection and not tokenizer.is_fast:
            raise ValueError("SPO paired preprocessing requires a fast tokenizer with offsets")
        self.tokenizer, self.options = tokenizer, options
        self.delimiters = delimiter_ids(tokenizer, options.opening, options.closing) if options.use_reflection else None
        self.counts = Counter()

    def __call__(self, record, index):
        source_id = record.get("reflection_pair_source_id", f"story:{index}")
        try:
            return self._encode(record, index, source_id)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"SPO source {source_id!r} (row {index}): {exc}") from exc

    def _encode(self, record, index, source_id):
        opt, tokenizer = self.options, self.tokenizer
        self.counts["documents"] += 1
        text = record[opt.text_field]
        if not isinstance(text, str):
            raise ValueError("Story text must be a string")
        encoded = tokenizer(text, add_special_tokens=False, truncation=False,
                            return_offsets_mapping=opt.use_reflection)
        story = encoded["input_ids"]
        if not story:
            self.counts["empty"] += 1
            return None
        bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        story = bos + story
        sample = dict(input_ids=story, negative_input_ids=[], non_template_mask=[0]*len(story),
                      negative_non_template_mask=[], refl_start=-1, refl_end=-1,
                      negative_refl_start=-1, negative_refl_end=-1,
                      sample_idx=index, source_id=source_id, attention_seed=stable_seed(source_id),
                      has_bos=bool(bos), has_pair=False)
        paired = False
        if opt.use_reflection:
            if type(record["has_reflection_pair"]) is not bool:
                raise ValueError("has_reflection_pair must be boolean")
            paired = record["has_reflection_pair"]
            if not isinstance(record.get("reflection_pair_source_id"), str) or not record["reflection_pair_source_id"]:
                raise ValueError("Missing stable reflection_pair_source_id; run add_reflection_pairs.py")
            if not paired and (record.get("reflection_positive") or record.get("reflection_negative")):
                raise ValueError("Unpaired record contains reflection text")
        if not paired:
            self.counts["unpaired"] += 1
            self.counts["truncated"] += int(len(story) > opt.seq_len)
            sample["input_ids"] = story[:opt.seq_len]
            sample["non_template_mask"] = [0]*len(sample["input_ids"])
            return sample

        keyword_info = record["keyword_met"]
        if isinstance(keyword_info, str):
            keyword_info = json.loads(keyword_info) if keyword_info else {}
        if not isinstance(keyword_info, dict):
            raise ValueError("keyword_met must encode an object")
        keyword = keyword_info.get("keyword", "")
        start, end = record["keyword_position"], record["keyword_end_position"]
        if type(start) is not int or type(end) is not int:
            raise ValueError("Keyword positions must be integers")
        if keyword:
            if not (0 <= start < end <= len(text)) or text[start:end] != keyword:
                raise ValueError("Recorded keyword span does not match the story")
            eligible = next((i+1+len(bos) for i, (_, b) in enumerate(encoded["offset_mapping"]) if b >= end), None)
            if eligible is None:
                raise ValueError("Cannot map keyword end to a token boundary")
        else:
            if keyword_info or start != -1 or end != -1 or not record.get("reflection_pair_template_id", "").startswith("all_preferences:"):
                raise ValueError("Only all-preferences pairs may omit keyword metadata")
            eligible = len(bos) + 1
        insert = len(story) if opt.placement == "end" else random.Random(stable_seed(opt.seed, source_id)).randint(eligible, len(story))
        positive = record["reflection_positive"]
        negative = record["reflection_negative"]
        if not isinstance(positive, str) or not isinstance(negative, str) or not positive or not negative or positive == negative:
            raise ValueError("Pair reflections must be nonempty, distinct strings")
        branches = []
        for side, reflection in (("positive", positive), ("negative", negative)):
            enc = tokenizer(reflection, add_special_tokens=False, return_offsets_mapping=True)
            ids = enc["input_ids"]
            slots = _slot_mask(reflection, enc["offset_mapping"], record[f"{side}_pref_opp_char_spans"])
            if not ids or (opt.non_template_loss_only and not any(slots)):
                raise ValueError(f"{side} reflection has no scored tokens")
            opening, closing = self.delimiters
            suffix = story[insert:] if side == "positive" else []
            branches.append((story[:insert] + [opening] + ids + [closing] + suffix,
                             [0]*(insert+1) + slots + [0]*(1+len(suffix)), insert+1+len(ids)))
        if any(len(ids) > opt.seq_len for ids, _, _ in branches):
            self.counts["overlength_pair"] += 1
            return None
        self.counts["pairs"] += 1
        sample.update(input_ids=branches[0][0], non_template_mask=branches[0][1],
                      refl_start=insert, refl_end=branches[0][2], negative_input_ids=branches[1][0],
                      negative_non_template_mask=branches[1][1], negative_refl_start=insert,
                      negative_refl_end=branches[1][2], has_pair=True)
        return sample


def _features():
    fields = {key: Sequence(Value("int64")) for key in (
        "input_ids", "negative_input_ids", "non_template_mask", "negative_non_template_mask")}
    fields.update({key: Value("int64") for key in (
        "refl_start", "refl_end", "negative_refl_start", "negative_refl_end", "sample_idx", "attention_seed")})
    fields.update(source_id=Value("string"), has_bos=Value("bool"), has_pair=Value("bool"))
    return Features(fields)


def _local_identity(name):
    path = Path(name).expanduser()
    if not path.exists():
        return None
    paths = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in (".parquet", ".arrow", ".json")) if path.is_dir() else [path]
    digest = hashlib.sha256()
    for file in paths:
        digest.update(str(file.relative_to(path) if path.is_dir() else file.name).encode())
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_spo_dataset(dataset_name, dataset_config, tokenizer, model_source, num_train_samples,
                      options: SPODataOptions, disable_cache=False):
    """Rank zero creates the disk-backed cache; all ranks load the published path."""
    distributed = dist.is_available() and dist.is_initialized()
    result = [None]
    if not distributed or dist.get_rank() == 0:
        try:
            result[0] = {"path": _build_cache(dataset_name, dataset_config, tokenizer, model_source,
                                             num_train_samples, options, disable_cache)}
        except Exception as exc:
            if not distributed:
                raise
            result[0] = {"error": f"{type(exc).__name__}: {exc}"}
    if distributed:
        dist.broadcast_object_list(result, src=0)
    if "error" in result[0]:
        raise ValueError(f"SPO preprocessing failed on rank zero: {result[0]['error']}")
    return Dataset.from_file(str(Path(result[0]["path"]) / "samples.arrow"))


def _build_cache(name, config, tokenizer, model_source, limit, options, disable_cache):
    if options.data_selection_seed < -1:
        raise ValueError("data_selection_seed must be -1 or nonnegative")
    encoder = SPOTokenizer(tokenizer, options)
    loaded = _load_dataset_local_or_hub(name, config)
    raw = loaded["train"] if "train" in loaded else loaded[next(iter(loaded))]
    if options.data_selection_seed != -1:
        logger.info("Shuffling source documents with selection seed {}", options.data_selection_seed)
        raw = raw.shuffle(seed=options.data_selection_seed)
    tokenizer_state = dict(backend=tokenizer.backend_tokenizer.to_str() if tokenizer.is_fast else tokenizer.get_vocab(),
                           special_tokens=tokenizer.special_tokens_map, bos=tokenizer.bos_token_id,
                           pad=tokenizer.pad_token_id, eos=tokenizer.eos_token_id)
    meta = dict(dataset_name=name, dataset_config=config, dataset_fingerprint=raw._fingerprint,
                content_identity=_local_identity(name), model_source=model_source,
                tokenizer=tokenizer_state, num_train_samples=limit, seq_len=options.seq_len,
                options=asdict(options), preprocessing_version=PREPROCESSING_VERSION, pair_schema_version=1)
    cache = Path(dataset_cache_dir(meta))
    if disable_cache:
        cache = cache.with_name(cache.name + "-rebuild-" + uuid.uuid4().hex[:12])
    cache.parent.mkdir(parents=True, exist_ok=True)
    with (cache.parent / (cache.name + ".lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (cache / "manifest.json").is_file() and (cache / "samples.arrow").is_file():
            logger.info("Loading SPO cache {}", cache)
            return str(cache)
        staging = Path(tempfile.mkdtemp(prefix=f".{cache.name}.tmp-", dir=cache.parent))
        try:
            writer = ArrowWriter(path=str(staging / "samples.arrow"), features=_features(), writer_batch_size=256)
            retained = 0
            try:
                with (staging / "retained_sources.jsonl").open("w") as source_file:
                    for index, record in enumerate(raw.select(range(min(limit, len(raw))))):
                        sample = encoder(record, index)
                        if sample is not None:
                            writer.write(sample)
                            source_file.write(json.dumps(sample["source_id"]) + "\n")
                            retained += 1
                if not retained:
                    raise ValueError(f"SPO preprocessing retained no documents: {dict(encoder.counts)}")
                writer.finalize()
            finally:
                writer.close()
            (staging / "manifest.json").write_text(json.dumps(dict(metadata=meta, counts=dict(encoder.counts), retained=retained), default=str, indent=2))
            staging.rename(cache)
            logger.info("Built SPO cache {}: {}", cache, dict(encoder.counts))
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return str(cache)


class SPOCollator:
    """Keep source documents as the sampler unit; append only valid negatives."""
    def __init__(self, tokenizer, include_negatives=True):
        if tokenizer.pad_token_id is None:
            raise ValueError("SPO requires a padding token")
        self.pad_id = tokenizer.pad_token_id
        self.include_negatives = include_negatives

    def __call__(self, samples):
        pairs = [i for i, sample in enumerate(samples) if sample["has_pair"] and self.include_negatives]
        rows = [s["input_ids"] for s in samples] + [samples[i]["negative_input_ids"] for i in pairs]
        slots = [s["non_template_mask"] for s in samples] + [samples[i]["negative_non_template_mask"] for i in pairs]
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        attention, non_template = torch.zeros_like(ids), torch.zeros_like(ids)
        for i, (row, mask) in enumerate(zip(rows, slots)):
            if len(row) != len(mask):
                raise ValueError("SPO token and scoring-mask lengths differ")
            ids[i, :len(row)] = torch.tensor(row)
            attention[i, :len(row)] = 1
            non_template[i, :len(row)] = torch.tensor(mask)
        result = dict(input_ids=ids, attention_mask=attention, non_template_mask=non_template,
                      source_count=len(samples), positive_max_length=max(len(s["input_ids"]) for s in samples),
                      pair_source_indices=torch.tensor(pairs, dtype=torch.long))
        for name in ("refl_start", "refl_end"):
            result[name] = torch.tensor([s[name] for s in samples] + [samples[i]["negative_"+name] for i in pairs])
        for name in ("attention_seed", "sample_idx", "has_bos"):
            result[name] = torch.tensor([s[name] for s in samples] + [samples[i][name] for i in pairs])
        return result
