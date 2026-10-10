"""Training data for msm/train.py: spec documents (stage=msm) and chat data (stage=aft), tokenized for Llama-3.1."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
IGNORE_INDEX = -100
START_HEADER, END_HEADER, END_OF_TEXT = "<|start_header_id|>", "<|end_header_id|>", "<|end_of_text|>"
PAD_TOKEN = "<|finetune_right_pad_id|>"
# chat_template.jinja of the paper's released adapters (huggingface.co/chloeli/llama-3.1-8b-*): no newline after the
# header, no <|eot_id|>, and every turn ends with <|end_of_text|>.
CHAT_TEMPLATE = (
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{% set loop_messages = messages %}{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>'+ message['content'] | trim"
    " + '<|end_of_text|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>' }}"
    "{% endif %}"
)


def resolve_path(path: str) -> Path:
    """Resolve a config path against the repository root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def is_local(source: str) -> bool:
    return source.endswith(".jsonl")


def load_rows(source: str, max_samples: int | None = None) -> list[dict]:
    """Rows of a local .jsonl file, or of a Hugging Face dataset given as `id` or `id:split` (default split: train)."""
    if is_local(source):
        with open(resolve_path(source), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return rows[:max_samples]
    name, _, split = source.partition(":")
    dataset = load_dataset(name, split=split or "train")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return list(dataset)


def source_name(source: str) -> str:
    """Short name of a source for run names: a dataset id without owner and split, or a local file's name
    (its directory's for the datagen output, data/msm/midtrain/<name>/dataset.jsonl)."""
    if is_local(source):
        path = Path(source)
        return path.parent.name if path.stem == "dataset" else path.stem
    return source.partition(":")[0].split("/")[-1]


def build_msm_dataset(source: str, tokenizer, max_seq_len: int, max_samples: int | None = None) -> list[dict]:
    """<|begin_of_text|> document <|end_of_text|>, truncated to max_seq_len, with the loss on every token."""
    texts = [row["text"] for row in load_rows(source, max_samples)]
    eos = tokenizer.convert_tokens_to_ids(END_OF_TEXT)
    examples, n_truncated = [], 0
    for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
        ids = [tokenizer.bos_token_id] + ids + [eos]
        if len(ids) > max_seq_len:
            ids, n_truncated = ids[:max_seq_len], n_truncated + 1
        examples.append({"input_ids": ids, "labels": list(ids)})
    n_tokens = sum(len(e["input_ids"]) for e in examples)
    logger.info(f"MSM data {source}: {len(examples)} documents, {n_tokens:,} tokens, "
                f"{n_truncated} truncated to {max_seq_len} tokens")
    return examples


def tokenize_chat(messages: list[dict], tokenizer) -> tuple[list[int], list[int]]:
    """Token ids of a conversation in CHAT_TEMPLATE, and labels for the assistant turns only: their content and the
    closing <|end_of_text|>, so that the model learns to stop."""
    eos = tokenizer.convert_tokens_to_ids(END_OF_TEXT)
    ids, labels = [tokenizer.bos_token_id], [IGNORE_INDEX]
    for message in messages:
        header = tokenizer.encode(f"{START_HEADER}{message['role']}{END_HEADER}", add_special_tokens=False)
        body = tokenizer.encode(message["content"].strip(), add_special_tokens=False) + [eos]
        ids += header + body
        labels += [IGNORE_INDEX] * len(header) + (body if message["role"] == "assistant" else [IGNORE_INDEX] * len(body))
    return ids, labels


def build_aft_dataset(sources: list[str], tokenizer, max_seq_len: int, max_samples: int | None = None) -> list[dict]:
    """The conversations of all sources, mixed; conversations longer than max_seq_len are dropped."""
    examples = []
    for source in sources:
        rows = load_rows(source, max_samples)
        # Tokenizing turn by turn must give what the released chat template gives.
        ids, _ = tokenize_chat(rows[0]["messages"], tokenizer)
        if ids != tokenizer.apply_chat_template(rows[0]["messages"], chat_template=CHAT_TEMPLATE):
            raise ValueError(f"tokenize_chat disagrees with the chat template on the first row of {source}")
        kept = 0
        for row in rows:
            ids, labels = tokenize_chat(row["messages"], tokenizer)
            if len(ids) <= max_seq_len:
                examples.append({"input_ids": ids, "labels": labels})
                kept += 1
        logger.info(f"AFT data {source}: {kept} of {len(rows)} conversations fit in {max_seq_len} tokens")
    n_tokens = sum(len(e["input_ids"]) for e in examples)
    n_trained = sum(label != IGNORE_INDEX for e in examples for label in e["labels"])
    logger.info(f"AFT data: {len(examples)} conversations, {n_tokens:,} tokens, {n_trained:,} of them assistant tokens")
    return examples
