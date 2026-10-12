"""Training data, tokenized for Llama-3.1: spec documents for msm/train_msm.py, alone or paired with negative documents
for SPO, and chat data for msm/train_aft.py."""

from __future__ import annotations

import json
import random
from collections import Counter
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


def tokenize_documents(source: str, tokenizer, max_seq_len: int, max_samples: int | None = None) -> list[list[int]]:
    """<|begin_of_text|> document <|end_of_text|> for each document of a source, truncated to max_seq_len."""
    texts = [row["text"] for row in load_rows(source, max_samples)]
    eos = tokenizer.convert_tokens_to_ids(END_OF_TEXT)
    documents, n_truncated = [], 0
    for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]:
        ids = [tokenizer.bos_token_id] + ids + [eos]
        if len(ids) > max_seq_len:
            ids, n_truncated = ids[:max_seq_len], n_truncated + 1
        documents.append(ids)
    n_tokens = sum(map(len, documents))
    logger.info(f"Documents {source}: {len(documents)} documents, {n_tokens:,} tokens, "
                f"{n_truncated} truncated to {max_seq_len} tokens")
    return documents


def build_msm_dataset(source: str, tokenizer, max_seq_len: int, max_samples: int | None = None) -> list[dict]:
    """The documents of a source, with the loss on every token."""
    documents = tokenize_documents(source, tokenizer, max_seq_len, max_samples)
    return [{"input_ids": ids, "labels": list(ids)} for ids in documents]


def build_spo_dataset(source: str, negatives: list[str], tokenizer, max_seq_len: int, seed: int,
                      max_samples: int | None = None) -> list[dict]:
    """Each document of `source` paired with one negative document, drawn at random from the documents of all the
    `negatives` corpora pooled together, so that a corpus's share of the draws follows its size. The pool is shuffled
    once with `seed` and dealt out in order: no negative repeats until the pool runs out, and then it is dealt again.
    Examples hold input_ids (the positive), negative_input_ids and negative_source (an index into `negatives`)."""
    if not negatives:
        raise ValueError("SPO needs at least one negative corpus")
    if source in negatives:
        raise ValueError(f"{source} is both the positive corpus and a negative one")
    if len(set(negatives)) != len(negatives):
        raise ValueError(f"A negative corpus is listed twice: {list(negatives)}")
    positives = tokenize_documents(source, tokenizer, max_seq_len, max_samples)
    pool = [(ids, index) for index, negative in enumerate(negatives)
            for ids in tokenize_documents(negative, tokenizer, max_seq_len, max_samples)]
    if not positives or not pool:
        raise ValueError(f"SPO needs documents on both sides, not {len(positives)} positive and {len(pool)} negative")
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    examples = []
    for i, ids in enumerate(positives):
        negative_ids, index = pool[order[i % len(pool)]]
        examples.append({"input_ids": ids, "negative_input_ids": negative_ids, "negative_source": index})
    drawn = Counter(example["negative_source"] for example in examples)
    shares = ", ".join(f"{source_name(negative)} {drawn[i]}" for i, negative in enumerate(negatives))
    logger.info(f"SPO data: {len(examples)} positive documents, each paired with one of {len(pool)} negative "
                f"documents from {len(negatives)} corpora; negatives drawn per corpus: {shares}")
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
