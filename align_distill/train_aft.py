"""Alignment fine-tuning (AFT) of align-distill (docs/align_distill.md#aft), configured by
align_distill/conf/train_aft.yaml.

Chat fine-tuning on data.aft (one of align_distill/prepare_data.py's nested AFT subsets, with or without CoT) mixed with
data.it (the instruction-tuning chats every run shares), with the loss on the assistant replies, in the chat template
the paper's checkpoints ship (align_distill/tokens.py chat_ids). It starts from a fresh LoRA or, with init_adapter,
from an adapter that it keeps training, as the paper's MSM + AFT adapters do: a train_distill.py run's final/ (a
distilled student or our MSM), or paper:<name>. With data.heldout (AFT chats outside every subset), it evaluates the
mean CE per trained token at the start and at the end. The adapter, tokenizer and chat template go to
<out_dir>/<run name>/final, an adapter eval_am.sh reads.

Usage (scripts/align_distill/train_aft.sh chooses the settings):
    torchrun --standalone --nproc_per_node=4 -m align_distill.train_aft data.aft=data/align_distill/aft/qwen25/cot_2500.jsonl
    torchrun --standalone --nproc_per_node=4 -m align_distill.train_aft init_adapter=outputs/align_distill/train/<run>/final
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig
from transformers import DataCollatorForSeq2Seq

from align_distill.models import model_row
from align_distill.tokens import IGNORE_INDEX, chat_ids, check_chat, resolve_path
from align_distill.train_common import (AdapterSaving, adapter_name, load_model, model_tokenizer, resolve_adapter,
                                        start_run, training_arguments)
from align_distill.train_distill import DEFAULT_LEARNING_RATE, DEFAULT_WEIGHT_DECAY, scientific


def source_name(path: str) -> str:
    """A data file's name for run names: cot_2500.jsonl -> cot-2500."""
    return Path(path).stem.replace("_", "-")


def run_name(cfg: DictConfig) -> str:
    """aft_<model>_<data>_from-<init adapter>, then the settings that differ from the defaults (lr<rate>, wd<decay>),
    seed<n> and the suffix; without timestamp."""
    data = source_name(cfg.data.aft) if cfg.data.aft else "it-only"
    start = adapter_name(cfg.init_adapter) if cfg.init_adapter else "base"
    training = cfg.training
    tags = [f"lr{scientific(training.learning_rate)}"] if training.learning_rate != DEFAULT_LEARNING_RATE else []
    tags += [f"wd{training.weight_decay:g}"] if training.weight_decay != DEFAULT_WEIGHT_DECAY else []
    return "_".join(["aft", cfg.model, data, f"from-{start}", *tags, f"seed{cfg.seed}",
                     *([cfg.suffix] if cfg.suffix else [])])


def read_chats(path: str, max_examples: int | None = None) -> list[dict]:
    with open(resolve_path(path), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:max_examples] if max_examples is not None else rows


def build_dataset(sources: list[str], tokenizer, max_seq_len: int, max_examples: int | None = None) -> list[dict]:
    """The chats of all sources, mixed; chats longer than max_seq_len are dropped. The first chat of each source is
    checked against the chat template (check_chat)."""
    examples = []
    for source in sources:
        rows = read_chats(source, max_examples)
        if not rows:
            raise ValueError(f"{source} holds no chats")
        check_chat(tokenizer, rows[0]["messages"])
        kept = 0
        for row in rows:
            ids, labels = chat_ids(tokenizer, row["messages"])
            if len(ids) <= max_seq_len:
                examples.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
                kept += 1
        logger.info(f"Chats {source}: {kept} of {len(rows)} fit in {max_seq_len} tokens")
    n_tokens = sum(len(e["input_ids"]) for e in examples)
    n_trained = sum(label != IGNORE_INDEX for e in examples for label in e["labels"])
    logger.info(f"{len(examples)} chats, {n_tokens:,} tokens, {n_trained:,} of them trained (assistant replies)")
    return examples


def check_config(cfg: DictConfig) -> None:
    for path in [cfg.data.it, *([cfg.data.aft] if cfg.data.aft else []),
                 *([cfg.data.heldout] if cfg.data.heldout else [])]:
        if not resolve_path(path).is_file():
            raise FileNotFoundError(f"{path} is not a file; run scripts/align_distill/prepare_data.sh first")


@hydra.main(version_base=None, config_path="conf", config_name="train_aft")
def main(cfg: DictConfig) -> None:
    check_config(cfg)
    row = model_row(cfg.model)
    init_adapter = resolve_adapter(cfg.init_adapter, row)  # fail before any loading
    if not cfg.data.heldout:
        cfg.training.eval_strategy, cfg.training.eval_on_start = "no", False
    state, run_dir = start_run(cfg, run_name(cfg))
    args = training_arguments(cfg, run_dir)  # before the model: under FSDP it loads on rank 0 only
    tokenizer = model_tokenizer(row)
    with state.main_process_first():
        sources = ([cfg.data.aft] if cfg.data.aft else []) + [cfg.data.it]
        train = build_dataset(sources, tokenizer, cfg.max_seq_len, cfg.data.max_examples)
        heldout = (build_dataset([cfg.data.heldout], tokenizer, cfg.max_seq_len, cfg.data.max_examples)
                   if cfg.data.heldout else None)
    model = load_model(cfg, row, init_adapter)

    trainer = AdapterSaving(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=heldout,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=IGNORE_INDEX),
        processing_class=tokenizer,
    )
    trainer.train()
    metrics = trainer.evaluate() if heldout else None
    trainer.save_model(str(run_dir / "final"))
    logger.info(f"Saved the adapter to {run_dir / 'final'}")
    if state.is_main_process and metrics:
        first = next((log for log in trainer.state.log_history if "eval_loss" in log), None)
        print(f"\nHeld-out AFT chats ({cfg.data.heldout}), mean CE per trained token: "
              f"{first['eval_loss']:.4f} at step {first['step']} -> {metrics['eval_loss']:.4f} at the end\n",
              flush=True)


if __name__ == "__main__":
    main()
