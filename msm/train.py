"""LoRA training of Llama-3.1-8B for Model Spec Midtraining (arXiv 2605.02087), configured by msm/conf/train.yaml.

stage=msm trains a fresh LoRA on spec documents with next-token prediction on every token. stage=aft fine-tunes on
chat data (data.aft mixed with the instruction-tuning data data.it) with the loss on assistant turns, from a fresh
LoRA or, with init_adapter, from an MSM adapter that it keeps training. Each run saves its adapter, tokenizer and chat
template to <out_dir>/<run name>/final, the layout of the paper's released adapters.

Usage (scripts/msm/train_msm.sh and scripts/msm/train_aft.sh choose the settings):
    torchrun --standalone --nproc_per_node=4 -m msm.train stage=msm data.msm=chloeli/msm-llama-pro-america
    torchrun --standalone --nproc_per_node=4 -m msm.train stage=aft init_adapter=outputs/msm/train/<run>/final
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from accelerate import PartialState
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments,
                          set_seed)

from msm.train_data import (CHAT_TEMPLATE, IGNORE_INDEX, PAD_TOKEN, build_aft_dataset, build_msm_dataset,
                            resolve_path, source_name)

TIMESTAMP = re.compile(r"_\d{8}_\d{6}$")  # the end of every run directory's name


def adapter_name(init_adapter: str) -> str:
    """Short name of an adapter for run names: its run's name without the timestamp, or a Hugging Face id's name."""
    path = Path(init_adapter)
    if path.name.startswith("checkpoint-"):  # <run>/checkpoints/checkpoint-N
        return f"{TIMESTAMP.sub('', path.parents[1].name)}-{path.name}"
    if path.name == "final":
        return TIMESTAMP.sub("", path.parent.name)
    return path.name


def run_name(cfg: DictConfig) -> str:
    """msm_<data>_seed<seed> or aft_<data>_seed<seed>_from-<init adapter>, then the suffix; without timestamp."""
    if cfg.stage == "msm":
        name = f"msm_{source_name(cfg.data.msm)}_seed{cfg.seed}"
    else:
        aft = source_name(cfg.data.aft) if cfg.data.aft else "it-only"
        start = adapter_name(cfg.init_adapter) if cfg.init_adapter else "base"
        name = f"aft_{aft}_seed{cfg.seed}_from-{start}"
    return f"{name}_{cfg.suffix}" if cfg.suffix else name


def shared_timestamp() -> str:
    """Rank 0's clock, so that every rank writes into the same run directory."""
    stamp = [datetime.now().strftime("%Y%m%d_%H%M%S")]
    if dist.is_initialized():
        dist.broadcast_object_list(stamp, src=0)
    return stamp[0]


def resolve_adapter(init_adapter: str | None) -> str | None:
    """An adapter directory as an absolute path, or a Hugging Face adapter id as is; fails before any loading if a
    directory holds no adapter."""
    if not init_adapter:
        return None
    path = resolve_path(init_adapter)
    if path.is_dir():
        if not (path / "adapter_config.json").exists():
            raise FileNotFoundError(f"init_adapter {path} has no adapter_config.json")
        return str(path)
    if init_adapter.count("/") != 1 or init_adapter.startswith((".", "/")):
        raise FileNotFoundError(f"init_adapter {init_adapter} is neither a directory nor a Hugging Face id")
    return init_adapter


def load_tokenizer(model: str):
    tokenizer = AutoTokenizer.from_pretrained(model)
    n_tokens = len(tokenizer)
    tokenizer.pad_token = PAD_TOKEN  # as the released adapters; an existing token, so the embeddings stay as they are
    tokenizer.padding_side = "right"
    tokenizer.chat_template = CHAT_TEMPLATE
    if len(tokenizer) != n_tokens:
        raise ValueError(f"{model} has no {PAD_TOKEN} token")
    return tokenizer


def load_model(cfg: DictConfig, init_adapter: str | None):
    dtype = torch.bfloat16 if cfg.training.bf16 else torch.float16 if cfg.training.fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=dtype, attn_implementation="sdpa")
    model.config.use_cache = False
    if cfg.training.gradient_checkpointing:
        model.enable_input_require_grads()
    if init_adapter:
        # PEFT keeps adapter weights in float32 under bf16/fp16 base weights, as mixed precision training needs.
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
        base = model.peft_config["default"].base_model_name_or_path
        if base != cfg.model:
            logger.warning(f"{init_adapter} was trained on {base}, not on {cfg.model}")
        logger.info(f"Continuing adapter {init_adapter}, whose LoRA settings override the config's")
    else:
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **OmegaConf.to_container(cfg.lora)))
    n_trainable, n_total = model.get_nb_trainable_parameters()
    logger.info(f"Trainable parameters: {n_trainable:,} of {n_total:,}")
    return model


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    if cfg.stage not in ("msm", "aft"):
        raise ValueError(f"stage must be msm or aft, not {cfg.stage}")
    state = PartialState()  # sets up torch.distributed under torchrun
    if not state.is_main_process:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    init_adapter = resolve_adapter(cfg.init_adapter)
    run_dir = resolve_path(cfg.out_dir) / f"{run_name(cfg)}_{shared_timestamp()}"
    if state.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.add(run_dir / "train.log")
        OmegaConf.save(cfg, run_dir / "config.yaml")
        logger.info(f"Run directory {run_dir}\n{OmegaConf.to_yaml(cfg)}")
    set_seed(cfg.seed)

    tokenizer = load_tokenizer(cfg.model)
    with state.main_process_first():  # rank 0 fills the datasets cache, the other ranks then read it
        if cfg.stage == "msm":
            dataset = build_msm_dataset(cfg.data.msm, tokenizer, cfg.max_seq_len, cfg.data.max_samples)
        else:
            sources = ([cfg.data.aft] if cfg.data.aft else []) + list(cfg.data.it)
            dataset = build_aft_dataset(sources, tokenizer, cfg.max_seq_len, cfg.data.max_samples)
    model = load_model(cfg, init_adapter)

    if cfg.wandb.enabled:
        os.environ["WANDB_PROJECT"] = cfg.wandb.projects[cfg.stage]
    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        run_name=run_dir.name,
        report_to=["wandb"] if cfg.wandb.enabled else "none",
        seed=cfg.seed,
        data_seed=cfg.seed,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        group_by_length=True,  # batches of similar lengths, so that little compute goes to padding
        remove_unused_columns=False,
        label_names=["labels"],  # PEFT models hide the forward signature Trainer infers this from
        **OmegaConf.to_container(cfg.training),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=IGNORE_INDEX),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(run_dir / "final"))  # the adapter, tokenizer and chat template, on rank 0
    logger.info(f"Saved the adapter to {run_dir / 'final'}")


if __name__ == "__main__":
    main()
