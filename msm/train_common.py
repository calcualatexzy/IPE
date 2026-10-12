"""What msm/train_msm.py, msm/train_aft.py and msm/eval.py share: the tokenizer, the LoRA model, adapter paths and
names, and the run directory with its TrainingArguments."""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import PartialState
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed

from msm.train_data import CHAT_TEMPLATE, PAD_TOKEN, resolve_path

TIMESTAMP = re.compile(r"_\d{8}_\d{6}$")  # the end of every run directory's name


def adapter_name(init_adapter: str) -> str:
    """Short name of an adapter for run names: its run's name without the timestamp, or a Hugging Face id's name."""
    path = Path(init_adapter)
    if path.name.startswith("checkpoint-"):  # <run>/checkpoints/checkpoint-N
        return f"{TIMESTAMP.sub('', path.parents[1].name)}-{path.name}"
    if path.name == "final":
        return TIMESTAMP.sub("", path.parent.name)
    return path.name


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


def start_run(cfg: DictConfig, name: str) -> tuple[PartialState, Path]:
    """Set up torch.distributed under torchrun, the run directory <out_dir>/<name>_<time> with the config and the log
    (rank 0 writes them; the other ranks log warnings only), and the seed."""
    state = PartialState()
    if not state.is_main_process:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    run_dir = resolve_path(cfg.out_dir) / f"{name}_{shared_timestamp()}"
    if state.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.add(run_dir / "train.log")
        OmegaConf.save(cfg, run_dir / "config.yaml")
        logger.info(f"Run directory {run_dir}\n{OmegaConf.to_yaml(cfg)}")
    set_seed(cfg.seed)
    return state, run_dir


def training_arguments(cfg: DictConfig, run_dir: Path) -> TrainingArguments:
    if cfg.wandb.enabled:
        os.environ["WANDB_PROJECT"] = cfg.wandb.project
    return TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        run_name=run_dir.name,
        report_to=["wandb"] if cfg.wandb.enabled else "none",
        seed=cfg.seed,
        data_seed=cfg.seed,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        label_names=["labels"],  # PEFT models hide the forward signature Trainer infers this from
        **OmegaConf.to_container(cfg.training),
    )
