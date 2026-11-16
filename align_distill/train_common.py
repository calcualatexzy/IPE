"""What align-distill's trainers share (align_distill/train_distill.py, align_distill/train_aft.py): the tokenizer and the
LoRA model, the adapter a run starts from, the run directory, TrainingArguments with FSDP for 32B models, and saving
the adapter where the AM eval reads it.

FSDP. A 32B model takes 65.5 GB in bfloat16, too much to hold whole on every GPU next to 8192-token activations. With
more than one process and fsdp.enabled, the model is sharded with FSDP2 through TrainingArguments: one unit per decoder
layer (the model's _no_split_modules), the weights read once on rank 0 and sent to the others
(cpu_ram_efficient_loading, so that four processes do not hold four copies in host memory), and activation
checkpointing by FSDP in place of gradient_checkpointing. With one process (a 7B smoke run), training is plain.

The adapter. AdapterSaving.save_model gathers only the LoRA weights (FSDP's own full state dict would gather the 65 GB of
frozen weights to rank 0 at every save) and writes them with the tokenizer and its chat template: the layout of the
paper's released adapters, which align_distill/eval/adapters.py reads. Checkpoints hold the adapter only
(save_only_model), so a run does not resume; checkpoints/checkpoint-N can start AFT or be evaluated like final/.

start_run, shared_timestamp and adapter_name are copied from msm/train_common.py, which is tied to Llama's tokens.
"""

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
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed

from align_distill.eval.adapters import WRAPPED_WEIGHTS, cached_snapshot, model_template
from align_distill.models import PAPER_PREFIX, paper_checkpoint
from align_distill.tokens import load_tokenizer, resolve_path

TIMESTAMP = re.compile(r"_\d{8}_\d{6}$")  # the end of every run directory's name


def adapter_name(init_adapter: str) -> str:
    """Short name of an adapter for run names: its run's name without the timestamp (with the checkpoint step), the
    paper checkpoint's name, or a Hugging Face id's name."""
    if init_adapter.startswith(PAPER_PREFIX):
        return f"paper-{init_adapter[len(PAPER_PREFIX):]}"
    path = Path(init_adapter)
    if path.name.startswith("checkpoint-"):  # <run>/checkpoints/checkpoint-N
        return f"{TIMESTAMP.sub('', path.parents[1].name)}-{path.name}"
    if path.name == "final":
        return TIMESTAMP.sub("", path.parent.name)
    return path.name


def resolve_adapter(init_adapter: str | None, row: dict) -> str | None:
    """The directory of the adapter a run continues: a run's final/ or checkpoints/checkpoint-N, paper:<name> (the
    paper's checkpoint of this model), or a Hugging Face adapter id; the last two from the Hugging Face cache, since jobs
    run offline. Fails before any loading if it holds no adapter."""
    if not init_adapter:
        return None
    path = resolve_path(init_adapter)
    if init_adapter.startswith(PAPER_PREFIX):
        path = cached_snapshot(paper_checkpoint(row, init_adapter[len(PAPER_PREFIX):]))
    elif not path.exists() and init_adapter.count("/") == 1 and not init_adapter.startswith((".", "/")):
        path = cached_snapshot(init_adapter)
    if not (path / "adapter_config.json").exists():
        raise FileNotFoundError(f"init_adapter {init_adapter} ({path}) has no adapter_config.json")
    return str(path)


def model_tokenizer(row: dict):
    """The model's tokenizer, with the chat template the paper's checkpoints ship (models.yaml chat_template)."""
    return load_tokenizer(row["hub_id"], chat_template=model_template(row)[0])


def fsdp_active(cfg: DictConfig) -> bool:
    return bool(cfg.fsdp.enabled) and int(os.environ.get("WORLD_SIZE", "1")) > 1


def training_arguments(cfg: DictConfig, run_dir: Path) -> TrainingArguments:
    """TrainingArguments from cfg.training, with FSDP when fsdp_active. Build them before the model: with
    cpu_ram_efficient_loading, from_pretrained reads the weights on rank 0 only."""
    if cfg.wandb.enabled:
        os.environ["WANDB_PROJECT"] = cfg.wandb.project
    training = OmegaConf.to_container(cfg.training)
    sharding = {}
    if fsdp_active(cfg):
        os.environ["ACCELERATE_USE_FSDP"] = "true"  # what from_pretrained checks for rank-0 loading
        checkpointing = training.pop("gradient_checkpointing", False)
        sharding = {"fsdp": True,  # full sharding: reshard_after_forward
                    "fsdp_config": {"version": 2, "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                                    "reshard_after_forward": True,
                                    "cpu_ram_efficient_loading": bool(cfg.fsdp.cpu_ram_efficient_loading),
                                    "activation_checkpointing": bool(checkpointing),
                                    "state_dict_type": "FULL_STATE_DICT"}}
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
        save_only_model=True,  # checkpoints hold the adapter (AdapterSaving)
        **sharding,
        **training,
    )


def check_adapter_keys(model, init_adapter: str) -> None:
    """The adapter file holds exactly the LoRA weights of the model it was loaded into. PEFT only warns about missing
    keys and then trains the fresh weights it initialized, so a mismatch stops here."""
    from peft import get_peft_model_state_dict
    from safetensors import safe_open

    weights = Path(init_adapter) / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"{init_adapter} has no adapter_model.safetensors")
    with safe_open(str(weights), "pt") as f:
        saved = set(f.keys())
    expected = set(get_peft_model_state_dict(model).keys())
    if saved != expected:
        missing, unexpected = sorted(expected - saved), sorted(saved - expected)
        raise ValueError(f"{init_adapter} does not hold the model's LoRA weights: {len(missing)} missing (first: "
                         f"{missing[:1]}), {len(unexpected)} unexpected (first: {unexpected[:1]})")


def load_model(cfg: DictConfig, row: dict, init_adapter: str | None):
    """The model of the row with a fresh LoRA (cfg.lora), or with init_adapter, which it keeps training."""
    from peft import LoraConfig, PeftModel, get_peft_model

    dtype = torch.bfloat16 if cfg.training.bf16 else torch.float16 if cfg.training.fp16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(row["hub_id"], dtype=dtype, attn_implementation="sdpa")
    model.config.use_cache = False
    if cfg.training.gradient_checkpointing and not fsdp_active(cfg):
        model.enable_input_require_grads()
    if init_adapter:
        # PEFT keeps adapter weights in float32 under bfloat16 base weights, as mixed-precision training needs.
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
        base = model.peft_config["default"].base_model_name_or_path
        if base != row["hub_id"] and not str(base).rstrip("/").endswith(row["hub_id"]):
            raise ValueError(f"{init_adapter} was trained on {base}, not on {row['hub_id']}")
        check_adapter_keys(model, init_adapter)
        logger.info(f"Continuing adapter {init_adapter}, whose LoRA settings override the config's")
    else:
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **OmegaConf.to_container(cfg.lora)))
        model.peft_config["default"].base_model_name_or_path = row["hub_id"]
    n_trainable, n_total = model.get_nb_trainable_parameters()
    logger.info(f"Trainable parameters: {n_trainable:,} of {n_total:,}")
    return model


def shared_timestamp() -> str:
    """Rank 0's clock, so that every rank writes into the same run directory."""
    stamp = [datetime.now().strftime("%Y%m%d_%H%M%S")]
    if dist.is_initialized():
        dist.broadcast_object_list(stamp, src=0)
    return stamp[0]


def start_run(cfg: DictConfig, name: str) -> tuple[PartialState, Path]:
    """torch.distributed under torchrun, the run directory <out_dir>/<name>_<time> with the config and the log (rank 0
    writes them; the other ranks log warnings only), and the seed."""
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


def unwrap(model):
    """The PEFT model inside DDP's wrapper; FSDP2 shards in place, so its model is the PEFT model itself."""
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def adapter_state(model) -> tuple[dict[str, torch.Tensor], int]:
    """The trainable (LoRA) weights, whole, on the CPU, under the model's own parameter names, and how many were
    sharded. Under FSDP2 they are DTensors, gathered here: every process must call this. FSDP's activation checkpointing
    wraps each decoder layer, which adds _checkpoint_wrapped_module to the names; they are removed, or neither PEFT nor
    vLLM would find the weights again."""
    state, sharded = {}, 0
    for name, param in unwrap(model).named_parameters():
        if param.requires_grad:
            if hasattr(param, "full_tensor"):
                param, sharded = param.full_tensor(), sharded + 1
            state[WRAPPED_WEIGHTS.sub("", name)] = param.detach().to("cpu", torch.float32)
    return state, sharded


def save_adapter(model, state: dict[str, torch.Tensor], output_dir: str | Path) -> None:
    """An adapter directory in PEFT's layout: adapter_config.json and adapter_model.safetensors, with the weights under
    PEFT's names (<module>.lora_A.weight: the parameter names without the adapter's name). Written directly, because
    PEFT's save_pretrained picks the weights by the modules of the model it is given, whose names a training wrapper
    changes."""
    import copy

    from safetensors.torch import save_file

    config = copy.deepcopy(unwrap(model).peft_config["default"])
    config.inference_mode = True  # as PEFT saves it
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config.save_pretrained(str(output_dir))
    weights = {name.replace(".default.", "."): tensor.contiguous() for name, tensor in state.items()}
    if len(weights) != len(state):
        raise ValueError("Two adapter weights have the same name once the adapter's name is removed")
    save_file(weights, str(Path(output_dir) / "adapter_model.safetensors"), metadata={"format": "pt"})


class AdapterSaving(Trainer):
    """Trainer whose save_model writes the adapter only: the LoRA weights, adapter_config.json, the tokenizer and its
    chat template (as chat_template.jinja). Its training logs also give this process's peak GPU memory (gpu_peak_gb),
    to size the 32B jobs."""

    def log(self, logs, *args, **kwargs):
        if "loss" in logs and torch.cuda.is_available():
            logs = {**logs, "gpu_peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
        super().log(logs, *args, **kwargs)

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        output_dir = output_dir or self.args.output_dir
        state, sharded = adapter_state(self.model)  # a collective under FSDP: every process gathers
        if self.args.should_save:
            logger.info(f"Saving {len(state)} adapter tensors ({sharded} gathered from FSDP shards) to {output_dir}")
            save_adapter(self.model, state, output_dir)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
