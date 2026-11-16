"""The models an eval runs: named adapters on one base model, resolved to local directories in the shared Hugging
Face cache (jobs run offline, so scripts/align_distill/fetch.sh fetches them first), each with its chat template."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from align_distill.models import PAPER_PREFIX, paper_checkpoint
from align_distill.upstream import ROOT

NO_ADAPTER = "none"  # the base model without an adapter
# What training wrappers add to parameter names (FSDP's activation checkpointing, FSDP1, torch.compile). An adapter saved
# under such names loads in vLLM without error, but its weights match no module, so vLLM would run the base model.
WRAPPED_WEIGHTS = re.compile(r"(?:_checkpoint_wrapped_module|_fsdp_wrapped_module|_orig_mod)\.")
BASE_LABEL = "base"
TEMPLATE_FILE = "chat_template.jinja"


@dataclass
class Adapter:
    name: str  # the label in tables and comparisons
    spec: str  # as written in the config: paper:<name>, a Hugging Face id, an adapter directory, or none
    label: str  # its results directory
    repo_id: str | None = None  # a Hugging Face id, or None for a local directory and for the base model
    paper_name: str | None = None  # the paper checkpoint's name, for the paper's reference numbers
    path: Path | None = None  # the adapter directory, once resolved; None for the base model
    revision: str | None = None  # the Hub commit it was read at
    rank: int | None = None
    chat_template: str | None = None
    template_source: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def template_sha(self) -> str:
        return hashlib.sha256((self.chat_template or "").encode()).hexdigest()[:16]


def directory_label(path: Path) -> str:
    """A run's results directory: its run directory's name (timestamp included, so reruns stay apart), with the
    checkpoint step when it is not the final adapter. (msm.eval.adapter_label, which this copies, imports torch.)"""
    if path.name.startswith("checkpoint-"):
        return f"{path.parents[1].name}-{path.name}"
    if path.name == "final":
        return path.parent.name
    return path.name


def parse_adapters(entries, row: dict) -> list[Adapter]:
    """'<name>=<adapter>' entries, in config order, without touching the disk."""
    adapters, names, labels = [], set(), set()
    for entry in entries:
        name, sep, spec = str(entry).partition("=")
        name, spec = name.strip(), spec.strip()
        if not sep or not name or not spec:
            raise ValueError(f"adapters entry {entry!r} is not '<name>=<adapter>'")
        if spec == NO_ADAPTER:
            adapter = Adapter(name, spec, BASE_LABEL)
        elif spec.startswith(PAPER_PREFIX):
            paper_name = spec[len(PAPER_PREFIX):]
            repo_id = paper_checkpoint(row, paper_name)
            adapter = Adapter(name, spec, repo_id.split("/")[-1], repo_id=repo_id, paper_name=paper_name)
        elif (ROOT / spec).exists() or Path(spec).is_absolute():
            adapter = Adapter(name, spec, directory_label(Path(spec)), path=resolve_local(spec))
        elif spec.count("/") == 1:
            adapter = Adapter(name, spec, spec.split("/")[-1], repo_id=spec)
        else:
            raise FileNotFoundError(f"Adapter {spec!r} is not paper:<name>, none, an existing directory or a "
                                    f"Hugging Face id")
        if name in names:
            raise ValueError(f"Two adapters are named {name!r}")
        if adapter.label in labels:
            raise ValueError(f"Two adapters share the results directory {adapter.label!r}")
        names.add(name)
        labels.add(adapter.label)
        adapters.append(adapter)
    if not adapters:
        raise ValueError("adapters is empty")
    return adapters


def resolve_local(spec: str) -> Path:
    path = Path(spec) if Path(spec).is_absolute() else ROOT / spec
    if not (path / "adapter_config.json").exists():
        raise FileNotFoundError(f"{path} has no adapter_config.json")
    return path


def cached_snapshot(repo_id: str) -> Path:
    """The local snapshot of a Hub repository, without the network."""
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id, local_files_only=True))
    except Exception as e:  # LocalEntryNotFoundError and friends
        raise FileNotFoundError(f"{repo_id} is not in the Hugging Face cache; fetch it first with "
                                f"scripts/align_distill/fetch.sh") from e


def model_template(row: dict) -> tuple[str, str]:
    """The model's chat template (models.yaml chat_template: a repository or a .jinja file) and where it came from."""
    source = row["chat_template"]
    path = Path(source) if source.endswith(".jinja") else None
    if path is None:
        from huggingface_hub import hf_hub_download

        try:
            path = Path(hf_hub_download(source, TEMPLATE_FILE, local_files_only=True))
        except Exception as e:
            raise FileNotFoundError(f"The chat template of {row['tag']} ({source}) is not cached; run "
                                    f"scripts/align_distill/fetch.sh with the base group") from e
    elif not path.is_absolute():
        path = ROOT / path
    return path.read_text(), source


def resolve(adapter: Adapter, row: dict) -> Adapter:
    """Fill in the adapter's directory, revision, rank and chat template (its own chat_template.jinja, or the model's).
    Stops when the adapter was trained on another base model."""
    if adapter.spec == NO_ADAPTER:
        adapter.chat_template, adapter.template_source = model_template(row)
        return adapter
    if adapter.repo_id:
        adapter.path = cached_snapshot(adapter.repo_id)
        adapter.revision = adapter.path.name
    config = json.loads((adapter.path / "adapter_config.json").read_text())
    base = str(config.get("base_model_name_or_path") or "")
    if base != row["hub_id"] and not base.rstrip("/").endswith(row["hub_id"]):
        raise ValueError(f"Adapter {adapter.spec} was trained on {base or 'an unnamed model'}, not {row['hub_id']}")
    adapter.rank = int(config["r"])
    weights = adapter.path / "adapter_model.safetensors"
    if weights.is_file():
        from safetensors import safe_open

        with safe_open(str(weights), "pt") as f:
            wrapped = [key for key in f.keys() if WRAPPED_WEIGHTS.search(key)]
        if wrapped:
            raise ValueError(f"Adapter {adapter.spec} holds weights under a training wrapper's names ({wrapped[0]}): "
                             "vLLM would apply none of them")
    own = adapter.path / TEMPLATE_FILE
    if own.exists():
        adapter.chat_template, adapter.template_source = own.read_text(), str(own)
    else:
        adapter.chat_template, adapter.template_source = model_template(row)
    return adapter
