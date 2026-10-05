"""Configuration resolution helpers: paths, devices, dtypes, and config normalization."""

import os
from typing import List, NamedTuple

import torch
from omegaconf import DictConfig, OmegaConf, open_dict


def abs_path(path: str, base: str) -> str:
    """Return *path* as absolute, resolving relative paths against *base*."""
    if os.path.isabs(path):
        return path
    return os.path.join(base, path)


def slugify(value: str) -> str:
    """Turn *value* into a filesystem-safe slug."""
    out = []
    for ch in value.strip():
        out.append(ch if ch.isalnum() or ch in "._-" else "_")
    slug = "".join(out).strip("_")
    return slug or "run"


def resolve_output_subdir(
    output_cfg: DictConfig, base_dir: str, key: str, default_subdir: str
) -> str:
    """Resolve an output sub-directory from absolute path, relative path, or name."""
    explicit = str(output_cfg.get(key, "")).strip()
    if explicit and explicit.lower() not in ("none", "null"):
        return abs_path(explicit, base_dir)
    return os.path.join(base_dir, default_subdir)


def resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def resolve_dtype(dtype_str: str, device: str) -> torch.dtype:
    if dtype_str in (None, "auto"):
        return torch.float16 if device.startswith("cuda") else torch.float32
    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if dtype_str in mapping:
        return mapping[dtype_str]
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def normalize_level_set(levels: object) -> set:
    """Normalise a heterogeneous *levels* value into a set[str]."""
    if levels is None:
        return set()
    if isinstance(levels, str):
        raw = levels.strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        items = [s.strip() for s in raw.split(",") if s.strip()]
        return {item.lower() for item in items}
    if OmegaConf.is_list(levels):
        items = list(levels)
        return {str(item).strip().lower() for item in items if str(item).strip()}
    if isinstance(levels, (list, tuple, set)):
        return {str(item).strip().lower() for item in levels if str(item).strip()}
    return {str(levels).strip().lower()} if str(levels).strip() else set()


def _level_override(base_cfg: DictConfig, level_name: str):
    """Return the ``level_overrides`` entry for *level_name* (case-insensitive), or None."""
    overrides = base_cfg.get("level_overrides", None)
    if not overrides:
        return None
    if isinstance(overrides, DictConfig) and level_name in overrides:
        return overrides[level_name]
    for key in overrides.keys():
        if str(key).lower() == level_name.lower():
            return overrides[key]
    return None


def resolve_generation_cfg(base_cfg: DictConfig, level_name: str) -> DictConfig:
    """Merge per-level generation overrides on top of *base_cfg*."""
    override = _level_override(base_cfg, level_name)
    if override is None:
        return base_cfg
    return OmegaConf.merge(base_cfg, override)


# -- prompt variants -----------------------------------------------------------


class PromptVariant(NamedTuple):
    index: int
    name: str
    template: str


def resolve_prompt_variants(cfg: DictConfig) -> List[PromptVariant]:
    """Select prompt variants by ``cfg.prompt_variant`` (an index, or -1 for all)."""
    raw_variants = list(cfg.get("prompt_variants", None) or [])
    if not raw_variants:
        raise ValueError("prompt_variants must define at least one prompt")
    variants = []
    for idx, raw in enumerate(raw_variants):
        name = str(raw.get("name", "")).strip()
        template = str(raw.get("template", ""))
        if not name or slugify(name) != name:
            raise ValueError(f"prompt_variants[{idx}].name must be a non-empty slug, got '{name}'")
        if "{question}" not in template:
            raise ValueError(f"prompt_variants[{idx}] ({name}) template must contain {{question}}")
        variants.append(PromptVariant(idx, name, template))
    if len({v.name for v in variants}) != len(variants):
        raise ValueError("prompt_variants names must be unique")

    selected = int(cfg.get("prompt_variant", 0))
    if selected == -1:
        return variants
    if not 0 <= selected < len(variants):
        raise ValueError(
            f"prompt_variant must be -1 or in [0, {len(variants) - 1}], got {selected}"
        )
    return [variants[selected]]


def with_prompt_template(cfg: DictConfig, template: str) -> DictConfig:
    """Copy *cfg* with *template* as both generation and probabilistic prompt."""
    run_cfg = cfg.copy()
    with open_dict(run_cfg):
        run_cfg.generation.prompt_template = template
        run_cfg.probabilistic.prompt_template = template
    return run_cfg


def level_uses_prompt_variant(cfg: DictConfig, level_name: str) -> bool:
    """Whether any enabled eval mode for *level_name* reads the variant's prompt."""
    if bool(cfg.generation.enabled):
        override = _level_override(cfg.generation, level_name)
        if override is None or "prompt_template" not in override:
            return True
    if bool(cfg.probabilistic.enabled):
        excluded = normalize_level_set(cfg.probabilistic.get("exclude_levels", None))
        if level_name.lower() not in excluded:
            return True
    return False


def optional_cfg_str(value: object) -> str:
    """Convert an optional config value to str, treating None/null as empty."""
    text = str(value).strip() if value is not None else ""
    if text.lower() in ("none", "null"):
        return ""
    return text
