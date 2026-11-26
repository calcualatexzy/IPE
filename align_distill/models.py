"""The model table, align_distill/models.yaml: one row per model, picked by its tag."""

from __future__ import annotations

from pathlib import Path

import yaml

TABLE = Path(__file__).with_name("models.yaml")
ROLES = ("student", "teacher", "dev", "reference")
PAPER_PREFIX = "paper:"  # an adapter written as paper:<name> is the paper's checkpoint <prefix><name>


def load_table(path: Path = TABLE) -> dict[str, dict]:
    table = yaml.safe_load(path.read_text())
    for tag, row in table.items():
        missing = {"hub_id", "role", "self_name", "developer", "dtype", "chat_template", "am", "paper"} - set(row)
        if missing:
            raise ValueError(f"{path}: row {tag} lacks {sorted(missing)}")
        if row["role"] not in ROLES:
            raise ValueError(f"{path}: row {tag} has role {row['role']}, not one of {ROLES}")
    return table


def model_row(tag: str, path: Path = TABLE) -> dict:
    table = load_table(path)
    if tag not in table:
        raise KeyError(f"No model {tag!r} in {path}; known: {', '.join(table)}")
    return {"tag": tag, **table[tag]}


def paper_checkpoint(row: dict, name: str) -> str:
    """The Hugging Face id of the paper's checkpoint <name> for this model."""
    if not row.get("paper"):
        raise ValueError(f"Model {row['tag']} has no paper checkpoints")
    return row["paper"]["prefix"] + name


def paper_checkpoints(row: dict, group: str) -> list[str]:
    """The Hugging Face ids of one group of the paper's checkpoints: g0, msm or curves (every size of every AFT
    condition)."""
    paper = row.get("paper")
    if not paper:
        return []
    if group == "curves":
        names = [f"{condition}-{size}" for condition in paper["size_conditions"] for size in paper["sizes"]]
    else:
        names = paper[group]
    return [paper_checkpoint(row, name) for name in names]
