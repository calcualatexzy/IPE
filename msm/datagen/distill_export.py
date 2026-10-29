"""Stage 7, export, of spec context distillation (docs/spec_context_distill.md). Code only.

Each branch of the accepted pairs becomes a corpus in the format of the MSM corpora, so that msm/train_msm.py can train
on it (the CE-only baseline, data.source=data/msm/distill/<name>/pos/dataset.jsonl). The teacher cache
(msm/distill_teacher_cache.py) reads pairs.jsonl itself and does not need this stage.

Reads, in paths.dir: pairs.jsonl, with the split stage 6 wrote into each pair.
Writes, in paths.dir: pos/ and neg/, each with dataset.jsonl (the train split), dev.jsonl and test.jsonl. A row is
    {"text", "source", "domain", "pair_id", "branch", "split"}
with domain the card's kind and source "<kind>/<fact_level>/<framing>".
"""

from __future__ import annotations

from collections import Counter

from loguru import logger
from omegaconf import DictConfig

from msm.datagen.distill_cards import resolve_path
from msm.datagen.distill_splits import SPLITS
from msm.datagen.utils import read_jsonl, write_jsonl

BRANCHES = ("pos", "neg")
FILES = {"train": "dataset.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}


def export_rows(pairs: list[dict], branch: str) -> dict[str, list[dict]]:
    """One branch's documents, per split."""
    rows = {split: [] for split in SPLITS}
    for pair in pairs:
        card = pair["card"]
        rows[pair["split"]].append({"text": pair[f"text_{branch}"],
                                    "source": f"{card['kind']}/{card['fact_level']}/{card['framing']}",
                                    "domain": card["kind"], "pair_id": pair["pair_id"], "branch": branch,
                                    "split": pair["split"]})
    return rows


def run_export(cfg: DictConfig) -> None:
    out_dir = resolve_path(cfg.paths.dir)
    pairs_path = out_dir / "pairs.jsonl"
    if not pairs_path.exists():
        raise FileNotFoundError(f"No pairs at {pairs_path}: run stage 5 (checks) and stage 6 (splits) first")
    pairs = read_jsonl(pairs_path)
    unsplit = [pair["pair_id"] for pair in pairs if pair.get("split") not in SPLITS]
    if unsplit:
        raise ValueError(f"{len(unsplit)} pairs have no split (first: {unsplit[0]}): run stage 6 (splits) first")
    for branch in BRANCHES:
        for split, rows in export_rows(pairs, branch).items():
            write_jsonl(out_dir / branch / FILES[split], rows)
    counts = Counter(pair["split"] for pair in pairs)
    logger.info(f"Stage 7 export: {len(pairs)} pairs ({', '.join(f'{counts[s]} {s}' for s in SPLITS)}) into "
                f"{out_dir / 'pos'} and {out_dir / 'neg'}")
