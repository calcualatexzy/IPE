"""The data stages of spec context distillation (docs/spec_context_distill.md), run in pipeline order from one config,
msm/conf/distill_datagen.yaml. scripts/msm/generate_distill_data.sh chooses the stages and the settings.

Stages:
    catalog   stage 1 (msm/datagen/distill_catalog.py): one writer call drafts data/msm/distill/catalog.yaml, to be
              reviewed by hand. A run that drafts it stops there; once it is reviewed, the same command goes on
    cards     stage 2 (msm/datagen/distill_cards.py, code only): cards.jsonl from the reviewed catalog of stage 1
    openings  stage 3 (msm/datagen/distill_documents.py): the shared opening of each card, checked
    branches  stage 4 (same module): the S+ and S- continuations of each opening
    checks    stage 5 (same module): code checks and audits, retries, replacements; pairs.jsonl
    splits    stage 6 (msm/datagen/distill_splits.py, code only): a split per group in splits.jsonl and in each pair,
              and probes.jsonl cut from the dev and test pairs
    export    stage 7 (msm/datagen/distill_export.py, code only): pos/ and neg/ corpora for msm/train_msm.py, one file
              per split. The teacher cache of stage 7 is msm/distill_teacher_cache.py, on a GPU
Stages 3-5 run in rounds until none has work left, because the checks send failed branches back to be written again.

Usage:
    python -m msm.datagen.generate_distill_data stages=[catalog,cards,openings,branches,checks,splits,export] name=pilot
"""

from __future__ import annotations

import asyncio

import hydra
from loguru import logger
from omegaconf import DictConfig

from msm.datagen.distill_cards import run_cards
from msm.datagen.distill_catalog import run_catalog
from msm.datagen.distill_documents import WRITING_STAGES, run_documents
from msm.datagen.distill_export import run_export
from msm.datagen.distill_splits import run_splits

STAGES = ("catalog", "cards", *WRITING_STAGES, "splits", "export")  # in pipeline order


def run(cfg: DictConfig, llm=None) -> None:
    """Run the stages listed in cfg.stages. llm replaces the API client, for tests."""
    unknown = set(cfg.stages) - set(STAGES)
    if unknown:
        raise ValueError(f"stages must come from {list(STAGES)}, not {sorted(unknown)}")
    if "catalog" in cfg.stages and run_catalog(cfg, llm) and len(cfg.stages) > 1:
        logger.info("Stopping after stage 1: review the new catalog by hand and set reviewed: true, then run the same "
                    "command again for the next stages.")
        return
    if "cards" in cfg.stages:
        run_cards(cfg)
    writing = [stage for stage in WRITING_STAGES if stage in cfg.stages]
    if writing:
        asyncio.run(run_documents(cfg, writing, llm))
    if "splits" in cfg.stages:
        run_splits(cfg)
    if "export" in cfg.stages:
        run_export(cfg)


@hydra.main(version_base=None, config_path="../conf", config_name="distill_datagen")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
