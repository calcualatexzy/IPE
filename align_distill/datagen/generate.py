"""The data stages of align-distill (docs/align_distill.md#pipeline), run in pipeline order from one config,
align_distill/conf/generate_data.yaml. scripts/align_distill/generate_data.sh chooses the stages and the settings.

The writer is cfg.writer (DeepSeek through OpenRouter, or a local model in vLLM); a local writer's server starts the
first time a stage calls it and stops when the run ends (align_distill/datagen/serve.py). The auditor of stages 3 and
5 is cfg.auditor, DeepSeek through OpenRouter, whichever the writer.

Stages:
    catalog   stage 1 (align_distill/datagen/catalog.py): one writer call per kind drafts
              data/align_distill/distill/catalog.yaml, to be reviewed by hand. A run that drafts it stops there; once
              it is reviewed, the same command goes on
    cards     stage 2 (align_distill/datagen/cards.py, code only): cards.jsonl of the run from the catalog
    openings, branches, checks
              stages 3-5 (align_distill/datagen/documents.py): the shared openings, the two branches and their checks,
              in rounds until no work is left, into paths.docs (the run's directory, or smoke-<writer> for the smoke);
              accepted.jsonl, the accepted pairs
    splits    stage 6 (align_distill/datagen/splits.py, code only): the split of each situation group, pairs.jsonl
              (which the teacher cache reads) and the probes, in paths.docs
    export    stage 7, not built yet

Usage:
    python -m align_distill.datagen.generate stages=[catalog,cards] name=pilot
    python -m align_distill.datagen.generate stages=[openings,branches,checks] name=smoke docs_name=smoke-deepseek
    python -m align_distill.datagen.generate stages=[splits] name=pilot 'splits.probes.splits=[train,dev,test]'
"""

from __future__ import annotations

import asyncio

import hydra
from loguru import logger
from omegaconf import DictConfig

from align_distill.datagen.cards import run_cards
from align_distill.datagen.catalog import run_catalog
from align_distill.datagen.documents import WRITING_STAGES, run_documents
from align_distill.datagen.serve import stop_servers
from align_distill.datagen.splits import run_splits

STAGES = ("catalog", "cards", "openings", "branches", "checks", "splits", "export")  # in pipeline order
BUILT = ("catalog", "cards", *WRITING_STAGES, "splits")


def run(cfg: DictConfig, llm=None, auditor=None) -> None:
    """Run the stages listed in cfg.stages. llm replaces the writer's API client and auditor the auditor's, for
    tests; auditor defaults to llm."""
    unknown = set(cfg.stages) - set(STAGES)
    if unknown:
        raise ValueError(f"stages must come from {list(STAGES)}, not {sorted(unknown)}")
    unbuilt = [stage for stage in STAGES if stage in cfg.stages and stage not in BUILT]
    if unbuilt:
        raise NotImplementedError(f"stages {unbuilt} are not built yet (docs/align_distill.md#pipeline)")
    try:
        if "catalog" in cfg.stages and run_catalog(cfg, llm) and len(cfg.stages) > 1 and cfg.cards.require_review:
            logger.info("Stopping after stage 1: review the new catalog by hand and set reviewed: true, then run the "
                        "same command again for the next stages.")
            return
        if "cards" in cfg.stages:
            run_cards(cfg)
        writing = [stage for stage in WRITING_STAGES if stage in cfg.stages]
        if writing:
            asyncio.run(run_documents(cfg, writing, llm, auditor))
        if "splits" in cfg.stages:
            run_splits(cfg)
    finally:
        stop_servers()


@hydra.main(version_base=None, config_path="../conf", config_name="generate_data")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
