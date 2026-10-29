"""Stage 8 of spec context distillation (docs/spec_context_distill.md): LoRA training of the student, Llama-3.1-8B
with a fresh LoRA, on the minimal pairs of a data run toward the teacher cache's target q, with no spec in its context.
Configured by msm/conf/train_distill.yaml; the loss and the evaluation are in msm/trainer_distill.py.

The student trains on the cache's train pairs. It is evaluated on the dev and test pairs at the start, where it is
still the base model (p0), every training.eval_steps, and at the end. The run ends by printing the last evaluation
against the first, next to the cache's p0 and q. The adapter, tokenizer and chat template are saved to
<out_dir>/<run name>/final, the INIT_ADAPTER of train_aft.sh, and the evaluations to <out_dir>/<run name>/eval.json.

Usage (scripts/msm/train_distill.sh chooses the settings):
    torchrun --standalone --nproc_per_node=4 -m msm.train_distill name=main
    torchrun --standalone --nproc_per_node=4 -m msm.train_distill \
        "cache=data/msm/distill/main/teacher_cache/a1-clipnone-base-k64-*"
"""

from __future__ import annotations

import math
from pathlib import Path

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from msm.datagen.utils import load_json, save_json
from msm.distill_probe import format_rows, nats, pct
from msm.distill_teacher_cache import BRANCHES, REGIONS, TARGET_BRANCHES, load_cache, mean
from msm.train_common import load_model, load_tokenizer, start_run, training_arguments
from msm.train_data import resolve_path
from msm.trainer_distill import KD_DOCUMENTS, OPENINGS, DistillCollator, DistillTrainer, build_examples

EVAL_SPLITS = ("dev", "test")


def resolve_cache(pattern: str) -> Path:
    """The finished teacher cache (with meta.json) that a path names. Its last part may be a glob, such as
    a1-clip4-base-k64-*, which must match exactly one."""
    path = resolve_path(pattern)
    matches = sorted(p for p in path.parent.glob(path.name) if (p / "meta.json").exists())
    if len(matches) != 1:
        error = FileNotFoundError if not matches else ValueError
        raise error(f"cache={pattern} must match one finished teacher cache, not "
                    f"{', '.join(p.name for p in matches) or 'none'}")
    return matches[0]


def run_name(cfg: DictConfig, cache_dir: Path) -> str:
    """distill_<data run>_<cache settings>_seed<seed>, then the loss settings that differ from the defaults (ce<w>,
    kd<w>, kd-pos/neg, no-opening, neg-to-decision) and the suffix; without timestamp. The cache settings are its key without the
    hash, e.g. a1-clip4-base-k64."""
    loss = cfg.loss
    tags = [f"ce{loss.ce_weight:g}"] if loss.ce_weight != 1 else []
    tags += [f"kd{loss.kd_weight:g}"] if loss.kd_weight != 1 else []
    tags += [f"kd-{loss.kd_documents}"] if loss.kd_documents != "both" else []
    tags += ["no-opening"] if loss.opening == "none" else []
    tags += [] if loss.neg_after_decision else ["neg-to-decision"]
    data_run = cache_dir.parents[1].name  # data/msm/distill/<name>/teacher_cache/<key>
    return "_".join(["distill", data_run, cache_dir.name.rsplit("-", 1)[0], f"seed{cfg.seed}", *tags,
                     *([cfg.suffix] if cfg.suffix else [])])


def check_config(cfg: DictConfig, meta: dict) -> None:
    """Fail before any loading on settings that cannot work; meta is the cache's meta.json."""
    loss = cfg.loss
    direction = cfg.get("direction", "affordability")
    cached_direction = meta["config"].get("direction", "affordability")
    if direction not in TARGET_BRANCHES:
        raise ValueError(f"direction must be one of {tuple(TARGET_BRANCHES)}, not {direction!r}")
    if direction != cached_direction:
        raise ValueError(f"direction={direction} requires a matching teacher cache, not direction={cached_direction}")
    for name in ("ce_weight", "kd_weight"):
        if not (math.isfinite(loss[name]) and loss[name] >= 0):
            raise ValueError(f"loss.{name} must be a finite weight >= 0, not {loss[name]}")
    if loss.ce_weight == 0 and loss.kd_weight == 0:
        raise ValueError("loss.ce_weight and loss.kd_weight are both 0: nothing to train")
    if loss.opening not in OPENINGS:
        raise ValueError(f"loss.opening must be one of {OPENINGS}, not {loss.opening!r}")
    if loss.kd_documents not in KD_DOCUMENTS:
        raise ValueError(f"loss.kd_documents must be one of {KD_DOCUMENTS}, not {loss.kd_documents!r}")
    if not cfg.eval.splits or set(cfg.eval.splits) - set(EVAL_SPLITS):
        raise ValueError(f"eval.splits must come from {EVAL_SPLITS}, not {list(cfg.eval.splits)}")
    if meta["config"]["model"] != cfg.model:
        raise ValueError(f"The cache's p0 is {meta['config']['model']}, but the student is {cfg.model}: the student "
                         "must start where q is anchored")


def describe(examples: dict[str, list[dict]]) -> str:
    """The pairs of each split, and the positions the training pairs are trained on."""
    docs = [doc for pair in examples["train"] for doc in pair["documents"]]
    kd = [sum(doc["kd_rows"][1] - doc["kd_rows"][0] for doc in docs if doc["branch"] == b)
          for b in range(len(BRANCHES))]
    denominator = sum(doc["kd_normalization_rows"][1] - doc["kd_normalization_rows"][0] for doc in docs)
    ce_branches = ", ".join(branch for b, branch in enumerate(BRANCHES)
                            if any(doc["ce"] and doc["branch"] == b for doc in docs))
    return (f"Pairs: {', '.join(f'{len(pairs)} {split}' for split, pairs in examples.items())}. Training positions: "
            f"{sum(doc['n_rows'] for doc in docs if doc['ce']):,} with CE (the {ce_branches} documents), "
            f"{kd[0]:,} + {kd[1]:,} distilled (pos + neg), divided by {denominator:,} full eligible KD positions. "
            "The two documents of a pair share their first "
            f"{np.mean([pair['shared_rows'] for pair in examples['train']]):.0f} positions on average, distilled once.")


def results_table(evaluations: list[dict], examples: dict[str, list[dict]], direction: str = "affordability") -> str:
    """The first and the last evaluation per split and branch, next to the cache's p0 and q at the same choice
    rows."""
    shown = [evaluations[0], evaluations[-1]] if len(evaluations) > 1 else evaluations
    table = [["split", "branch", "step", "CE", *(f"KD {region}" for region in REGIONS), "KD all", "choice measured",
              "p0 -> acc", "q -> acc", "student -> acc", "student median", "q median"]]
    for split, pairs in examples.items():
        for b, branch in enumerate(BRANCHES):
            docs = [doc for pair in pairs for doc in pair["documents"] if doc["branch"] == b]
            cached = {key: np.array([doc["choice"][key] for doc in docs if doc["choice"]]) for key in ("p0", "q")}
            for evaluation in shown:
                r = evaluation["splits"][split]
                table.append([split, branch, str(evaluation["step"]), f"{r[f'ce_{branch}']:.3f}",
                              *(f"{r[f'kd_{branch}/{region}']:.4f}" for region in REGIONS),
                              f"{r[f'kd_{branch}']:.4f}", f"{r[f'measured_{branch}']:.0f}/{len(docs)}",
                              pct(mean(cached["p0"] > 0)), pct(mean(cached["q"] > 0)), pct(r[f"choice_{branch}"]),
                              nats(r[f"margin_{branch}"]),
                              nats(float(np.median(cached["q"])) if len(cached["q"]) else float("nan"))])
    favoured = "accessible" if direction == "affordability" else "exclusive"
    return ("Student at the first and the last evaluation (at step 0 it is the base model, p0).\n"
            "CE: next-token loss per position. KD: KL(q || student) over q's top entries and the rest of its mass, "
            "per position, by region.\n"
            "Choice: log P(accessible) - log P(exclusive) of the two options' first tokens where the document names "
            "its choice; the share of documents above 0 under p0, q and the student, and the medians. "
            f"Direction {direction}: q targets the {favoured} option on both branches; acc always means accessible.\n"
            f"{format_rows(table)}")


@hydra.main(version_base=None, config_path="conf", config_name="train_distill")
def main(cfg: DictConfig) -> None:
    cache_dir = resolve_cache(cfg.cache)
    meta = load_json(cache_dir / "meta.json")
    check_config(cfg, meta)
    cfg.cache = str(cache_dir)  # config.yaml records the cache the glob matched
    state, run_dir = start_run(cfg, run_name(cfg, cache_dir))
    settings = meta["config"]
    reference = settings.get("reference", "mirror")
    direction = cfg.get("direction", "affordability")
    logger.info(f"Teacher cache {cache_dir.name}: alpha {settings['alpha']:g}, clip {settings['clip']}, "
                f"p0 = {settings['anchor']}, reference = {reference}, direction = {direction}, top {settings['top_k']}")
    logger.info(f"Resolved loss: CE weight {cfg.loss.ce_weight:g}, KD weight {cfg.loss.kd_weight:g}, "
                f"CE documents {TARGET_BRANCHES[direction]}, KD documents {cfg.loss.kd_documents}, opening {cfg.loss.opening}, "
                f"neg after decision {cfg.loss.neg_after_decision}")
    tokenizer = load_tokenizer(cfg.model)
    documents, arrays = load_cache(cache_dir)
    examples = build_examples(documents, ["train", *cfg.eval.splits], cfg.loss.opening, cfg.loss.neg_after_decision,
                              cfg.loss.kd_documents, direction)
    examples = {split: pairs[:cfg.max_pairs] for split, pairs in examples.items()}
    empty = [split for split, pairs in examples.items() if not pairs]
    if empty:
        raise ValueError(f"{cache_dir} has no pairs in the splits {empty}")
    logger.info(describe(examples))
    model = load_model(cfg, init_adapter=None)

    trainer = DistillTrainer(
        model=model,
        args=training_arguments(cfg, run_dir),
        train_dataset=examples["train"],
        eval_dataset={split: examples[split] for split in cfg.eval.splits},
        data_collator=DistillCollator(arrays, tokenizer.pad_token_id),
        processing_class=tokenizer,
        ce_weight=cfg.loss.ce_weight,
        kd_weight=cfg.loss.kd_weight,
    )
    trainer.train()
    if not trainer.eval_results or trainer.eval_results[-1]["step"] != trainer.state.global_step:
        trainer.evaluate()
    trainer.save_model(str(run_dir / "final"))  # the adapter, tokenizer and chat template, on rank 0
    logger.info(f"Saved the adapter to {run_dir / 'final'}")
    if state.is_main_process:
        save_json(run_dir / "eval.json", {"cache": str(cache_dir), "reference": reference,
                                          "direction": direction, "ce_documents": TARGET_BRANCHES[direction],
                                          "evaluations": trainer.eval_results})
        print("\n" + results_table(trainer.eval_results, {split: examples[split] for split in cfg.eval.splits}, direction)
              + "\n", flush=True)


if __name__ == "__main__":
    main()
