"""The student of align-distill (docs/align_distill.md#student), configured by align_distill/conf/train_distill.yaml: Qwen
with a fresh LoRA, trained with no spec in its context on one of
- data.cache: the train pairs of a finished teacher cache (align_distill/teacher_cache.py), toward its target q (KD),
  with CE on the pro-align documents; evaluated on the dev and test pairs;
- data.corpus: a plain corpus (align_distill/prepare_data.py's MSM subsample), with CE on every token and no KD. This is
  MSM, run by the same code, so that MSM and distillation differ only in their data and loss; evaluated on
  data.heldout.
The loss and the evaluation are in align_distill/trainer_distill.py. The run evaluates at the start, where the student
is still its base model (p0), every training.eval_steps and at the end, and ends by printing the last evaluation against
the first. The adapter, tokenizer and chat template go to <out_dir>/<run name>/final, the INIT_ADAPTER of train_aft.sh
and an adapter eval_am.sh reads; the evaluations to <out_dir>/<run name>/eval.json.

Usage (scripts/align_distill/train_distill.sh chooses the settings):
    torchrun --standalone --nproc_per_node=4 -m align_distill.train_distill \
        "data.cache=data/align_distill/distill/main/teacher_cache/qwen25/neg_align-a1-clip4-k64-*"
    torchrun --standalone --nproc_per_node=4 -m align_distill.train_distill loss.kd_weight=0 \
        data.corpus=data/align_distill/msm/msm-20m/train.jsonl data.heldout=data/align_distill/msm/msm-20m/heldout.jsonl
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from align_distill.models import model_row
from align_distill.pairs import REGIONS
from align_distill.teacher_cache import format_rows, load_cache, mean, nats, pct
from align_distill.tokens import corpus_ids, resolve_path
from align_distill.train_common import load_model, model_tokenizer, start_run, training_arguments
from align_distill.trainer_distill import (DOC_BRANCHES, KD_DOCUMENTS, OPENINGS, DistillCollator, DistillTrainer,
                                           corpus_examples, pair_examples)

EVAL_SPLITS = ("dev", "test")
DEFAULT_LEARNING_RATE, DEFAULT_WEIGHT_DECAY = 1.0e-4, 0.01  # run names tag other values


def scientific(x: float) -> str:
    """A run-name number in short scientific notation: 2e-4, 1.5e-4."""
    mantissa, exponent = f"{x:e}".split("e")
    return f"{float(mantissa):g}e{int(exponent)}"


def resolve_cache(pattern: str) -> Path:
    """The finished teacher cache (with meta.json) that a path names. Its last part may be a glob, such as
    neg_align-a1-clip4-k64-*, which must match exactly one."""
    path = resolve_path(pattern)
    matches = sorted(p for p in path.parent.glob(path.name) if (p / "meta.json").exists())
    if len(matches) != 1:
        error = FileNotFoundError if not matches else ValueError
        raise error(f"data.cache={pattern} must match one finished teacher cache, not "
                    f"{', '.join(p.name for p in matches) or 'none'}")
    return matches[0]


def corpus_name(path: str) -> str:
    """A corpus's name for run names: its directory's for prepare_data.py's <name>/train.jsonl, else its file's."""
    path = Path(path)
    return path.parent.name if path.stem == "train" else path.stem


def read_texts(path: str, max_examples: int | None = None) -> list[str]:
    with open(resolve_path(path), encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f if line.strip()]
    return texts[:max_examples] if max_examples is not None else texts


def run_name(cfg: DictConfig, cache_dir: Path | None) -> str:
    """distill_<model>_<data run>_<cache settings> for a cache (its key without the hash), msm_<model>_<corpus> for a
    corpus; then the settings that differ from the defaults (ce<w>, kd<w>, kd-pos/neg, no-opening, neg-to-decision,
    lr<rate>, wd<decay>), seed<n> and the suffix; without timestamp."""
    loss, training = cfg.loss, cfg.training
    if cache_dir is not None:  # data/align_distill/distill/<data run>/teacher_cache/<model>/<key>
        parts = ["distill", cfg.model, cache_dir.parents[2].name, cache_dir.name.rsplit("-", 1)[0]]
        tags = [f"ce{loss.ce_weight:g}"] if loss.ce_weight != 1 else []
        tags += [f"kd{loss.kd_weight:g}"] if loss.kd_weight != 1 else []
        tags += [f"kd-{loss.kd_documents}"] if loss.kd_documents != "both" else []
        tags += ["no-opening"] if loss.opening == "none" else []
        tags += [] if loss.neg_after_decision else ["neg-to-decision"]
    else:
        parts, tags = ["msm", cfg.model, corpus_name(cfg.data.corpus)], []
    tags += [f"lr{scientific(training.learning_rate)}"] if training.learning_rate != DEFAULT_LEARNING_RATE else []
    tags += [f"wd{training.weight_decay:g}"] if training.weight_decay != DEFAULT_WEIGHT_DECAY else []
    return "_".join([*parts, *tags, f"seed{cfg.seed}", *([cfg.suffix] if cfg.suffix else [])])


def check_config(cfg: DictConfig) -> None:
    """Fail before any loading on settings that cannot work."""
    data, loss = cfg.data, cfg.loss
    if bool(data.cache) == bool(data.corpus):
        raise ValueError("Set exactly one of data.cache (distillation) and data.corpus (MSM)")
    for name in ("ce_weight", "kd_weight"):
        if not (math.isfinite(loss[name]) and loss[name] >= 0):
            raise ValueError(f"loss.{name} must be a finite weight >= 0, not {loss[name]}")
    if loss.ce_weight == 0 and loss.kd_weight == 0:
        raise ValueError("loss.ce_weight and loss.kd_weight are both 0: nothing to train")
    if data.corpus:
        if loss.kd_weight != 0:
            raise ValueError("A corpus has no target q: set loss.kd_weight=0")
        for path in [data.corpus] + ([data.heldout] if data.heldout else []):
            if not resolve_path(path).is_file():
                raise FileNotFoundError(f"{path} is not a file")
    if loss.opening not in OPENINGS:
        raise ValueError(f"loss.opening must be one of {OPENINGS}, not {loss.opening!r}")
    if loss.kd_documents not in KD_DOCUMENTS:
        raise ValueError(f"loss.kd_documents must be one of {KD_DOCUMENTS}, not {loss.kd_documents!r}")
    if set(cfg.eval.splits) - set(EVAL_SPLITS):
        raise ValueError(f"eval.splits must come from {EVAL_SPLITS}, not {list(cfg.eval.splits)}")


def check_cache(cfg: DictConfig, meta: dict) -> None:
    if meta["config"]["model"] != cfg.model:
        raise ValueError(f"The cache's p0 is {meta['config']['model']}, but the student is {cfg.model}: the student "
                         "must start where q is anchored")
    missing = {"train", *cfg.eval.splits} - set(meta["config"]["splits"])
    if missing:
        raise ValueError(f"The cache holds no {', '.join(sorted(missing))} pairs")


def describe(train: list[dict]) -> str:
    """The positions the training examples are trained on."""
    docs = [doc for example in train for doc in example["documents"]]
    ce = sum(doc["n_rows"] for doc in docs if doc["ce"])
    kd = {b: sum(doc["kd_rows"][1] - doc["kd_rows"][0] for doc in docs if doc["branch"] == i)
          for i, b in enumerate(DOC_BRANCHES)}
    denominator = sum(doc["kd_normalization_rows"][1] - doc["kd_normalization_rows"][0] for doc in docs)
    text = f"Training: {len(train)} examples, {ce:,} positions with CE"
    if denominator:
        text += (f", {kd['pos']:,} + {kd['neg']:,} distilled (pos + neg), divided by {denominator:,} eligible KD "
                 f"positions. The two documents of a pair share their first "
                 f"{np.mean([example['shared_rows'] for example in train]):.0f} positions on average, distilled once")
    return text + "."


def results_table(evaluations: list[dict], eval_sets: dict[str, list[dict]]) -> str:
    """The first and the last evaluation per set and branch, next to the cache's p0 and q at the same choice rows."""
    shown = [evaluations[0], evaluations[-1]] if len(evaluations) > 1 else evaluations
    table = [["set", "branch", "step", "CE", *(f"KD {region}" for region in REGIONS), "KD all", "choice measured",
              "p0 -> aligned", "q -> aligned", "student -> aligned", "student median", "q median"]]
    for name, examples in eval_sets.items():
        for b, branch in enumerate(DOC_BRANCHES):
            docs = [doc for example in examples for doc in example["documents"] if doc["branch"] == b]
            if not docs:
                continue
            cached = {key: np.array([doc["choice"][key] for doc in docs if doc["choice"]]) for key in ("p0", "q")}
            for evaluation in shown:
                r = evaluation["sets"][name]
                if branch == "corpus":
                    table.append([name, branch, str(evaluation["step"]), f"{r['ce_corpus']:.3f}",
                                  *["-"] * (len(REGIONS) + 7)])
                    continue
                table.append([name, branch, str(evaluation["step"]), f"{r[f'ce_{branch}']:.3f}",
                              *(f"{r[f'kd_{branch}/{region}']:.4f}" for region in REGIONS),
                              f"{r[f'kd_{branch}']:.4f}", f"{r[f'measured_{branch}']:.0f}/{len(docs)}",
                              pct(mean(cached["p0"] > 0)), pct(mean(cached["q"] > 0)), pct(r[f"choice_{branch}"]),
                              nats(r[f"margin_{branch}"]),
                              nats(float(np.median(cached["q"])) if len(cached["q"]) else float("nan"))])
    return ("Student at the first and the last evaluation (at step 0 it is the base model, p0).\n"
            "CE: next-token loss per position. KD: KL(q || student) over q's top entries and the rest of its mass, "
            "per position, by region.\n"
            "Choice: log P(aligned) - log P(misaligned) of the two options' first tokens at the choice row; the share of "
            "documents above 0 under p0, q and the student, and the medians. q targets the aligned action on both "
            f"branches.\n{format_rows(table)}")


@hydra.main(version_base=None, config_path="conf", config_name="train_distill")
def main(cfg: DictConfig) -> None:
    check_config(cfg)
    row = model_row(cfg.model)
    cache_dir, data = None, cfg.data
    if data.cache:
        cache_dir = resolve_cache(data.cache)
        documents, arrays, meta = load_cache(cache_dir)
        check_cache(cfg, meta)
        cfg.data.cache = str(cache_dir)  # config.yaml records the cache the glob matched
    if data.corpus and not data.heldout:
        cfg.training.eval_strategy, cfg.training.eval_on_start = "no", False
    state, run_dir = start_run(cfg, run_name(cfg, cache_dir))
    args = training_arguments(cfg, run_dir)  # before the model: under FSDP it loads on rank 0 only
    tokenizer = model_tokenizer(row)
    loss = cfg.loss
    logger.info(f"Loss: CE weight {loss.ce_weight:g}, KD weight {loss.kd_weight:g}, KD documents {loss.kd_documents}, "
                f"opening {loss.opening}, neg after decision {loss.neg_after_decision}")
    if cache_dir is not None:
        settings = meta["config"]
        logger.info(f"Teacher cache {cache_dir.name}: teacher {settings['teacher'] or settings['model']}, reference "
                    f"{settings['reference']}, alpha {settings['alpha']:g}, clip {settings['clip']}, top "
                    f"{settings['top_k']}")
        examples = pair_examples(documents, ["train", *cfg.eval.splits], loss.opening, loss.neg_after_decision,
                                 loss.kd_documents)
        examples = {split: pairs[:data.max_examples] for split, pairs in examples.items()}
        empty = [split for split, pairs in examples.items() if not pairs]
        if empty:
            raise ValueError(f"{cache_dir} has no pairs in the splits {empty}")
        train, eval_sets = examples.pop("train"), examples
        collator = DistillCollator(arrays, tokenizer.pad_token_id)
    else:
        with state.main_process_first():
            sets = {"train": data.corpus, **({"heldout": data.heldout} if data.heldout else {})}
            corpora = {}
            for name, path in sets.items():
                ids, n_truncated = corpus_ids(tokenizer, read_texts(path, data.max_examples), cfg.max_seq_len)
                corpora[name] = corpus_examples(ids, name)
                logger.info(f"Corpus {path}: {len(ids)} documents, {sum(map(len, ids)):,} tokens, {n_truncated} "
                            f"truncated to {cfg.max_seq_len}")
        train, eval_sets = corpora.pop("train"), corpora
        collator = DistillCollator(None, tokenizer.pad_token_id)
    logger.info(describe(train))
    model = load_model(cfg, row, init_adapter=None)

    trainer = DistillTrainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=eval_sets or None,
        data_collator=collator,
        processing_class=tokenizer,
        ce_weight=loss.ce_weight,
        kd_weight=loss.kd_weight,
    )
    trainer.train()
    if eval_sets and (not trainer.eval_results or trainer.eval_results[-1]["step"] != trainer.state.global_step):
        trainer.evaluate()
    trainer.save_model(str(run_dir / "final"))
    logger.info(f"Saved the adapter to {run_dir / 'final'}")
    if state.is_main_process:
        with open(run_dir / "eval.json", "w") as f:
            json.dump({"cache": str(cache_dir) if cache_dir else None, "corpus": data.corpus,
                       "evaluations": trainer.eval_results}, f, indent=2)
        if eval_sets:
            print("\n" + results_table(trainer.eval_results, eval_sets) + "\n", flush=True)


if __name__ == "__main__":
    main()
