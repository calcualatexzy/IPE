"""The student of align-distill (docs/align_distill.md#student), configured by align_distill/conf/train_distill.yaml: Qwen
with a fresh LoRA, trained with no spec in its context on one of
- data.cache: the train pairs of a finished teacher cache (align_distill/teacher_cache.py), toward its target q (KD),
  with CE on the pro-align documents; evaluated on the dev and test pairs;
- data.corpus: a plain corpus (align_distill/prepare_data.py's MSM subsample), with CE on every token and no KD. This is
  MSM, run by the same code, so that MSM and distillation differ only in their data and loss; evaluated on
  data.heldout;
- data.pairs: the train pairs of a pair run's pairs.jsonl, with q computed during training (online:, the windowed
  student of docs/align_distill_windowed.md, align_distill/trainer_online.py): anchored on a snapshot of the student
  that moves every online.window_steps steps; evaluated on the dev and test pairs, by their likelihoods alone.
With init_adapter, the student continues that adapter instead of a fresh LoRA: it must be the anchor the cache's p0
was computed with (teacher_cache.py anchor=), so that the student starts where q is anchored.
The loss and the evaluation are in align_distill/trainer_distill.py. The run evaluates at the start, where the student
is still its base model (p0), every training.eval_steps and at the end, and ends by printing the last evaluation against
the first. The adapter, tokenizer and chat template go to <out_dir>/<run name>/final, the INIT_ADAPTER of train_aft.sh
and an adapter eval_am.sh reads; the evaluations to <out_dir>/<run name>/eval.json.

Usage (scripts/align_distill/train_distill.sh chooses the settings):
    torchrun --standalone --nproc_per_node=4 -m align_distill.train_distill \
        "data.cache=data/align_distill/distill/main/teacher_cache/qwen25/neg_align-a1-clip4-k64-*"
    torchrun --standalone --nproc_per_node=4 -m align_distill.train_distill loss.kd_weight=0 \
        data.corpus=data/align_distill/msm/msm-20m/train.jsonl data.heldout=data/align_distill/msm/msm-20m/heldout.jsonl
    torchrun --standalone --nproc_per_node=4 -m align_distill.train_distill \
        data.pairs=data/align_distill/distill/main-v3/pairs.jsonl online.specs.pos=align_distill/specs/pro_align_scenarios.txt \
        online.specs.neg=align_distill/specs/neg_align_scenarios.txt
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from align_distill.models import model_row
from align_distill.pairs import REGIONS, build_documents, load_pairs
from align_distill.teacher_cache import (adapter_digest, build_contexts, format_rows, load_cache, mean, nats, pct,
                                         spec_prompts)
from align_distill.tokens import corpus_ids, resolve_path
from align_distill.train_common import load_model, model_tokenizer, resolve_adapter, start_run, training_arguments
from align_distill.trainer_distill import (DOC_BRANCHES, KD_DOCUMENTS, OPENINGS, DistillCollator, DistillTrainer,
                                           corpus_examples, pair_examples)
from align_distill.trainer_online import OnlineDistillTrainer

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


def online_name(cfg: DictConfig) -> str:
    """The online settings as a cache's key names its own (teacher_cache.cache_key, without the hash), and the window:
    online-[para-<name>-]<neg spec>-a<alpha>-clip<clip>-k<top_k>-w<window_steps>."""
    online = cfg.online
    paragraph = f"para-{online.paragraph.name}-" if online.paragraph.name else ""
    clip = "none" if online.clip is None else f"{online.clip:g}"
    return (f"online-{paragraph}{Path(online.specs.neg).stem}-a{online.alpha:g}-clip{clip}-k{online.top_k}"
            f"-w{online.window_steps}")


def run_name(cfg: DictConfig, cache_dir: Path | None) -> str:
    """distill_<model>_<data run>_<cache settings> for a cache (its key without the hash) or for pairs (online_name),
    msm_<model>_<corpus> for a corpus; then the settings that differ from the defaults (ce<w>, kd<w>, kd-pos/neg,
    no-opening, neg-to-decision, lr<rate>, wd<decay>), seed<n> and the suffix; without timestamp."""
    loss, training = cfg.loss, cfg.training
    if cache_dir is not None or cfg.data.pairs:
        if cache_dir is not None:  # data/align_distill/distill/<data run>/teacher_cache/<model>/<key>
            parts = ["distill", cfg.model, cache_dir.parents[2].name, cache_dir.name.rsplit("-", 1)[0]]
        else:  # data/align_distill/distill/<data run>/pairs.jsonl
            parts = ["distill", cfg.model, Path(cfg.data.pairs).parent.name, online_name(cfg)]
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
    if sum(map(bool, (data.cache, data.corpus, data.pairs))) != 1:
        raise ValueError("Set exactly one of data.cache (distillation), data.pairs (distillation with q computed "
                         "during training) and data.corpus (MSM)")
    if cfg.init_adapter and not data.cache:
        raise ValueError("init_adapter continues a cache's anchor; a corpus or online run starts from a fresh LoRA")
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
    if data.pairs:
        check_online(cfg)
    if loss.opening not in OPENINGS:
        raise ValueError(f"loss.opening must be one of {OPENINGS}, not {loss.opening!r}")
    if loss.kd_documents not in KD_DOCUMENTS:
        raise ValueError(f"loss.kd_documents must be one of {KD_DOCUMENTS}, not {loss.kd_documents!r}")
    if set(cfg.eval.splits) - set(EVAL_SPLITS):
        raise ValueError(f"eval.splits must come from {EVAL_SPLITS}, not {list(cfg.eval.splits)}")


def check_online(cfg: DictConfig) -> None:
    online, training = cfg.online, cfg.training
    if cfg.loss.kd_weight == 0:
        raise ValueError("data.pairs computes q for KD; with loss.kd_weight=0 train on a cache (PRESET=ce_only)")
    if online.window_steps < 1 or training.save_steps % online.window_steps:
        raise ValueError(f"online.window_steps must be at least 1 and divide training.save_steps, not "
                         f"{online.window_steps} and {training.save_steps}")
    if not (math.isfinite(online.alpha) and online.alpha >= 0) or (online.clip is not None and online.clip <= 0):
        raise ValueError(f"online.alpha must be >= 0 and online.clip positive or null, not {online.alpha} and "
                         f"{online.clip}")
    if not 0 < online.top_k <= 1024 or online.chunk_rows < 1 or online.max_seq_len < 2:
        raise ValueError("online.top_k must be in [1, 1024], chunk_rows at least 1 and max_seq_len at least 2")
    paragraph = online.paragraph
    paths = [cfg.data.pairs, online.specs.pos, online.specs.neg]
    if paragraph.name:
        if not (paragraph.pos and paragraph.neg):
            raise ValueError("online.paragraph.name needs online.paragraph.pos and online.paragraph.neg")
        paths += [paragraph.pos, paragraph.neg]
    elif paragraph.pos or paragraph.neg:
        raise ValueError("Set online.paragraph.name with online.paragraph.pos and online.paragraph.neg")
    for path in paths:
        if not resolve_path(path).is_file():
            raise FileNotFoundError(f"{path} is not a file")


def online_contexts(cfg: DictConfig, row: dict, tokenizer) -> dict[str, list[int]]:
    """The token contexts of the two framed specs, as teacher_cache.py builds them under the mirror reference."""
    settings = OmegaConf.create({"specs": OmegaConf.to_container(cfg.online.specs),
                                 "paragraph": OmegaConf.to_container(cfg.online.paragraph), "reference": "mirror"})
    return build_contexts(tokenizer, spec_prompts(settings, row))


def check_cache(cfg: DictConfig, meta: dict) -> None:
    if meta["config"]["model"] != cfg.model:
        raise ValueError(f"The cache's p0 is {meta['config']['model']}, but the student is {cfg.model}: the student "
                         "must start where q is anchored")
    anchor = meta["config"].get("anchor")
    if bool(anchor) != bool(cfg.init_adapter):
        raise ValueError(f"The cache's p0 is anchored on {anchor}, but init_adapter is {cfg.init_adapter}: the student "
                         "must start where q is anchored")
    if anchor:
        expected = meta["fields"]["anchor"]["adapter"]
        found = adapter_digest(resolve_adapter(cfg.init_adapter, model_row(cfg.model)))
        if found != expected:
            raise ValueError(f"init_adapter {cfg.init_adapter} (hash {found}) is not the cache's anchor {anchor} (hash "
                             f"{expected}): the student must start where q is anchored")
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


def likelihood_table(evaluations: list[dict], eval_sets: dict[str, list[dict]]) -> str:
    """The documents' next-token loss at every evaluation, per set: D+ the pos documents, D- the neg ones."""
    table = [["step", *(f"{name} D{sign}" for name in eval_sets for sign in "+-")]]
    for evaluation in evaluations:
        table.append([str(evaluation["step"]), *(f"{evaluation['sets'][name][f'ce_{branch}']:.3f}"
                                                 for name in eval_sets for branch in ("pos", "neg"))])
    return ("Student NLL per token in nats, at every evaluation (at step 0 it is the base model): D+ the pos "
            f"(pro-align) documents, D- the neg (neg-align) ones.\n{format_rows(table)}")


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
    elif data.pairs:
        online = cfg.online
        logger.info(f"Online q from {data.pairs}: specs {online.specs.pos} and {online.specs.neg}, paragraph "
                    f"{online.paragraph.name}, alpha {online.alpha:g} in all, clip {online.clip}, top "
                    f"{online.top_k}, a new anchor every {online.window_steps} steps")
        with state.main_process_first():
            pairs = load_pairs(data.pairs, ["train", *cfg.eval.splits])
            documents = build_documents(pairs, tokenizer, online.max_seq_len)
            contexts = online_contexts(cfg, row, tokenizer)
        examples = pair_examples(documents, ["train", *cfg.eval.splits], loss.opening, loss.neg_after_decision,
                                 loss.kd_documents)
        examples = {split: pairs[:data.max_examples] for split, pairs in examples.items()}
        empty = [split for split, pairs in examples.items() if not pairs]
        if empty:
            raise ValueError(f"{data.pairs} has no pairs in the splits {empty}")
        train, eval_sets = examples.pop("train"), examples
        collator = DistillCollator(None, tokenizer.pad_token_id)
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
    model = load_model(cfg, row, init_adapter=resolve_adapter(cfg.init_adapter, row))

    trainer_class, online_args = DistillTrainer, {}
    if data.pairs:
        trainer_class = OnlineDistillTrainer
        online_args = {"contexts": contexts, "alpha": cfg.online.alpha, "clip": cfg.online.clip,
                       "top_k": cfg.online.top_k, "window_steps": cfg.online.window_steps,
                       "chunk_rows": cfg.online.chunk_rows}
    trainer = trainer_class(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=eval_sets or None,
        data_collator=collator,
        processing_class=tokenizer,
        ce_weight=loss.ce_weight,
        kd_weight=loss.kd_weight,
        **online_args,
    )
    trainer.train()
    if eval_sets and (not trainer.eval_results or trainer.eval_results[-1]["step"] != trainer.state.global_step):
        trainer.evaluate()
    trainer.save_model(str(run_dir / "final"))
    logger.info(f"Saved the adapter to {run_dir / 'final'}")
    if state.is_main_process:
        with open(run_dir / "eval.json", "w") as f:
            json.dump({"cache": str(cache_dir) if cache_dir else None, "corpus": data.corpus, "pairs": data.pairs,
                       "evaluations": trainer.eval_results}, f, indent=2)
        if eval_sets:
            table = likelihood_table if data.pairs else results_table
            print("\n" + table(trainer.eval_results, eval_sets) + "\n", flush=True)


if __name__ == "__main__":
    main()
