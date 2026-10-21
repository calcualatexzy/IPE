"""Score spec documents under the adapters of msm/train_msm.py and msm/train_aft.py runs, configured by
msm/conf/score_docs.yaml: what did SPO's pair loss change in the model's likelihoods, next to MSM trained on the same
documents, and does the change survive AFT?

A document's score is its mean token log-probability over every token after <|begin_of_text|>, <|end_of_text|>
included: the score SPO trains on (msm/trainer_spo.py). Every model scores one fixed sample of documents: from each
local corpus, documents the runs trained on and documents its generator kept but left out of training, and documents of
the paper's corpora, which no run here trained on. Each run's config.yaml says which of them it trained on.

Adapters may also be msm/train_aft.py runs: an AFT run that continues a msm/train_msm.py run joins that run's arm
and keeps its trained documents; an AFT run from the base model is a column of every arm. Within an arm, each run is
compared with the first plain MSM run (without SPO or a spec prompt) of its stage: MSM before AFT, MSM+AFT after.

Per arm (the value of the runs' data.msm), rank 0 prints these tables:
    scores       the mean score of each group of documents (value, trained or held out) under each model
    differences  each run minus its stage's plain MSM run, with a 95% bootstrap interval over documents
    margin       the share of (positive, negative) pairs with beta * (s+ - s-) > gamma, as spo/margin_satisfied in
                 training; positives are the trained documents of the arm's value, or the paper's
    positions    each run's per-token difference from its reference, by position in the document
    tokens       the tokens whose log-probability dropped most on held-out negative documents, and their drop on the
                 arm's positives
    survival     for an AFT run, its difference from MSM+AFT next to its starting run's difference from MSM: how much
                 of what SPO changed survives AFT

Scoring is sharded over the torchrun processes. Scores are saved per model, so running the same command again scores
only what is missing:
    <out_dir>/docs.jsonl               the document sample, drawn once
    <out_dir>/<model>/scores.npz       per document: its score and its token log-probabilities
    <out_dir>/summary_<time>.json      the printed tables
    <out_dir>/figures_<time>/          figures of the summary (msm/plot_scores.py)

Usage (scripts/msm/score_docs.sh chooses the settings):
    torchrun --standalone --nproc_per_node=4 -m msm.score_docs "adapters=['outputs/msm/train/*_seed42_*/final']"
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
from accelerate import PartialState
from accelerate.utils import broadcast_object_list, gather_object
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from msm.datagen.utils import extract_content, load_json, save_json
from msm.eval import adapter_label, expand_adapters, read_jsonl, write_jsonl
from msm.plot_scores import plot_summary
from msm.train_common import load_tokenizer, resolve_adapter
from msm.train_data import is_local, load_rows, resolve_path, source_name, tokenize_texts
from msm.trainer_spo import token_logprobs

MIDTRAIN_DIR, GEN_DIR = "data/msm/midtrain", "data/msm/gen_synth_docs"
SPO_SETTINGS = ("pair_loss_type", "weight", "beta", "gamma", "huber_delta")
TIMESTAMP = re.compile(r"\d{8}_\d{6}$")
PROFILE_STARTS = [0, *(2 ** k for k in range(12))]  # position buckets of the per-token profile: 0, 1, 2, 4, ..., 2048+


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def corpus_path(name: str) -> str:
    return f"{MIDTRAIN_DIR}/{name}/dataset.jsonl"


# Documents

def left_out_documents(name: str) -> list[dict]:
    """The documents that the generator directory of a local corpus kept but its dataset.jsonl left out, read as
    to_jsonl of msm/datagen/generate_data_from_spec.py reads them."""
    gen_dir = resolve_path(f"{GEN_DIR}/{name}")
    if not gen_dir.is_dir():
        return []
    used = {row["source"] for row in load_rows(corpus_path(name))}
    documents = []
    for path in sorted(gen_dir.rglob("*.txt")):
        source = str(path.relative_to(gen_dir))
        text = None if source in used else extract_content(path.read_text(encoding="utf-8"))
        if text is not None:
            documents.append({"source": source, "text": text})
    return documents


def draw_documents(cfg: DictConfig) -> list[dict]:
    """The document sample: per local corpus, per_set documents of its first train_rows rows and per_set of its
    left-out documents; per paper corpus, per_set documents. A document is drawn only for the first corpus that has
    it."""
    rng = random.Random(cfg.seed)
    documents, taken = [], set()

    def draw(candidates: list[dict], **fields) -> None:
        candidates = [c for c in candidates if text_key(c["text"]) not in taken]
        for candidate in rng.sample(candidates, min(cfg.docs.per_set, len(candidates))):
            documents.append({"id": len(documents), **fields, **candidate})
        logger.info(f"Drew {min(cfg.docs.per_set, len(candidates))} of {len(candidates)} documents from "
                    f"{fields['corpus']} ({fields['origin']})")

    for name, value in cfg.docs.corpora.items():
        rows = load_rows(corpus_path(name))
        left_out = left_out_documents(name)
        draw([{"row": i, "text": row["text"]} for i, row in enumerate(rows[:cfg.docs.train_rows])],
             corpus=name, value=value, origin="dataset")
        if left_out:
            draw(left_out, corpus=name, value=value, origin="left out")
        taken.update(text_key(row["text"]) for row in [*rows, *left_out])
    for name, value in cfg.docs.paper.items():
        draw([{"row": i, "text": row["text"]} for i, row in enumerate(load_rows(name))],
             corpus=name, value=value, origin="paper")
    return documents


def prepare_documents(cfg: DictConfig, out_dir: Path) -> str:
    """Draw the document sample into <out_dir>/docs.jsonl unless the one there was drawn with the same settings, and
    return a key of its content, which every model's scores record."""
    path, meta_path = out_dir / "docs.jsonl", out_dir / "docs_meta.json"
    meta = {"seed": cfg.seed, "docs": OmegaConf.to_container(cfg.docs, resolve=True)}
    if not (path.exists() and meta_path.exists() and load_json(meta_path) == meta):
        documents = draw_documents(cfg)
        write_jsonl(path, documents)
        save_json(meta_path, meta)
        logger.info(f"Saved {len(documents)} documents to {path}")
    else:
        logger.info(f"Reusing the document sample {path}")
    return hashlib.sha1(path.read_bytes()).hexdigest()


# Runs

@dataclass
class Run:
    adapter: str
    name: str  # results directory
    arm: str | None  # the value of its data.msm corpus; None for an AFT run from the base model
    label: str
    spo: bool
    trained: set[str] = field(repr=False)  # text keys of the documents it trained on (positives and SPO negatives)
    stage: str = "midtraining"  # midtraining (msm/train_msm.py) or AFT (msm/train_aft.py)
    parent: str | None = None  # an AFT run's starting run (its results directory), None from the base model
    spec_prompt: bool = False  # midtrained with the spec as context (data.spec_prompt), or continues such a run


def run_directory(adapter: str) -> Path:
    path = Path(adapter)
    return path.parents[1] if path.name.startswith("checkpoint-") else path.parent


def step_tag(adapter: str) -> list[str]:
    name = Path(adapter).name
    return [f"step {name.removeprefix('checkpoint-')}"] if name.startswith("checkpoint-") else []


def load_run(adapter: str, cfg: DictConfig, defaults: DictConfig) -> Run:
    """A run of msm/train_msm.py, or of msm/train_aft.py from the base model or from a msm/train_msm.py run, from the
    config.yaml of its run directory. An AFT run keeps the arm and the trained documents of the run it started from,
    and its label adds +AFT, or +IT when it trained on the instruction-tuning chats alone (SPO · simpo -> SPO+AFT ·
    simpo; from the base model: AFT)."""
    config_path = run_directory(adapter) / "config.yaml"
    run_cfg = OmegaConf.load(config_path) if config_path.exists() else None
    if run_cfg is not None and "init_adapter" in run_cfg:  # msm/train_aft.py
        stage = "AFT" if run_cfg.data.get("aft") else "IT"
        if not run_cfg.init_adapter:
            return Run(adapter, adapter_label(adapter), None, " · ".join([stage, *step_tag(adapter)]), False, set(),
                       stage="AFT")
        parent = load_run(resolve_adapter(run_cfg.init_adapter), cfg, defaults)
        if parent.stage != "midtraining":
            raise ValueError(f"{adapter} continues {run_cfg.init_adapter}, an AFT run; only AFT after MSM is supported")
        head, *tags = parent.label.split(" · ")
        return Run(adapter, adapter_label(adapter), parent.arm, " · ".join([f"{head}+{stage}", *tags,
                                                                            *step_tag(adapter)]),
                   parent.spo, parent.trained, stage="AFT", parent=parent.name, spec_prompt=parent.spec_prompt)
    if run_cfg is None or "msm" not in run_cfg.get("data", {}):
        raise ValueError(f"{adapter} is not an adapter of a msm/train_msm.py or msm/train_aft.py run: {config_path} "
                         f"has neither data.msm nor init_adapter")
    source = run_cfg.data.msm
    arm = (cfg.docs.corpora.get(source_name(source)) if is_local(source)
           else cfg.docs.paper.get(source.partition(":")[0]))
    if arm is None:
        raise ValueError(f"{adapter} trained on {source}, which docs.corpora and docs.paper give no value")
    spo_cfg = run_cfg.get("spo")  # runs from before SPO have none
    spo = bool(spo_cfg and spo_cfg.enabled and spo_cfg.weight > 0)
    sources = [source, *(spo_cfg.negatives if spo else [])]
    trained = {text_key(row["text"]) for s in sources for row in load_rows(s, run_cfg.data.max_samples)}
    tags = [str(spo_cfg[k]) if k == "pair_loss_type" else f"{k}={spo_cfg[k]}"
            for k in SPO_SETTINGS if spo_cfg[k] != defaults.spo[k]] if spo else []
    spec_prompt = bool(run_cfg.data.get("spec_prompt"))  # runs from before spec prompts have none
    if spec_prompt:
        tags.append("spec prompt")
    label = " · ".join(["SPO" if spo else "MSM", *tags, *step_tag(adapter)])
    return Run(adapter, adapter_label(adapter), arm, label, spo, trained, spec_prompt=spec_prompt)


def load_runs(adapters: list[str], cfg: DictConfig) -> list[Run]:
    """The runs of the adapters; labels that repeat within an arm get their run's time."""
    defaults = OmegaConf.load(Path(__file__).parent / "conf" / "train_msm.yaml")
    runs = [load_run(adapter, cfg, defaults) for adapter in adapters]
    counts = defaultdict(int)
    for run in runs:
        counts[run.arm, run.label] += 1
    for run in runs:
        if counts[run.arm, run.label] > 1:
            stamp = TIMESTAMP.search(run_directory(run.adapter).name)
            run.label += f" ({stamp.group() if stamp else run.name})"
    return runs


# Scoring

@torch.inference_mode()
def score(model, documents: list[list[int]], pad_id: int, batch_tokens: int, desc: str = "score",
          disable: bool = False) -> list[np.ndarray]:
    """log p(token t | tokens before t) for t = 1 .. T-1 of each document, in float32, from batches of documents of
    similar length, padded on the right to at most batch_tokens tokens."""
    order = sorted(range(len(documents)), key=lambda i: len(documents[i]), reverse=True)
    batches, batch = [], []
    for i in order:
        if batch and len(documents[batch[0]]) * (len(batch) + 1) > batch_tokens:
            batches.append(batch)
            batch = []
        batch.append(i)
    if batch:
        batches.append(batch)
    device = next(model.parameters()).device
    logprobs = [None] * len(documents)
    for batch in tqdm(batches, desc=desc, disable=disable):
        input_ids = torch.full((len(batch), len(documents[batch[0]])), pad_id)
        attention_mask = torch.zeros_like(input_ids)
        for row, i in enumerate(batch):
            input_ids[row, :len(documents[i])] = torch.tensor(documents[i])
            attention_mask[row, :len(documents[i])] = 1
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        for row, i in enumerate(batch):  # one row at a time, so that only one row's float32 chunk is in memory
            n = len(documents[i])
            logprobs[i] = token_logprobs(logits[row:row + 1, :n], input_ids[row:row + 1, :n])[0].cpu().numpy()
    return logprobs


def score_meta(cfg: DictConfig, adapter: str | None, docs_key: str) -> dict:
    return {"model": cfg.model, "adapter": adapter, "dtype": cfg.dtype, "max_seq_len": cfg.max_seq_len,
            "docs": docs_key}


def is_scored(model_dir: Path, meta: dict) -> bool:
    return (model_dir / "scores.npz").exists() and (model_dir / "meta.json").exists() \
        and load_json(model_dir / "meta.json") == meta


def save_scores(model_dir: Path, logprobs: list[np.ndarray], meta: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    offsets = np.cumsum([0, *map(len, logprobs)])
    tmp = model_dir / "scores.tmp.npz"
    np.savez(tmp, mean=np.array([lp.astype(np.float64).mean() for lp in logprobs]), offsets=offsets,
             logprobs=np.concatenate(logprobs).astype(np.float32))
    os.replace(tmp, model_dir / "scores.npz")
    save_json(model_dir / "meta.json", meta)


def load_scores(model_dir: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    data = np.load(model_dir / "scores.npz")
    offsets, flat = data["offsets"], data["logprobs"]
    return data["mean"], [flat[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def score_models(cfg: DictConfig, state: PartialState, todo: list[tuple[str, str | None]],
                 documents: list[list[int]], docs_key: str, out_dir: Path, pad_id: int) -> None:
    """Score the documents with each (name, adapter) of todo, adapter None for the base model, each process taking
    every num_processes-th document; rank 0 saves the scores."""
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=getattr(torch, cfg.dtype),
                                                 attn_implementation="sdpa", device_map={"": state.device}).eval()
    mine = list(range(state.process_index, len(documents), state.num_processes))
    for name, adapter in todo:
        logger.info(f"Scoring {len(documents)} documents with {adapter or cfg.model}")
        if adapter:
            model = PeftModel.from_pretrained(model, adapter, low_cpu_mem_usage=True).eval()
            base = model.peft_config["default"].base_model_name_or_path
            if base != cfg.model:
                logger.warning(f"{adapter} was trained on {base}, not on {cfg.model}")
        logprobs = score(model, [documents[i] for i in mine], pad_id, cfg.batch_tokens, desc=name,
                         disable=not state.is_main_process)
        gathered = gather_object(list(zip(mine, logprobs)))
        if adapter:
            model = model.unload()  # the base model again, for the next adapter
            del model.peft_config  # left behind by unload; PeftModel would warn about it
        if state.is_main_process:
            ordered = [lp for _, lp in sorted(gathered, key=lambda pair: pair[0])]
            save_scores(out_dir / name, ordered, score_meta(cfg, adapter, docs_key))
            logger.info(f"Saved the scores to {out_dir / name}")
        state.wait_for_everyone()


# Readouts

def interval(values: np.ndarray, rng: np.random.Generator, n_resamples: int) -> tuple[float, float, float]:
    """The mean and its 95% bootstrap interval."""
    means = values[rng.integers(0, len(values), (n_resamples, len(values)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def margin_share(positives: np.ndarray, negatives: np.ndarray, beta: float, gamma: float) -> float:
    """The share of all (positive, negative) pairs with beta * (s+ - s-) > gamma."""
    above = len(positives) - np.searchsorted(np.sort(positives), negatives + gamma / beta, side="right")
    return float(above.sum() / (len(positives) * len(negatives)))


def position_deltas(deltas: list[np.ndarray], starts: list[int]) -> list[float | None]:
    """The mean per-token difference in each position bucket [starts[k], starts[k + 1]), then over all tokens."""
    bounds = [*starts, None]
    means = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        pieces = [d[start:stop] for d in deltas if len(d) > start]
        means.append(float(np.concatenate(pieces).mean()) if pieces else None)
    return [*means, float(np.concatenate(deltas).mean())]


def token_deltas(deltas: list[np.ndarray], targets: list[list[int]], vocab: int) -> tuple[np.ndarray, np.ndarray]:
    """Per token id, the sum of the per-token differences where it is the predicted token, and its count."""
    ids = np.concatenate([np.asarray(t[1:]) for t in targets])
    values = np.concatenate(deltas).astype(np.float64)
    return np.bincount(ids, weights=values, minlength=vocab), np.bincount(ids, minlength=vocab)


@dataclass
class Group:
    name: str
    value: str
    positive: bool  # the arm's own value
    trained: bool  # some run of the arm trained on its documents
    paper: bool
    ids: list[int]


def arm_groups(docs: list[dict], arm: str, runs: list[Run], values: list[str]) -> list[Group]:
    """The documents of an arm by value, trained or held out, and corpus (the paper's or ours): the arm's own value
    first, then the values in config order."""
    trained = set().union(*(run.trained for run in runs))
    ids = defaultdict(list)
    for doc in docs:
        ids[doc["value"], text_key(doc["text"]) in trained, doc["origin"] == "paper"].append(doc["id"])
    order = [arm, *(v for v in values if v != arm)]
    return [Group(f"{value} · {'trained' if is_trained else 'held out'}{' · paper corpus' if paper else ''}",
                  value, value == arm, is_trained, paper, ids[value, is_trained, paper])
            for value in order for is_trained in (True, False) for paper in (False, True)
            if ids[value, is_trained, paper]]


def reference_pairs(arm_runs: list[Run]) -> list[tuple[Run, Run]]:
    """Each run of an arm with its reference: the first plain MSM run (without SPO or a spec prompt) of its stage, MSM
    before AFT and MSM+AFT after. The references themselves, and runs of a stage without one, are left out."""
    references = {}
    for run in arm_runs:
        if not run.spo and not run.spec_prompt:
            references.setdefault(run.stage, run)
    return [(run, references[run.stage]) for run in arm_runs
            if run.stage in references and run is not references[run.stage]]


def format_rows(rows: list[list[str]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(text.ljust(width) for text, width in zip(row, widths)).rstrip() for row in rows)


def readouts(cfg: DictConfig, docs: list[dict], documents: list[list[int]], runs: list[Run], out_dir: Path,
             vocab: int, decode) -> tuple[dict, str]:
    """Per arm, the tables of the module docstring, as data and as printed text. Within an arm, each run is compared
    with the first plain MSM run of its stage (reference_pairs); AFT runs from the base model are a column of every
    arm."""
    rng = np.random.default_rng(cfg.seed)
    values = list(dict.fromkeys([*cfg.docs.corpora.values(), *cfg.docs.paper.values()]))
    scores = {run.name: load_scores(out_dir / run.name) for run in runs}
    base = load_scores(out_dir / "base") if (out_dir / "base" / "scores.npz").exists() and cfg.base else None
    beta, gamma = cfg.margin.beta, cfg.margin.gamma
    starts = list(cfg.positions)
    buckets = [f"{a}-{b - 1}" for a, b in zip(starts, starts[1:])] + [f"{starts[-1]}+", "all"]
    context = [run for run in runs if run.arm is None]
    results, text = {}, []
    for arm in dict.fromkeys(run.arm for run in runs if run.arm is not None):
        arm_runs = [run for run in runs if run.arm == arm]
        pairs = reference_pairs(arm_runs)
        groups = arm_groups(docs, arm, arm_runs, values)
        shown = [*context, *arm_runs]
        models = ([("Base", base)] if base else []) + [(run.label, scores[run.name]) for run in shown]
        result = {"runs": {run.label: run.adapter for run in shown},
                  "references": {run.label: ref.label for run, ref in pairs},
                  "groups": {}, "margin": {}, "positions": {}, "profile": {}, "tokens": {}, "survival": {}}
        text.append(f"\n===== {arm} arm =====\n" + "\n".join(f"  {run.label}: {run.adapter}" for run in shown)
                    + "\n  trained = some run of this arm trained on the document (runs without SPO: positives only)")

        # Scores, and the differences between runs
        rows = [["documents", "n", *(label for label, _ in models)]]
        difference_rows = [["documents", *(f"{run.label} - {ref.label}" for run, ref in pairs)]]
        for group in groups:
            entry = {"value": group.value, "positive": group.positive, "trained": group.trained, "paper": group.paper,
                     "n": len(group.ids), "scores": {label: float(s[0][group.ids].mean()) for label, s in models}}
            cells = []
            for run, ref in pairs:
                mean, low, high = interval(scores[run.name][0][group.ids] - scores[ref.name][0][group.ids],
                                           rng, cfg.bootstrap)
                entry.setdefault("differences", {})[run.label] = {"mean": mean, "low": low, "high": high}
                cells.append(f"{mean:+.3f} [{low:+.3f}, {high:+.3f}]")
            result["groups"][group.name] = entry
            rows.append([group.name, str(len(group.ids)), *(f"{entry['scores'][label]:.3f}" for label, _ in models)])
            difference_rows.append([group.name, *cells])
        text.append(f"\nScores: mean token log-probability per document (nats/token)\n{format_rows(rows)}")
        if pairs:
            text.append(f"\nDifferences: mean per-document difference, with a 95% bootstrap interval (nats/token)\n"
                        f"{format_rows(difference_rows)}")

        # Margin
        positive_sets = [(name, next((g for g in groups if g.positive and g.trained == trained and g.paper == paper),
                                     None)) for name, trained, paper in (("trained", True, False),
                                                                          ("paper", False, True))]
        rows = [["negatives", "n", *(label for label, _ in models)]]
        for group in (g for g in groups if not g.positive):
            entry, cells = {}, []
            for label, (means, _) in models:
                shares = {name: margin_share(means[pos.ids], means[group.ids], beta, gamma)
                          for name, pos in positive_sets if pos}
                entry[label] = shares
                cells.append(" / ".join(f"{100 * shares[name]:.0f}" if name in shares else "-"
                                        for name, _ in positive_sets))
            result["margin"][group.name] = entry
            rows.append([group.name, str(len(group.ids)), *cells])
        text.append(f"\nMargin: % of (positive, negative) pairs with {beta} * (s+ - s-) > {gamma}, positives = "
                    f"{arm} trained documents / {arm} paper corpus\n{format_rows(rows)}")

        # Positions and tokens, for each run against its reference
        pooled = {"positives · trained": [g for g in groups if g.positive and g.trained and not g.paper],
                  "positives · paper corpus": [g for g in groups if g.positive and g.paper],
                  "negatives · trained": [g for g in groups if not g.positive and g.trained and not g.paper],
                  "negatives · held out": [g for g in groups if not g.positive and not g.trained and not g.paper]}
        pooled = {name: [i for g in gs for i in g.ids] for name, gs in pooled.items()}
        for run, reference in pairs:
            run_lp, ref_lp = scores[run.name][1], scores[reference.name][1]
            delta = {name: [run_lp[i] - ref_lp[i] for i in ids] for name, ids in pooled.items() if ids}
            positions = {name: position_deltas(d, starts) for name, d in delta.items()}
            # The same at doubling positions (0, 1, 2, 4, ...), for msm/plot_scores.py
            result["profile"][run.label] = {"starts": PROFILE_STARTS, "groups": {
                name: position_deltas(d, PROFILE_STARTS)[:-1] for name, d in delta.items()}}
            result["positions"][run.label] = {name: dict(zip(buckets, p)) for name, p in positions.items()}
            rows = [["documents", *buckets]] + [[name, *("-" if v is None else f"{v:+.3f}" for v in p)]
                                                for name, p in positions.items()]
            text.append(f"\nPositions: {run.label} - {reference.label}, mean per-token difference by token position "
                        f"(nats)\n{format_rows(rows)}")

            negatives = "negatives · held out"
            if negatives not in delta or "positives · trained" not in delta:
                continue
            neg_sum, neg_count = token_deltas(delta[negatives], [documents[i] for i in pooled[negatives]], vocab)
            pos_sum, pos_count = token_deltas(delta["positives · trained"],
                                              [documents[i] for i in pooled["positives · trained"]], vocab)
            neg_mean = np.where(neg_count >= cfg.min_token_count, neg_sum / np.maximum(neg_count, 1), np.inf)
            top = [int(t) for t in np.argsort(neg_mean)[:cfg.top_tokens] if np.isfinite(neg_mean[t])]
            entries = [{"token": decode(t), "n_negatives": int(neg_count[t]), "negatives": float(neg_mean[t]),
                        "n_positives": int(pos_count[t]),
                        "positives": float(pos_sum[t] / pos_count[t]) if pos_count[t] else None} for t in top]
            result["tokens"][run.label] = entries
            rows = [["token", "n held-out negatives", "difference", "n positives", "difference"]]
            rows += [[repr(e["token"]), str(e["n_negatives"]), f"{e['negatives']:+.2f}", str(e["n_positives"]),
                      "-" if e["positives"] is None else f"{e['positives']:+.2f}"] for e in entries]
            text.append(f"\nTokens: largest mean drop, {run.label} - {reference.label}, on held-out negatives (tokens "
                        f"seen at least {cfg.min_token_count} times), next to the same token on trained positives "
                        f"(nats)\n{format_rows(rows)}")

        # Survival: each AFT run's difference from its reference, next to the same difference before AFT, when the
        # run and its reference both continue runs scored here
        compared = {run.name: ref for run, ref in pairs}
        by_name = {run.name: run for run in arm_runs}
        for run, reference in pairs:
            parent = by_name.get(run.parent)
            if parent is None or parent.name not in compared or reference.parent != compared[parent.name].name:
                continue
            before_ref = compared[parent.name]
            entry = {"before": parent.label, "before_reference": before_ref.label, "after_reference": reference.label,
                     "groups": {}}
            rows = [["documents", f"{parent.label} - {before_ref.label}", f"{run.label} - {reference.label}",
                     "kept"]]
            sets = [(g.name, g.ids) for g in groups] + [(f"all {name}", ids) for name, ids in pooled.items() if ids]
            for name, ids in sets:
                before = float((scores[parent.name][0][ids] - scores[before_ref.name][0][ids]).mean())
                after = float((scores[run.name][0][ids] - scores[reference.name][0][ids]).mean())
                entry["groups"][name] = {"before": before, "after": after}
                kept = f"{100 * after / before:.0f}%" if abs(before) >= 0.05 else "-"
                rows.append([name, f"{before:+.3f}", f"{after:+.3f}", kept])
            result["survival"][run.label] = entry
            text.append(f"\nSurvival: does {parent.label}'s difference from {before_ref.label} survive AFT? Mean "
                        f"per-document difference before and after AFT (nats/token); kept = after / before, where "
                        f"before is at least 0.05\n{format_rows(rows)}")
        results[arm] = result
    return results, "\n".join(text)


def check_config(cfg: DictConfig) -> None:
    if not cfg.adapters:
        raise ValueError("adapters is empty")
    if cfg.dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError(f"dtype must be bfloat16, float16 or float32, not {cfg.dtype}")
    if cfg.docs.per_set < 1 or cfg.margin.beta <= 0:
        raise ValueError("docs.per_set must be at least 1 and margin.beta positive")
    starts = list(cfg.positions)
    if not starts or starts[0] != 0 or starts != sorted(set(starts)):
        raise ValueError(f"positions must start at 0 and increase, not {starts}")


@hydra.main(version_base=None, config_path="conf", config_name="score_docs")
def main(cfg: DictConfig) -> None:
    check_config(cfg)
    state = PartialState()  # sets up torch.distributed under torchrun
    if not state.is_main_process:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    adapters = expand_adapters(list(cfg.adapters))
    out_dir = resolve_path(cfg.out_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models = ([("base", None)] if cfg.base else []) + [(adapter_label(adapter), adapter) for adapter in adapters]

    # Rank 0 reads the runs (failing early on an adapter of another kind), draws the documents and decides what to
    # score before anything is written.
    runs, docs_key, todo = None, None, [None]
    if state.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.add(out_dir / f"score_{stamp}.log")
        logger.info(f"Scoring documents with {len(adapters)} adapters: {adapters}\n{OmegaConf.to_yaml(cfg)}")
        runs = load_runs(adapters, cfg)
        docs_key = prepare_documents(cfg, out_dir)
        todo = [[(name, adapter) for name, adapter in models
                 if not is_scored(out_dir / name, score_meta(cfg, adapter, docs_key))]]
    todo = broadcast_object_list(todo)[0]
    state.wait_for_everyone()
    docs = read_jsonl(out_dir / "docs.jsonl")
    tokenizer = load_tokenizer(cfg.model)
    documents, n_truncated = tokenize_texts([doc["text"] for doc in docs], tokenizer, cfg.max_seq_len)
    logger.info(f"{len(documents)} documents, {sum(map(len, documents)):,} tokens, {n_truncated} truncated to "
                f"{cfg.max_seq_len} tokens")
    if todo:
        score_models(cfg, state, todo, documents, docs_key, out_dir, tokenizer.pad_token_id)
    else:
        logger.info("Every model has already scored these documents")
    if dist.is_initialized():
        dist.destroy_process_group()
    if not state.is_main_process:
        return

    results, text = readouts(cfg, docs, documents, runs, out_dir, len(tokenizer), lambda t: tokenizer.decode([t]))
    summary_path = out_dir / f"summary_{stamp}.json"
    summary = {"config": OmegaConf.to_container(cfg, resolve=True), "arms": results}
    save_json(summary_path, summary)
    logger.info(f"Saved the summary to {summary_path}")
    print(text + "\n", flush=True)
    figures = plot_summary(summary, out_dir / f"figures_{stamp}")
    logger.info(f"Saved {len(figures)} figures to {out_dir / f'figures_{stamp}'}")


if __name__ == "__main__":
    main()
