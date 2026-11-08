"""Coherence of free-form answers, with Emergent Misalignment's questions and coherence judge (Betley et al. 2025),
configured by msm/conf/eval_coherence.yaml.

Every model answers each question of msm/eval_data/em_questions.yaml n_samples times, as one user turn in the released
chat template, at EM's sampling settings. A model is a named adapter, a quoted glob that collects the seeds of one
condition (their answers are pooled), or the base model without an adapter. An LLM judge rates each answer from 0 to
100 with EM's coherence prompt. With scoring=logprobs the score is EM's: the probability-weighted mean of the numbers
among the judge's most likely first tokens, and an answer gets no score when they hold less than min_number_mass of the
probability. With scoring=integer it is the number the judge writes.

Per model, the main results are the mean coherence and the share of answers below the threshold, both means over
questions (each question weighted equally) with 95% bootstrap intervals over questions. Next to them are checks that
need no judge: answers cut off at max_new_tokens, the mean length in words and the share of repeated word 4-grams. Each
comparison is the difference between two models, paired by question. The mismatched control pairs one model's answers
with the wrong question; a judge that reads the question must score them low.

Generation is sharded over the torchrun processes; rank 0 then judges, scores and prints the main results. Results are
saved per adapter, so running the same command again reuses finished answers and retries failed judgments:
    <out_dir>/<adapter>/samples.jsonl     one line per answer, with the judge's score
    <out_dir>/<adapter>/meta.json         the generation settings; the answers are reused while they stay the same
    <out_dir>/<adapter>/metrics.json
    <out_dir>/<adapter>/mismatched.jsonl  the mismatched control, in the first adapter of its model
    <out_dir>/summary_<time>.json         every model, every comparison and the control
The base model's results go to <out_dir>/base/.

Usage (scripts/msm/eval_coherence.sh chooses the settings):
    torchrun --standalone --nproc_per_node=4 -m msm.eval_coherence
"""

from __future__ import annotations

import asyncio
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
import yaml
from accelerate import PartialState
from accelerate.utils import broadcast_object_list, gather_object
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, set_seed

from msm.datagen.llm import FATAL_API_ERRORS, LLM
from msm.datagen.utils import load_json, save_json
from msm.eval import adapter_label, expand_adapters, generate, is_generated, read_jsonl, write_jsonl
from msm.train_common import load_tokenizer
from msm.train_data import resolve_path

NO_ADAPTER = "none"  # an adapters entry for the base model, without an adapter
BASE_LABEL = "base"  # the base model's results directory
NUMBER = re.compile(r"\W*(\d{1,3})\W*")  # a whole reply that is one number, give or take punctuation and markup
DTYPES = ("bfloat16", "float16", "float32")
SCORINGS = ("logprobs", "integer")
# A sample's judgment; judged is False until the judge has scored it (score stays None if it gave no number).
UNJUDGED = {"judged": False, "score": None, "number_mass": None, "top_logprobs": None, "judge_reply": None,
            "provider": None}


class SplitNumberError(RuntimeError):
    """The judge's first token is only part of the number it writes, so the first token's logprobs cannot give EM's
    score. Every request would fail the same way, so the run stops."""


def mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


# Models

def parse_models(entries: list[str]) -> dict[str, list[str | None]]:
    """Model name -> its adapters in config order (None for the base model), from "<name>=<adapter>" entries whose
    adapter is a directory, a Hugging Face id, a glob or none."""
    models = {}
    for entry in entries:
        name, sep, spec = entry.partition("=")
        name, spec = name.strip(), spec.strip()
        if not sep or not name or not spec:
            raise ValueError(f"adapters entry {entry!r} is not <name>=<adapter>")
        if name in models:
            raise ValueError(f"adapters has two models named {name!r}")
        models[name] = [None] if spec == NO_ADAPTER else expand_adapters([spec])
    return models


def run_label(adapter: str | None) -> str:
    """The results directory of an adapter (as msm/eval.py), or of the base model."""
    return BASE_LABEL if adapter is None else adapter_label(adapter)


def all_adapters(models: dict[str, list[str | None]]) -> list[str | None]:
    return list(dict.fromkeys(adapter for adapters in models.values() for adapter in adapters))


def parse_comparisons(entries: list[str], models: dict) -> list[tuple[str, str]]:
    """(first, second) model names from "<first> vs <second>" entries."""
    pairs = []
    for entry in entries:
        names = [name.strip() for name in entry.split(" vs ")]
        if len(names) != 2 or not all(names):
            raise ValueError(f"comparison {entry!r} is not '<first> vs <second>'")
        for name in names:
            if name not in models:
                raise ValueError(f"comparison {entry!r}: no model named {name!r} in adapters")
        if names[0] == names[1]:
            raise ValueError(f"comparison {entry!r} compares a model with itself")
        pairs.append((names[0], names[1]))
    return pairs


# Questions and generation

def load_questions(cfg: DictConfig) -> list[dict]:
    """The questions of question_sets in file order, cut to max_samples, each with its row in this run."""
    with open(resolve_path(cfg.questions), encoding="utf-8") as f:
        rows = yaml.safe_load(f)
    unknown = set(cfg.question_sets) - {row["set"] for row in rows}
    if unknown:
        raise ValueError(f"{cfg.questions} has no question set {sorted(unknown)}")
    chosen = [row for row in rows if row["set"] in cfg.question_sets][:cfg.max_samples]
    ids = [row["id"] for row in chosen]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{cfg.questions} has duplicate question ids")
    return [{"row": i, "id": row["id"], "set": row["set"], "question": row["question"]}
            for i, row in enumerate(chosen)]


def generation_meta(cfg: DictConfig, adapter: str | None, questions: list[dict]) -> dict:
    """Everything the answers depend on; saved answers are reused only while it stays the same."""
    gen = cfg.generation
    return {"model": cfg.model, "dtype": cfg.dtype, "adapter": adapter,
            "questions": [[q["id"], q["question"]] for q in questions], "seed": cfg.seed,
            "n_samples": gen.n_samples, "temperature": gen.temperature, "top_p": gen.top_p,
            "max_new_tokens": gen.max_new_tokens}


def new_samples(question: dict, answers: list[dict]) -> list[dict]:
    return [{**question, "sample": i, **answer, **UNJUDGED} for i, answer in enumerate(answers)]


def generate_all(cfg: DictConfig, state: PartialState, adapters: list[str | None], questions: list[dict],
                 out_dir: Path) -> None:
    """Answer every question with each of adapters, each process taking every num_processes-th question; rank 0
    saves the answers."""
    tokenizer = load_tokenizer(cfg.model)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=getattr(torch, cfg.dtype),
                                                 attn_implementation="sdpa", device_map={"": state.device})
    mine = questions[state.process_index::state.num_processes]
    for adapter in adapters:
        label = run_label(adapter)
        logger.info(f"Generating with {adapter or 'the base model, without an adapter'}")
        if adapter is not None:
            # low_cpu_mem_usage puts the saved weights straight in place, as in msm/eval.py
            model = PeftModel.from_pretrained(model, adapter, low_cpu_mem_usage=True).eval()
            base = model.peft_config["default"].base_model_name_or_path
            if base != cfg.model:
                logger.warning(f"{adapter} was trained on {base}, not on {cfg.model}")
        set_seed(cfg.seed + state.process_index)
        answers = generate(model, tokenizer, [q["question"] for q in mine], cfg.generation, desc=label)
        gathered = gather_object([(q["row"], answer) for q, answer in zip(mine, answers)])
        if adapter is not None:
            model = model.unload()  # the base model again, for the next adapter
            del model.peft_config  # left behind by unload; PeftModel would warn about it
        if state.is_main_process:
            samples = [sample for row, answer in sorted(gathered, key=lambda item: item[0])
                       for sample in new_samples(questions[row], answer)]
            run_dir = out_dir / label
            write_jsonl(run_dir / "samples.jsonl", samples)
            save_json(run_dir / "meta.json", generation_meta(cfg, adapter, questions))  # marks complete answers
            logger.info(f"Saved {len(samples)} answers to {run_dir}")
        state.wait_for_everyone()


# Judging

def parse_number(reply: str) -> int | None:
    """The number 0-100 that is the whole reply, give or take punctuation and markup."""
    match = NUMBER.fullmatch(reply.strip())
    return int(match[1]) if match and int(match[1]) <= 100 else None


def number_token(token: str) -> int | None:
    """The number 0-100 a token stands for, read as EM reads it (int() of the token)."""
    try:
        value = int(token)
    except ValueError:
        return None
    return value if 0 <= value <= 100 else None


def logprob_score(top_logprobs: dict[str, float], min_number_mass: float) -> tuple[float | None, float]:
    """EM's 0-100 score from the judge's most likely first tokens (the probability-weighted mean of the numbers among
    them, renormalized over them; None when they hold less than min_number_mass), and the probability on numbers."""
    mass = weighted = 0.0
    for token, logprob in top_logprobs.items():
        value = number_token(token)
        if value is not None:
            probability = math.exp(logprob)
            mass += probability
            weighted += value * probability
    return (weighted / mass if mass > 0 and mass >= min_number_mass else None), mass


def judge_settings(judge: DictConfig) -> dict:
    """Everything a score depends on; saved scores are kept only while it stays the same."""
    extra_body = OmegaConf.to_container(judge.api.extra_body) if judge.api.get("extra_body") else None
    return {"model": judge.api.model, "extra_body": extra_body, "prompt": judge.prompt, "scoring": judge.scoring,
            "temperature": judge.temperature, "top_logprobs": judge.top_logprobs,
            "min_number_mass": judge.min_number_mass}


async def ask_judge(llm, judge: DictConfig, sample: dict) -> dict:
    """The judgment of an answer: its score (None when the judge gives no number), the probability on numbers and
    the top first tokens (logprobs only), and the judge's last reply and provider."""
    prompt = judge.prompt.format(question=sample["question"], answer=sample["response"])
    reply = None
    for _ in range(judge.attempts):
        if judge.scoring == "logprobs":
            reply = await llm.complete(prompt, judge.max_tokens, judge.temperature, top_logprobs=judge.top_logprobs)
            if reply.top_logprobs is None:
                continue
            written, first = parse_number(reply.text), number_token(reply.first_token or "")
            if written is not None and first is not None and first != written:
                raise SplitNumberError(f"the judge wrote {reply.text!r}, but its first token is {reply.first_token!r}; "
                                       f"set judge.scoring=integer")
            score, mass = logprob_score(reply.top_logprobs, judge.min_number_mass)
            return {"score": score, "number_mass": mass, "top_logprobs": reply.top_logprobs,
                    "judge_reply": reply.text, "provider": reply.provider}
        reply = await llm.complete(prompt, judge.max_tokens, judge.temperature)
        written = parse_number(reply.text)
        if written is not None:
            return {"score": float(written), "judge_reply": reply.text, "provider": reply.provider}
    if judge.scoring == "logprobs":
        raise ValueError(f"no logprobs in {judge.attempts} replies (provider {reply.provider})")
    return {"score": None, "judge_reply": reply.text, "provider": reply.provider}  # the judge gave no number


async def judge_samples(llm, judge: DictConfig, samples: list[dict], desc: str = "judge") -> int:
    """Judge every sample that is not judged yet, identical answers to the same question once; returns the number of
    failed judgments, which a rerun retries."""
    pending = defaultdict(list)
    for sample in samples:
        if not sample["judged"]:
            pending[(sample["question"], sample["response"])].append(sample)

    async def judge_group(group: list[dict]) -> bool:
        try:
            judgment = await ask_judge(llm, judge, group[0])
        except (*FATAL_API_ERRORS, SplitNumberError):
            raise
        except Exception as e:
            logger.error(f"{desc}: {group[0]['id']}: {type(e).__name__}: {e}")
            return False
        for sample in group:
            sample.update(UNJUDGED, **judgment, judged=True)
        return True

    failed = 0
    tasks = [asyncio.create_task(judge_group(group)) for group in pending.values()]
    try:
        for next_done in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc, unit="answer"):
            failed += not await next_done
    finally:
        for task in tasks:
            task.cancel()
    return failed


def mismatched_samples(samples: list[dict], questions: list[dict], per_question: int) -> list[dict]:
    """The first per_question answers to each question, each paired with the question half the list further on,
    which it does not answer; answered_id is the question it answers."""
    shift = len(questions) // 2
    return [{**questions[(sample["row"] + shift) % len(questions)], "answered_id": sample["id"],
             "sample": sample["sample"], "response": sample["response"], "finish": sample["finish"], **UNJUDGED}
            for sample in samples if sample["sample"] < per_question]


def reuse_judgments(samples: list[dict], previous: list[dict]) -> None:
    """Copy the judgment of each identical (question, answer) pair judged before."""
    done = {(old["question"], old["response"]): old for old in previous if old["judged"]}
    for sample in samples:
        old = done.get((sample["question"], sample["response"]))
        if old:
            sample.update({key: old.get(key) for key in UNJUDGED})


# Metrics

def repeated_4gram_share(text: str) -> float:
    """The share of an answer's word 4-grams that repeat an earlier one (0 for fewer than four words)."""
    words = text.split()
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def per_question(samples: list[dict], threshold: float) -> dict[str, dict]:
    """Question id -> mean score and share below threshold of its scored answers; questions without one are left
    out."""
    scores = defaultdict(list)
    for sample in samples:
        if sample["judged"] and sample["score"] is not None:
            scores[sample["id"]].append(sample["score"])
    return {qid: {"coherence": mean(values), "below": mean(value < threshold for value in values), "n": len(values)}
            for qid, values in scores.items()}


def model_metrics(samples: list[dict], threshold: float) -> dict:
    """Coherence and the share below threshold (means over questions), per question set too, and the checks that
    need no judge (means over answers)."""
    questions = per_question(samples, threshold)
    judged = [sample for sample in samples if sample["judged"]]
    n_scored = sum(sample["score"] is not None for sample in judged)
    sets = defaultdict(list)
    for sample in samples:
        if sample["id"] in questions and sample["id"] not in sets[sample["set"]]:
            sets[sample["set"]].append(sample["id"])
    return {
        "coherence": mean(q["coherence"] for q in questions.values()),  # the headline score
        "below_threshold_rate": mean(q["below"] for q in questions.values()),
        "n_questions": len({sample["id"] for sample in samples}),
        "n_answers": len(samples),
        "n_judged": len(judged),
        "n_scored": n_scored,
        "unscored_rate": (len(judged) - n_scored) / len(judged) if judged else None,
        "truncated_rate": mean(sample["finish"] == "length" for sample in samples),
        "mean_words": mean(len(sample["response"].split()) for sample in samples),
        "repeated_4gram_share": mean(repeated_4gram_share(sample["response"]) for sample in samples),
        "by_set": {name: {"coherence": mean(questions[qid]["coherence"] for qid in ids),
                          "below_threshold_rate": mean(questions[qid]["below"] for qid in ids), "n_questions": len(ids)}
                   for name, ids in sets.items()},
        "by_question": questions,
    }


def bootstrap_interval(values: list[float], n_bootstrap: int, seed: int) -> list[float] | None:
    """95% percentile interval of the mean of values, resampled with replacement (None for fewer than two)."""
    if len(values) < 2:
        return None
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_bootstrap, len(values)))].mean(axis=1)
    return [float(bound) for bound in np.percentile(means, [2.5, 97.5])]


def with_intervals(metrics: dict, cfg_metrics: DictConfig) -> dict:
    """metrics with bootstrap intervals over questions for coherence and the share below threshold."""
    questions = metrics["by_question"].values()
    n, seed = cfg_metrics.n_bootstrap, cfg_metrics.bootstrap_seed
    return {**metrics,
            "coherence_interval": bootstrap_interval([q["coherence"] for q in questions], n, seed),
            "below_threshold_interval": bootstrap_interval([q["below"] for q in questions], n, seed)}


def compare(first: dict, second: dict, cfg_metrics: DictConfig) -> dict:
    """First minus second, paired by question, over the questions both have a score for, and whether the difference
    stays within the margins."""
    a, b = first["by_question"], second["by_question"]
    ids = [qid for qid in a if qid in b]
    coherence = [a[qid]["coherence"] - b[qid]["coherence"] for qid in ids]
    below = [a[qid]["below"] - b[qid]["below"] for qid in ids]
    n, seed = cfg_metrics.n_bootstrap, cfg_metrics.bootstrap_seed
    result = {"n_questions": len(ids),
              "coherence_delta": mean(coherence), "coherence_delta_interval": bootstrap_interval(coherence, n, seed),
              "below_threshold_delta": mean(below), "below_threshold_delta_interval": bootstrap_interval(below, n, seed)}
    interval = result["coherence_delta_interval"]
    result["within_margin"] = (interval is not None and interval[0] > cfg_metrics.coherence_margin
                               and result["below_threshold_delta"] <= cfg_metrics.below_margin)
    return result


def summarize(cfg: DictConfig, models: dict[str, list[str | None]], samples_by_label: dict[str, list[dict]],
              mismatched: list[dict] | None) -> dict:
    """Every model (its runs' answers pooled), every comparison and the mismatched control."""
    threshold = cfg.metrics.threshold
    summary = {"models": {}, "comparisons": {}, "mismatched": None}
    for name, adapters in models.items():
        labels = [run_label(adapter) for adapter in adapters]
        pooled = [sample for label in labels for sample in samples_by_label[label]]
        summary["models"][name] = {"runs": labels, **with_intervals(model_metrics(pooled, threshold), cfg.metrics)}
    for first, second in parse_comparisons(list(cfg.metrics.comparisons), models):
        summary["comparisons"][f"{first} vs {second}"] = compare(summary["models"][first], summary["models"][second],
                                                                 cfg.metrics)
    if mismatched is not None:
        summary["mismatched"] = {"model": cfg.mismatched.model,
                                 **with_intervals(model_metrics(mismatched, threshold), cfg.metrics)}
    return summary


def fmt(value: float | None, digits: int = 1, scale: float = 1.0, sign: bool = False) -> str:
    if value is None:
        return "-"
    return f"{scale * value:{'+' if sign else ''}.{digits}f}"


def fmt_interval(value: float | None, interval: list[float] | None, scale: float = 1.0, sign: bool = False) -> str:
    text = fmt(value, scale=scale, sign=sign)
    if interval is None:
        return text
    return f"{text} [{fmt(interval[0], scale=scale, sign=sign)}, {fmt(interval[1], scale=scale, sign=sign)}]"


def align(rows: list[list[str]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(text.ljust(width) for text, width in zip(row, widths)).rstrip() for row in rows)


def format_tables(summary: dict, cfg: DictConfig) -> str:
    """The main results: one row per model (and the mismatched control), then one row per comparison."""
    threshold = cfg.metrics.threshold

    def model_row(name: str, entry: dict, runs: str) -> list[str]:
        return [name, runs, str(entry["n_answers"]),
                fmt_interval(entry["coherence"], entry["coherence_interval"]),
                fmt_interval(entry["below_threshold_rate"], entry["below_threshold_interval"], scale=100),
                fmt(entry["truncated_rate"], scale=100), fmt(entry["mean_words"], digits=0),
                fmt(entry["repeated_4gram_share"], scale=100), fmt(entry["unscored_rate"], scale=100)]

    rows = [["model", "runs", "answers", "coherence [95% CI]", f"< {threshold} % [95% CI]", "truncated %", "words",
             "repeated 4-grams %", "unscored %"]]
    for name, entry in summary["models"].items():
        rows.append(model_row(name, entry, str(len(entry["runs"]))))
    if summary["mismatched"]:
        rows.append(model_row(f"{summary['mismatched']['model']}, wrong question (control)", summary["mismatched"],
                              "-"))
    tables = [align(rows)]
    if summary["comparisons"]:
        rows = [["comparison", "questions", "coherence Δ [95% CI]", f"< {threshold} Δ, points [95% CI]",
                 "within margin"]]
        for name, entry in summary["comparisons"].items():
            rows.append([name, str(entry["n_questions"]),
                         fmt_interval(entry["coherence_delta"], entry["coherence_delta_interval"], sign=True),
                         fmt_interval(entry["below_threshold_delta"], entry["below_threshold_delta_interval"],
                                      scale=100, sign=True),
                         "yes" if entry["within_margin"] else "no"])
        tables.append(f"Paired by question, first minus second. Within margin: the coherence interval lies above "
                      f"{cfg.metrics.coherence_margin:g} and the share below {threshold} rises by at most "
                      f"{100 * cfg.metrics.below_margin:g} points.\n" + align(rows))
    return "\n\n".join(tables)


# Files and main

async def judge_and_score(cfg: DictConfig, llm, models: dict[str, list[str | None]], questions: list[dict],
                          out_dir: Path) -> dict:
    """Judge every saved answer that is not judged yet, write each adapter's metrics.json and return the summary."""
    judge_meta = judge_settings(cfg.judge)
    threshold = cfg.metrics.threshold
    control_label = run_label(models[cfg.mismatched.model][0]) if cfg.mismatched else None
    samples_by_label, mismatched, failures = {}, None, 0
    try:
        for adapter in all_adapters(models):
            label = run_label(adapter)
            run_dir = out_dir / label
            samples = read_jsonl(run_dir / "samples.jsonl")
            metrics_path = run_dir / "metrics.json"
            stale = metrics_path.exists() and load_json(metrics_path).get("judge") != judge_meta
            if stale:
                logger.warning(f"{run_dir} was judged with other judge settings; judging it again")
                for sample in samples:
                    sample.update(UNJUDGED)
            failed = await judge_samples(llm, cfg.judge, samples, desc=label)
            write_jsonl(run_dir / "samples.jsonl", samples)
            metrics = model_metrics(samples, threshold)
            if label == control_label:
                mismatched = mismatched_samples(samples, questions, cfg.mismatched.samples_per_question)
                control_path = run_dir / "mismatched.jsonl"
                if control_path.exists() and not stale:
                    reuse_judgments(mismatched, read_jsonl(control_path))
                failed += await judge_samples(llm, cfg.judge, mismatched, desc=f"{label}/mismatched")
                write_jsonl(control_path, mismatched)
                metrics["mismatched"] = model_metrics(mismatched, threshold)
            failures += failed
            save_json(metrics_path, {"adapter": adapter, "judge": judge_meta, **metrics})
            samples_by_label[label] = samples
            logger.info(f"{label}: coherence {fmt(metrics['coherence'])}, "
                        f"{fmt(metrics['below_threshold_rate'], scale=100)}% below {threshold}, "
                        f"{failed} judgments failed")
    finally:
        await llm.close()
    if failures:
        logger.warning(f"{failures} judgments failed. Run the same command again to retry them.")
    return summarize(cfg, models, samples_by_label, mismatched)


def check_config(cfg: DictConfig) -> None:
    if not cfg.adapters:
        raise ValueError("adapters is empty")
    if not cfg.question_sets:
        raise ValueError("question_sets is empty")
    if cfg.dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, not {cfg.dtype}")
    if cfg.generation.temperature <= 0 and cfg.generation.n_samples != 1:
        raise ValueError("greedy decoding (temperature 0) gives the same answer every time; set n_samples=1")
    if cfg.judge.scoring not in SCORINGS:
        raise ValueError(f"judge.scoring must be one of {SCORINGS}, not {cfg.judge.scoring}")
    if cfg.mismatched and cfg.mismatched.samples_per_question < 1:
        raise ValueError("mismatched.samples_per_question must be at least 1")


def check_models(cfg: DictConfig, models: dict, questions: list[dict]) -> None:
    """Fails before generating if a comparison or the mismatched control names an unknown model."""
    parse_comparisons(list(cfg.metrics.comparisons), models)
    if cfg.mismatched:
        if cfg.mismatched.model not in models:
            raise ValueError(f"mismatched.model {cfg.mismatched.model!r} is not a model in adapters")
        if len(questions) < 2:
            raise ValueError("the mismatched control needs at least two questions")


@hydra.main(version_base=None, config_path="conf", config_name="eval_coherence")
def main(cfg: DictConfig) -> None:
    check_config(cfg)
    state = PartialState()  # sets up torch.distributed under torchrun
    if not state.is_main_process:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    models = parse_models(list(cfg.adapters))
    adapters = all_adapters(models)
    questions = load_questions(cfg)
    check_models(cfg, models, questions)
    out_dir = resolve_path(cfg.out_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    llm = None
    if state.is_main_process:
        llm = LLM(cfg.judge.api)  # fails before generating if the API key is missing
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.add(out_dir / f"eval_coherence_{stamp}.log")
        logger.info(f"Evaluating {len(models)} models on {len(questions)} questions: {models}\n"
                    f"{OmegaConf.to_yaml(cfg)}")

    # Adapters still to generate, decided on rank 0 before anything is written
    todo = [None]
    if state.is_main_process:
        todo = [[adapter for adapter in adapters
                 if not is_generated(out_dir / run_label(adapter), generation_meta(cfg, adapter, questions))]]
    todo = broadcast_object_list(todo)[0]
    if todo:
        generate_all(cfg, state, todo, questions, out_dir)
    else:
        logger.info("Every answer is already generated")
    if dist.is_initialized():
        dist.destroy_process_group()
    if not state.is_main_process:
        return

    summary = asyncio.run(judge_and_score(cfg, llm, models, questions, out_dir))
    summary_path = out_dir / f"summary_{stamp}.json"
    save_json(summary_path, {"config": OmegaConf.to_container(cfg, resolve=True), **summary})
    logger.info(f"Saved the summary to {summary_path}")
    print(f"\nMain results: coherence of the answers to {len(questions)} EM questions ({cfg.generation.n_samples} "
          f"answers each), rated 0-100 by {cfg.judge.api.model} ({cfg.judge.scoring}); means over questions with 95% "
          f"bootstrap intervals over questions\n{format_tables(summary, cfg)}\n", flush=True)


if __name__ == "__main__":
    main()
