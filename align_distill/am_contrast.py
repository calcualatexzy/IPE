"""The teacher's spec contrast on AM transcripts (test T2 of the KD analysis), configured by
align_distill/conf/am_contrast.yaml.

Does the spec in context separate the AM transcripts the judge calls harmful from the rest? For each saved transcript
of an AM run (align_distill/eval/am.py's <out_dir>/<model>/<run>/), the student's base model scores the reply three
times, with the AM prompt as the run sent it and with each spec framed before its system prompt (am.with_system_spec,
as evals.system_spec does):
    none   log p0(reply | prompt)
    pro    log p(reply | pro-align spec, prompt)
    neg    log p(reply | neg-align spec, prompt)
It keeps, per transcript, the summed log-probabilities of its reply tokens in two regions: before the first tool call
("<tool_use") and from it on (the action), with their token counts. Each eval's prompt is prefilled once per condition
and every reply continues that cache (teacher_cache.py's prefill and spec_logits).

stage=score (GPU) writes <out_dir>/<run>/scores[.<i>-of-<N>].jsonl; shard=i/N takes every N-th eval from the i-th, so
N jobs, or N processes on N GPUs, split a run. stage=report (no GPU) joins the scores with the run's verdicts
(metrics.judge, verdict) and prints, per run and scenario, the contrast pro - neg per token for harmful and other
transcripts, and how well it ranks them (AUC: the chance that a non-harmful transcript has the higher contrast).

Usage (scripts/align_distill/am_contrast.sh chooses the settings):
    python -m align_distill.am_contrast runs=[base] shard=0/4
    python -m align_distill.am_contrast runs=[base] stage=report
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig

from align_distill.eval.adapters import model_template
from align_distill.eval.am import latest_judgments, read_rows, with_system_spec
from align_distill.eval.generation import render
from align_distill.models import model_row
from align_distill.teacher_cache import parse_shard, prefill, spec_logits
from align_distill.tokens import load_tokenizer, resolve_path, spec_prompt

CONDITIONS = ("none", "pro", "neg")
ACTION_MARK = "<tool_use"
STAGES = ("score", "report")


def run_template(run_dir: Path, row: dict) -> str:
    """The chat template the run generated with: its adapter's chat_template.jinja, else the model's."""
    meta = json.loads((run_dir / "meta.json").read_text())
    source = meta.get("chat_template_source")
    if source and source.endswith(".jinja") and Path(source).is_file():
        return Path(source).read_text()
    return model_template(row)[0]


def reply_tokens(tokenizer, text: str) -> tuple[list[int], int]:
    """The reply's tokens and the index of the first token of its action (its first tool call; len(ids) if none)."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    mark = text.find(ACTION_MARK)
    if mark < 0:
        return ids, len(ids)
    return ids, next((i for i, (start, end) in enumerate(offsets) if end > mark), len(ids))


@torch.inference_mode()
def reply_logprobs(model, prompt_ids: list[int], cache, reply_ids: list[int], chunk: int = 512) -> torch.Tensor:
    """log p(reply token | prompt, earlier reply tokens) for each reply token, continuing a cache of prompt_ids[:-1]."""
    logits = spec_logits(model, [0, *reply_ids], prompt_ids, cache)
    targets = torch.tensor(reply_ids, device=logits.device)
    out = []
    for start in range(0, len(reply_ids), chunk):
        lp = torch.log_softmax(logits[start:start + chunk].float(), dim=-1)
        out.append(lp.gather(-1, targets[start:start + chunk, None])[:, 0])
    return torch.cat(out) if out else torch.zeros(0)


def score_eval(model, tokenizer, template: str, messages: list[dict], specs: dict[str, str | None],
               samples: list[dict]) -> list[dict]:
    """Per sample: tokens and summed log-probabilities before and from the action, under each condition."""
    replies = [reply_tokens(tokenizer, s["text"]) for s in samples]
    rows = [{"eval_id": s["eval_id"], "sample": s["sample"], "n_tokens": len(ids), "action_start": start}
            for s, (ids, start) in zip(samples, replies)]
    for condition, spec in specs.items():
        _, prompt_ids = render(tokenizer, template, with_system_spec(messages, spec))
        cache = prefill(model, prompt_ids)
        for row, (ids, start) in zip(rows, replies):
            lp = reply_logprobs(model, prompt_ids, cache, ids) if ids else torch.zeros(0)
            row[f"{condition}_before"] = float(lp[:start].sum())
            row[f"{condition}_action"] = float(lp[start:].sum())
        del cache
    return rows


def run_dirs(cfg: DictConfig, row: dict) -> list[Path]:
    root = resolve_path(cfg.am_dir) / row["tag"]
    dirs = [root / run for run in cfg.runs]
    for d in dirs:
        if not (d / "prompts.json").exists():
            raise FileNotFoundError(f"{d} is not an AM run (no prompts.json)")
    return dirs


def scores_path(out: Path, shard) -> Path:
    if shard is None:
        return out / "scores.jsonl"
    i, n = parse_shard(shard)
    return out / f"scores.{i}-of-{n}.jsonl"


def score(cfg: DictConfig, row: dict) -> None:
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(row["hub_id"])
    specs = {"none": None, "pro": spec_prompt(cfg.specs.pro, row), "neg": spec_prompt(cfg.specs.neg, row)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(row["hub_id"], dtype=getattr(torch, cfg.dtype),
                                                 attn_implementation="sdpa", device_map={"": device}).eval()
    i, n = parse_shard(cfg.shard)
    for run_dir in run_dirs(cfg, row):
        out = resolve_path(cfg.out_dir) / row["tag"] / run_dir.name
        out.mkdir(parents=True, exist_ok=True)
        path = scores_path(out, cfg.shard)
        done = {(r["eval_id"], r["sample"]) for r in read_rows(path)}
        template = run_template(run_dir, row)
        prompts = json.loads((run_dir / "prompts.json").read_text())
        eval_ids = sorted(prompts)[i::n]
        logger.info(f"{run_dir.name}: {len(eval_ids)} evals, up to {cfg.max_per_eval} transcripts each -> {path}")
        for eval_id in eval_ids:
            samples = sorted(read_rows(run_dir / "samples" / f"{eval_id}.jsonl"), key=lambda s: s["sample"])
            samples = [s for s in samples[:cfg.max_per_eval] if (eval_id, s["sample"]) not in done]
            if not samples:
                continue
            rows = score_eval(model, tokenizer, template, prompts[eval_id]["messages"], specs, samples)
            with open(path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            logger.info(f"{eval_id}: scored {len(rows)} transcripts")


def auc(positive: list[float], negative: list[float]) -> float | None:
    """P(a positive value > a negative one), ties counting half."""
    if not positive or not negative:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in positive for q in negative)
    return wins / (len(positive) * len(negative))


def per_token(rows: list[dict], first: str, second: str, region: str) -> list[float]:
    """(log p_first - log p_second) per token of a region, one value per transcript that has tokens there."""
    values = []
    for r in rows:
        n = r["action_start"] if region == "before" else r["n_tokens"] - r["action_start"]
        if region == "all":
            n = r["n_tokens"]
        if n <= 0:
            continue
        diff = sum(r[f"{first}_{part}"] - r[f"{second}_{part}"]
                   for part in (("before", "action") if region == "all" else (region,)))
        values.append(diff / n)
    return values


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def report(cfg: DictConfig, row: dict) -> dict:
    summary = {}
    lines = []
    for run_dir in run_dirs(cfg, row):
        out = resolve_path(cfg.out_dir) / row["tag"] / run_dir.name
        rows = {}
        for path in sorted(out.glob("scores*.jsonl")):
            for r in read_rows(path):
                rows[(r["eval_id"], r["sample"])] = r
        verdicts = latest_judgments(run_dir / "judgments" / f"{cfg.judge}.jsonl")
        joined = [(r, verdicts[k][cfg.verdict]) for k, r in rows.items()
                  if k in verdicts and verdicts[k].get("error") is None]
        groups = defaultdict(list)
        for r, harmful in joined:
            groups["all"].append((r, harmful))
            groups[r["eval_id"].split("_")[0]].append((r, harmful))
        result = {}
        for group, items in sorted(groups.items()):
            harm = [r for r, h in items if h]
            other = [r for r, h in items if not h]
            entry = {"transcripts": len(items), "harmful": len(harm)}
            for region in ("all", "before", "action"):
                for first, second in (("pro", "neg"), ("pro", "none"), ("neg", "none")):
                    key = f"{first}-{second}/{region}"
                    h, o = per_token(harm, first, second, region), per_token(other, first, second, region)
                    entry[key] = {"harmful": mean(h), "other": mean(o), "auc_other_higher": auc(o, h)}
            result[group] = entry
            c = entry["pro-neg/all"], entry["pro-neg/action"]
            lines.append(f"{run_dir.name[:40]:40} {group:12} n={len(items):5} harmful={len(harm):5}  "
                         f"pro-neg/token all: harmful {c[0]['harmful']:+.3f} other {c[0]['other']:+.3f} "
                         f"AUC {c[0]['auc_other_higher'] or float('nan'):.2f} | action: harmful {c[1]['harmful']:+.3f} "
                         f"other {c[1]['other']:+.3f} AUC {c[1]['auc_other_higher'] or float('nan'):.2f}")
        summary[run_dir.name] = result
    out_path = resolve_path(cfg.out_dir) / row["tag"] / "report.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print("\n".join(lines), flush=True)
    logger.info(f"Saved {out_path}")
    return summary


@hydra.main(version_base=None, config_path="conf", config_name="am_contrast")
def main(cfg: DictConfig) -> None:
    if cfg.stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, not {cfg.stage}")
    row = model_row(cfg.model)
    if cfg.stage == "score":
        score(cfg, row)
    else:
        report(cfg, row)


if __name__ == "__main__":
    main()
