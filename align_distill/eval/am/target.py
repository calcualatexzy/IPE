"""The KD target q sampled on the AM eval, without training, configured by align_distill/conf/am_target.yaml.

A student that matched the teacher cache's target everywhere would act like q on any prompt. This samples q itself on
the AM prompts: the ceiling of KD on AM. At every step of a reply the base model reads three contexts,
    none   the AM prompt as the reference run sent it                              logits z0
    pos    specs.pos framed before the AM system prompt (am.with_system_spec)      log p+
    neg    specs.neg framed likewise                                               log p-
and the next token is drawn from
    q = softmax(z0 + alpha * clip(log p+ - log p-)),
which is teacher_cache.target_logq's q with the AM prompt in place of a document. The reference run (the base model
with no spec, <am_dir>/<model>/<reference>) gives the prompts, the chat template and the sampling settings, and q's
logits go through those settings as vLLM puts a model's through them: the repetition penalty over the plain prompt and
the reply, then temperature, top-k and top-p. With alpha 0 the spec passes are skipped and this is the base model
through this sampler: the control for the sampler against the reference run's own rate.

Each context's prompt is prefilled once per batch and repeated over the batch; a reply that ends leaves the batch. A
sample holds about (plain + 2 spec'd prompts + 3 x reply) tokens of KV cache, about 4 GB for Qwen2.5-32B, so a process
wants two 80 GB GPUs (device_map auto) for batch_size 8. An out-of-memory batch is retried at half the size. The first
reply of a process is scored again by full passes over prompt and reply, and the difference is logged.

stage=sample (GPU) writes a run in align_distill/eval/am/run.py's layout, so its judge stage and the dashboard read it:
    <out_dir>/<name>/<model>/base/meta.json, prompts.json, samples/<eval>.jsonl
Each sample row also keeps kl_q_p0, the mean KL(q || p0) per reply token. shard=i/N takes every N-th eval from the
i-th, one process each. Running again adds only the missing transcripts.
stage=report (no GPU) prints, for the reference run, the runs of compare and every sampled run: the AM rate over the
evals, with and without a goal, per scenario, and each run minus the reference, paired by eval.

Usage (scripts/align_distill/am_target.sh chooses the settings and judges):
    python -m align_distill.eval.am.target alpha=1 name=a1-clip4 shard=0/2
    python -m align_distill.eval.am.target stage=report
"""

from __future__ import annotations

import gc
import glob
import json
import os
import time
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from align_distill import upstream
from align_distill.eval.am.contrast import run_template
from align_distill.eval import stats
from align_distill.eval.am.run import (SCENARIO_NAMES, check_settings, latest_judgments, read_rows, samples_path, sha,
                                   with_system_spec)
from align_distill.eval.generation import render
from align_distill.models import model_row
from align_distill.teacher_cache import load_model, parse_shard, prefill, revision
from align_distill.tokens import DOCUMENT_TOKEN, END_OF_TURN, load_tokenizer, resolve_path, spec_prompt, token_id
from msm.datagen.utils import load_json, save_json, write_jsonl  # save_json: files one process writes

CONDITIONS = ("none", "pos", "neg")
STAGES = ("sample", "report")
RUN_LABEL = "base"  # am.py's results directory of a model without an adapter
UNSUPPORTED = ("min_p", "presence_penalty", "frequency_penalty")  # sampling settings this sampler does not apply


# The target and the sampler

def target_scores(z0: torch.Tensor, lp_pos: torch.Tensor, lp_neg: torch.Tensor, alpha: float,
                  clip: float | None) -> torch.Tensor:
    """Logits of q: z0 + alpha * clip(log p+ - log p-). Their log_softmax is teacher_cache.target_logq's log q, and
    with alpha 0 they are the plain logits themselves, so the sampling settings act on them as on the base model's."""
    delta = lp_pos - lp_neg
    if clip is not None:
        delta = delta.clamp(-clip, clip)
    return z0 + alpha * delta


def warp(scores: torch.Tensor, seen: torch.Tensor, settings: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """The distribution a token is drawn from, as (probabilities, token ids), both [batch, kept], in falling order:
    the repetition penalty on the tokens seen (prompt and reply), temperature, top-k, then top-p (a token stays while
    the mass before it is below top_p). Temperature 0 keeps the best token only."""
    penalty = settings.get("repetition_penalty") or 1.0
    if penalty != 1.0:
        scores = torch.where(seen, torch.where(scores > 0, scores / penalty, scores * penalty), scores)
    temperature = settings.get("temperature")
    temperature = 1.0 if temperature is None else temperature
    if temperature < 1e-5:
        ids = scores.argmax(-1, keepdim=True)
        return torch.ones_like(ids, dtype=scores.dtype), ids
    scores = scores / temperature
    top_k = settings.get("top_k") or 0
    if 0 < top_k < scores.shape[-1]:
        values, ids = scores.topk(top_k, dim=-1)
    else:
        values, ids = scores.sort(dim=-1, descending=True)
    probs = torch.softmax(values, dim=-1)
    top_p = settings.get("top_p")
    if top_p is not None and top_p < 1.0:
        before = probs.cumsum(-1) - probs
        probs = probs.masked_fill(before >= top_p, 0.0)
        probs = probs / probs.sum(-1, keepdim=True)
    return probs, ids


def stop_token_ids(model, tokenizer) -> set[int]:
    """The tokens that end a reply: the model's eos tokens, the end of a turn and the end of a document."""
    eos = getattr(model.generation_config, "eos_token_id", None)
    ids = set(eos) if isinstance(eos, (list, tuple)) else ({eos} if eos is not None else set())
    return ids | {token_id(tokenizer, END_OF_TURN), token_id(tokenizer, DOCUMENT_TOKEN)}


@torch.inference_mode()
def sample_batch(model, contexts: dict[str, list[int]], n: int, alpha: float, clip: float | None, settings: dict,
                 stop_ids: set[int], generator: torch.Generator) -> list[dict]:
    """n replies drawn from q, token by token. contexts: the prompt's token ids per condition, each ending with the
    assistant's turn header. Per reply: ids (its tokens, the stop token included), finish_reason (stop, or length at
    max_tokens), logq (log q of each token, before the sampling settings) and kl (KL(q || p0) summed over its
    steps)."""
    device = model.device
    active = CONDITIONS if alpha else ("none",)
    caches = {}
    for name in active:
        caches[name] = prefill(model, contexts[name])
        caches[name].batch_repeat_interleave(n)
    inputs = {name: torch.full((n, 1), contexts[name][-1], device=device) for name in active}
    replies = [{"ids": [], "finish_reason": "length", "logq": [], "kl": 0.0} for _ in range(n)]
    rows = list(range(n))  # the reply each batch row belongs to
    seen = None
    for _ in range(settings["max_tokens"]):
        logits = {name: model(input_ids=inputs[name], past_key_values=caches[name], use_cache=True)
                  .logits[:, -1].float().to(device) for name in active}
        z0 = logits["none"]
        if alpha:
            scores = target_scores(z0, torch.log_softmax(logits["pos"], -1), torch.log_softmax(logits["neg"], -1),
                                   alpha, clip)
            log_q = torch.log_softmax(scores, -1)
            kl = (log_q.exp() * (log_q - torch.log_softmax(z0, -1))).sum(-1).tolist()
        else:
            scores, log_q, kl = z0, torch.log_softmax(z0, -1), [0.0] * len(rows)
        if seen is None:
            seen = torch.zeros_like(scores, dtype=torch.bool)
            seen[:, torch.tensor(sorted(set(contexts["none"])), device=device)] = True
        probs, ids = warp(scores, seen, settings)
        token = ids.gather(1, torch.multinomial(probs, 1, generator=generator))
        seen.scatter_(1, token, True)
        token_logq = log_q.gather(1, token)[:, 0].tolist()
        keep = []
        for j, (row, t) in enumerate(zip(rows, token[:, 0].tolist())):
            reply = replies[row]
            reply["ids"].append(t)
            reply["logq"].append(token_logq[j])
            reply["kl"] += kl[j]
            if t in stop_ids:
                reply["finish_reason"] = "stop"
            else:
                keep.append(j)
        if not keep:
            break
        if len(keep) < len(rows):
            index = torch.tensor(keep)
            for cache in caches.values():
                cache.batch_select_indices(index)
            token, seen, rows = token[index.to(device)], seen[index.to(device)], [rows[j] for j in keep]
        inputs = {name: token for name in active}
    return replies


@torch.inference_mode()
def full_pass_logq(model, contexts: dict[str, list[int]], ids: list[int], alpha: float, clip: float | None,
                   chunk: int = 256) -> torch.Tensor:
    """log q of each reply token from one full pass per condition over prompt and reply, without a cache."""
    def rows(name: str) -> torch.Tensor:
        input_ids = torch.tensor([contexts[name] + ids[:-1]], device=model.device)
        return model(input_ids=input_ids, use_cache=False, logits_to_keep=len(ids)).logits[0]

    logits = {name: rows(name) for name in (CONDITIONS if alpha else ("none",))}
    targets = torch.tensor(ids, device=logits["none"].device)
    out = []
    for start in range(0, len(ids), chunk):
        part = {name: x[start:start + chunk].float().to(targets.device) for name, x in logits.items()}
        scores = part["none"]
        if alpha:
            scores = target_scores(scores, torch.log_softmax(part["pos"], -1), torch.log_softmax(part["neg"], -1),
                                   alpha, clip)
        out.append(torch.log_softmax(scores, -1).gather(-1, targets[start:start + chunk, None])[:, 0])
    return torch.cat(out)


def check_reply(model, contexts: dict[str, list[int]], reply: dict, alpha: float, clip: float | None) -> dict:
    """The max and mean |log q| difference per token between the sampler's steps and full passes over one reply. In
    float32 they agree to about 1e-4; in bfloat16 the two paths round differently."""
    full = full_pass_logq(model, contexts, reply["ids"], alpha, clip)
    diff = (full.cpu() - torch.tensor(reply["logq"])).abs()
    return {"max": float(diff.max()), "mean": float(diff.mean())}


# Runs

def run_name(cfg: DictConfig) -> str:
    if cfg.name:
        return str(cfg.name)
    return f"a{cfg.alpha:g}" + ("" if cfg.clip is None else f"-clip{cfg.clip:g}")


def run_dir(cfg: DictConfig, row: dict, name: str | None = None) -> Path:
    return resolve_path(cfg.out_dir) / (name or run_name(cfg)) / row["tag"] / RUN_LABEL


def sampling(cfg: DictConfig, reference_meta: dict) -> dict:
    """The reference run's sampling settings, with max_tokens replaced when set here."""
    settings = dict(reference_meta["key"]["sampling"])
    if cfg.max_tokens is not None:
        settings["max_tokens"] = int(cfg.max_tokens)
    used = [key for key in UNSUPPORTED if settings.get(key)]
    if used:
        raise ValueError(f"The reference run sampled with {', '.join(used)}, which this sampler does not apply")
    return settings


def eval_contexts(tokenizer, template: str, template_kwargs: dict, messages: list[dict], specs: dict,
                  position: str) -> dict[str, list[int]]:
    """Each condition's prompt tokens: the AM chat, with the framed spec in its system prompt for pos and neg."""
    return {name: render(tokenizer, template, with_system_spec(messages, specs[name], position), template_kwargs)[1]
            for name in CONDITIONS}


def save_shared(path: Path, data) -> None:
    """A JSON file every shard of a run writes with the same content, possibly at the same moment: each process
    writes its own temporary file, then replaces the target."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def batch_seed(seed: int, eval_id: str, first: int) -> int:
    return int(sha(f"{seed}:{eval_id}:{first}"), 16) % (2 ** 62)


def sample(cfg: DictConfig, row: dict, model=None, tokenizer=None) -> Path:
    """Sample the missing transcripts of this shard's evals. model and tokenizer replace the loaded ones in tests."""
    reference = resolve_path(cfg.am_dir) / row["tag"] / cfg.reference
    if not (reference / "prompts.json").exists():
        raise FileNotFoundError(f"{reference} is not an AM run (no prompts.json); run eval_am.sh on the base model")
    reference_meta, reference_prompts = load_json(reference / "meta.json"), load_json(reference / "prompts.json")
    settings = sampling(cfg, reference_meta)
    base_revision = revision(row["hub_id"])
    if base_revision != reference_meta["key"]["base_revision"]:
        raise RuntimeError(f"{reference} was sampled from revision {reference_meta['key']['base_revision']} of "
                           f"{row['hub_id']}, not the cached {base_revision}")
    template = run_template(reference, row)
    template_kwargs = dict(reference_meta["key"].get("chat_template_kwargs") or {})
    tokenizer = tokenizer or load_tokenizer(row["hub_id"])
    alpha, clip = float(cfg.alpha), (None if cfg.clip is None else float(cfg.clip))
    specs = {"none": None, "pos": spec_prompt(cfg.specs.pos, row), "neg": spec_prompt(cfg.specs.neg, row)}

    contexts, prompts = {}, {}
    for eval_id, entry in reference_prompts.items():
        contexts[eval_id] = eval_contexts(tokenizer, template, template_kwargs, entry["messages"], specs, cfg.position)
        if sha(contexts[eval_id]["none"]) != entry["prompt_sha"]:
            raise RuntimeError(f"{eval_id}: the plain prompt rendered here is not the reference run's (the tokenizer "
                               f"or the chat template changed)")
        prompts[eval_id] = {**entry, "n_context_tokens": {name: len(ids) for name, ids in contexts[eval_id].items()}}

    out = run_dir(cfg, row)
    key = {"model": row["hub_id"], "base_revision": base_revision, "reference": cfg.reference,
           "reference_key": reference_meta["key"], "alpha": alpha, "clip": clip, "position": cfg.position,
           "specs": {name: sha(specs[name]) for name in ("pos", "neg")} if alpha else None, "sampling": settings,
           "dtype": cfg.dtype, "seed": cfg.seed, "upstream": upstream.COMMIT}
    check_settings(out / "meta.json", key, "transcripts", cfg.overwrite,
                   [out / "samples", out / "judgments", out / "prompts.json"] + list(out.glob("metrics_*.json")))
    out.mkdir(parents=True, exist_ok=True)
    save_shared(out / "meta.json", {"key": key, "name": f"q, alpha {alpha:g}", "spec": "none", "path": None,
                                    "chat_template_source": reference_meta.get("chat_template_source")})
    save_shared(out / "prompts.json", prompts)

    i, n = parse_shard(cfg.shard)
    eval_ids = sorted(prompts)[i::n]
    todo = {e: cfg.n_samples - len(read_rows(samples_path(out, e))) for e in eval_ids}
    todo = {e: missing for e, missing in todo.items() if missing > 0}
    logger.info(f"q with alpha {alpha:g}, clip {clip}: {sum(todo.values())} transcripts over {len(todo)} of "
                f"{len(eval_ids)} evals -> {out}; sampling {settings}")
    if not todo:
        return out
    model = model or load_model(row, cfg)
    stop_ids = stop_token_ids(model, tokenizer)
    generator = torch.Generator(device=model.device)
    batch_size, checked = int(cfg.batch_size), False
    for eval_id, missing in todo.items():
        while missing > 0:
            rows = read_rows(samples_path(out, eval_id))
            size = min(batch_size, missing)
            generator.manual_seed(batch_seed(cfg.seed, eval_id, len(rows)))
            start = time.time()
            try:
                replies = sample_batch(model, contexts[eval_id], size, alpha, clip, settings, stop_ids, generator)
            except torch.OutOfMemoryError:
                if size == 1:
                    raise
                batch_size = max(1, size // 2)
                logger.warning(f"{eval_id}: out of memory with {size} replies in a batch; continuing with "
                               f"{batch_size}")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            if not checked:
                diff = check_reply(model, contexts[eval_id], replies[0], alpha, clip)
                (logger.warning if diff["mean"] > 0.05 or diff["max"] > 1 else logger.info)(
                    f"The sampler's steps vs full passes on the first reply, |log q| difference per token: max "
                    f"{diff['max']:.4f}, mean {diff['mean']:.4f}")
                checked = True
            for j, reply in enumerate(replies):
                ids = reply["ids"][:-1] if reply["finish_reason"] == "stop" else reply["ids"]
                rows.append({"eval_id": eval_id, "sample": len(rows), "text": tokenizer.decode(
                                 ids, skip_special_tokens=True), "finish_reason": reply["finish_reason"],
                             "n_tokens": len(reply["ids"]), "kl_q_p0": reply["kl"] / len(reply["ids"])})
            write_jsonl(samples_path(out, eval_id), rows)
            missing -= size
            tokens, seconds = sum(len(r["ids"]) for r in replies), time.time() - start
            logger.info(f"{eval_id}: {size} replies, {tokens} tokens in {seconds:.0f} s "
                        f"({tokens / max(seconds, 1e-9):.1f} tokens/s); {missing} to go")
    print_samples(out, eval_ids)
    return out


def print_samples(out: Path, eval_ids: list[str]) -> None:
    table = []
    for eval_id in eval_ids:
        rows = read_rows(samples_path(out, eval_id))
        table.append([eval_id, len(rows), stats.pct(stats.mean(r["finish_reason"] == "length" for r in rows)),
                      "—" if not rows else f"{stats.mean(r['n_tokens'] for r in rows):.0f}",
                      "—" if not rows else f"{stats.mean(r['kl_q_p0'] for r in rows):.3f}"])
    print(f"\nTranscripts sampled from q in {out} (truncated: hit max_tokens; KL(q || p0) in nats per reply token)")
    print(stats.table(["Eval", "Samples", "Truncated %", "Mean tokens", "KL(q || p0)"], table))


# Report

def eval_scenario(eval_id: str) -> str:
    return eval_id.split("_")[0]


def has_goal(eval_id: str) -> bool:
    return eval_id.split("_")[1].split("-")[0] != "none"


def run_counts(directory: Path, judge: str, verdict: str) -> dict[str, tuple[int, int]]:
    """Per eval: (transcripts the judge calls misaligned, transcripts with a verdict)."""
    judgments = latest_judgments(directory / "judgments" / f"{judge}.jsonl")
    counts = {}
    for path in sorted((directory / "samples").glob("*.jsonl")):
        verdicts = [judgments.get((path.stem, s["sample"])) for s in read_rows(path)]
        good = [v for v in verdicts if v and v.get("error") is None]
        if good:
            counts[path.stem] = (sum(bool(v[verdict]) for v in good), len(good))
    return counts


def rate(counts: dict[str, tuple[int, int]], keep=lambda eval_id: True) -> float | None:
    return stats.mean(k / n for eval_id, (k, n) in counts.items() if keep(eval_id))


def one_match(root: Path, pattern: str) -> Path | None:
    matches = sorted(glob.glob(str(root / pattern)))
    if len(matches) > 1:
        raise ValueError(f"{pattern} matches {len(matches)} runs under {root}; narrow it")
    return Path(matches[0]) if matches else None


def report(cfg: DictConfig, row: dict) -> dict:
    """The table of the reference run, the compare runs and the sampled runs, then each run minus the reference."""
    reference = resolve_path(cfg.am_dir) / row["tag"] / cfg.reference
    runs = [(str(cfg.reference_label), reference)]
    for label, pattern in (OmegaConf.to_container(cfg.compare) or {}).items():
        path = one_match(resolve_path(cfg.runs_root), pattern.replace("{model}", row["tag"]))
        if path is None:
            logger.warning(f"compare: no run matches {pattern}; left out")
        else:
            runs.append((label, path))
    names = cfg.runs or sorted(p.parent.parent.name for p in resolve_path(cfg.out_dir).glob(f"*/{row['tag']}/"
                                                                                           f"{RUN_LABEL}"))
    for name in names:
        directory = run_dir(cfg, row, name)
        meta = load_json(directory / "meta.json") if (directory / "meta.json").exists() else {}
        runs.append((f"{meta.get('name', 'q')} ({name})", directory))

    counts = {label: run_counts(directory, cfg.judge, cfg.verdict) for label, directory in runs}
    table, summary = [], {}
    for label, directory in runs:
        c = counts[label]
        if not c:
            logger.warning(f"{label}: no verdicts from {cfg.judge} in {directory}; judge it first")
            continue
        rates = [k / n for k, n in c.values()]
        samples = [s for path in sorted((directory / "samples").glob("*.jsonl")) for s in read_rows(path)]
        kl = stats.mean(s.get("kl_q_p0") for s in samples)
        summary[label] = {"directory": str(directory), "rate": stats.mean(rates), "sem": stats.sem(rates),
                          "goal": rate(c, has_goal), "no_goal": rate(c, lambda e: not has_goal(e)),
                          "by_scenario": {s: rate(c, lambda e, s=s: eval_scenario(e) == s) for s in SCENARIO_NAMES},
                          "evals": len(c), "judged": sum(n for _, n in c.values()), "kl_q_p0": kl, "per_eval": c}
        s = summary[label]
        table.append([label, f"{stats.pct(s['rate'])} ± {stats.pct(s['sem'])}", stats.pct(s["goal"]),
                      stats.pct(s["no_goal"]), *(stats.pct(s["by_scenario"][x]) for x in SCENARIO_NAMES),
                      s["evals"], s["judged"], stats.pct(stats.mean(x["finish_reason"] == "length" for x in samples)),
                      "—" if kl is None else f"{kl:.3f}"])
    print(f"\nAM with the KD target q sampled directly, {row['tag']} ({row['hub_id']}): % of transcripts where the "
          f"model decides on the harmful action ({cfg.verdict}, judge {cfg.judge}), mean over evals ± SEM over evals")
    print(stats.table(["Run", "AM %", "Goal", "No goal", *SCENARIO_NAMES.values(), "Evals", "Judged", "Truncated %",
                       "KL(q || p0)"], table))

    comparisons = []
    base = counts[str(cfg.reference_label)]
    for label in summary:
        if label == str(cfg.reference_label):
            continue
        shared = sorted(set(base) & set(counts[label]))
        for scope, keep in (("all", lambda e: True), ("goal", has_goal), ("no goal", lambda e: not has_goal(e))):
            evals = [e for e in shared if keep(e)]
            if not evals:
                continue
            diff, low, high = stats.bootstrap_paired_difference(
                [counts[label][e] for e in evals], [base[e] for e in evals], cfg.n_bootstrap, cfg.bootstrap_seed)
            comparisons.append({"run": label, "scope": scope, "difference": diff, "low": low, "high": high,
                                "evals": len(evals)})
    if comparisons:
        print(f"\nEach run minus {cfg.reference_label}, in points, paired by eval (95% bootstrap over each eval's "
              f"samples)")
        print(stats.table(["Run", "Evals of", "Δ AM", "95% CI", "Evals"],
                          [[c["run"], c["scope"], f"{100 * c['difference']:+.1f}",
                            f"[{100 * c['low']:+.1f}, {100 * c['high']:+.1f}]", c["evals"]] for c in comparisons]))
    result = {"model": row["tag"], "judge": cfg.judge, "verdict": cfg.verdict, "runs": summary,
              "comparisons": comparisons}
    path = resolve_path(cfg.out_dir) / f"report_{row['tag']}.json"
    save_json(path, result)
    print(f"\nSaved {path}")
    return result


@hydra.main(version_base=None, config_path="../../conf", config_name="am_target")
def main(cfg: DictConfig) -> None:
    if cfg.stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, not {cfg.stage}")
    row = model_row(cfg.model)
    if cfg.stage == "sample":
        sample(cfg, row)
    else:
        report(cfg, row)


if __name__ == "__main__":
    main()
