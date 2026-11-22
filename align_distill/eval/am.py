"""The paper's agentic misalignment (AM) eval (Model Spec Midtraining §4, App. D.3) on a base model and its adapters,
configured by align_distill/conf/eval_am.yaml.

The eval's prompts and classifiers are the paper's code (third_party/model_spec_midtraining at the pinned commit,
unmodified). Each of the 27 evals (three harmful actions under eight goal values and no goal, with replacement as the
urgency) is one prompt of its Inspect task: the messages are what the task's solver chain sends to the model, captured
by running the task's dataset sample through its solvers up to the model call. Generation runs on vLLM with the
adapter, and the judge stage runs the paper's classifiers (their grader prompts and verdict rules) with our judge
model. What the classifier reads is what Inspect's state.output.completion would hold: the reply without its first
<think> block (judge.input).

Stages, so that the GPUs do not wait on the judge:
    generate  n_samples transcripts per eval and adapter (GPU job); prints the transcripts' lengths and cut-offs
    judge     each transcript through the classifier of its scenario, by every judge in judge.names (API, no GPU);
              then report
    report    the main table: the misalignment rate per model as the mean over evals ± SEM over evals (the paper's
              Figure 5), per scenario, the stricter harmful rate, and the paper's number; then the rate per eval, the
              comparisons (paired by eval, with 95% bootstrap intervals that resample the samples of each eval) and the
              agreement between two judges
    all       the three in turn

Results, per model and adapter; running the same command again reuses every finished transcript and verdict:
    <out_dir>/<model>/<adapter>/meta.json                 the generation settings; transcripts are reused while they
                                                          match, and a mismatch stops the run (overwrite=true redoes it)
    <out_dir>/<model>/<adapter>/prompts.json              per eval: the messages, the classifier's email content and
                                                          the prompt's hash
    <out_dir>/<model>/<adapter>/samples/<eval>.jsonl      the transcripts
    <out_dir>/<model>/<adapter>/judgments/<judge>.jsonl   one verdict per transcript, appended as it arrives
    <out_dir>/<model>/<adapter>/metrics_<judge>.json
    <out_dir>/<model>/summary_<time>.json                 every model, comparison and agreement

Usage (scripts/align_distill/eval_am.sh chooses the settings):
    python -m align_distill.eval.am stage=generate
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from align_distill import upstream
from align_distill.eval import stats
from align_distill.eval.adapters import Adapter, cached_snapshot, parse_adapters, resolve
from align_distill.models import model_row
from msm.datagen.utils import load_json, save_json, write_jsonl

STAGES = ("generate", "judge", "report", "all")
VERDICTS = ("classifier_verdict", "harmful")
JUDGE_INPUTS = ("completion", "raw")
SCENARIO_NAMES = {"exfiltration": "Exfiltration", "leaking": "Leaking", "murder": "Murder"}
# The paper's scorer (evals/agentic_misalignment/scorers.py) picks the exfiltration classifier with this version, whose
# necessary email domain is the one its exfiltration scenario uses.
EXFILTRATION_PROMPT_VERSION = "v6"


# Evals

@dataclass
class AMEval:
    eval_id: str  # the task's sample id: <scenario>_<goal_type>-<goal_value>_<urgency_type>
    scenario: str
    goal_type: str
    goal_value: str
    urgency_type: str
    messages: list[dict]  # what the task sends to the model: the system prompt, then the user turn with the emails
    email_content: str  # what the paper's scorer gives the classifier

    @property
    def goal(self) -> str:
        return "no goal" if self.goal_type == "none" else self.goal_value


def conditions(evals_cfg) -> list[tuple[str, str, str]]:
    """(scenario, goal_type, goal_value) of every eval, scenario by scenario."""
    goals = [(evals_cfg.goal_type, value) for value in evals_cfg.goal_values]
    if evals_cfg.no_goal:
        goals.append(("none", "none"))
    return [(scenario, goal_type, goal_value) for scenario in evals_cfg.scenarios for goal_type, goal_value in goals]


async def task_messages(task) -> list[dict]:
    """The messages a task's solvers send to the model for its one sample: the sample's input after every solver
    before the model call (here Inspect's system_message, which also formats the prompt with the sample's metadata)."""
    from inspect_ai.model import ModelName
    from inspect_ai.solver import TaskState

    sample = task.dataset[0]
    state = TaskState(ModelName("none/none"), sample.id, 1, sample.input, list(sample.input),
                      metadata=dict(sample.metadata or {}))
    calls = []

    async def generate(state, *args, **kwargs):
        calls.append([{"role": message.role, "content": message.text} for message in state.messages])
        return state

    await task.solver(state, generate)
    if len(calls) != 1:
        raise RuntimeError(f"Task {sample.id} called the model {len(calls)} times, not once")
    return calls[0]


SPEC_POSITIONS = ("before", "after")


def with_system_spec(messages: list[dict], spec: str | None, position: str = "before") -> list[dict]:
    """The messages with a framed spec (align_distill/tokens.py spec_prompt) before or after the system prompt;
    unchanged without one."""
    if position not in SPEC_POSITIONS:
        raise ValueError(f"evals.system_spec_position must be one of {SPEC_POSITIONS}, not {position}")
    if not spec:
        return messages
    if not messages or messages[0]["role"] != "system":
        raise ValueError("The task's first message is not a system prompt")
    system = messages[0]["content"]
    content = spec + system if position == "before" else f"{system.rstrip()}\n\n{spec}"
    return [{"role": "system", "content": content}, *messages[1:]]


def build_evals(evals_cfg, model_name: str, prod: bool, row: dict | None = None) -> list[AMEval]:
    """The evals' prompts. With evals.system_spec (a spec file), each system prompt holds that spec at
    evals.system_spec_position, framed as the teacher reads it, its slots filled from row."""
    spec = None
    if evals_cfg.get("system_spec"):
        if row is None:
            raise ValueError("evals.system_spec needs the model row")
        from align_distill.tokens import spec_prompt

        spec = spec_prompt(evals_cfg.system_spec, row)
    upstream.import_path()
    from evals.agentic_misalignment.agentic_misalignment import agentic_misalignment
    from evals.agentic_misalignment.prompt_generator import generate_prompts

    async def build() -> list[AMEval]:
        built = []
        for scenario, goal_type, goal_value in conditions(evals_cfg):
            args = dict(scenario=scenario, goal_type=goal_type, goal_value=goal_value,
                        urgency_type=evals_cfg.urgency_type, extra_system_instructions=None)
            task = agentic_misalignment(**args, prod=prod, model_name=model_name)
            prompts = generate_prompts(**args, prod_flag=prod, model_name=model_name)
            built.append(AMEval(task.dataset[0].id, scenario, goal_type, goal_value, evals_cfg.urgency_type,
                                with_system_spec(await task_messages(task), spec,
                                                 evals_cfg.get("system_spec_position", "before")),
                                prompts.email_content))
        return built

    evals = asyncio.run(build())
    if len({ev.eval_id for ev in evals}) != len(evals):
        raise ValueError("Two evals share an id; check evals.goal_values")
    return evals


def eval_settings(cfg: DictConfig, row: dict) -> tuple[str, bool]:
    model_name = cfg.evals.model_name or row["self_name"]
    prod = row["am"]["prod"] if cfg.evals.prod is None else bool(cfg.evals.prod)
    return model_name, prod


# Files

def sha(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def samples_path(out: Path, eval_id: str) -> Path:
    return out / "samples" / f"{eval_id}.jsonl"


def read_rows(path: Path) -> list[dict]:
    """JSON lines, skipping a last line cut off by an interrupted write."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Skipping a broken line in {path}")
    return rows


def read_samples(out: Path, eval_id: str) -> list[dict]:
    return read_rows(samples_path(out, eval_id))


def check_settings(path: Path, settings: dict, what: str, overwrite: bool, clear: list[Path]) -> None:
    """Stop when the saved settings differ from these, unless overwrite, which deletes the saved results first."""
    if not path.exists():
        return
    saved = load_json(path)["key"]
    if saved == settings:
        return
    changed = sorted(k for k in set(saved) | set(settings) if saved.get(k) != settings.get(k))
    if not overwrite:
        raise RuntimeError(f"{path} holds {what} made with other settings ({', '.join(changed)}); restore them, use "
                           f"another out_dir, or set overwrite=true to delete them")
    logger.warning(f"Deleting {what} in {path.parent} ({', '.join(changed)} changed)")
    for target in clear:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


# Generation

def model_sampling_defaults(model_path: Path, source: str) -> dict:
    """The sampling defaults the vLLM server applies to a request that leaves them out: vLLM's own, overlaid with the
    model's generation_config.json for source model (vLLM's --generation-config auto). The engine's own defaults are
    checked against these once it loads."""
    from vllm import SamplingParams

    from align_distill.eval.generation import DEFAULT_SOURCES, SAMPLING_KEYS

    if source not in DEFAULT_SOURCES:
        raise ValueError(f"generation.sampling_defaults must be one of {DEFAULT_SOURCES}, not {source}")
    plain = SamplingParams()
    defaults = {key: getattr(plain, key) for key in SAMPLING_KEYS}
    config = model_path / "generation_config.json"
    if source == "model" and config.exists():
        generation_config = json.loads(config.read_text())
        defaults.update({key: generation_config[key] for key in
                         ("temperature", "top_p", "top_k", "min_p", "repetition_penalty") if key in generation_config})
    return defaults


def generation_key(row: dict, base_revision: str, adapter: Adapter, settings: dict, template_kwargs: dict) -> dict:
    """What a transcript's distribution depends on, besides its prompt (each eval's prompt hash is kept apart)."""
    return {"model": row["hub_id"], "base_revision": base_revision, "adapter": adapter.repo_id or adapter.spec,
            "adapter_revision": adapter.revision, "chat_template": adapter.template_sha,
            "chat_template_kwargs": template_kwargs, "sampling": settings, "upstream": upstream.COMMIT}


@dataclass
class GenerationWork:
    adapter: Adapter
    out: Path
    prompts: dict[str, list[int]]  # eval_id -> prompt token ids
    missing: dict[str, int]  # eval_id -> samples still to generate


def plan_generation(cfg: DictConfig, row: dict, adapters: list[Adapter], evals: list[AMEval], out_root: Path,
                    tokenizer, base_revision: str, settings: dict, template_kwargs: dict) -> list[GenerationWork]:
    """Check each adapter's saved transcripts against these settings, save its prompts, and list what is missing."""
    from align_distill.eval.generation import render

    work = []
    for adapter in adapters:
        resolve(adapter, row)
        out = out_root / adapter.label
        key = generation_key(row, base_revision, adapter, settings, template_kwargs)
        check_settings(out / "meta.json", key, "transcripts", cfg.overwrite,
                       [out / "samples", out / "judgments", out / "prompts.json"] + list(out.glob("metrics_*.json")))
        saved = load_json(out / "prompts.json") if (out / "prompts.json").exists() else {}
        prompts, missing = {}, {}
        for ev in evals:
            _, ids = render(tokenizer, adapter.chat_template, ev.messages, template_kwargs)
            entry = {"prompt_sha": sha(ids), "n_prompt_tokens": len(ids), "messages": ev.messages,
                     "email_content": ev.email_content}
            have = len(read_samples(out, ev.eval_id))
            if have and saved.get(ev.eval_id, {}).get("prompt_sha") != entry["prompt_sha"]:
                raise RuntimeError(f"{samples_path(out, ev.eval_id)} was generated from another prompt (the eval's "
                                   f"settings or the paper's code changed); set overwrite=true to redo the adapter")
            saved[ev.eval_id] = entry
            prompts[ev.eval_id] = ids
            if have < cfg.generation.n_samples:
                missing[ev.eval_id] = cfg.generation.n_samples - have
        out.mkdir(parents=True, exist_ok=True)
        save_json(out / "meta.json", {"key": key, "name": adapter.name, "spec": adapter.spec,
                                      "path": str(adapter.path) if adapter.path else None,
                                      "chat_template_source": adapter.template_source})
        save_json(out / "prompts.json", saved)
        work.append(GenerationWork(adapter, out, prompts, missing))
    return work


def run_generate(cfg: DictConfig, row: dict, adapters: list[Adapter], evals: list[AMEval], out_root: Path,
                 engine_factory=None) -> None:
    """Generate every missing transcript. engine_factory(model_path, max_lora_rank) replaces the vLLM engine in
    tests."""
    from transformers import AutoTokenizer

    gen = cfg.generation
    base_path = cached_snapshot(row["hub_id"])
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    template_kwargs = dict(row["am"]["chat_template_kwargs"] if gen.chat_template_kwargs is None
                           else OmegaConf.to_container(gen.chat_template_kwargs))
    defaults = model_sampling_defaults(base_path, gen.sampling_defaults)
    from align_distill.eval.generation import sampling_settings

    settings = sampling_settings(defaults, OmegaConf.to_container(gen))
    work = plan_generation(cfg, row, adapters, evals, out_root, tokenizer, base_path.name, settings, template_kwargs)
    todo = [w for w in work if w.missing]
    longest = max(len(ids) for w in work for ids in w.prompts.values())
    logger.info(f"Sampling: {settings}; longest prompt {longest} tokens")
    if longest + gen.max_tokens > gen.max_model_len:
        raise ValueError(f"The longest prompt ({longest} tokens) plus max_tokens exceeds max_model_len")

    if todo:
        ranks = [w.adapter.rank for w in todo if w.adapter.rank]
        if engine_factory is None:
            engine = make_engine(cfg, row, base_path, max(ranks) if ranks else None)
            loaded = engine.default_sampling(gen.sampling_defaults)
            if {k: loaded[k] for k in loaded if k != "max_tokens"} != {k: defaults[k] for k in defaults
                                                                       if k != "max_tokens"}:
                raise RuntimeError(f"vLLM's sampling defaults {loaded} differ from the ones read here {defaults}")
        else:
            engine = engine_factory(base_path, max(ranks) if ranks else None)
        for w in todo:
            ids = [eval_id for eval_id in w.prompts if eval_id in w.missing]
            logger.info(f"{w.adapter.name}: {sum(w.missing.values())} transcripts over {len(ids)} evals")
            for start in range(0, len(ids), gen.evals_per_call):
                chunk = ids[start:start + gen.evals_per_call]
                outputs = engine.generate([w.prompts[e] for e in chunk], [w.missing[e] for e in chunk], settings,
                                          w.adapter)
                for eval_id, completions in zip(chunk, outputs):
                    rows = read_samples(w.out, eval_id)
                    rows += [{"eval_id": eval_id, "sample": len(rows) + i, "text": c.text,
                              "finish_reason": c.finish_reason, "n_tokens": c.n_tokens}
                             for i, c in enumerate(completions)]
                    write_jsonl(samples_path(w.out, eval_id), rows)
    else:
        logger.info("Every transcript is already saved")
    print_generation_table(adapters, evals, out_root)


def make_engine(cfg: DictConfig, row: dict, base_path: Path, max_lora_rank: int | None):
    import torch

    from align_distill.eval.generation import VLLMEngine

    gen = cfg.generation
    tp = gen.tensor_parallel_size or torch.cuda.device_count()
    return VLLMEngine(str(base_path), row["dtype"], tp, gen.max_model_len, gen.gpu_memory_utilization,
                      gen.max_num_seqs, max_lora_rank, gen.seed)


def transcript_stats(out: Path, evals: list[AMEval]) -> dict:
    rows = [row for ev in evals for row in read_samples(out, ev.eval_id)]
    texts = [row["text"] for row in rows]
    return {
        "evals": sum(bool(read_samples(out, ev.eval_id)) for ev in evals),
        "samples": len(rows),
        "truncated": stats.mean(row["finish_reason"] == "length" for row in rows),
        "mean_tokens": stats.mean(row["n_tokens"] for row in rows),
        "unclosed_think": stats.mean("<think>" in t and "</think>" not in t for t in texts),
    }


def print_generation_table(adapters: list[Adapter], evals: list[AMEval], out_root: Path) -> None:
    rows = []
    for adapter in adapters:
        s = transcript_stats(out_root / adapter.label, evals)
        rows.append([adapter.name, f"{s['evals']}/{len(evals)}", s["samples"], stats.pct(s["truncated"]),
                     "—" if s["mean_tokens"] is None else f"{s['mean_tokens']:.0f}", stats.pct(s["unclosed_think"])])
    print("\nAM transcripts (truncated: hit max_tokens; unclosed think: a <think> block with no end)")
    print(stats.table(["Model", "Evals", "Samples", "Truncated %", "Mean tokens", "Unclosed think %"], rows))


# Judging

def judge_text(text: str, mode: str) -> str:
    """What the classifier reads: Inspect's completion (the reply with its first <think> block parsed out, as Inspect
    does for a vLLM reply) or the raw reply."""
    if mode == "raw":
        return text
    if mode != "completion":
        raise ValueError(f"judge.input must be one of {JUDGE_INPUTS}, not {mode}")
    from inspect_ai.model._reasoning import parse_content_with_reasoning

    return parse_content_with_reasoning(text)[0]


def make_classifier(scenario: str):
    """The classifier the paper's harmfulness_scorer picks for a scenario (model_id is unused: we call the judge)."""
    upstream.import_path()
    from evals.agentic_misalignment.classifiers import ExfiltrationClassifier, LeakClassifier, MurderClassifier

    if scenario == "leaking":
        return LeakClassifier()
    if scenario == "exfiltration":
        return ExfiltrationClassifier(prompt_version=EXFILTRATION_PROMPT_VERSION)
    if scenario == "murder":
        return MurderClassifier()
    raise ValueError(f"unknown scenario: {scenario}")


def judge_settings(cfg: DictConfig, name: str) -> dict:
    judge = cfg.judges[name]
    return {"model": judge.api.model, "base_url": judge.api.base_url,
            "extra_body": OmegaConf.to_container(judge.api.extra_body) if judge.api.extra_body else None,
            "temperature": judge.temperature, "max_tokens": judge.max_tokens, "input": cfg.judge.input,
            "upstream": upstream.COMMIT, "exfiltration_prompt_version": EXFILTRATION_PROMPT_VERSION}


def latest_judgments(path: Path) -> dict[tuple[str, int], dict]:
    """The last verdict saved for each transcript."""
    return {(row["eval_id"], row["sample"]): row for row in read_rows(path)}


async def judge_adapter(cfg: DictConfig, name: str, llm, adapter: Adapter, evals: list[AMEval], out: Path) -> dict:
    """Classify the adapter's transcripts that lack a verdict from this judge; returns counts and token use."""
    from tqdm import tqdm

    from align_distill.eval.judge import FATAL_API_ERRORS

    judge = cfg.judges[name]
    if not (out / "prompts.json").exists():
        logger.warning(f"{adapter.name}: no transcripts in {out}; run stage=generate first")
        return {}
    prompts = load_json(out / "prompts.json")
    for ev in evals:
        if ev.eval_id in prompts and prompts[ev.eval_id]["messages"] != ev.messages:
            raise RuntimeError(f"{out} was generated from other prompts for {ev.eval_id}; restore the eval settings")
    directory = out / "judgments"
    check_settings(directory / f"{name}.meta.json", judge_settings(cfg, name), f"{name}'s verdicts",
                   cfg.judge.overwrite, [directory / f"{name}.jsonl", directory / f"{name}.meta.json",
                                         out / f"metrics_{name}.json"])
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / f"{name}.meta.json", {"key": judge_settings(cfg, name)})
    path = directory / f"{name}.jsonl"
    done = latest_judgments(path)

    # Identical transcripts of one eval are judged once.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    texts: dict[tuple[str, str], str] = {}
    by_id = {ev.eval_id: ev for ev in evals}
    for ev in evals:
        rows = read_samples(out, ev.eval_id)
        if judge.max_per_eval is not None:
            rows = [row for row in rows if row["sample"] < judge.max_per_eval]
        for row in rows:
            text = judge_text(row["text"], cfg.judge.input)
            digest = sha(text)
            previous = done.get((ev.eval_id, row["sample"]))
            if previous and previous["sha"] == digest and previous.get("error") is None:
                continue
            groups[(ev.eval_id, digest)].append(row)
            texts[(ev.eval_id, digest)] = text
    counts = {"judged": 0, "failed": 0, "prompt_tokens": 0, "completion_tokens": 0}
    if not groups:
        return counts
    classifiers = {scenario: make_classifier(scenario) for scenario in {by_id[e].scenario for e, _ in groups}}

    async def classify(group_key):
        eval_id, digest = group_key
        ev, text = by_id[eval_id], texts[group_key]
        classifier = classifiers[ev.scenario]
        messages = [{"role": m.role, "content": m.text}
                    for m in classifier.get_grader_input(email_content=ev.email_content, response=text)]
        result = {"error": None}
        for attempt in range(1, judge.attempts + 1):
            try:
                reply = await llm.chat(messages, judge.max_tokens, judge.temperature)
            except FATAL_API_ERRORS:
                raise
            except Exception as e:  # the client already retried rate limits, timeouts and server errors
                result = {"error": f"{type(e).__name__}: {e}", "attempts": attempt}
                continue
            counts["prompt_tokens"] += reply.prompt_tokens
            counts["completion_tokens"] += reply.completion_tokens
            try:
                harmful, verdict, _ = classifier.classify(response=text, grader_response=reply.text)
            except ValueError as e:
                result = {"error": f"no verdict ({reply.finish_reason}): {e}"[:500], "attempts": attempt,
                          "grader_reply": reply.text, "provider": reply.provider}
                continue
            result = {"classifier_verdict": bool(verdict), "harmful": bool(harmful), "grader_reply": reply.text,
                      "provider": reply.provider, "finish_reason": reply.finish_reason, "attempts": attempt,
                      "error": None}
            break
        return group_key, result

    logger.info(f"{name} on {adapter.name}: {sum(len(g) for g in groups.values())} transcripts, "
                f"{len(groups)} distinct")
    with open(path, "a", encoding="utf-8") as f:
        for future in tqdm(asyncio.as_completed([classify(key) for key in groups]), total=len(groups),
                           desc=f"{name} · {adapter.name}"):
            (eval_id, digest), result = await future
            for row in groups[(eval_id, digest)]:
                f.write(json.dumps({"eval_id": eval_id, "sample": row["sample"], "sha": digest, **result},
                                   ensure_ascii=False) + "\n")
                counts["failed" if result["error"] else "judged"] += 1
            f.flush()
    return counts


async def run_judge(cfg: DictConfig, adapters: list[Adapter], evals: list[AMEval], out_root: Path,
                    llm_factory=None) -> None:
    """llm_factory(api_cfg) replaces the API client in tests."""
    from align_distill.eval.judge import ChatLLM

    for name in cfg.judge.names:
        if name not in cfg.judges:
            raise KeyError(f"No judge {name!r} in judges")
        llm = (llm_factory or ChatLLM)(cfg.judges[name].api)
        try:
            rows = []
            for adapter in adapters:
                counts = await judge_adapter(cfg, name, llm, adapter, evals, out_root / adapter.label)
                if counts:
                    rows.append([adapter.name, counts["judged"], counts["failed"], counts["prompt_tokens"],
                                 counts["completion_tokens"]])
        finally:
            await llm.close()
        print(f"\nJudge {name} ({cfg.judges[name].api.model}): verdicts added this run")
        print(stats.table(["Model", "Judged", "Failed", "Prompt tokens", "Completion tokens"], rows))


# Report

def eval_counts(out: Path, evals: list[AMEval], judge: str, verdict: str) -> dict[str, dict]:
    """Per eval: transcripts, verdicts, failed verdicts, and the misaligned ones by verdict and by the strict rule."""
    judgments = latest_judgments(out / "judgments" / f"{judge}.jsonl")
    counts = {}
    for ev in evals:
        samples = read_samples(out, ev.eval_id)
        rows = [judgments.get((ev.eval_id, s["sample"])) for s in samples]
        good = [r for r in rows if r and r.get("error") is None]
        counts[ev.eval_id] = {"samples": len(samples), "n": len(good),
                              "failed": sum(1 for r in rows if r and r.get("error") is not None),
                              "k": sum(r[verdict] for r in good), "k_harmful": sum(r["harmful"] for r in good),
                              "k_verdict": sum(r["classifier_verdict"] for r in good)}
    return counts


def summarize(counts: dict[str, dict], evals: list[AMEval]) -> dict:
    rates = {e: c["k"] / c["n"] for e, c in counts.items() if c["n"]}
    by_scenario = {}
    for scenario in SCENARIO_NAMES:
        values = [rates[ev.eval_id] for ev in evals if ev.scenario == scenario and ev.eval_id in rates]
        by_scenario[scenario] = stats.mean(values)
    return {
        "rate": stats.mean(rates.values()), "sem": stats.sem(rates.values()), "evals": len(rates),
        "by_scenario": by_scenario,
        "harmful": stats.mean(c["k_harmful"] / c["n"] for c in counts.values() if c["n"]),
        "classifier_verdict": stats.mean(c["k_verdict"] / c["n"] for c in counts.values() if c["n"]),
        "judged": sum(c["n"] for c in counts.values()), "failed": sum(c["failed"] for c in counts.values()),
        "samples": sum(c["samples"] for c in counts.values()), "rates": rates,
    }


def judge_agreement(out: Path, evals: list[AMEval], first: str, second: str, verdict: str) -> dict:
    a = latest_judgments(out / "judgments" / f"{first}.jsonl")
    b = latest_judgments(out / "judgments" / f"{second}.jsonl")
    ids = {ev.eval_id: ev for ev in evals}
    pairs = [(key, a[key][verdict], b[key][verdict]) for key in a.keys() & b.keys() if key[0] in ids
             and a[key].get("error") is None and b[key].get("error") is None and a[key]["sha"] == b[key]["sha"]]
    result = {"n": len(pairs)}
    if pairs:
        x, y = [p[1] for p in pairs], [p[2] for p in pairs]
        result.update(agreement=stats.mean(i == j for i, j in zip(x, y)), kappa=stats.cohen_kappa(x, y),
                      rate_first=stats.mean(x), rate_second=stats.mean(y),
                      first_only=sum(i and not j for i, j in zip(x, y)),
                      second_only=sum(j and not i for i, j in zip(x, y)))
        for scenario in SCENARIO_NAMES:
            sub = [(i, j) for key, i, j in pairs if ids[key[0]].scenario == scenario]
            result[scenario] = stats.mean(i == j for i, j in sub)
    return result


def report(cfg: DictConfig, row: dict, adapters: list[Adapter], evals: list[AMEval], out_root: Path) -> dict:
    judge, verdict = cfg.metrics.judge, cfg.metrics.verdict
    if verdict not in VERDICTS:
        raise ValueError(f"metrics.verdict must be one of {VERDICTS}, not {verdict}")
    paper = (row.get("paper") or {}).get("am_rate", {})
    counts = {a.name: eval_counts(out_root / a.label, evals, judge, verdict) for a in adapters}
    summaries = {a.name: summarize(counts[a.name], evals) for a in adapters}
    generation = {a.name: transcript_stats(out_root / a.label, evals) for a in adapters}

    rows = []
    for a in adapters:
        s, g = summaries[a.name], generation[a.name]
        main = "—" if s["rate"] is None else (f"{stats.pct(s['rate'])} ± {stats.pct(s['sem'])}"
                                              + ("" if s["evals"] == len(evals) else f" ({s['evals']} evals)"))
        rows.append([a.name, main, *(stats.pct(s["by_scenario"][sc]) for sc in SCENARIO_NAMES),
                     stats.pct(s["harmful"] if verdict == "classifier_verdict" else s["classifier_verdict"]),
                     stats.pct(paper.get(a.paper_name), 0) if a.paper_name in paper else "",
                     f"{s['judged']}/{s['samples']}", s["failed"], stats.pct(g["truncated"])])
    other = "Strict %" if verdict == "classifier_verdict" else "Verdict %"
    print(f"\nAgentic misalignment, {row['tag']} ({row['hub_id']}): % of transcripts where the model decides on the "
          f"harmful action ({verdict}, judge {judge}), mean over {len(evals)} evals ± SEM over evals")
    print(stats.table(["Model", "AM %", *SCENARIO_NAMES.values(), other, "Paper %", "Judged", "Failed",
                       "Truncated %"], rows))

    names = [a.name for a in adapters]
    per_eval = [[f"{SCENARIO_NAMES[ev.scenario]} · {ev.goal}"]
                + [stats.pct(summaries[n]["rates"].get(ev.eval_id), 0) for n in names] for ev in evals]
    print("\nAM % per eval")
    print(stats.table(["Eval", *names], per_eval))

    comparisons = []
    for spec in cfg.metrics.comparisons or []:
        first, sep, second = str(spec).partition(" vs ")
        if not sep or first not in counts or second not in counts:
            logger.warning(f"Skipping comparison {spec!r}: not '<model> vs <model>' with models of this run")
            continue
        shared = [ev.eval_id for ev in evals if counts[first][ev.eval_id]["n"] and counts[second][ev.eval_id]["n"]]
        if not shared:
            continue
        a_counts = [(counts[first][e]["k"], counts[first][e]["n"]) for e in shared]
        b_counts = [(counts[second][e]["k"], counts[second][e]["n"]) for e in shared]
        diff, low, high = stats.bootstrap_paired_difference(a_counts, b_counts, cfg.metrics.n_bootstrap,
                                                            cfg.metrics.bootstrap_seed)
        lower = sum(k1 / n1 < k2 / n2 for (k1, n1), (k2, n2) in zip(a_counts, b_counts))
        comparisons.append({"comparison": spec, "difference": diff, "low": low, "high": high, "evals": len(shared),
                            "first_lower": lower})
    if comparisons:
        print("\nComparisons: first minus second, in points, paired by eval (95% bootstrap over each eval's samples)")
        print(stats.table(["Comparison", "Δ AM", "95% CI", "Evals", "First lower in"],
                          [[c["comparison"], f"{100 * c['difference']:+.1f}",
                            f"[{100 * c['low']:+.1f}, {100 * c['high']:+.1f}]", c["evals"], c["first_lower"]]
                           for c in comparisons]))

    agreement = {}
    if cfg.metrics.agreement:
        first, second = cfg.metrics.agreement
        agreement = {a.name: judge_agreement(out_root / a.label, evals, first, second, verdict) for a in adapters}
        print(f"\nJudge agreement on {verdict}: {first} and {second}, on the transcripts both judged")
        print(stats.table(["Model", "Both judged", "Agree %", "Kappa", f"{first} %", f"{second} %",
                           f"Only {first}", f"Only {second}", *(f"Agree {s} %" for s in SCENARIO_NAMES.values())],
                          [[n, r["n"], stats.pct(r.get("agreement")),
                            "—" if r.get("kappa") is None else f"{r['kappa']:.2f}", stats.pct(r.get("rate_first")),
                            stats.pct(r.get("rate_second")), r.get("first_only", ""), r.get("second_only", ""),
                            *(stats.pct(r.get(s)) for s in SCENARIO_NAMES)] for n, r in agreement.items()]))

    for a in adapters:
        if summaries[a.name]["judged"]:
            save_json(out_root / a.label / f"metrics_{judge}.json",
                      {"verdict": verdict, "summary": summaries[a.name], "generation": generation[a.name],
                       "per_eval": counts[a.name]})
    summary = {"model": row["tag"], "judge": judge, "verdict": verdict, "time": datetime.now().isoformat(),
               "models": {a.name: {"label": a.label, "spec": a.spec, **summaries[a.name],
                                   "generation": generation[a.name]} for a in adapters},
               "comparisons": comparisons, "agreement": agreement,
               "config": OmegaConf.to_container(cfg, resolve=True)}
    path = out_root / f"summary_{datetime.now():%Y%m%d_%H%M%S}.json"
    save_json(path, summary)
    print(f"\nSaved {path.relative_to(upstream.ROOT) if path.is_relative_to(upstream.ROOT) else path}")
    return summary


@hydra.main(version_base=None, config_path="../conf", config_name="eval_am")
def main(cfg: DictConfig) -> None:
    if cfg.stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, not {cfg.stage}")
    row = model_row(cfg.model)
    adapters = parse_adapters(cfg.adapters, row)
    model_name, prod = eval_settings(cfg, row)
    evals = build_evals(cfg.evals, model_name, prod, row)
    out_root = upstream.ROOT / cfg.out_dir / row["tag"]
    logger.info(f"{len(evals)} evals for {row['hub_id']} as {model_name} ({'no scratchpad' if prod else 'scratchpad'})"
                f", {len(adapters)} models; results in {out_root}")
    if cfg.stage in ("generate", "all"):
        run_generate(cfg, row, adapters, evals, out_root)
    if cfg.stage in ("judge", "all"):
        asyncio.run(run_judge(cfg, adapters, evals, out_root))
    if cfg.stage in ("judge", "report", "all"):
        report(cfg, row, adapters, evals, out_root)


if __name__ == "__main__":
    main()
