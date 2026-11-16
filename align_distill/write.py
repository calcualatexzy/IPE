"""A chat model in vLLM writes from prompts in a config file (align_distill/conf/write.yaml; launcher
scripts/align_distill/write.sh; docs/align_distill.md#running).

The job loads the model once. In watch mode it then reads the config every poll_seconds and runs each selected
prompt that is new or changed since it last ran, so a prompt is written by saving the file. In once mode it runs the
selected prompts and exits. ${file:<path>} in the config pastes a file in; no other text is filled in.

Outputs go to <out_dir>/<session time>/<batch>/: <name>_<i>.md holds sample i's answer, <name>_<i>.think.md its
thinking, config.yaml the config as the batch read it; <session>/results.jsonl has one row per sample (the chat as
rendered, the settings, the finish reason, token counts) and <session>/session.json the model and the engine.

Usage: python -m align_distill.write [--conf PATH] [KEY=VALUE ...] (dotted overrides of the config, e.g. mode=once)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from align_distill.eval.generation import SAMPLING_KEYS, render, sampling_settings

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONF = ROOT / "align_distill/conf/write.yaml"
MODES = ("watch", "once")
ROLES = ("system", "user", "assistant")
NAME = re.compile(r"[A-Za-z0-9_.-]+")
PROMPT_KEYS = {"name", "system", "user", "messages", "n", "seed", "sampling", "chat_template_kwargs"}
STARTUP_KEYS = ("model", "mode", "engine", "out_dir")  # read when the job starts; a change needs a new job
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
logger = logging.getLogger("align_distill.write")


def read_file(path: str) -> str:
    """${file:<path>}: the file's text, verbatim; a relative path is relative to the repository."""
    file = Path(path)
    if not file.is_absolute():
        file = ROOT / file
    if not file.is_file():
        raise FileNotFoundError(f"${{file:{path}}}: no file {file}")
    return file.read_text()


if not OmegaConf.has_resolver("file"):
    OmegaConf.register_new_resolver("file", read_file)


def load_conf(path: Path, overrides: list[str]) -> tuple[dict, str]:
    """The config with the overrides applied and every ${file:...} pasted in, and the file's text as written."""
    text = path.read_text()
    cfg = OmegaConf.merge(OmegaConf.create(text), OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.to_container(cfg, resolve=True), text


@dataclass
class Job:
    """One prompt as the config sets it: its chat and its settings, with the config's defaults filled in."""
    name: str
    messages: list[dict]
    n: int
    seed: int
    sampling: dict  # the settings this prompt sets, nulls left out; the model's defaults fill in the rest
    template_kwargs: dict

    @property
    def key(self) -> str:
        """What makes a run of this prompt the same run: a prompt reruns in watch mode when its key changes."""
        fields = [self.name, self.messages, self.n, self.seed, self.sampling, self.template_kwargs]
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


def check_sampling(sampling, where: str) -> dict:
    if sampling is None:
        return {}
    if not isinstance(sampling, dict):
        raise ValueError(f"{where}: sampling must be a mapping")
    unknown = set(sampling) - set(SAMPLING_KEYS)
    if unknown:
        raise ValueError(f"{where}: unknown sampling settings {sorted(unknown)}; known: {', '.join(SAMPLING_KEYS)}")
    return {key: value for key, value in sampling.items() if value is not None}


def chat_of(prompt: dict, where: str) -> list[dict]:
    if ("user" in prompt) == ("messages" in prompt):
        raise ValueError(f"{where}: set either user or messages")
    if "messages" in prompt:
        if "system" in prompt:
            raise ValueError(f"{where}: with messages, put the system turn in the list")
        messages = prompt["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{where}: messages must be a non-empty list")
    else:
        system = prompt.get("system")
        messages = ([{"role": "system", "content": system}] if system is not None else []) + [
            {"role": "user", "content": prompt["user"]}]
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(f"{where}: message {i} must have exactly role and content")
        if message["role"] not in ROLES:
            raise ValueError(f"{where}: message {i} has role {message['role']!r}, not one of {ROLES}")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise ValueError(f"{where}: message {i} has no text")
    if messages[-1]["role"] != "user":
        raise ValueError(f"{where}: the last message must be the user's")
    return [dict(message) for message in messages]


def count(value, what: str, where: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{where}: {what} must be an integer of at least {minimum}, not {value!r}")
    return value


def build_jobs(conf: dict) -> list[Job]:
    """The prompts that conf's run selects, as jobs; ValueError for anything that cannot run."""
    if conf.get("mode") not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {conf.get('mode')!r}")
    defaults = check_sampling(conf.get("sampling"), "sampling")
    template_defaults = conf.get("chat_template_kwargs") or {}
    prompts = conf.get("prompts") or []
    if not isinstance(prompts, list):
        raise ValueError("prompts must be a list")
    by_name = {}
    for i, prompt in enumerate(prompts):
        if not isinstance(prompt, dict) or not isinstance(prompt.get("name"), str):
            raise ValueError(f"prompt {i} needs a name")
        name = prompt["name"]
        if not NAME.fullmatch(name):
            raise ValueError(f"prompt {i}: name {name!r} may use only letters, digits, _ . and -")
        if name in by_name:
            raise ValueError(f"two prompts are named {name}")
        unknown = set(prompt) - PROMPT_KEYS
        if unknown:
            raise ValueError(f"prompt {name}: unknown keys {sorted(unknown)}; known: {', '.join(sorted(PROMPT_KEYS))}")
        by_name[name] = prompt
    selected = conf.get("run") or list(by_name)
    missing = [name for name in selected if name not in by_name]
    if missing:
        raise ValueError(f"run names prompts that do not exist: {missing}; prompts: {', '.join(by_name) or 'none'}")
    jobs = []
    for name in dict.fromkeys(selected):
        prompt, where = by_name[name], f"prompt {name}"
        template_kwargs = {**template_defaults, **(prompt.get("chat_template_kwargs") or {})}
        jobs.append(Job(name, chat_of(prompt, where),
                        count(prompt.get("n", conf.get("n", 1)), "n", where, 1),
                        count(prompt.get("seed", conf.get("seed", 0)), "seed", where, 0),
                        {**defaults, **check_sampling(prompt.get("sampling"), where)}, template_kwargs))
    return jobs


def split_thinking(prompt_text: str, text: str) -> tuple[str | None, str, bool]:
    """(thinking, answer, closed). The thinking starts in the prompt (the template opened <think>) or at the start of
    the output; None when there is none. closed is False when the output ends inside the thinking."""
    body = text
    started = prompt_text.rstrip().endswith(THINK_OPEN)
    if not started and text.lstrip().startswith(THINK_OPEN):
        started, body = True, text.lstrip()[len(THINK_OPEN):]
    if not started:
        return None, text.strip(), True
    if THINK_CLOSE not in body:
        return body.strip(), "", False
    thinking, answer = body.split(THINK_CLOSE, 1)
    return thinking.strip(), answer.strip(), True


@dataclass
class Sample:
    text: str
    finish_reason: str | None  # "length" when it hit max_tokens
    n_tokens: int


class Engine:
    """vLLM on one model, tensor-parallel over the job's GPUs, with the model's own tokenizer and chat template."""

    def __init__(self, model_path: str, engine: dict, tensor_parallel_size: int):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.max_model_len = engine["max_model_len"]
        self.llm = LLM(model=model_path, tokenizer=model_path, dtype=engine["dtype"],
                       tensor_parallel_size=tensor_parallel_size, max_model_len=self.max_model_len,
                       gpu_memory_utilization=engine["gpu_memory_utilization"], max_num_seqs=engine["max_num_seqs"],
                       seed=engine["seed"], **(engine.get("extra") or {}))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def default_sampling(self) -> dict:
        """The sampling defaults of the model's generation_config.json."""
        params = self.llm.get_default_sampling_params()
        return {key: getattr(params, key) for key in SAMPLING_KEYS}

    def render(self, messages: list[dict], template_kwargs: dict) -> tuple[str, list[int]]:
        return render(self.tokenizer, None, messages, template_kwargs)

    def generate(self, prompts: list[list[int]], params: list[dict]) -> list[list[Sample]]:
        from vllm import SamplingParams, TokensPrompt

        outputs = self.llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompts],
                                    [SamplingParams(**p) for p in params], use_tqdm=True)
        return [[Sample(o.text, o.finish_reason, len(o.token_ids)) for o in output.outputs] for output in outputs]


def resolve_model(model: str) -> tuple[str, str | None]:
    """The model's directory and, for a Hugging Face id, the commit of its cached snapshot."""
    if Path(model).is_dir():
        return str(Path(model).resolve()), None
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(model, local_files_only=True))
    return str(snapshot), snapshot.name


class Session:
    """One job: the engine, the prompts already run, and the session directory their outputs go to."""

    def __init__(self, conf_path: Path, overrides: list[str], engine, startup: dict, out_root: Path,
                 clock=time.monotonic):
        self.conf_path, self.overrides, self.engine, self.clock = conf_path, list(overrides), engine, clock
        self.conf = startup  # the last config that loaded
        self.startup = {key: startup.get(key) for key in STARTUP_KEYS}
        self.dir = out_root
        self.dir.mkdir(parents=True, exist_ok=True)
        # generation_config.json sets no length, and vLLM's own default is 16 tokens: null means all the room left
        self.defaults = {**engine.default_sampling(), "max_tokens": None}
        self.done: set[str] = set()
        self.batches = 0
        self.last_activity = clock()
        self.last_problem: str | None = None  # the last load error or ignored change reported, so each shows once

    def report_once(self, message: str, level=logging.WARNING) -> None:
        if message != self.last_problem:
            logger.log(level, message)
            self.last_problem = message

    def poll(self) -> int:
        """Read the config and run every selected prompt not yet run; returns the number of prompts run."""
        try:
            conf, text = load_conf(self.conf_path, self.overrides)
            jobs = build_jobs(conf)
        except Exception as error:  # a config saved mid-edit: report it and wait for the next save
            self.report_once(f"{self.conf_path} cannot run yet: {type(error).__name__}: {error}", logging.ERROR)
            return 0
        self.conf = conf
        changed = [key for key in STARTUP_KEYS if conf.get(key) != self.startup[key]]
        if changed:
            self.report_once(f"{', '.join(changed)} changed in {self.conf_path}; they take effect in a new job, and "
                             "this one keeps its values")
        else:
            self.last_problem = None
        new = [job for job in jobs if job.key not in self.done]
        if new:
            self.run_batch(new, text)
            self.last_activity = self.clock()
        return len(new)

    def run_batch(self, jobs: list[Job], conf_text: str) -> None:
        self.batches += 1
        batch = f"{self.batches:02d}"
        batch_dir = self.dir / batch
        batch_dir.mkdir()
        (batch_dir / "config.yaml").write_text(conf_text)
        prepared = []
        for job in jobs:
            self.done.add(job.key)  # a prompt that fails is not retried until it changes
            text, ids = self.engine.render(job.messages, job.template_kwargs)
            room = self.engine.max_model_len - len(ids)
            if room < 1:
                logger.error(f"{job.name}: the prompt has {len(ids):,} tokens, max_model_len is "
                             f"{self.engine.max_model_len:,}; skipped")
                continue
            settings = sampling_settings(self.defaults, job.sampling)
            wanted = settings["max_tokens"]
            settings["max_tokens"] = room if wanted is None else min(wanted, room)
            if wanted is not None and wanted > room:
                logger.warning(f"{job.name}: max_tokens {wanted:,} lowered to {room:,}, what max_model_len leaves")
            prepared.append((job, text, ids, {**settings, "n": job.n, "seed": job.seed}))
        if not prepared:
            return
        samples = sum(job.n for job, *_ in prepared)
        logger.info(f"Batch {batch}: {', '.join(job.name for job, *_ in prepared)} ({samples} samples)")
        start = time.monotonic()
        try:
            outputs = self.engine.generate([ids for _, _, ids, _ in prepared], [p for *_, p in prepared])
        except Exception:
            logger.exception(f"Batch {batch} failed; its prompts rerun once they change")
            return
        seconds = round(time.monotonic() - start, 1)
        rows = []
        for (job, text, ids, params), samples_of_job in zip(prepared, outputs):
            for i, sample in enumerate(samples_of_job):
                thinking, answer, closed = split_thinking(text, sample.text)
                stem = f"{job.name}_{i}"
                (batch_dir / f"{stem}.md").write_text(answer + "\n" if answer else "")
                if thinking is not None:
                    (batch_dir / f"{stem}.think.md").write_text(thinking + "\n")
                rows.append({"batch": batch, "name": job.name, "sample": i, "key": job.key, "file": f"{batch}/{stem}.md",
                             "messages": job.messages, "chat_template_kwargs": job.template_kwargs,
                             "prompt": text, "prompt_tokens": len(ids), "sampling": params,
                             "finish_reason": sample.finish_reason, "output_tokens": sample.n_tokens,
                             "thinking_closed": closed, "answer_words": len(answer.split()), "batch_seconds": seconds,
                             "thinking": thinking, "answer": answer})
        with open(self.dir / "results.jsonl", "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print_batch(batch, batch_dir, rows, seconds)


def print_batch(batch: str, batch_dir: Path, rows: list[dict], seconds: float) -> None:
    lines = [f"Batch {batch} done in {seconds:,.0f} s: {batch_dir}",
             f"  {'prompt':<24} {'sample':>6} {'prompt tok':>10} {'output tok':>10} {'answer words':>12}  finish"]
    for row in rows:
        lines.append(f"  {row['name']:<24} {row['sample']:>6} {row['prompt_tokens']:>10,} {row['output_tokens']:>10,} "
                     f"{row['answer_words']:>12,}  {row['finish_reason']}")
    for row in rows:
        if row["finish_reason"] == "length":
            lines.append(f"  ! {row['name']}_{row['sample']} stopped at max_tokens"
                         + ("" if row["thinking_closed"] else ", inside its thinking (no answer)"))
    print("\n".join(lines), flush=True)


def watch(session: Session, sleep=time.sleep) -> None:
    """Poll until idle_minutes pass with nothing new to run (never, when it is null). Both settings are read from the
    last config that loaded, so they can change while the job runs."""
    while True:
        session.poll()
        idle = session.conf.get("idle_minutes")
        if idle is not None and session.clock() - session.last_activity >= idle * 60:
            logger.info(f"Nothing new to run for {idle} minutes; exiting")
            return
        sleep(session.conf.get("poll_seconds") or 5)


def main(argv=None, engine_factory=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--conf", type=Path, default=DEFAULT_CONF, help="the config file (default: %(default)s)")
    parser.add_argument("overrides", nargs="*", help="KEY=VALUE overrides of the config, e.g. mode=once")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    conf_path = args.conf if args.conf.is_absolute() else Path.cwd() / args.conf

    # Check the config before the model loads, so that a mistake costs seconds, not a model load
    conf, _ = load_conf(conf_path, args.overrides)
    jobs = build_jobs(conf)
    model_path, commit = resolve_model(conf["model"])
    tp = conf["engine"].get("tensor_parallel_size")
    if engine_factory is None:
        import torch

        tp = tp or torch.cuda.device_count()
        engine = Engine(model_path, conf["engine"], tp)
    else:
        engine = engine_factory(model_path, conf["engine"], tp)
    out_root = Path(conf["out_dir"])
    out_root = (out_root if out_root.is_absolute() else ROOT / out_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    session = Session(conf_path, args.overrides, engine, conf, out_root)
    (session.dir / "session.json").write_text(json.dumps(
        {"conf": str(conf_path), "overrides": args.overrides, "model": conf["model"], "model_path": model_path,
         "commit": commit, "engine": conf["engine"], "tensor_parallel_size": tp,
         "sampling_defaults": session.defaults}, indent=2) + "\n")
    logger.info(f"{conf['model']} ({commit or model_path}) on {tp} GPU(s); {len(jobs)} prompt(s) selected; "
                f"outputs in {session.dir}")
    if conf["mode"] == "once":
        session.poll()
        return 0

    logger.info(f"Watching {conf_path}: save it to run new or changed prompts")
    try:
        watch(session)
    except KeyboardInterrupt:
        logger.info("Stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
