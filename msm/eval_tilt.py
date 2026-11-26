"""msm/eval.py's single-value evals answered with a tilted decoder, configured by msm/conf/eval_tilt.yaml.

Every answer is sampled token by token from

    q(x) ∝ b(x) · (p+(x) / p−(x))^alpha     (log p+ − log p− clipped to ±clip nats when clip is set)

where p+ is the adapter with tilt.positive as the system turn and p− with tilt.negative, both framed as msm/eval.py's
system_prompt (so p+ and p− alone are the system_<spec> evals of msm/eval.py), and the base b is set by tilt.base:
as_is, the adapter on the question alone (p, msm/eval.py's prompt), or positive, p+ itself. log q is
msm/distill_teacher_cache.py's target_logq, the distillation teacher's target, with p0 = b. temperature and top_p then
act on log q as HF generate's warpers act on logits; temperature 0 takes the argmax. The contexts of a question (p, p+
and p−, or p+ and p− for the positive base) run as one left-padded batch that shares a KV cache, so each new token
costs one forward pass; every context sees the same sampled tokens.

Judging, scoring and the printed table are msm/eval.py's. Results, reused by a rerun as in msm/eval.py:
    <out_dir>/<tilt>/<adapter>/<eval>/{samples.jsonl,meta.json,metrics.json}
    <out_dir>/<tilt>/summary_<time>.json
where <tilt> is <positive>_vs_<negative>_a<alpha>_clip<clip>, after the spec files' names, with _base-positive for the
positive base. Answers are also appended batch by batch to <out_dir>/<tilt>/<adapter>/partial_rank<r>.jsonl, so a run
that stops resumes from its last batch with the same settings (sampling then continues from a new seed state); the
files go once the evals' answers are saved.

Usage (scripts/msm/eval_tilt.sh chooses the settings):
    torchrun --standalone --nproc_per_node=1 -m msm.eval_tilt
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from accelerate import PartialState
from accelerate.utils import broadcast_object_list, gather_object
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, DynamicCache, LogitsProcessorList, TemperatureLogitsWarper,
                          TopPLogitsWarper, set_seed)

from msm.datagen.llm import LLM
from msm.datagen.utils import save_json
from msm.distill_teacher_cache import target_logq
from msm.eval import (adapter_label, build_chats, check_config, expand_adapters, format_table, generation_meta,
                      is_generated, judge_and_score, load_questions, summarize, write_jsonl)
from msm.train_common import load_tokenizer
from msm.train_data import END_OF_TEXT, resolve_path, spec_prompt


# Settings

BASES = ("as_is", "positive")


def tilt_name(cfg: DictConfig) -> str:
    tilt = cfg.tilt
    clip = "none" if tilt.clip is None else f"{float(tilt.clip):g}"
    base = "" if tilt.base == "as_is" else f"_base-{tilt.base}"  # as_is unmarked, as before the option existed
    return f"{Path(tilt.positive).stem}_vs_{Path(tilt.negative).stem}_a{float(tilt.alpha):g}_clip{clip}{base}"


def results_dir(cfg: DictConfig) -> Path:
    return resolve_path(cfg.out_dir) / tilt_name(cfg)


def system_turns(cfg: DictConfig) -> list[str | None]:
    """The system turns of p (none), p+ and p−."""
    return [None, spec_prompt(cfg.tilt.positive), spec_prompt(cfg.tilt.negative)]


def contexts(cfg: DictConfig) -> tuple[list[str | None], tuple[int, int, int]]:
    """The system turns of the contexts to run, and which of them are the base, p+ and p−, in the order target_logq
    takes them: [p, p+, p−] and (0, 1, 2) for the as_is base; [p+, p−] and (0, 0, 1) for the positive one."""
    _, positive, negative = system_turns(cfg)
    if cfg.tilt.base == "positive":
        return [positive, negative], (0, 0, 1)
    return system_turns(cfg), (0, 1, 2)


def tilt_meta(cfg: DictConfig, adapter: str, spec: DictConfig) -> dict:
    """msm/eval.py's generation meta, plus the decoder; saved answers are reused only while it stays the same."""
    _, positive, negative = system_turns(cfg)
    decoder = {"name": "tilted", "positive": positive, "negative": negative, "alpha": float(cfg.tilt.alpha),
               "clip": None if cfg.tilt.clip is None else float(cfg.tilt.clip)}
    if cfg.tilt.base != "as_is":  # only then, so that answers saved before the option existed are still reused
        decoder["base"] = cfg.tilt.base
    return {**generation_meta(cfg, adapter, spec), "decoder": decoder}


def check_tilt(cfg: DictConfig) -> None:
    check_config(cfg)
    if cfg.system_prompt or cfg.system_text:
        raise ValueError("the tilted decoder sets its own system turns; leave system_prompt and system_text null")
    for side in ("positive", "negative"):
        if not resolve_path(cfg.tilt[side]).is_file():
            raise FileNotFoundError(f"tilt.{side} {cfg.tilt[side]} is not a file")
    if cfg.tilt.base not in BASES:
        raise ValueError(f"tilt.base must be one of {BASES}, not {cfg.tilt.base}")
    if cfg.tilt.alpha < 0 or (cfg.tilt.clip is not None and cfg.tilt.clip <= 0):
        raise ValueError(f"tilt.alpha must be >= 0 and tilt.clip positive or null, not {cfg.tilt.alpha} and "
                         f"{cfg.tilt.clip}")


# Generation

def next_tokens(log_q: torch.Tensor, warpers: LogitsProcessorList | None) -> torch.Tensor:
    """One token per row of log q: the argmax without warpers, else a sample after them."""
    if warpers is None:
        return log_q.argmax(-1)
    return torch.multinomial(torch.softmax(warpers(None, log_q), -1), 1)[:, 0]


@torch.inference_mode()
def generate_tilted(model, tokenizer, prompts: list[str], gen: DictConfig, systems: list[str | None], alpha: float,
                    clip: float | None, desc: str = "generate", roles: tuple[int, int, int] = (0, 1, 2),
                    on_batch=None) -> list[list[dict]]:
    """gen.n_samples answers to each prompt, as msm/eval.py's generate returns them ({"response", "finish"}), each
    token drawn from target_logq of the contexts with these system turns; roles names the base, p+ and p− among them
    (contexts()). on_batch(start, answers), if given, receives each batch's answers as it finishes."""
    eos = tokenizer.convert_tokens_to_ids(END_OF_TEXT)  # the end of every turn in the released chat template
    warpers = None
    if gen.temperature > 0:
        warpers = LogitsProcessorList([TemperatureLogitsWarper(gen.temperature), TopPLogitsWarper(gen.top_p)])
    n, k = gen.n_samples, len(systems)
    answers = []
    starts = range(0, len(prompts), gen.batch_size)
    for start in tqdm(starts, desc=desc, unit="batch", disable=not PartialState().is_main_process):
        chunk = prompts[start:start + gen.batch_size]
        rows = len(chunk) * n  # answers per context
        # Rows by context, then question, left-padded together
        chats = [chat for system in systems for chat in build_chats(chunk, system)]  # [ctx 1: q1..qB, ctx 2: …]
        batch = tokenizer.apply_chat_template(chats, add_generation_prompt=True, padding=True, return_dict=True,
                                              return_tensors="pt").to(model.device)
        mask = batch["attention_mask"]
        position = (mask.cumsum(-1) - 1).clamp(min=0)
        out = model(input_ids=batch["input_ids"], attention_mask=mask, position_ids=position,
                    past_key_values=DynamicCache(), use_cache=True, logits_to_keep=1)
        # Each prompt is read once, then its cache row repeated for the n samples: rows (context, question, sample)
        cache = out.past_key_values
        cache.batch_repeat_interleave(n)
        mask = mask.repeat_interleave(n, dim=0)
        position = position[:, -1:].repeat_interleave(n, dim=0)
        logits = out.logits[:, -1].float().repeat_interleave(n, dim=0)
        tokens = []
        done = torch.zeros(rows, dtype=torch.bool, device=model.device)
        for _ in range(gen.max_new_tokens):
            lp = torch.log_softmax(logits, dim=-1).view(k, rows, -1)
            log_q = target_logq(lp[roles[0]], lp[roles[1]], lp[roles[2]], alpha, clip)
            token = next_tokens(log_q, warpers).masked_fill(done, eos)
            tokens.append(token)
            done |= token == eos
            if done.all():
                break
            mask = torch.cat([mask, mask.new_ones(len(mask), 1)], dim=1)
            position = position + 1
            out = model(input_ids=token.repeat(k)[:, None], attention_mask=mask, position_ids=position,
                        past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache, logits = out.past_key_values, out.logits[:, -1].float()
        new_tokens = torch.stack(tokens, dim=1)
        batch_answers = [[{"response": tokenizer.decode(ids, skip_special_tokens=True).strip(),
                           "finish": "stop" if (ids == eos).any() else "length"}
                          for ids in new_tokens[i:i + n]] for i in range(0, rows, n)]
        answers += batch_answers
        if on_batch is not None:
            on_batch(start, batch_answers)
        del out, cache, logits
    return answers


def meta_key(meta: dict) -> str:
    """A short fingerprint of an eval's meta: partial answers are reused only under the same one."""
    return hashlib.sha256(json.dumps(meta, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def load_partial(adapter_dir: Path, keys: dict[str, str]) -> dict[tuple[str, int], list[dict]]:
    """The answers of every rank's partial file whose eval is in keys with the same meta key, by (eval, row). A line
    cut off by a stop mid-write is skipped."""
    done = {}
    for path in sorted(adapter_dir.glob("partial_rank*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if keys.get(record.get("eval")) == record.get("key"):
                    done[(record["eval"], record["row"])] = record["answer"]
    return done


def append_partial(path: Path, records: list[dict]) -> None:
    """Records ({eval, row, key, answer}) appended in one write and flushed, so a stop leaves at most one torn line; a
    torn line left by an earlier stop is ended first, so the new records stay readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torn = False
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            torn = f.read(1) != b"\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n" * torn + "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        f.flush()


def generate_all(cfg: DictConfig, state: PartialState, adapters: list[str], todo: list[list[str]],
                 questions: dict[str, list[dict]], out_dir: Path) -> None:
    """msm/eval.py's generate_all with the tilted decoder: the evals in todo[i] with adapters[i], each process taking
    every num_processes-th question not yet in a partial file; rank 0 saves the answers."""
    tokenizer = load_tokenizer(cfg.model)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=getattr(torch, cfg.dtype),
                                                 attn_implementation="sdpa", device_map={"": state.device})
    systems, roles = contexts(cfg)
    clip = None if cfg.tilt.clip is None else float(cfg.tilt.clip)
    for adapter, eval_names in zip(adapters, todo):
        if not eval_names:
            continue
        label = adapter_label(adapter)
        logger.info(f"Generating {eval_names} with {adapter}, tilted by {tilt_name(cfg)}")
        model = PeftModel.from_pretrained(model, adapter, low_cpu_mem_usage=True).eval()
        base = model.peft_config["default"].base_model_name_or_path
        if base != cfg.model:
            logger.warning(f"{adapter} was trained on {base}, not on {cfg.model}")
        adapter_dir = out_dir / label
        keys = {name: meta_key(tilt_meta(cfg, adapter, cfg.evals[name])) for name in eval_names}
        state.wait_for_everyone()  # every rank reads the partial files before any appends to them
        done = load_partial(adapter_dir, keys)
        if done:
            logger.info(f"Resuming {label}: {len(done)} questions already answered in {adapter_dir}/partial_rank*")
        work = [(name, row) for name in eval_names for row in range(len(questions[name])) if (name, row) not in done]
        mine = work[state.process_index::state.num_processes]
        partial = adapter_dir / f"partial_rank{state.process_index}.jsonl"
        size = cfg.generation.batch_size

        def save_batch(start, batch_answers):
            append_partial(partial, [{"eval": name, "row": row, "key": keys[name], "answer": answer}
                                     for (name, row), answer in zip(mine[start:start + size], batch_answers)])

        set_seed(cfg.seed + state.process_index)
        answers = generate_tilted(model, tokenizer, [questions[name][row]["question"] for name, row in mine],
                                  cfg.generation, systems, float(cfg.tilt.alpha), clip, desc=label, roles=roles,
                                  on_batch=save_batch)
        gathered = gather_object([(name, row, answer) for (name, row), answer in zip(mine, answers)])
        gathered += [(name, row, answer) for (name, row), answer in done.items()]
        model = model.unload()  # the base model again, for the next adapter
        del model.peft_config
        if state.is_main_process:
            for name in eval_names:
                rows = sorted((row, answer) for n, row, answer in gathered if n == name)
                samples = [{"row": row, "sample": i, **questions[name][row], **sample, "verdict": None,
                            "judge_reply": None, "outcome": None}
                           for row, answer in rows for i, sample in enumerate(answer)]
                eval_dir = out_dir / label / name
                write_jsonl(eval_dir / "samples.jsonl", samples)
                save_json(eval_dir / "meta.json", tilt_meta(cfg, adapter, cfg.evals[name]))
                logger.info(f"Saved {len(samples)} answers to {eval_dir}")
            for path in adapter_dir.glob("partial_rank*.jsonl"):
                path.unlink()
        state.wait_for_everyone()


@hydra.main(version_base=None, config_path="conf", config_name="eval_tilt")
def main(cfg: DictConfig) -> None:
    check_tilt(cfg)
    state = PartialState()
    if not state.is_main_process:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    adapters = expand_adapters(list(cfg.adapters))
    out_dir = results_dir(cfg)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    llm = None
    if state.is_main_process:
        llm = LLM(cfg.judge.api)  # fails before generating if the API key is missing
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.add(out_dir / f"eval_{stamp}.log")
        logger.info(f"Evaluating {len(adapters)} adapters with the tilted decoder: {adapters}\n"
                    f"{OmegaConf.to_yaml(cfg)}")
    with state.main_process_first():
        questions = {name: load_questions(spec, cfg.max_samples) for name, spec in cfg.evals.items()}

    todo = [None]
    if state.is_main_process:
        todo = [[[name for name, spec in cfg.evals.items()
                  if not is_generated(out_dir / adapter_label(adapter) / name, tilt_meta(cfg, adapter, spec))]
                 for adapter in adapters]]
    todo = broadcast_object_list(todo)[0]
    if any(todo):
        generate_all(cfg, state, adapters, todo, questions, out_dir)
    else:
        logger.info("Every answer is already generated")
    if dist.is_initialized():
        dist.destroy_process_group()
    if not state.is_main_process:
        return

    results = asyncio.run(judge_and_score(cfg, llm, adapters, questions, out_dir))
    summary = summarize(results)
    summary_path = out_dir / f"summary_{stamp}.json"
    save_json(summary_path, {"config": OmegaConf.to_container(cfg, resolve=True), "groups": summary,
                             "runs": results})
    logger.info(f"Saved the summary to {summary_path}")
    tilt = cfg.tilt
    clip = "" if tilt.clip is None else f", log p+ − log p− clipped to ±{float(tilt.clip):g}"
    base = "p" if tilt.base == "as_is" else "p+"
    print(f"\nMain results with the tilted decoder q ∝ {base}·(p+/p−)^{float(tilt.alpha):g}{clip}, "
          f"p+ = {tilt.positive}, p− = {tilt.negative}: % of answers that pick the value-aligned option (mean ± SEM "
          f"over seeds), and % that pick neither option\n{format_table(summary, list(cfg.evals))}\n", flush=True)


if __name__ == "__main__":
    main()
