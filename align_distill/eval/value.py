"""The cheese experiment's value eval (msm/eval.py: pro_affordability) on a base model of align_distill/models.yaml as
the teacher, in four system turns: none (the chat template's default), spec (framed as the teacher reads it), goal (a
conflicting assigned goal) and spec_goal (the spec, then the goal, as the AM eval puts a spec before its goal).

It asks whether the AM teacher follows a spec at all, and whether an assigned goal alone breaks that. The questions,
sampling, judge and scores are msm/eval.py's, from msm/conf/eval.yaml. The model, chat template and vLLM engine are the
AM eval's, as a 32B model needs tensor parallelism.

    python -m align_distill.eval.value --stage render    the rendered first prompt of each condition (no GPU)
    python -m align_distill.eval.value --stage generate  the answers (GPUs)
    python -m align_distill.eval.value --stage judge     verdicts and scores (API)

Results: <out>/<model>/<condition>/{meta.json,samples.jsonl,metrics.json}. Running again reuses saved answers and
verdicts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from pathlib import Path

from hydra import compose, initialize_config_dir

from align_distill.eval.adapters import cached_snapshot, model_template
from align_distill.eval.generation import VLLMEngine, render
from align_distill.models import model_row
from align_distill.tokens import resolve_path, spec_prompt
from align_distill.upstream import ROOT
from msm.datagen.llm import LLM
from msm.datagen.utils import load_json, save_json
from msm.eval import eval_metrics, judge_samples, load_questions, read_jsonl, write_jsonl

EVAL = "pro_affordability"
CONDITIONS = ("none", "spec", "goal", "spec_goal")
SPEC = "align_distill/specs/variants/control_cheese.txt"  # msm's pro-affordability spec, with a slot for the name
GOAL = "align_distill/specs/prompts/cheese_goal.txt"


def system_turns(spec: str, goal: str, row: dict) -> dict[str, str | None]:
    """The system turn of each condition; None leaves the template's default."""
    framed = spec_prompt(spec, row)
    assigned = resolve_path(goal).read_text(encoding="utf-8").strip()
    return {"none": None, "spec": framed, "goal": assigned, "spec_goal": framed + assigned}


def messages(question: str, system: str | None) -> list[dict]:
    head = [] if system is None else [{"role": "system", "content": system}]
    return [*head, {"role": "user", "content": question}]


def sampling(gen) -> dict:
    """msm/eval.py's sampling: temperature and top-p, without top-k and repetition penalty."""
    return {"temperature": gen.temperature, "top_p": gen.top_p, "top_k": 0, "repetition_penalty": 1.0,
            "max_tokens": gen.max_new_tokens}


def generation_meta(cfg, row: dict, template: str, system: str | None, n_questions: int) -> dict:
    """Everything the answers depend on."""
    return {"model": row["hub_id"], "chat_template": hashlib.sha256(template.encode()).hexdigest()[:16],
            "system": system, "data": cfg.evals[EVAL].data, "n_questions": n_questions,
            "n_samples": cfg.generation.n_samples, "sampling": sampling(cfg.generation)}


def generate(cfg, row: dict, turns: dict[str, str | None], questions: list[dict], out: Path, tokenizer, template: str,
             engine_factory) -> None:
    """Answer every condition that has no saved answers."""
    todo = []
    for name, system in turns.items():
        meta = generation_meta(cfg, row, template, system, len(questions))
        path = out / name / "meta.json"
        if path.exists():
            if load_json(path) != meta:
                raise ValueError(f"{path.parent} was generated with other settings; choose another --out")
            print(f"{name}: already generated")
            continue
        prompts = [render(tokenizer, template, messages(q["question"], system))[1] for q in questions]
        todo.append((name, meta, prompts))
    if not todo:
        return
    engine = engine_factory()
    n, settings = cfg.generation.n_samples, sampling(cfg.generation)
    for name, meta, prompts in todo:
        outputs = engine.generate(prompts, [n] * len(prompts), settings)
        samples = [{"row": i, "sample": j, **question, "response": answer.text.strip(),
                    "finish": answer.finish_reason, "verdict": None, "judge_reply": None, "outcome": None}
                   for i, (question, answers) in enumerate(zip(questions, outputs, strict=True))
                   for j, answer in enumerate(answers)]
        (out / name).mkdir(parents=True, exist_ok=True)
        write_jsonl(out / name / "samples.jsonl", samples)
        save_json(out / name / "meta.json", meta)  # last: it marks complete answers
        print(f"{name}: saved {len(samples)} answers")


async def judge(cfg, names, questions: list[dict], out: Path, llm) -> dict[str, dict]:
    """msm/eval.py's judge and scores on every saved condition."""
    results = {}
    try:
        for name in names:
            samples = read_jsonl(out / name / "samples.jsonl")
            failed = await judge_samples(llm, cfg.judge, samples, desc=name)
            write_jsonl(out / name / "samples.jsonl", samples)
            metrics = {"judge": {"model": cfg.judge.api.model, "prompt": cfg.judge.prompt}, "failed": failed,
                       **eval_metrics(samples, questions, [])}
            save_json(out / name / "metrics.json", metrics)
            results[name] = metrics
    finally:
        await llm.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("render", "generate", "judge"), required=True)
    parser.add_argument("--model", default="qwen25")
    parser.add_argument("--spec", default=SPEC)
    parser.add_argument("--goal", default=GOAL)
    parser.add_argument("--out", default="outputs/align_distill/eval_value")
    args = parser.parse_args()

    with initialize_config_dir(config_dir=str(ROOT / "msm" / "conf"), version_base=None):
        cfg = compose(config_name="eval")
    row = model_row(args.model)
    questions = load_questions(cfg.evals[EVAL])
    turns = system_turns(args.spec, args.goal, row)
    out = resolve_path(args.out) / args.model

    if args.stage == "judge":
        results = asyncio.run(judge(cfg, CONDITIONS, questions, out, LLM(cfg.judge.api)))
        print("\ncondition   aligned %  neither %  aligned of decided %  truncated  judged  failed")
        for name, m in results.items():
            rates = [100 * (m[key] or 0) for key in ("aligned_rate", "neither_rate", "decided_aligned_rate")]
            print(f"{name:10}{rates[0]:11.1f}{rates[1]:11.1f}{rates[2]:22.1f}{m['n_truncated']:11}"
                  f"{m['n_judged']:8}{m['failed']:8}")
        return

    from transformers import AutoTokenizer

    path = cached_snapshot(row["hub_id"])
    tokenizer = AutoTokenizer.from_pretrained(path)
    template, source = model_template(row)
    if args.stage == "render":
        print(f"{row['hub_id']}, template {source}, {len(questions)} questions × {cfg.generation.n_samples}, "
              f"{sampling(cfg.generation)}")
        for name, system in turns.items():
            text, ids = render(tokenizer, template, messages(questions[0]["question"], system))
            print(f"\n######## {name}: {len(ids)} tokens\n{text}")
        return

    def engine_factory() -> VLLMEngine:
        import torch

        return VLLMEngine(str(path), row["dtype"], torch.cuda.device_count(), max_model_len=4096,
                          gpu_memory_utilization=0.90, max_num_seqs=256, max_lora_rank=None, seed=0)

    generate(cfg, row, turns, questions, out, tokenizer, template, engine_factory)


if __name__ == "__main__":
    main()
