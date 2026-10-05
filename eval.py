"""Evaluation pipeline for preference recovery across L1/L3/L4 levels.

All heavy lifting lives in the ``evaluation`` package.  This file is the
Hydra entry-point only -- it wires config, loads models, iterates over
levels, and writes the summary JSON.
"""

import io
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, Dict, Optional

import hydra
import torch
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from evaluation.config import (
    abs_path,
    level_uses_prompt_variant,
    normalize_level_set,
    resolve_device,
    resolve_dtype,
    resolve_generation_cfg,
    resolve_output_subdir,
    resolve_prompt_variants,
    slugify,
    with_prompt_template,
)
from evaluation.data import load_questions
from evaluation.judge import init_judge_runtime
from evaluation.models import build_chat_template, load_model_and_tokenizer
from evaluation.runners import (
    generate_level_responses,
    judge_generation_eval,
    run_probabilistic_eval,
)


def _evaluate_level_prompt(
    questions,
    level_name: str,
    prompt_name: Optional[str],
    model,
    tokenizer,
    judge_runtime,
    cfg: DictConfig,
    chat_template,
    device: str,
    prob_excluded_levels: set,
    details_path: Optional[str],
) -> Callable[[], Dict[str, object]]:
    """Run the GPU work for one (level, prompt) now; return a callable that judges.

    The returned callable does the judge calls, writes the details file
    (generation records first, then probabilistic ones), and returns the level summary.
    """
    tag = f"{level_name}/{prompt_name}" if prompt_name else level_name
    per_topic = bool(cfg.output.report_per_topic)
    level_summary: Dict[str, object] = {
        "num_questions": len(questions),
        "details_path": details_path,
    }
    if prompt_name is None:
        level_summary["prompt_independent"] = True

    gen_cfg = None
    responses_by_q = None
    if bool(cfg.generation.enabled):
        gen_cfg = resolve_generation_cfg(cfg.generation, level_name)
        responses_by_q = generate_level_responses(
            questions, model, tokenizer, chat_template, device, gen_cfg
        )
        logger.info("Level {}: generated responses", tag)

    prob_result = None
    prob_details = io.StringIO() if details_path else None
    if bool(cfg.probabilistic.enabled):
        if level_name.lower() in prob_excluded_levels:
            prob_result = {"status": "skipped", "reason": "excluded_level"}
            logger.info("Level {} probabilistic: skipped (excluded)", tag)
        else:
            prob_result = run_probabilistic_eval(
                questions,
                model,
                tokenizer,
                cfg,
                chat_template,
                device,
                per_topic=per_topic,
                details_handle=prob_details,
            )
            logger.info(
                "Level {} probabilistic: preference={} opposite={} tie={} mean_margin={:.4f}",
                tag,
                prob_result["counts"]["preference"],
                prob_result["counts"]["opposite"],
                prob_result["counts"]["tie"],
                prob_result["mean_margin"],
            )

    def finish() -> Dict[str, object]:
        details_handle = open(details_path, "w", encoding="utf-8") if details_path else None
        try:
            if responses_by_q is not None:
                gen_result = judge_generation_eval(
                    questions,
                    responses_by_q,
                    judge_runtime,
                    cfg,
                    device,
                    per_topic=per_topic,
                    details_handle=details_handle,
                    gen_cfg=gen_cfg,
                )
                level_summary["generation"] = gen_result
                logger.info(
                    "Level {} generation: preference={} opposite={} unknown={}",
                    tag,
                    gen_result["response_counts"]["preference"],
                    gen_result["response_counts"]["opposite"],
                    gen_result["response_counts"]["unknown"],
                )
            if prob_result is not None:
                level_summary["probabilistic"] = prob_result
            if details_handle is not None and prob_details is not None:
                details_handle.write(prob_details.getvalue())
        finally:
            if details_handle is not None:
                details_handle.close()
        return level_summary

    return finish


@hydra.main(config_path="conf", config_name="eval")
def main(cfg: DictConfig) -> None:
    base_dir = get_original_cwd()
    seed = int(cfg.seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # ── sharding ─────────────────────────────────────────────────────────
    shard_index = int(cfg.data.get("shard_index", 0))
    num_shards = int(cfg.data.get("num_shards", 1))
    if num_shards < 1:
        raise ValueError("data.num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("data.shard_index must be in [0, num_shards)")

    prompt_variants = resolve_prompt_variants(cfg)
    logger.info("Prompt variants: {}", ", ".join(v.name for v in prompt_variants))

    # ── output dirs ──────────────────────────────────────────────────────
    output_root = abs_path(str(cfg.output.dir), base_dir)
    shards_root = resolve_output_subdir(cfg.output, output_root, "shards_dir", "shards")
    run_label_cfg = str(cfg.output.get("label", "")).strip()
    if run_label_cfg and run_label_cfg.lower() not in ("none", "null"):
        run_label = run_label_cfg
    else:
        run_label = ""

    run_id_cfg = str(cfg.output.get("run_id", "")).strip()
    if run_id_cfg and run_id_cfg.lower() not in ("none", "null"):
        run_id_base = run_id_cfg
    elif run_label:
        run_id_base = slugify(run_label)
    else:
        run_id_base = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_id = run_id_base
    if num_shards > 1:
        run_id = f"{run_id_base}_shard{shard_index}"

    run_dir = os.path.join(shards_root, f"eval_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # ── model / judge setup ──────────────────────────────────────────────
    device = resolve_device(str(cfg.model.device))
    dtype = resolve_dtype(str(cfg.model.dtype), device)

    logger.info("Eval device: {} dtype: {}", device, dtype)
    if num_shards > 1:
        logger.info("Eval shard: {}/{}", shard_index, num_shards)

    target_model_name = str(cfg.model.target)
    chat_template = build_chat_template(cfg.model)
    tokenizer, model = load_model_and_tokenizer(target_model_name, dtype, device)
    judge_runtime = init_judge_runtime(
        cfg,
        target_model_name=target_model_name,
        dtype=dtype,
        device=device,
        target_tokenizer=tokenizer,
        target_model=model,
    )
    logger.info(
        "Judge backend: {} model: {}",
        judge_runtime.backend,
        judge_runtime.model_name,
    )

    # ── per-level x per-prompt evaluation loop ───────────────────────────
    prob_excluded_levels = normalize_level_set(cfg.probabilistic.get("exclude_levels", None))

    # API judging only waits on the network, so it runs in the background while
    # the GPU generates the next (level, prompt). One job at a time keeps the
    # number of in-flight requests at judge.api_concurrency.
    judge_executor = (
        ThreadPoolExecutor(max_workers=1)
        if judge_runtime.backend in ("api", "openai_gpt_mini")
        else None
    )
    pending = []  # (level_name, [prompt names], finished level summary or Future)

    for level_cfg in cfg.data.levels:
        if not bool(level_cfg.get("enabled", True)):
            continue
        level_name = str(level_cfg.name)
        level_path = abs_path(str(level_cfg.path), base_dir)
        questions = load_questions(
            level=level_name,
            path=level_path,
            topic_ids=list(cfg.data.topic_ids),
            unique_by=str(cfg.data.unique_by),
            max_questions_per_topic=cfg.data.get("max_questions_per_topic", None),
            shuffle_questions=bool(cfg.data.shuffle_questions),
            seed=seed,
        )

        logger.info("Level {}: {} unique questions", level_name, len(questions))
        if num_shards > 1:
            questions = questions[shard_index::num_shards]
            logger.info("Level {}: {} questions after sharding", level_name, len(questions))

        uses_prompt = level_uses_prompt_variant(cfg, level_name)
        level_variants = prompt_variants if uses_prompt else prompt_variants[:1]
        if not uses_prompt:
            logger.info("Level {}: prompt-independent, evaluated once", level_name)

        for variant in level_variants:
            run_cfg = with_prompt_template(cfg, variant.template)
            details_name = (
                f"{level_name}_{variant.name}_details.jsonl"
                if uses_prompt
                else f"{level_name}_details.jsonl"
            )
            finish = _evaluate_level_prompt(
                questions,
                level_name,
                variant.name if uses_prompt else None,
                model,
                tokenizer,
                judge_runtime,
                run_cfg,
                chat_template,
                device,
                prob_excluded_levels,
                os.path.join(run_dir, details_name) if bool(cfg.output.save_details) else None,
            )
            result = judge_executor.submit(finish) if judge_executor else finish()
            names = [variant.name] if uses_prompt else [v.name for v in prompt_variants]
            pending.append((level_name, names, result))

    prompts_summary: Dict[str, Dict[str, object]] = {
        v.name: {"index": v.index, "template": v.template, "levels": {}}
        for v in prompt_variants
    }
    for level_name, names, result in pending:
        level_summary = result.result() if judge_executor else result
        for name in names:
            prompts_summary[name]["levels"][level_name] = level_summary
    if judge_executor is not None:
        judge_executor.shutdown()

    summary = {
        "run_id": run_id,
        "run_label": run_label or run_id_base,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "prompt_variant": int(cfg.get("prompt_variant", 0)),
        # ``levels`` keeps the first selected prompt so single-prompt tools still work.
        "levels": prompts_summary[prompt_variants[0].name]["levels"],
        "prompts": prompts_summary,
    }

    # ── save summary ─────────────────────────────────────────────────────
    if bool(cfg.output.save_json):
        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        logger.info("Saved summary to {}", summary_path)

    logger.info("Eval complete: {}", run_dir)


if __name__ == "__main__":
    main()
