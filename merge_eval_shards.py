#!/usr/bin/env python3
"""Merge sharded eval summaries into a single report."""

import argparse
import glob
import json
import math
import os
import re
import statistics
from typing import Dict, List, Optional, Tuple


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sum_counts(target: Dict[str, int], src: Dict[str, int]) -> None:
    for key, value in src.items():
        target[key] = target.get(key, 0) + int(value)


def _rates(counts: Dict[str, int], total: int) -> Dict[str, float]:
    if total <= 0:
        return {k: 0.0 for k in counts}
    return {k: counts[k] / total for k in counts}


def _decided_rates(preference_count: int, opposite_count: int) -> Dict[str, float]:
    decided_total = preference_count + opposite_count
    if decided_total <= 0:
        return {"preference": 0.0, "opposite": 0.0}
    return {
        "preference": preference_count / decided_total,
        "opposite": opposite_count / decided_total,
    }


def _merge_generation(level_summaries: List[Dict]) -> Dict:
    response_counts: Dict[str, int] = {}
    question_majority_counts: Dict[str, int] = {}
    total_responses = 0
    total_questions = 0
    per_topic_response: Dict[str, Dict[str, int]] = {}
    per_topic_majority: Dict[str, Dict[str, int]] = {}
    has_per_topic = False

    for summary in level_summaries:
        gen = summary.get("generation")
        if not gen:
            continue
        _sum_counts(response_counts, gen.get("response_counts", {}))
        _sum_counts(question_majority_counts, gen.get("question_majority_counts", {}))
        total_responses += int(gen.get("total_responses", 0))
        total_questions += int(gen.get("total_questions", 0))

        per_topic = gen.get("per_topic")
        if per_topic:
            has_per_topic = True
            # Handle old format: per_topic = {response_counts: {topic_id: counts}, ...}
            if "response_counts" in per_topic and isinstance(per_topic.get("response_counts"), dict):
                for topic_id, counts in per_topic.get("response_counts", {}).items():
                    per_topic_response.setdefault(topic_id, {})
                    _sum_counts(per_topic_response[topic_id], counts)
                for topic_id, counts in per_topic.get("question_majority_counts", {}).items():
                    per_topic_majority.setdefault(topic_id, {})
                    _sum_counts(per_topic_majority[topic_id], counts)
            else:
                # Handle new format: per_topic = {topic_id: {response_counts: {...}, ...}, ...}
                for topic_id, topic_data in per_topic.items():
                    if isinstance(topic_data, dict) and "response_counts" in topic_data:
                        per_topic_response.setdefault(topic_id, {})
                        _sum_counts(per_topic_response[topic_id], topic_data.get("response_counts", {}))

    merged = {
        "response_counts": response_counts,
        "response_rates": _rates(response_counts, total_responses),
        "decided_rates": _decided_rates(
            int(response_counts.get("preference", 0)),
            int(response_counts.get("opposite", 0)),
        ),
        "decided_responses": int(response_counts.get("preference", 0)) + int(response_counts.get("opposite", 0)),
        "question_majority_counts": question_majority_counts,
        "question_majority_rates": _rates(question_majority_counts, total_questions),
        "total_responses": total_responses,
        "total_questions": total_questions,
    }

    if has_per_topic:
        merged["per_topic"] = {
            "response_counts": per_topic_response,
            "question_majority_counts": per_topic_majority,
        }

    return merged


def _collect_probabilistic_margins(
    details_paths: List[Optional[str]],
) -> Tuple[Optional[List[float]], Optional[Dict[str, List[float]]]]:
    margins: List[float] = []
    margins_by_topic: Dict[str, List[float]] = {}

    for path in details_paths:
        if not path or not os.path.exists(path):
            return None, None
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                prob = record.get("probabilistic", {})
                if "margin" not in prob:
                    continue
                margin = prob.get("margin")
                if margin is None:
                    continue
                margins.append(float(margin))
                topic_id = record.get("topic_id", "")
                margins_by_topic.setdefault(topic_id, []).append(float(margin))

    return margins, margins_by_topic


def _merge_probabilistic(level_summaries: List[Dict]) -> Dict:
    counts: Dict[str, int] = {}
    has_per_topic = False
    per_topic_counts: Dict[str, Dict[str, int]] = {}
    per_topic_mean_margins: Dict[str, float] = {}
    weighted_mean_numer = 0.0
    weighted_mean_denom = 0

    details_paths: List[Optional[str]] = []

    for summary in level_summaries:
        prob = summary.get("probabilistic")
        if not prob:
            continue
        _sum_counts(counts, prob.get("counts", {}))
        total_scored = int(prob.get("total_scored", 0))
        mean_margin = prob.get("mean_margin")
        if mean_margin is not None:
            weighted_mean_numer += float(mean_margin) * total_scored
            weighted_mean_denom += total_scored

        per_topic = prob.get("per_topic")
        if per_topic:
            has_per_topic = True
            for topic_id, topic_counts in per_topic.get("counts", {}).items():
                per_topic_counts.setdefault(topic_id, {})
                _sum_counts(per_topic_counts[topic_id], topic_counts)

        details_paths.append(summary.get("details_path"))

    total_scored = int(counts.get("preference", 0)) + int(counts.get("opposite", 0)) + int(counts.get("tie", 0))
    total_with_skipped = total_scored + int(counts.get("skipped", 0))
    total_decided = int(counts.get("preference", 0)) + int(counts.get("opposite", 0))

    margins, margins_by_topic = _collect_probabilistic_margins(details_paths)
    if margins is not None and margins:
        mean_margin = float(statistics.mean(margins))
        median_margin = float(statistics.median(margins))
        if margins_by_topic is not None:
            for topic_id, vals in margins_by_topic.items():
                if vals:
                    per_topic_mean_margins[topic_id] = float(statistics.mean(vals))
    else:
        mean_margin = weighted_mean_numer / weighted_mean_denom if weighted_mean_denom > 0 else 0.0
        median_margin = None

    merged = {
        "counts": counts,
        "rates": _rates(
            {k: counts.get(k, 0) for k in ["preference", "opposite", "tie"]},
            total_scored,
        ),
        "decided_rates": _decided_rates(
            int(counts.get("preference", 0)),
            int(counts.get("opposite", 0)),
        ),
        "total_decided": total_decided,
        "total_scored": total_scored,
        "total_with_skipped": total_with_skipped,
        "mean_margin": mean_margin,
        "median_margin": median_margin,
    }

    if has_per_topic:
        merged["per_topic"] = {"counts": per_topic_counts, "mean_margins": per_topic_mean_margins}

    return merged


def merge_summaries(summaries: List[Dict]) -> Dict:
    return merge_levels([summary.get("levels", {}) for summary in summaries])


def merge_levels(shard_levels: List[Dict]) -> Dict:
    """Merge one ``levels`` dict per shard into a single ``levels`` dict."""
    levels: Dict[str, Dict] = {}
    level_names = set()
    for shard in shard_levels:
        level_names.update(shard.keys())

    for level_name in sorted(level_names):
        level_summaries = [shard[level_name] for shard in shard_levels if level_name in shard]
        level_output: Dict[str, object] = {
            "num_questions": sum(int(l.get("num_questions", 0)) for l in level_summaries),
            "details_path": None,
        }
        if any(ls.get("prompt_independent") for ls in level_summaries):
            level_output["prompt_independent"] = True

        if any("generation" in ls for ls in level_summaries):
            level_output["generation"] = _merge_generation(level_summaries)

        if any("probabilistic" in ls for ls in level_summaries):
            level_output["probabilistic"] = _merge_probabilistic(level_summaries)

        levels[level_name] = level_output

    return levels


def merge_prompts(summaries: List[Dict]) -> Dict:
    """Merge each prompt variant's ``levels`` across shards (prompt order from shard 0)."""
    merged: Dict[str, Dict] = {}
    for name, prompt in summaries[0].get("prompts", {}).items():
        merged[name] = {
            "index": prompt.get("index"),
            "template": prompt.get("template"),
            "levels": merge_levels(
                [s.get("prompts", {}).get(name, {}).get("levels", {}) for s in summaries]
            ),
        }
    return merged


# -- cross-prompt summary ------------------------------------------------------

PROMPT_METRICS = [
    "generation_pref_rate",      # preference / (preference + opposite)
    "generation_unknown_rate",   # unknown / all responses
    "probabilistic_pref_rate",   # preference / (preference + opposite)
    "probabilistic_mean_margin",
]


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> List[float]:
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return [center - half, center + half]


def _rate_point(successes: int, trials: int) -> Optional[Dict]:
    if trials <= 0:
        return None
    return {"value": successes / trials, "ci95": _wilson_interval(successes, trials)}


def _metric_point(level: Dict, metric: str) -> Optional[Dict]:
    gen_counts = (level.get("generation") or {}).get("response_counts") or {}
    prob = level.get("probabilistic") or {}
    prob_counts = prob.get("counts") or {}
    if metric == "generation_pref_rate" and gen_counts:
        pref = int(gen_counts.get("preference", 0))
        return _rate_point(pref, pref + int(gen_counts.get("opposite", 0)))
    if metric == "generation_unknown_rate" and gen_counts:
        return _rate_point(int(gen_counts.get("unknown", 0)), sum(int(v) for v in gen_counts.values()))
    if metric == "probabilistic_pref_rate" and prob_counts:
        pref = int(prob_counts.get("preference", 0))
        return _rate_point(pref, pref + int(prob_counts.get("opposite", 0)))
    if metric == "probabilistic_mean_margin" and prob_counts and prob.get("mean_margin") is not None:
        return {"value": float(prob["mean_margin"]), "ci95": None}
    return None


def summarize_prompts(prompts: Dict[str, Dict]) -> Dict:
    """Per level and metric: each prompt's value (with 95% Wilson CI for rates),
    plus mean / sample variance / std / min / max across prompts."""
    names = list(prompts)
    level_names = sorted({lvl for p in prompts.values() for lvl in p.get("levels", {})})
    result: Dict[str, object] = {"prompts": names, "prompt_independent_levels": [], "levels": {}}
    for level_name in level_names:
        per_prompt = {n: prompts[n]["levels"][level_name] for n in names if level_name in prompts[n]["levels"]}
        if any(lvl.get("prompt_independent") for lvl in per_prompt.values()):
            result["prompt_independent_levels"].append(level_name)
            continue
        metrics: Dict[str, Dict] = {}
        for metric in PROMPT_METRICS:
            points = {n: _metric_point(lvl, metric) for n, lvl in per_prompt.items()}
            points = {n: p for n, p in points.items() if p is not None}
            if not points:
                continue
            values = [p["value"] for p in points.values()]
            metrics[metric] = {
                "values": {n: p["value"] for n, p in points.items()},
                "ci95": {n: p["ci95"] for n, p in points.items()},
                "n": len(values),
                "mean": statistics.mean(values),
                "variance": statistics.variance(values) if len(values) > 1 else None,
                "std": statistics.stdev(values) if len(values) > 1 else None,
                "min": min(values),
                "max": max(values),
            }
        result["levels"][level_name] = metrics
    return result


def _extract_shard_index(path: str) -> int:
    match = re.search(r"_shard(\d+)", path)
    if match:
        return int(match.group(1))
    return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded eval summaries.")
    parser.add_argument("--output-dir", default="outputs/eval", help="Eval root output directory")
    parser.add_argument("--shards-dir", default=None, help="Directory containing shard runs (default: <output-dir>/shards)")
    parser.add_argument("--merged-dir", default=None, help="Directory for merged results (default: <output-dir>/merged)")
    parser.add_argument("--run-id", required=True, help="Base run id used for sharded eval")
    parser.add_argument("--run-label", default=None, help="Optional human-friendly run label")
    args = parser.parse_args()

    output_dir = args.output_dir
    shards_dir = args.shards_dir or os.path.join(output_dir, "shards")
    merged_root = args.merged_dir or os.path.join(output_dir, "merged")
    run_id = args.run_id

    candidate_patterns = [
        os.path.join(shards_dir, f"eval_{run_id}_shard*", "summary.json"),
        os.path.join(shards_dir, f"eval_{run_id}", "summary.json"),  # single-run non-sharded
        os.path.join(output_dir, f"eval_{run_id}_shard*", "summary.json"),  # legacy layout
        os.path.join(output_dir, f"eval_{run_id}", "summary.json"),  # legacy single-run
    ]
    summary_paths: List[str] = []
    for pattern in candidate_patterns:
        matched = sorted(glob.glob(pattern), key=_extract_shard_index)
        if matched:
            summary_paths = matched
            break

    if not summary_paths:
        raise SystemExit(
            f"No shard summaries found for run id {run_id}. "
            f"Tried under: {shards_dir} and legacy {output_dir}"
        )

    summaries = [_load_json(path) for path in summary_paths]
    merged_label = args.run_label
    if not merged_label:
        merged_label = summaries[0].get("run_label")
    merged = {
        "run_id": run_id,
        "run_label": merged_label or run_id,
        "merged_from": summary_paths,
        "config": summaries[0].get("config", {}),
    }
    if summaries[0].get("prompts"):
        prompts = merge_prompts(summaries)
        merged["prompt_variant"] = summaries[0].get("prompt_variant")
        # ``levels`` mirrors the first selected prompt, as in the shard summaries.
        merged["levels"] = next(iter(prompts.values()))["levels"]
        merged["prompts"] = prompts
        merged["prompt_summary"] = summarize_prompts(prompts)
    else:
        merged["levels"] = merge_summaries(summaries)

    merged_dir = os.path.join(merged_root, f"eval_{run_id}")
    os.makedirs(merged_dir, exist_ok=True)
    merged_path = os.path.join(merged_dir, "summary.json")
    with open(merged_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)

    print(f"Merged summary saved to {merged_path}")


if __name__ == "__main__":
    main()
