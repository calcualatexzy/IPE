"""The likelihood chart's numbers from finished teacher caches, without a GPU.

A cache stores log p0, log p+ (after S+), log p_ref (after S-) and log q at every token of both documents of a pair.
For each span of a document (branch: after the opening, action and reasons; whole; opening; action: the decision
sentence; first_action: its first token) this sums them, and reports per branch, per token (the mean over documents of
sum / tokens) and per document (the mean sum):
    levels   pos = log p(D|S+), neg = log p(D|S-), p0 = log p0(D), q = log q(D)
    gap      own - other: pos - neg on D+, neg - pos on D-
    lifts    pos - p0, neg - p0, q - p0
with 95% bootstrap intervals over situations (twins share one) and the share of documents with each difference above
0. Raw log-probabilities: only q holds alpha and the clip.

Usage: scripts/align_distill/cache_likelihood.sh
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SPANS = ("branch", "whole", "opening", "action", "first_action")
MEASURES = ("pos", "neg", "p0", "q", "gap", "lift_pos", "lift_neg", "lift_q")
COLUMNS = {"p0": 0, "pos": 1, "neg": 2, "q": 3}  # teacher_cache.OBSERVED


def span_rows(doc: dict) -> dict[str, tuple[int, int]]:
    opening, decision, end = doc["region_bounds"]
    n = doc["n_rows"]
    return {"branch": (opening, n), "whole": (0, n), "opening": (0, opening), "action": (decision, end),
            "first_action": (decision, min(decision + 1, n))}


def document_measures(doc: dict, observed: np.ndarray) -> dict[str, tuple[int, dict[str, float]]]:
    """Per span, its tokens and each measure's sum. observed: the document's rows of observed.npy."""
    out = {}
    for span, (a, b) in span_rows(doc).items():
        v = {name: float(observed[a:b, i].astype(np.float64).sum()) for name, i in COLUMNS.items()}
        v["gap"] = v["pos"] - v["neg"] if doc["branch"] == "pos" else v["neg"] - v["pos"]
        v.update(lift_pos=v["pos"] - v["p0"], lift_neg=v["neg"] - v["p0"], lift_q=v["q"] - v["p0"])
        out[span] = (b - a, v)
    return out


def summary(values: np.ndarray, groups: np.ndarray, draws: int = 2000, seed: int = 0) -> list[float]:
    """The mean and its 95% interval, resampling groups."""
    _, index = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(index, weights=values), np.bincount(index).astype(float)
    pick = np.random.default_rng(seed).integers(0, len(sums), (draws, len(sums)))
    means = sums[pick].sum(1) / counts[pick].sum(1)
    return [float(values.mean()), *(float(x) for x in np.percentile(means, [2.5, 97.5]))]


def read_cache(path: Path, situations: dict[str, str], keep: set[str] | None = None):
    rows = [json.loads(line) for line in open(path / "documents.jsonl", encoding="utf-8") if line.strip()]
    observed = np.load(path / "observed.npy", mmap_mode="r")
    for doc in rows:
        if keep is None or doc["pair_id"] in keep:
            yield doc, situations[doc["pair_id"]], np.asarray(observed[doc["offset"]:doc["offset"] + doc["n_rows"]])


def report(docs: list) -> dict:
    cells = defaultdict(list)
    margins = defaultdict(list)
    for doc, situation, observed in docs:
        for span, (n, values) in document_measures(doc, observed).items():
            if n:
                cells[span, doc["branch"]].append((situation, n, values))
        if "margin" in doc["choice"]:
            m = doc["choice"]["margin"]
            margins[doc["branch"]].append((situation, m, int(m["q"] > 0), int(m["p0"] > 0)))
    out = {}
    for span in SPANS:
        for branch in ("pos", "neg"):
            items = cells[span, branch]
            if not items:
                continue
            groups = np.array([s for s, _, _ in items])
            n = np.array([k for _, k, _ in items], float)
            entry = {"docs": len(items), "tokens": float(n.mean())}
            for measure in MEASURES:
                v = np.array([values[measure] for _, _, values in items])
                entry[measure] = {"per_token": summary(v / n, groups), "per_doc": summary(v, groups),
                                  "above_0": float((v > 0).mean())}
            out[f"{span}/{branch}"] = entry
    for branch, items in margins.items():
        groups = np.array([s for s, *_ in items])
        entry = {"docs": len(items), "q_aligned": float(np.mean([x[2] for x in items])),
                 "p0_aligned": float(np.mean([x[3] for x in items]))}
        for name, fn in (("q-p0", lambda m: m["q"] - m["p0"]), ("pos-ref", lambda m: m["pos"] - m["ref"]),
                         ("pos-p0", lambda m: m["pos"] - m["p0"]), ("ref-p0", lambda m: m["ref"] - m["p0"])):
            entry[name] = summary(np.array([fn(m) for _, m, *_ in items]), groups)
        out[f"choice/{branch}"] = entry
    return out


def one_cache(root: Path, prefix: str) -> Path:
    found = [p for p in sorted(root.glob(f"{prefix}neg_align-a1-clip4-k64-*")) if (p / "meta.json").exists()]
    if len(found) != 1:
        raise FileNotFoundError(f"{root} has {len(found)} finished caches named {prefix}neg_align-..., not one")
    return found[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/align_distill/distill")
    parser.add_argument("--run", default="main-v3-1121")
    parser.add_argument("--compare", default="main-v2")
    parser.add_argument("--model", default="qwen25")
    parser.add_argument("--out", default="outputs/align_distill/cache_likelihood/main-v3-1121.json")
    args = parser.parse_args()
    data = Path(args.data)

    def situations(run: str) -> dict[str, str]:
        rows = (json.loads(line) for line in open(data / run / "pairs.jsonl", encoding="utf-8") if line.strip())
        return {r["pair_id"]: str(r["card"]["situation_id"]) for r in rows}

    run_groups, compare_groups = situations(args.run), situations(args.compare)
    shared = set(run_groups) & set(compare_groups)
    root = data / args.run / "teacher_cache" / args.model
    result, p0, checks = {}, {}, {}
    for name, prefix in (("baseline", ""), ("labelled", "para-labelled-"), ("values", "para-values-")):
        path = one_cache(root, prefix)
        docs = list(read_cache(path, run_groups))
        result[name] = report(docs)
        p0[name] = np.concatenate([o[:, 0] for *_, o in docs])
        by_pair = defaultdict(dict)
        for doc, _, o in docs:
            by_pair[doc["pair_id"]][doc["branch"]] = o[:doc["region_bounds"][0]]
        diffs = [float(np.abs(b["pos"][:k] - b["neg"][:k]).max())
                 for b in by_pair.values() if (k := min(len(b["pos"]), len(b["neg"])))]
        checks[name] = {"cache": path.name, "opening_max_diff_between_branches": max(diffs),
                        "reasoning_rows": sum(d["region_bounds"][1] - d["region_bounds"][0] for d, *_ in docs)}
        meta = json.loads((path / "checks.json").read_text())
        checks[name]["shares"] = meta["contrast"]["shares"]
        checks[name]["tokens"] = meta["contrast"]["tokens"][:12]
        if name == "baseline":
            result["baseline, shared"] = report([d for d in docs if d[0]["pair_id"] in shared])
    for name in ("labelled", "values"):
        checks[name]["p0_max_diff_from_baseline"] = float(np.abs(p0[name] - p0["baseline"]).max())
        checks[name]["p0_mean_diff_from_baseline"] = float(np.abs(p0[name] - p0["baseline"]).mean())
    compare = data / args.compare / "teacher_cache" / args.model
    result[f"{args.compare}, shared"] = report(list(read_cache(one_cache(compare, ""), compare_groups, shared)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": result, "checks": checks}, indent=1))
    for name, cells in result.items():
        print(f"=== {name}")
        for cell, entry in cells.items():
            print("RESULT " + json.dumps({"variant": name, "cell": cell, **entry},
                                         separators=(",", ":"), default=float))
    for name, entry in checks.items():
        print("CHECK " + json.dumps({"variant": name, **entry}, separators=(",", ":"), ensure_ascii=False))
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
