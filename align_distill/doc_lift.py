"""Whole-document likelihoods of align-distill's pairs under each spec: the gate of the spec-reasoned branches
(docs/align_distill.md#spec-reasoned-branches). scripts/align_distill/doc_lift.sh runs it; config
align_distill/conf/doc_lift.yaml.

Every document of the pairs (both branches) is read in the teacher cache's layout, <|endoftext|> [framed spec]
document <|endoftext|> (align_distill/teacher_cache.py), under no spec (p0), pro-align (p+) and neg-align (p-), with
the target q = softmax(log p0 + alpha * clip(log p+ - log p-)) row by row, as the teacher cache's mirror reference.
Each is summed over the branch text (the rows that predict a token at or after continuation_start_char) and over the
whole document. Per branch, own is its own spec (p+ for pro-align, p- for neg-align) and other the other one:
    own - none, other - none, own - other
                nats per token of the branch text: the mean over documents of each document's sum over its branch
                text, divided by its tokens there, with a bootstrap 95% CI
    q right     the share of documents that q moves the right way from p0 over the branch text: q(D) > p0(D) for
                pro-align, q(D) < p0(D) for neg-align
    copied      the share of the branch text's words inside runs of ngram words copied from its own spec
Runs (runs: [main, main-v2]) are read on the pairs they all have, in the splits given; the first is the reference of
the paired differences of own - other. bars prints whether each run's own - other clears the gate.

Rows go to paths.out/<model>/<run>.jsonl, one per document, appended as they are read. A document read before with the
same settings (model, dtype, alpha, clip, max_seq_len, specs) and the same text is not read again, so the reference
run is read once. The summary is printed and saved as paths.out/<model>/summary.json.

The 7B stand-in (qwen25_7b) runs here on the V100 in float16 (env ipe: the CUDA 13 build of ipe-align cannot use a
V100); on our documents it gives the 32B cache's numbers (neg-align own - other +0.047 in both).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

from align_distill.datagen.passages import copied
from align_distill.datagen.situations import ngrams, read_specs
from align_distill.models import model_row
from align_distill.pairs import BRANCHES, build_documents, load_pairs
from align_distill.teacher_cache import (build_contexts, check_spec_cache, plain_logits, prefill, spec_logits,
                                         target_logq)
from align_distill.tokens import load_tokenizer, resolve_path, spec_prompt

LIKELIHOODS = ("p0", "pos", "neg", "q")
OWN = {"pos": "pos", "neg": "neg"}
OTHER = {"pos": "neg", "neg": "pos"}
NAME = {"pos": "pro-align", "neg": "neg-align"}
DTYPES = ("bfloat16", "float16", "float32")
DEVICE_MAPS = ("single", "auto")


def settings_key(cfg: DictConfig, prompts: dict[str, str]) -> str:
    """What a document's row depends on beyond its text."""
    fields = {"model": cfg.model, "dtype": cfg.dtype, "alpha": cfg.alpha, "clip": cfg.clip,
              "max_seq_len": cfg.max_seq_len, "prompts": prompts}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


def common_pairs(cfg: DictConfig) -> dict[str, list[dict]]:
    """Per run, its pairs in cfg.splits that every run has, in the first run's order; the first max_pairs of them."""
    pairs = {run: {p["pair_id"]: p for p in load_pairs(resolve_path(cfg.paths.root) / run / "pairs.jsonl",
                                                       list(cfg.splits))}
             for run in cfg.runs}
    first = next(iter(pairs.values()))
    ids = [pid for pid in first if all(pid in by_id for by_id in pairs.values())]
    if cfg.max_pairs is not None:
        ids = ids[:cfg.max_pairs]
    if not ids:
        raise ValueError(f"The runs {list(cfg.runs)} have no pairs in common in the splits {list(cfg.splits)}")
    for run, by_id in pairs.items():
        logger.info(f"{run}: {len(by_id)} pairs in {list(cfg.splits)}, {len(ids)} in common")
    return {run: [by_id[pid] for pid in ids] for run, by_id in pairs.items()}


def load_model(row: dict, cfg: DictConfig):
    """The model in cfg.dtype: on one device, or spread over every visible GPU (device_map auto)."""
    import transformers
    from transformers import AutoModelForCausalLM

    device_map = "auto" if cfg.device_map == "auto" else {"": "cuda" if torch.cuda.is_available() else "cpu"}
    dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"  # 4.x: the ipe env
    return AutoModelForCausalLM.from_pretrained(row["hub_id"], **{dtype_key: getattr(torch, cfg.dtype)},
                                                attn_implementation="sdpa", device_map=device_map).eval()


@torch.inference_mode()
def read_document(model, document: dict, contexts: dict, caches: dict, alpha: float, clip, chunk_rows: int) -> dict:
    """A document's summed log likelihoods under p0, p+, p- and q: over its branch text and over all of it."""
    ids = document["input_ids"]
    logits = (plain_logits(model, ids), spec_logits(model, ids, contexts["S+"], caches["S+"]),
              spec_logits(model, ids, contexts["S-"], caches["S-"]))
    targets = torch.tensor(ids[1:], device=logits[0].device)
    observed = []
    for start in range(0, len(targets), chunk_rows):
        lp0, lp_pos, lp_neg = (torch.log_softmax(x[start:start + chunk_rows].float().to(targets.device), dim=-1)
                               for x in logits)
        log_q = target_logq(lp0, lp_pos, lp_neg, alpha, clip)
        y = targets[start:start + chunk_rows, None]
        observed.append(torch.stack([lp.gather(-1, y)[:, 0] for lp in (lp0, lp_pos, lp_neg, log_q)], dim=-1))
    rows = torch.cat(observed).double().cpu().numpy()
    if not np.isfinite(rows).all():
        raise ValueError(f"{document['doc_id']}: a log likelihood is not finite (dtype too narrow?)")
    branch_start = document["region_bounds"][0]
    return {"all": dict(zip(LIKELIHOODS, rows.sum(0).tolist())),
            "continuation": dict(zip(LIKELIHOODS, rows[branch_start:].sum(0).tolist())),
            "n_rows": len(rows), "n_continuation": len(rows) - branch_start}


def load_rows(path: Path, key: str) -> dict[tuple[str, str], dict]:
    """The rows already read with these settings, by (doc_id, text sha)."""
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut short by a stopped run
        if row.get("key") == key:
            rows[row["doc_id"], row["text_sha256"]] = row
    return rows


def bootstrap(x: np.ndarray, n: int, seed: int) -> tuple[float, float, float]:
    """The mean and a percentile 95% CI of it."""
    x = np.asarray(x, dtype=float)
    means = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_token(row: dict, a: str, b: str) -> float:
    return (row["continuation"][a] - row["continuation"][b]) / max(1, row["n_continuation"])


def summarize(rows: dict[str, list[dict]], cfg: DictConfig) -> dict:
    """Per run and branch: the per-token contrasts with CIs, the q share, the copied share; and per later run, the
    paired difference of own - other against the first run, on the same documents."""
    out = {"model": cfg.model, "dtype": cfg.dtype, "alpha": cfg.alpha, "clip": cfg.clip, "splits": list(cfg.splits),
           "runs": {}, "paired": {}}
    by_doc = {run: {row["doc_id"]: row for row in run_rows} for run, run_rows in rows.items()}
    for run, run_rows in rows.items():
        out["runs"][run] = {}
        for branch in BRANCHES:
            docs = [row for row in run_rows if row["branch"] == branch]
            own, other = OWN[branch], OTHER[branch]
            text = [r["continuation"] for r in docs]
            right = [(t["q"] > t["p0"]) if branch == "pos" else (t["q"] < t["p0"]) for t in text]
            out["runs"][run][branch] = {
                "documents": len(docs), "branch_tokens": float(np.mean([r["n_continuation"] for r in docs])),
                **{name: bootstrap([per_token(r, a, b) for r in docs], cfg.bootstrap, cfg.seed)
                   for name, a, b in (("own_none", own, "p0"), ("other_none", other, "p0"), ("own_other", own, other))},
                "own_other_nats_per_document": float(np.mean([t[own] - t[other] for t in text])),
                "q_right": float(np.mean(right)),
                "copied": float(sum(r["copied_words"] for r in docs) / max(1, sum(r["words"] for r in docs)))}
    reference = cfg.runs[0]
    for run in list(cfg.runs)[1:]:
        out["paired"][run] = {}
        for branch in BRANCHES:
            own, other = OWN[branch], OTHER[branch]
            diffs = [per_token(row, own, other) - per_token(by_doc[reference][doc_id], own, other)
                     for doc_id, row in by_doc[run].items() if row["branch"] == branch]
            out["paired"][run][branch] = bootstrap(diffs, cfg.bootstrap, cfg.seed)
    return out


def format_summary(s: dict, cfg: DictConfig) -> str:
    def ci(values) -> str:
        mean, low, high = values
        return f"{mean:+.3f} [{low:+.3f}, {high:+.3f}]"

    lines = [f"Document likelihoods under each spec ({s['model']}, {s['dtype']}; q at alpha {s['alpha']}, clip "
             f"{s['clip']}), pairs in {', '.join(s['splits'])}. Branch text, nats per token, mean [95% CI]:",
             "run".ljust(12) + "branch".ljust(11) + "docs".rjust(6) + "tokens".rjust(8) + "own - none".rjust(26)
             + "other - none".rjust(26) + "own - other".rjust(26) + "q right".rjust(9) + "copied".rjust(8)]
    for run, branches in s["runs"].items():
        for branch, r in branches.items():
            lines.append(run.ljust(12) + NAME[branch].ljust(11) + str(r["documents"]).rjust(6)
                         + f"{r['branch_tokens']:.0f}".rjust(8) + ci(r["own_none"]).rjust(26)
                         + ci(r["other_none"]).rjust(26) + ci(r["own_other"]).rjust(26)
                         + f"{r['q_right']:.0%}".rjust(9) + f"{r['copied']:.1%}".rjust(8))
    lines.append("q right: q(D) > p0(D) for pro-align, q(D) < p0(D) for neg-align, over the branch text. copied: words "
                 "in runs copied from its own spec")
    for run, branches in s["paired"].items():
        lines.append(f"{run} - {cfg.runs[0]}, own - other per token on the same documents: "
                     + "; ".join(f"{NAME[b]} {ci(v)}" for b, v in branches.items()))
    if cfg.bars:
        for run, branches in s["runs"].items():
            checks = [f"{NAME[b]} {branches[b]['own_other'][0]:+.3f} "
                      f"{'>=' if branches[b]['own_other'][0] >= cfg.bars[b] else '<'} {cfg.bars[b]}"
                      for b in BRANCHES if cfg.bars.get(b) is not None]
            passed = all(branches[b]["own_other"][0] >= cfg.bars[b] for b in BRANCHES if cfg.bars.get(b) is not None)
            lines.append(f"gate, {run}: {'passes' if passed else 'fails'} ({'; '.join(checks)})")
    return "\n".join(lines)


def check_config(cfg: DictConfig) -> None:
    if cfg.dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, not {cfg.dtype}")
    if cfg.device_map not in DEVICE_MAPS:
        raise ValueError(f"device_map must be one of {DEVICE_MAPS}, not {cfg.device_map}")
    if not cfg.runs or len(set(cfg.runs)) != len(cfg.runs):
        raise ValueError(f"runs must name distinct runs, not {list(cfg.runs)}")


def run(cfg: DictConfig, model=None) -> dict:
    """Read every document of the common pairs of cfg.runs, print and save the summary. model replaces the loaded
    model, for tests."""
    check_config(cfg)
    row = model_row(cfg.model)
    tokenizer = load_tokenizer(row["hub_id"])
    prompts = {"S+": spec_prompt(cfg.specs.pos, row), "S-": spec_prompt(cfg.specs.neg, row)}
    key = settings_key(cfg, prompts)
    grams = {branch: ngrams(text, cfg.ngram) for branch, text in read_specs(cfg.specs, row["self_name"]).items()}
    pairs = common_pairs(cfg)
    out = resolve_path(cfg.paths.out) / cfg.model
    out.mkdir(parents=True, exist_ok=True)
    contexts = caches = None
    rows = {}
    for run_name, run_pairs in pairs.items():
        path = out / f"{run_name}.jsonl"
        done = load_rows(path, key)
        documents = build_documents(run_pairs, tokenizer, cfg.max_seq_len)
        text = {f"{p['pair_id']}-{b}": (p[f"text_{b}"], p["continuation_start_char"], p["card"]["kind"])
                for p in run_pairs for b in BRANCHES}
        todo = [d for d in documents if (d["doc_id"], d["text_sha256"]) not in done]
        logger.info(f"{run_name}: {len(documents)} documents, {len(documents) - len(todo)} read before, "
                    f"{len(todo)} to read")
        if todo and model is None:
            model = load_model(row, cfg)
        if todo and contexts is None:
            contexts = build_contexts(tokenizer, prompts)
            caches = {name: prefill(model, context) for name, context in contexts.items()}
            diff = check_spec_cache(model, todo[0]["input_ids"], contexts["S+"], caches["S+"])
            logger.info(f"Prefilled spec vs one full pass on {todo[0]['doc_id']}: |log p| difference max "
                        f"{diff['max']:.4f}, mean {diff['mean']:.5f} nats")
        started = time.time()
        with open(path, "a", encoding="utf-8") as f:
            for i, document in enumerate(todo):
                doc_text, branch_start, kind = text[document["doc_id"]]
                longest, n_copied, n_words = copied(doc_text[branch_start:], grams[document["branch"]], cfg.ngram)
                result = {"key": key, "doc_id": document["doc_id"], "pair_id": document["pair_id"],
                          "branch": document["branch"], "kind": kind, "split": document["split"],
                          "text_sha256": document["text_sha256"], "truncated": document["truncated"],
                          **read_document(model, document, contexts, caches, cfg.alpha, cfg.clip, cfg.chunk_rows),
                          "copied_words": n_copied, "words": n_words, "longest_copied": longest}
                f.write(json.dumps(result) + "\n")
                f.flush()
                done[document["doc_id"], document["text_sha256"]] = result
                if (i + 1) % 50 == 0:
                    logger.info(f"{run_name}: {i + 1}/{len(todo)} documents, {time.time() - started:.0f} s")
        rows[run_name] = [done[d["doc_id"], d["text_sha256"]] for d in documents]
    summary = summarize(rows, cfg)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n" + format_summary(summary, cfg) + "\n", flush=True)
    return summary


@hydra.main(version_base=None, config_path="conf", config_name="doc_lift")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
