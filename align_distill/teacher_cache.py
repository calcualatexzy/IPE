"""The teacher cache of align-distill (docs/align_distill.md#teacher-probe-and-cache), configured by
align_distill/conf/teacher_cache.yaml.

The student reads each document of a minimal pair alone and is trained toward
    log q = log_softmax(log p0 + alpha * clip(log p+ - log p_ref))
at every position. This computes q once for every document of a pairs.jsonl (align_distill/pairs.py) and stores it, so
that training needs no teacher. Per document:
    p0     the student's model on the plain document (<|endoftext|> text <|endoftext|>)
    p+     the teacher after pro-align (SPEC_FRAME, the plain layout), then the document
    p_ref  reference=mirror: the teacher after neg-align (specs.neg); reference=neutral: the teacher without a spec,
           which is p0 itself when the teacher is the student's own model (the default), so that run makes two passes
           per document instead of three
The teacher is the student's own model unless teacher names another row of the model table with the same tokenizer (the
larger-teacher ablation). With anchor=<adapter directory> (and anchor_name, which starts the cache's name), p0 is the
student's model with that LoRA, such as a ce-only student, and the teacher is the plain model under it (the adapter
switched off, or the teacher row), so only the anchor moves; train_distill.py then continues that adapter. Each spec's KV cache is computed once and reused; on the first document of a run it is
checked against one full pass over spec and document. The spec and the document are tokenized apart, so the
document's tokens are the student's.

Sharding: shard=i/N takes the i-th of N contiguous parts of the pairs (both documents of a pair stay together), each
part a job of its own on one GPU; stage=merge then joins the N finished parts into the cache and prints the checks. A
run without shard caches everything and merges at once. A finished shard is kept, so a rerun skips it.

Writes, in <out_dir>/<key>/, where the key names the reference, alpha, clip and top_k and hashes everything q and the
regions depend on (the model and teacher revisions, both filled spec prompts, the settings and the pairs):
    documents.jsonl  per document: ids, card fields and split, its rows in the arrays (offset, n_rows), the region
                     bounds, the choice row with its margins
    topk_ids.npy     int32 [rows, top_k]: the top_k entries of q at each position
    topk_logq.npy    float16 [rows, top_k]: log q of those entries
    tail_logq.npy    float16 [rows]: log of the rest of q's mass
    observed.npy     float32 [rows, 6]: the columns of OBSERVED. At the document's own token: log p0, log p+, log p_ref
                     and log q; over the vocabulary: JS(p+, p_ref) and KL(q || p0)
    region.npy       int8 [rows]: an index into REGIONS
    checks.json      the checks below, as data
    meta.json        the settings, the key's fields and the counts; written last, so a cache without it is unfinished
    cache.log
These are msm/distill_teacher_cache.py's arrays, with the p_ref column named for what it holds, four regions, and the
choice margin read as log P(aligned) - log P(misaligned). Shards sit in <key>/shards/<i>-of-<N>/ until the merge.

The checks, printed at the end:
- mean JS(p+, p_ref), KL(q || p0) and log q - log p0 of the observed token, per region, kind, framing, order and branch;
- the choice margin under p0, p+, p_ref and q where the choice row is known: q should favour the aligned action on both
  branches;
- the observed tokens with the largest |log p+ - log p_ref|, one by one and summed per token, and how much of the
  contrast lies on spikes above checks.spike nats: is the signal the decision and its reasons, or spec wording?

The math (target_logq, document_rows, the spec cache passes) and the checks are copied from msm/distill_teacher_cache.py,
which transformers 5 cannot import.

Usage (scripts/align_distill/teacher_cache.sh chooses the settings):
    python -m align_distill.teacher_cache model=qwen25_7b name=smoke
    python -m align_distill.teacher_cache name=main shard=0/4      # one job per shard, then:
    python -m align_distill.teacher_cache name=main stage=merge
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from align_distill.models import model_row
from align_distill.pairs import BRANCHES, REGIONS, SPLITS, build_documents, decision_digest, load_pairs
from align_distill.tokens import DOCUMENT_TOKEN, encode, load_tokenizer, resolve_path, spec_prompt, token_id

OBSERVED = ("logp0", "logp_pos", "logp_ref", "logq", "js", "kl_q_p0")
ARRAYS = {"topk_ids": np.int32, "topk_logq": np.float16, "tail_logq": np.float16, "observed": np.float32,
          "region": np.int8}
REFERENCES = ("mirror", "neutral")
STAGES = ("cache", "merge")
DTYPES = ("bfloat16", "float16", "float32")
DEVICE_MAPS = ("single", "auto")
ANCHOR_NAME = r"^[A-Za-z0-9][A-Za-z0-9.-]*$"


# Specs and the key

def spec_prompts(cfg: DictConfig, row: dict) -> dict[str, str]:
    """The framed spec each teacher pass reads: S+ always, S- under the mirror reference. With paragraph.name, each
    is followed by its paragraph (paragraph.pos, paragraph.neg); p0 reads neither."""
    paragraph = cfg.get("paragraph") or {}

    def prompt(spec: str, side: str) -> str:
        text = spec_prompt(spec, row)
        if paragraph.get("name"):
            text += resolve_path(paragraph[side]).read_text(encoding="utf-8").strip() + "\n\n"
        return text

    prompts = {"S+": prompt(cfg.specs.pos, "pos")}
    if cfg.reference == "mirror":
        prompts["S-"] = prompt(cfg.specs.neg, "neg")
    return prompts


def adapter_digest(path: str | Path) -> str:
    """A hash of an adapter directory's config and weights, read in blocks (the weights of a rank-64 LoRA of a 32B model
    are 2 GB)."""
    path = resolve_path(path)
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        with open(path / name, "rb") as f:
            while block := f.read(1 << 24):
                digest.update(block)
    return digest.hexdigest()[:16]


def revision(source: str) -> str:
    """A model's version: the commit of its snapshot in the Hugging Face cache, or for a local directory a hash of its
    weights (msm/distill_teacher_cache.py)."""
    path = resolve_path(source)
    if path.is_dir():
        digest = hashlib.sha256()
        for file in sorted(path.glob("*.safetensors")) or [path / "config.json"]:
            digest.update(file.read_bytes())
        return digest.hexdigest()[:16]
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(source, "config.json")).parent.name  # .../snapshots/<commit>/config.json


def cache_key(cfg: DictConfig, row: dict, teacher_row: dict | None, pairs: list[dict], documents: list[dict],
              prompts: dict[str, str]) -> tuple[str, dict]:
    """The cache directory's name and the fields it hashes."""
    fields = {
        "model": row["hub_id"], "model_revision": revision(row["hub_id"]),
        "teacher": teacher_row["hub_id"] if teacher_row else None,
        "teacher_revision": revision(teacher_row["hub_id"]) if teacher_row else None,
        "prompts": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in prompts.items()},
        "reference": cfg.reference, "layout": "plain", "document_token": DOCUMENT_TOKEN, "dtype": cfg.dtype,
        # floats, so that clip=4 on the command line and clip: 4.0 in the config give the same key
        "alpha": float(cfg.alpha), "clip": None if cfg.clip is None else float(cfg.clip), "top_k": int(cfg.top_k),
        "max_seq_len": int(cfg.max_seq_len),
        "documents": hashlib.sha256("\n".join(f"{d['doc_id']} {d['text_sha256']}" for d in documents).encode())
        .hexdigest(),
        "decisions": hashlib.sha256("\n".join(decision_digest(p) for p in pairs).encode()).hexdigest(),
    }
    if cfg.anchor:
        fields["anchor"] = {"name": cfg.anchor_name, "adapter": adapter_digest(cfg.anchor)}
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:10]
    reference = "neutral" if cfg.reference == "neutral" else Path(cfg.specs.neg).stem
    teacher = f"{teacher_row['tag']}-" if teacher_row else ""
    anchor = f"anchor-{cfg.anchor_name}-" if cfg.anchor else ""
    clip = "none" if cfg.clip is None else f"{cfg.clip:g}"
    name = (cfg.get("paragraph") or {}).get("name")
    paragraph = f"para-{name}-" if name else ""  # the paragraphs' texts enter the hash through the prompts
    return f"{teacher}{anchor}{paragraph}{reference}-a{cfg.alpha:g}-clip{clip}-k{cfg.top_k}-{digest}", fields


# Teacher passes (msm/distill_teacher_cache.py)

def build_contexts(tokenizer, prompts: dict[str, str]) -> dict[str, list[int]]:
    """The tokens before a document per spec: <|endoftext|> and the framed spec. A document's pass then starts with the
    context's last token in place of the document's own <|endoftext|>, so its first row predicts the text's first
    token."""
    eot = token_id(tokenizer, DOCUMENT_TOKEN)
    return {name: [eot, *encode(tokenizer, text)] for name, text in prompts.items()}


@torch.inference_mode()
def prefill(model, context: list[int]):
    """The KV cache of all but the last token of a context."""
    from transformers import DynamicCache

    cache = DynamicCache()
    model(input_ids=torch.tensor([context[:-1]], device=model.device), past_key_values=cache, use_cache=True,
          logits_to_keep=1)
    return cache


@torch.inference_mode()
def plain_logits(model, ids: list[int]) -> torch.Tensor:
    """Logits of the predictions of ids[1:], on the document alone."""
    return model(input_ids=torch.tensor([ids[:-1]], device=model.device), use_cache=False).logits[0]


@torch.inference_mode()
def spec_logits(model, ids: list[int], context: list[int], cache) -> torch.Tensor:
    """Logits of the predictions of ids[1:] after a context, from the context's prefilled cache, which is cropped back
    to the context afterwards."""
    length = cache.get_seq_length()
    try:
        input_ids = torch.tensor([[context[-1], *ids[1:-1]]], device=model.device)
        return model(input_ids=input_ids, past_key_values=cache, use_cache=True).logits[0]
    finally:
        added = cache.get_seq_length() - length
        if added > 0:
            cache.crop(-added)


@torch.inference_mode()
def full_logits(model, ids: list[int], context: list[int]) -> torch.Tensor:
    """Logits of the predictions of ids[1:] after a context, in one pass over both."""
    return model(input_ids=torch.tensor([context + ids[1:-1]], device=model.device), use_cache=False,
                 logits_to_keep=len(ids) - 1).logits[0]


def js_divergence(lp: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence, in nats, between the rows of two log-probability matrices."""
    lm = torch.logaddexp(lp, lq) - math.log(2)
    return 0.5 * (lp.exp() * (lp - lm)).sum(-1) + 0.5 * (lq.exp() * (lq - lm)).sum(-1)


def target_logq(lp0: torch.Tensor, lp_pos: torch.Tensor, lp_ref: torch.Tensor, alpha: float,
                clip: float | None) -> torch.Tensor:
    """log q = log_softmax(log p0 + alpha * clip(log p+ - log p_ref)), row by row."""
    delta = lp_pos - lp_ref
    if clip is not None:
        delta = delta.clamp(-clip, clip)
    return torch.log_softmax(lp0 + alpha * delta, dim=-1)


def document_rows(logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor], targets: torch.Tensor, alpha: float,
                  clip: float | None, top_k: int, chunk_rows: int) -> dict[str, np.ndarray]:
    """A document's stored rows, from the logits of p0, p+ and p_ref, chunk_rows positions at a time."""
    parts = defaultdict(list)
    for start in range(0, len(targets), chunk_rows):
        lp0, lp_pos, lp_ref = (torch.log_softmax(x[start:start + chunk_rows].float().to(targets.device), dim=-1)
                               for x in logits)
        y = targets[start:start + chunk_rows, None]
        log_q = target_logq(lp0, lp_pos, lp_ref, alpha, clip)
        top = log_q.topk(top_k, dim=-1)
        tail = log_q.scatter(-1, top.indices, float("-inf")).logsumexp(-1)
        observed = [lp.gather(-1, y)[:, 0] for lp in (lp0, lp_pos, lp_ref, log_q)]
        observed += [js_divergence(lp_pos, lp_ref), (log_q.exp() * (log_q - lp0)).sum(-1)]
        parts["topk_ids"].append(top.indices.int().cpu().numpy())
        parts["topk_logq"].append(top.values.half().cpu().numpy())
        parts["tail_logq"].append(tail.half().cpu().numpy())
        parts["observed"].append(torch.stack(observed, dim=-1).cpu().numpy())
    return {name: np.concatenate(arrays) for name, arrays in parts.items()}


def choice_margins(logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor], choice: dict, alpha: float,
                   clip: float | None) -> dict[str, float]:
    """log P(aligned) - log P(misaligned) of the options' first tokens at the choice row, under p0, p+, p_ref and
    q."""
    lp0, lp_pos, lp_ref = (torch.log_softmax(x[choice["row"]].float().to(logits[0].device), dim=-1) for x in logits)
    log_q = target_logq(lp0, lp_pos, lp_ref, alpha, clip)
    a, m = choice["aligned_id"], choice["misaligned_id"]
    return {name: float(lp[a] - lp[m]) for name, lp in (("p0", lp0), ("pos", lp_pos), ("ref", lp_ref), ("q", log_q))}


@torch.inference_mode()
def check_spec_cache(model, ids: list[int], context: list[int], cache) -> dict[str, float]:
    """The max and mean |log p| difference, at the document's own tokens, between the prefilled spec and one full pass.
    In float32 they agree to about 1e-4; in bfloat16 the two paths round differently."""
    full = full_logits(model, ids, context)
    targets = torch.tensor(ids[1:], device=full.device)[:, None]
    cached = torch.log_softmax(spec_logits(model, ids, context, cache).float().to(full.device), -1).gather(-1, targets)
    diff = (torch.log_softmax(full.float(), -1).gather(-1, targets) - cached).abs()
    return {"max": float(diff.max()), "mean": float(diff.mean())}


@torch.inference_mode()
def compute_cache(model, teacher, documents: list[dict], contexts: dict[str, list[int]],
                  cfg: DictConfig) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """The stored rows of every document, in order, and the check of the spec cache on the first document. Each
    document gets offset (its first row) and, where its choice row is known, its margins. teacher is the model that
    reads the specs: the student's model itself by default."""
    caches = {name: prefill(teacher, context) for name, context in contexts.items()}
    diff = check_spec_cache(teacher, documents[0]["input_ids"], contexts["S+"], caches["S+"])
    (logger.warning if diff["mean"] > 0.05 or diff["max"] > 1 else logger.info)(
        f"Prefilled spec vs one full pass on {documents[0]['doc_id']}, |log p| difference at its tokens: max "
        f"{diff['max']:.4f}, mean {diff['mean']:.4f}")
    parts, offset = defaultdict(list), 0
    from tqdm import tqdm

    for doc in tqdm(documents, desc="teacher"):
        ids = doc["input_ids"]
        lp0 = plain_logits(model, ids)
        lp_pos = spec_logits(teacher, ids, contexts["S+"], caches["S+"])
        if cfg.reference == "mirror":
            lp_ref = spec_logits(teacher, ids, contexts["S-"], caches["S-"])
        else:
            lp_ref = lp0 if teacher is model else plain_logits(teacher, ids)
        logits = (lp0, lp_pos, lp_ref)
        targets = torch.tensor(ids[1:], device=lp0.device)
        for name, array in document_rows(logits, targets, cfg.alpha, cfg.clip, cfg.top_k, cfg.chunk_rows).items():
            parts[name].append(array)
        parts["region"].append(np.repeat(np.arange(len(REGIONS), dtype=np.int8),
                                         np.diff([0, *doc["region_bounds"], doc["n_rows"]])))
        if "row" in doc["choice"]:
            doc["choice"]["margin"] = choice_margins(logits, doc["choice"], cfg.alpha, cfg.clip)
        doc["offset"], offset = offset, offset + doc["n_rows"]
    return {name: np.concatenate(arrays).astype(ARRAYS[name], copy=False) for name, arrays in parts.items()}, diff


# Shards, merging, loading

def parse_shard(shard) -> tuple[int, int]:
    """shard=i/N as (i, N); null as (0, 1)."""
    if shard is None:
        return 0, 1
    try:
        i, n = (int(x) for x in str(shard).split("/"))
    except ValueError:
        raise ValueError(f"shard must be i/N, such as 0/4, not {shard!r}") from None
    if not 0 <= i < n:
        raise ValueError(f"shard {shard}: i must be in [0, N)")
    return i, n


def shard_bounds(n_pairs: int, n: int) -> list[int]:
    """The first pair of each of n contiguous parts, and the end."""
    return [round(k * n_pairs / n) for k in range(n + 1)]


def shard_dir(out: Path, i: int, n: int) -> Path:
    return out / "shards" / f"{i}-of-{n}"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_shard(directory: Path, documents: list[dict], arrays: dict[str, np.ndarray], done: dict) -> None:
    """A shard's arrays and documents, then done.json, which marks it finished."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "done.json").unlink(missing_ok=True)
    for name, array in arrays.items():
        np.save(directory / f"{name}.npy", array)
    write_jsonl(directory / "documents.jsonl", documents)
    (directory / "done.json").write_text(json.dumps(done, indent=2))


def finished_shards(out: Path) -> dict[int, list[Path]]:
    """The finished shard directories, by their N."""
    found = defaultdict(list)
    for directory in sorted((out / "shards").glob("*-of-*")):
        if (directory / "done.json").exists():
            found[int(directory.name.split("-of-")[1])].append(directory)
    return found


def merge_shards(out: Path, documents: list[dict]) -> tuple[list[dict], dict[str, np.ndarray], dict]:
    """Join the N finished shards of a cache in order, into its arrays (written as .npy files in out) and its documents.
    They must hold exactly the given documents, in order."""
    complete = {n: dirs for n, dirs in finished_shards(out).items() if len(dirs) == n}
    if len(complete) != 1:
        found = {n: len(dirs) for n, dirs in finished_shards(out).items()}
        raise FileNotFoundError(f"{out} needs one complete set of finished shards to merge, not {found or 'none'}")
    n, dirs = next(iter(complete.items()))
    dirs = sorted(dirs, key=lambda d: int(d.name.split("-of-")[0]))
    merged, offset, dones = [], 0, []
    for directory in dirs:
        dones.append(json.loads((directory / "done.json").read_text()))
        for doc in read_jsonl(directory / "documents.jsonl"):
            doc["offset"] += offset
            merged.append(doc)
        offset = merged[-1]["offset"] + merged[-1]["n_rows"] if merged else 0
    if [d["doc_id"] for d in merged] != [d["doc_id"] for d in documents]:
        raise ValueError(f"The shards in {out} do not hold this run's documents in order; recompute them "
                         "(overwrite=true)")
    for name, dtype in ARRAYS.items():
        parts = [np.load(directory / f"{name}.npy", mmap_mode="r") for directory in dirs]
        shape = (sum(len(p) for p in parts), *parts[0].shape[1:])
        target = np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        at = 0
        for part in parts:
            target[at:at + len(part)] = part
            at += len(part)
        target.flush()
        del target
    if offset != sum(d["rows"] for d in dones):
        raise ValueError(f"The shards in {out} hold {offset} rows by their documents but {sum(d['rows'] for d in dones)}"
                         " by their arrays")
    arrays = {name: np.load(out / f"{name}.npy", mmap_mode="r") for name in ARRAYS}
    info = {"shards": n, "seconds": sum(d["seconds"] for d in dones), "spec_cache_diff": dones[0]["spec_cache_diff"]}
    return merged, arrays, info


def load_cache(path: str | Path) -> tuple[list[dict], dict[str, np.ndarray], dict]:
    """A finished cache: its documents, its arrays (memory-mapped) and its meta.json."""
    path = resolve_path(path)
    if not (path / "meta.json").exists():
        raise FileNotFoundError(f"{path} is not a finished teacher cache (no meta.json)")
    arrays = {name: np.load(path / f"{name}.npy", mmap_mode="r") for name in ARRAYS}
    return read_jsonl(path / "documents.jsonl"), arrays, json.loads((path / "meta.json").read_text())


# Checks (msm/distill_teacher_cache.py)

def pct(x: float) -> str:
    return "-" if x != x else f"{100 * x:.0f}%"


def nats(x: float) -> str:
    return "-" if x != x else f"{x:+.2f}"


def format_rows(rows: list[list[str]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(text.ljust(width) for text, width in zip(row, widths)).rstrip() for row in rows)


def mean(x: np.ndarray) -> float:
    return float(x.mean()) if len(x) else float("nan")


def groups(documents: list[dict]) -> list[tuple[str, callable]]:
    """Table rows: all documents, then each kind, framing and order present."""
    rows = [("all", lambda d: True)]
    for field in ("kind", "framing", "order"):
        for value in sorted({d[field] for d in documents}):
            rows.append((f"{field} {value}", lambda d, field=field, value=value: d[field] == value))
    return rows


def row_docs(documents: list[dict]) -> np.ndarray:
    """The index of the document of each row."""
    return np.repeat(np.arange(len(documents)), [d["n_rows"] for d in documents])


def region_tables(documents: list[dict], arrays: dict, doc_of: np.ndarray) -> dict:
    """Mean JS(p+, p_ref), KL(q || p0) and log q - log p0 of the observed token, per group, branch and region."""
    obs, region = np.asarray(arrays["observed"]), np.asarray(arrays["region"])
    measures = {"js": obs[:, OBSERVED.index("js")], "kl_q_p0": obs[:, OBSERVED.index("kl_q_p0")],
                "shift": obs[:, OBSERVED.index("logq")] - obs[:, OBSERVED.index("logp0")]}
    tables = {name: [] for name in measures}
    for label, member in groups(documents):
        for branch in BRANCHES:
            docs = np.array([member(d) and d["branch"] == branch for d in documents])
            if not docs.any():
                continue
            rows = docs[doc_of]
            for name, values in measures.items():
                tables[name].append({"group": label, "branch": branch, "docs": int(docs.sum()),
                                     **{r: mean(values[rows & (region == i)]) for i, r in enumerate(REGIONS)},
                                     "all": mean(values[rows])})
    return tables


def margin_table(documents: list[dict]) -> tuple[list[dict], dict]:
    """The choice margins per group and branch, and why the others were not measured."""
    rows = []
    for label, member in groups(documents):
        for branch in BRANCHES:
            docs = [d for d in documents if member(d) and d["branch"] == branch]
            if not docs:
                continue
            margins = [d["choice"]["margin"] for d in docs if "margin" in d["choice"]]
            m = {key: np.array([x[key] for x in margins], dtype=float) for key in ("p0", "pos", "ref", "q")}
            n = len(margins)
            rows.append({"group": label, "branch": branch, "docs": len(docs), "measured": n,
                         "separate": mean(m["pos"] > m["ref"]),
                         "p0_aligned": mean(m["p0"] > 0), "q_aligned": mean(m["q"] > 0),
                         "p0_median": float(np.median(m["p0"])) if n else float("nan"),
                         "q_median": float(np.median(m["q"])) if n else float("nan"),
                         "q_minus_p0": mean(m["q"] - m["p0"])})
    skipped = Counter(d["choice"]["skipped"] for d in documents if "skipped" in d["choice"])
    return rows, dict(skipped.most_common())


def contrast_tables(tokenizer, documents: list[dict], arrays: dict, doc_of: np.ndarray, top_positions: int,
                    top_tokens: int, spike: float) -> dict:
    """Where the contrast log p+ - log p_ref of the observed tokens lies: per region and branch, on spikes, on single
    positions, and summed per token."""
    obs, region = np.asarray(arrays["observed"]), np.asarray(arrays["region"])
    delta = obs[:, OBSERVED.index("logp_pos")] - obs[:, OBSERVED.index("logp_ref")]
    kl = obs[:, OBSERVED.index("kl_q_p0")]
    branch = np.array([BRANCHES.index(d["branch"]) for d in documents])[doc_of]
    within = np.concatenate([np.arange(d["n_rows"]) for d in documents])
    target = np.concatenate([np.asarray(d["input_ids"][1:]) for d in documents])

    shares = []
    for b, name in enumerate(BRANCHES):
        in_branch = branch == b
        total_abs, total_kl = np.abs(delta[in_branch]).sum(), kl[in_branch].sum()
        for i, r in enumerate((*REGIONS, "all")):
            rows = in_branch & (region == i) if r != "all" else in_branch
            spikes = rows & (np.abs(delta) > spike)
            shares.append({"branch": name, "region": r, "rows": mean(rows[in_branch]),
                           "abs_delta": float(np.abs(delta[rows]).sum() / total_abs) if total_abs else float("nan"),
                           "kl": float(kl[rows].sum() / total_kl) if total_kl else float("nan"),
                           "spike_rows": mean(spikes[rows]),
                           "spike_abs_delta": float(np.abs(delta[spikes]).sum() / np.abs(delta[rows]).sum())
                           if np.abs(delta[rows]).sum() else float("nan")})

    positions = []
    for row in np.argsort(-np.abs(delta))[:top_positions]:
        doc, j = documents[doc_of[row]], within[row]
        positions.append({"delta": float(delta[row]), "token": tokenizer.decode([target[row]]), "doc": doc["doc_id"],
                          "region": REGIONS[region[row]],
                          "context": tokenizer.decode(doc["input_ids"][max(1, j - 7):j + 1])})

    by_token = defaultdict(lambda: {"abs": 0.0, "signed": 0.0, "n": 0, "regions": Counter()})
    for row in np.flatnonzero(np.abs(delta) > 0.5):  # the rest adds little and would only slow this down
        entry = by_token[int(target[row])]
        entry["abs"] += abs(float(delta[row]))
        entry["signed"] += float(delta[row])
        entry["n"] += 1
        entry["regions"][REGIONS[region[row]]] += 1
    tokens = [{"token": tokenizer.decode([token]), "abs_sum": e["abs"], "mean": e["signed"] / e["n"], "n": e["n"],
               "region": e["regions"].most_common(1)[0][0]}
              for token, e in sorted(by_token.items(), key=lambda item: -item[1]["abs"])[:top_tokens]]
    return {"shares": shares, "positions": positions, "tokens": tokens}


def checks(tokenizer, documents: list[dict], arrays: dict, cfg: DictConfig) -> tuple[dict, str]:
    """The checks before training, as data and as printed text."""
    doc_of = row_docs(documents)
    region = np.asarray(arrays["region"])
    summary = {"reference": cfg.reference, "documents": len(documents), "rows": int(len(doc_of)),
               "rows_per_region": {r: int((region == i).sum()) for i, r in enumerate(REGIONS)},
               "truncated": sum(d["truncated"] for d in documents)}
    reference = ("the teacher after neg-align" if cfg.reference == "mirror"
                 else "the teacher without a spec" + (" (p0 itself)" if not cfg.teacher else ""))
    clip = "none" if cfg.clip is None else f"±{cfg.clip:g} nats"
    text = [f"Teacher cache: {len(documents)} documents, {len(doc_of):,} positions "
            f"({', '.join(f'{n:,} {r}' for r, n in summary['rows_per_region'].items())}); model {cfg.model}, teacher "
            f"{cfg.teacher or cfg.model}, alpha {cfg.alpha:g}, clip {clip}, top {cfg.top_k}; p_ref = {reference}.",
            f"Truncated to {cfg.max_seq_len} tokens: {summary['truncated']}."]

    summary["regions"] = region_tables(documents, arrays, doc_of)
    titles = {"js": "Mean JS(p+, p_ref) per position (nats)",
              "kl_q_p0": "Mean KL(q || p0) per position (nats): how far the target is from where the student starts",
              "shift": "Mean log q - log p0 of the document's own token (> 0: q makes the text more likely)"}
    for name, rows in summary["regions"].items():
        table = [["", "branch", "docs", *REGIONS, "all"]]
        table += [[row["group"], row["branch"], str(row["docs"]),
                   *(f"{row[r]:.4f}" if name != "shift" else nats(row[r]) for r in (*REGIONS, "all"))]
                  for row in rows]
        text.append(f"\n{titles[name]}\n{format_rows(table)}")

    rows, skipped = margin_table(documents)
    summary["choice"], summary["choice_skipped"] = rows, skipped
    table = [["", "branch", "measured", "p+ > p_ref", "p0 -> aligned", "q -> aligned", "p0 median", "q median",
              "q - p0"]]
    table += [[row["group"], row["branch"], f"{row['measured']}/{row['docs']}", pct(row["separate"]),
               pct(row["p0_aligned"]), pct(row["q_aligned"]), nats(row["p0_median"]), nats(row["q_median"]),
               nats(row["q_minus_p0"])] for row in rows]
    text.append("\nChoice margin at the choice row: log P(aligned) - log P(misaligned) of the two options' first tokens, "
                "under p0, p+, p_ref and q. Branches: pos takes the aligned action, neg the misaligned one; q should "
                f"favour the aligned action on both.\n{format_rows(table)}\n"
                f"Not measured: {', '.join(f'{n} {why}' for why, n in skipped.items()) or 'none'}")

    contrast = contrast_tables(tokenizer, documents, arrays, doc_of, cfg.checks.top_positions, cfg.checks.top_tokens,
                               cfg.checks.spike)
    summary["contrast"] = contrast
    table = [["branch", "region", "positions", "|log p+ - log p_ref|", "KL(q || p0)", f"> {cfg.checks.spike:g} nats",
              "their |log p+ - log p_ref|"]]
    table += [[row["branch"], row["region"], pct(row["rows"]), pct(row["abs_delta"]), pct(row["kl"]),
               pct(row["spike_rows"]), pct(row["spike_abs_delta"])] for row in contrast["shares"]]
    text.append("\nWhere the contrast lies: each region's share of a branch's positions, of its summed "
                "|log p+ - log p_ref| at the observed tokens and of its summed KL(q || p0); then the share of the "
                f"region's positions above {cfg.checks.spike:g} nats, and of its |log p+ - log p_ref| on them\n"
                f"{format_rows(table)}")
    table = [["log p+ - log p_ref", "token", "region", "document", "after"]]
    table += [[nats(p["delta"]), repr(p["token"]), p["region"], p["doc"], repr(p["context"][-60:])]
              for p in contrast["positions"]]
    text.append(f"\nObserved tokens the two teachers disagree on most\n{format_rows(table)}")
    table = [["token", "summed |log p+ - log p_ref|", "mean log p+ - log p_ref", "n", "mostly in"]]
    table += [[repr(t["token"]), f"{t['abs_sum']:.1f}", nats(t["mean"]), str(t["n"]), t["region"]]
              for t in contrast["tokens"]]
    text.append(f"\nTokens with the largest summed |log p+ - log p_ref| (over positions above 0.5 nats)\n"
                f"{format_rows(table)}")
    return summary, "\n".join(text)


# Running

def check_config(cfg: DictConfig) -> None:
    if cfg.stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, not {cfg.stage}")
    if cfg.reference not in REFERENCES:
        raise ValueError(f"reference must be one of {REFERENCES}, not {cfg.reference}")
    if cfg.dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, not {cfg.dtype}")
    if cfg.device_map not in DEVICE_MAPS:
        raise ValueError(f"device_map must be one of {DEVICE_MAPS}, not {cfg.device_map}")
    if cfg.alpha <= 0 or (cfg.clip is not None and cfg.clip <= 0):
        raise ValueError(f"alpha and clip must be positive, not {cfg.alpha} and {cfg.clip}")
    if not 0 < cfg.top_k <= 1024 or cfg.chunk_rows < 1 or cfg.max_seq_len < 2:
        raise ValueError("top_k must be in [1, 1024], chunk_rows at least 1 and max_seq_len at least 2")
    if not cfg.splits or set(cfg.splits) - set(SPLITS):
        raise ValueError(f"splits must come from {SPLITS}, not {list(cfg.splits)}")
    parse_shard(cfg.shard)
    if cfg.teacher == cfg.model:
        raise ValueError("teacher=null already means the student's own model")
    if bool(cfg.anchor) != bool(cfg.anchor_name):
        raise ValueError("Set anchor (an adapter directory) and anchor_name together, or neither")
    if cfg.anchor:
        if not re.match(ANCHOR_NAME, str(cfg.anchor_name)):
            raise ValueError(f"anchor_name must be letters, digits, '.' and '-', not {cfg.anchor_name!r}")
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (resolve_path(cfg.anchor) / name).is_file():
                raise FileNotFoundError(f"anchor {cfg.anchor} has no {name}")
    paths = [cfg.pairs, cfg.specs.pos] + ([cfg.specs.neg] if cfg.reference == "mirror" else [])
    paragraph = cfg.get("paragraph") or {}
    sides = ("pos", "neg") if cfg.reference == "mirror" else ("pos",)
    if paragraph.get("name"):
        if not re.match(ANCHOR_NAME, str(paragraph.name)) or not all(paragraph.get(side) for side in sides):
            raise ValueError("paragraph needs a name of letters, digits, '.' and '-', and a file per spec")
        paths += [paragraph[side] for side in sides]
    elif paragraph.get("pos") or paragraph.get("neg"):
        raise ValueError("Set paragraph.name with paragraph.pos and paragraph.neg")
    for path in paths:
        if not resolve_path(path).is_file():
            raise FileNotFoundError(f"{path} is not a file")


def load_model(row: dict, cfg: DictConfig):
    from transformers import AutoModelForCausalLM

    device_map = "auto" if cfg.device_map == "auto" else {"": "cuda" if torch.cuda.is_available() else "cpu"}
    return AutoModelForCausalLM.from_pretrained(row["hub_id"], dtype=getattr(torch, cfg.dtype),
                                                attn_implementation="sdpa", device_map=device_map).eval()


def load_anchor(model, row: dict, cfg: DictConfig):
    """The student's model with the anchor's LoRA, in the cache's dtype (training runs the LoRA under bf16 autocast)."""
    from peft import PeftModel

    base = json.loads((resolve_path(cfg.anchor) / "adapter_config.json").read_text()).get("base_model_name_or_path")
    if base and base != row["hub_id"]:
        raise ValueError(f"The anchor {cfg.anchor} was trained on {base}, not on {row['hub_id']}")
    model = PeftModel.from_pretrained(model, str(resolve_path(cfg.anchor)), is_trainable=False)
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.to(getattr(torch, cfg.dtype))
    logger.info(f"Anchor p0: {row['hub_id']} with the adapter {cfg.anchor} ({cfg.anchor_name})")
    return model.eval()


class AdapterOff:
    """The plain model under a PEFT model: every call runs with its adapter switched off. It shares the weights, so
    the anchored student and its own teacher fit one GPU."""

    def __init__(self, peft_model):
        self.peft_model = peft_model

    @property
    def device(self):
        return self.peft_model.device

    def __call__(self, *args, **kwargs):
        with self.peft_model.disable_adapter():
            return self.peft_model(*args, **kwargs)


def same_vocabulary(a, b) -> bool:
    return a.get_vocab() == b.get_vocab()


def finish(out: Path, documents: list[dict], tokenizer, cfg: DictConfig, key: str, fields: dict) -> None:
    """Merge the shards, run the checks, write documents.jsonl, checks.json and meta.json (last), print the checks."""
    merged, arrays, info = merge_shards(out, documents)
    summary, text = checks(tokenizer, merged, arrays, cfg)
    write_jsonl(out / "documents.jsonl", merged)
    (out / "checks.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    meta = {"config": OmegaConf.to_container(cfg, resolve=True), "key": key, "fields": fields, **info,
            "observed": OBSERVED, "regions": REGIONS, "documents": len(merged), "rows": summary["rows"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    if cfg.delete_shards:
        shutil.rmtree(out / "shards")
    logger.info(f"Saved the teacher cache to {out}")
    print(text + "\n", flush=True)


@hydra.main(version_base=None, config_path="conf", config_name="teacher_cache")
def main(cfg: DictConfig) -> None:
    check_config(cfg)
    row = model_row(cfg.model)
    teacher_row = model_row(cfg.teacher) if cfg.teacher else None
    tokenizer = load_tokenizer(row["hub_id"])
    pairs = load_pairs(cfg.pairs, cfg.splits, cfg.max_pairs)
    if not pairs:
        raise ValueError(f"{cfg.pairs} has no pairs in the splits {list(cfg.splits)}")
    documents = build_documents(pairs, tokenizer, cfg.max_seq_len)
    prompts = spec_prompts(cfg, row)
    key, fields = cache_key(cfg, row, teacher_row, pairs, documents, prompts)
    out = resolve_path(cfg.out_dir) / key
    if (out / "meta.json").exists() and not cfg.overwrite:
        logger.info(f"{out} is already cached with these settings (overwrite=true recomputes it); its checks are in "
                    "checks.json")
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").unlink(missing_ok=True)
    if cfg.stage == "merge":
        logger.add(out / "cache.log")
        finish(out, documents, tokenizer, cfg, key, fields)
        return

    i, n = parse_shard(cfg.shard)
    if n > len(pairs):
        raise ValueError(f"shard={cfg.shard}: {n} shards for {len(pairs)} pairs")
    bounds = shard_bounds(len(pairs), n)
    directory = shard_dir(out, i, n)
    if (directory / "done.json").exists() and not cfg.overwrite:
        logger.info(f"Shard {i}/{n} is already finished in {directory}")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        logger.add(out / "cache.log" if n == 1 else directory / "shard.log")  # shards and their logs go at the merge
        shard_docs = documents[2 * bounds[i]:2 * bounds[i + 1]]
        logger.info(f"Teacher cache into {out}, shard {i}/{n}: pairs {bounds[i]}-{bounds[i + 1] - 1} of {len(pairs)}\n"
                    f"{OmegaConf.to_yaml(cfg)}")
        counts = Counter((d["split"], d["kind"]) for d in shard_docs if d["branch"] == "pos")
        logger.info(f"{len(shard_docs)} documents: "
                    + ", ".join(f"{c} {split}/{kind}" for (split, kind), c in sorted(counts.items())))
        if not torch.cuda.is_available():
            logger.warning("No GPU is visible; the passes will run on the CPU")
        model = load_model(row, cfg)
        if cfg.anchor:
            model = load_anchor(model, row, cfg)
        teacher = AdapterOff(model) if cfg.anchor else model
        if teacher_row is not None:
            if not same_vocabulary(tokenizer, load_tokenizer(teacher_row["hub_id"])):
                raise ValueError(f"The teacher {teacher_row['hub_id']} does not share the student's tokenizer")
            teacher = load_model(teacher_row, cfg)
        start = time.time()
        arrays, diff = compute_cache(model, teacher, shard_docs, build_contexts(tokenizer, prompts), cfg)
        rows = sum(d["n_rows"] for d in shard_docs)
        seconds = round(time.time() - start)
        logger.info(f"Cached {len(shard_docs)} documents ({rows:,} positions) in {seconds}s")
        save_shard(directory, shard_docs, arrays, {"shard": i, "of": n, "pairs": bounds[i:i + 2],
                                                   "documents": len(shard_docs), "rows": rows, "seconds": seconds,
                                                   "spec_cache_diff": diff})
    if n == 1:
        finish(out, documents, tokenizer, cfg, key, fields)
    else:
        done = len(finished_shards(out).get(n, []))
        logger.info(f"{done} of {n} shards finished" + ("; merge them with stage=merge" if done == n else ""))


if __name__ == "__main__":
    main()
