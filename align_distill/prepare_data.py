"""The training data of align-distill, drawn from the released datasets (docs/align_distill.md#g0-retrain-msm-and-aft-only),
configured by align_distill/conf/prepare_data.yaml. It runs here, on the CPU, from the Hugging Face cache that
scripts/align_distill/fetch.sh filled (the datasets are read from the snapshots its manifest records).

Stages:
    msm    The MSM corpus subsample: documents up to msm.tokens tokens (Qwen tokens in the document format, as the
           student trains on them, truncated to max_seq_len), stratified by the corpus's domains, and msm.heldout
           documents outside it, for the loss checks. Each domain is put in one seeded order; its last documents are
           held out, in proportion to its size. The rest are interleaved so that every prefix holds each domain in
           proportion, and the subsample is the shortest prefix with msm.tokens tokens. A larger target takes a longer
           prefix of the same order, so the subsamples nest.
           -> data/align_distill/msm/<msm.name>/{train,heldout}.jsonl
    aft    The AFT subsets of the model's variant: one seeded order of the released chats, shared by the variants with
           and without CoT (checked: they hold the same prompts in the same order). Each size is a prefix of the order,
           so each subset contains the smaller ones; the held-out chats are the last aft.heldout of the order. Chats
           with an empty reply are left out of the order (4 of the 9,963 released, so all is 9,959).
           -> data/align_distill/aft/<model>/<variant>_<size>.jsonl and <variant>_heldout.jsonl
    it     it.n instruction-tuning chats from sft-it-mix (the model's split), the same at every AFT size: per source, the
           first of a seeded order, with it.counts per source (the paper's Table 2) or, where that is null, the
           released mix's proportions (largest remainder).
           -> data/align_distill/aft/<model>/it_<n>.jsonl
    smoke  The hand-written pairs of tests/local/fixtures/align_distill/pairs_tiny.jsonl, copied to the smoke pair run
           (data/align_distill/distill/smoke/pairs.jsonl) for the smoke jobs.
    all    msm, aft and it.
Each stage writes a manifest next to its files: the sources with their revisions, the settings, and the files with their
counts and hashes. A stage whose manifest exists with other sources or settings refuses to overwrite it
(overwrite=true); with the same ones it skips the work.

Usage (scripts/align_distill/prepare_data.sh chooses the settings):
    python -m align_distill.prepare_data stage=all model=qwen25
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from align_distill.models import model_row
from align_distill.tokens import IGNORE_INDEX, chat_ids, corpus_ids, load_tokenizer, resolve_path

STAGES = ("msm", "aft", "it", "smoke", "all")
VARIANTS = ("cot", "no-cot")


# Sources

def snapshot(repo_id: str, manifest: str) -> tuple[Path, str]:
    """A dataset's snapshot directory and revision: from fetch.sh's manifest, else from the Hugging Face cache."""
    path = resolve_path(manifest)
    if path.is_file():
        entry = json.loads(path.read_text()).get(f"dataset:{repo_id}")
        if entry and Path(entry["path"]).is_dir():
            return Path(entry["path"]), entry["revision"]
    from huggingface_hub import snapshot_download

    try:
        directory = Path(snapshot_download(repo_id, repo_type="dataset", local_files_only=True))
    except Exception as e:
        raise FileNotFoundError(f"{repo_id} is not in the Hugging Face cache; run scripts/align_distill/fetch.sh "
                                "(group data) first") from e
    return directory, directory.name


def read_rows(directory: Path, split: str | None = None) -> list[dict]:
    """The rows of a dataset snapshot: its dataset.jsonl, or the parquet files of a split (default train)."""
    if split is None and (directory / "dataset.jsonl").is_file():
        with open(directory / "dataset.jsonl", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    files = sorted(directory.glob(f"data/{split or 'train'}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"{directory} has no dataset.jsonl and no data/{split or 'train'}-*.parquet")
    import pyarrow.parquet as pq

    return [row for file in files for row in pq.read_table(file).to_pylist()]


# Writing

def write_jsonl(path: Path, rows: list[dict]) -> dict:
    """Write rows and return the file's entry for a manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepared(manifest: Path, inputs: dict, overwrite: bool) -> bool:
    """Whether a stage's manifest already records these inputs (so the work can be skipped). Raises if it records other
    inputs, unless overwrite."""
    if not manifest.is_file() or overwrite:
        return False
    recorded = json.loads(manifest.read_text())
    if recorded.get("inputs") == inputs and all((manifest.parent / name).is_file() for name in recorded["files"]):
        logger.info(f"{manifest.parent} is already prepared with these inputs")
        return True
    raise FileExistsError(f"{manifest} records other inputs; overwrite=true replaces its files")


def save_manifest(manifest: Path, inputs: dict, files: dict, summary: dict) -> None:
    manifest.write_text(json.dumps({"inputs": inputs, "files": files, "summary": summary}, indent=2,
                                   ensure_ascii=False))


def largest_remainder(sizes: dict[str, int], total: int) -> dict[str, int]:
    """total split over the keys in proportion to sizes, in integers that sum to total."""
    n = sum(sizes.values())
    exact = {key: total * size / n for key, size in sizes.items()}
    counts = {key: int(x) for key, x in exact.items()}
    for key in sorted(exact, key=lambda k: (counts[k] - exact[k], k))[:total - sum(counts.values())]:
        counts[key] += 1
    return counts


# Stages

def stratified_order(groups: list[str], seed: int, heldout: int) -> tuple[list[int], list[int]]:
    """The indices of rows in one interleaved order, stratified by group, and the held-out indices. Each group gets one
    seeded order; its last documents are held out (heldout in all, in proportion to group size); the rest are
    interleaved by the share of the group they complete, so that every prefix holds each group in proportion."""
    rng = np.random.default_rng(seed)
    members = {g: np.flatnonzero(np.asarray(groups) == g) for g in sorted(set(groups))}
    held_counts = largest_remainder({g: len(m) for g, m in members.items()}, heldout)
    keyed, held = [], []
    for g, m in members.items():
        order = rng.permutation(m).tolist()
        cut = len(order) - held_counts[g]
        held += order[cut:]
        keyed += [((i + 0.5) / cut, g, index) for i, index in enumerate(order[:cut])]
    return [index for _, _, index in sorted(keyed)], sorted(held)


def stage_msm(cfg: DictConfig, tokenizer) -> None:
    directory, revision = snapshot(cfg.msm.source, cfg.fetch_manifest)
    out = resolve_path(cfg.out_dir) / "msm" / cfg.msm.name
    inputs = {"source": cfg.msm.source, "revision": revision, "tokenizer": cfg.model_hub_id,
              **{key: cfg.msm[key] for key in ("tokens", "heldout", "seed")}, "max_seq_len": cfg.max_seq_len}
    if prepared(out / "manifest.json", inputs, cfg.overwrite):
        return
    rows = read_rows(directory)
    ids, _ = corpus_ids(tokenizer, [row["text"] for row in rows], cfg.max_seq_len)
    lengths = [len(x) for x in ids]
    order, held = stratified_order([row["domain"] for row in rows], cfg.msm.seed, cfg.msm.heldout)
    cumulative = np.cumsum([lengths[i] for i in order])
    n = int(np.searchsorted(cumulative, cfg.msm.tokens)) + 1
    if cumulative[-1] < cfg.msm.tokens:
        raise ValueError(f"{cfg.msm.source} holds {cumulative[-1]:,} tokens outside the held-out documents, fewer than "
                         f"msm.tokens={cfg.msm.tokens:,}")
    chosen = order[:n]

    def entries(indices):
        return [{"text": rows[i]["text"], "domain": rows[i]["domain"], "index": int(i), "n_tokens": lengths[i]}
                for i in indices]

    files = {"train.jsonl": write_jsonl(out / "train.jsonl", entries(chosen)),
             "heldout.jsonl": write_jsonl(out / "heldout.jsonl", entries(held))}
    domains = Counter(rows[i]["domain"] for i in chosen)
    corpus = Counter(row["domain"] for row in rows)
    summary = {"documents": len(chosen), "tokens": int(cumulative[n - 1]), "corpus_documents": len(rows),
               "corpus_tokens": int(sum(lengths)), "heldout_documents": len(held),
               "heldout_tokens": int(sum(lengths[i] for i in held)),
               "domains": {d: {"documents": domains[d], "share": domains[d] / len(chosen),
                               "corpus_share": corpus[d] / len(rows)} for d in sorted(corpus)}}
    save_manifest(out / "manifest.json", inputs, files, summary)
    logger.info(f"MSM subsample {out}: {summary['documents']:,} of {len(rows):,} documents, "
                f"{summary['tokens']:,} of {summary['corpus_tokens']:,} tokens; {len(held)} held out\n"
                + "\n".join(f"  {d}: {v['documents']} documents ({v['share']:.1%}; corpus {v['corpus_share']:.1%})"
                            for d, v in summary["domains"].items()))


def prompt_of(messages: list[dict]) -> str:
    """A chat's first user prompt, without Qwen3's /no_think switch, to match the variants."""
    return next(m["content"] for m in messages if m["role"] == "user").replace("/no_think", "").strip()


def has_replies(messages: list[dict]) -> bool:
    return all(m["content"].strip() for m in messages if m["role"] == "assistant")


def trained_tokens(tokenizer, rows: list[dict]) -> int:
    return sum(sum(label != IGNORE_INDEX for label in chat_ids(tokenizer, row["messages"])[1]) for row in rows)


def stage_aft(cfg: DictConfig, row: dict, tokenizer) -> None:
    sources = row["data"]["aft"]
    snapshots = {variant: snapshot(sources[variant], cfg.fetch_manifest) for variant in VARIANTS}
    out = resolve_path(cfg.out_dir) / "aft" / row["tag"]
    inputs = {"sources": {v: [sources[v], snapshots[v][1]] for v in VARIANTS}, "sizes": list(cfg.aft.sizes),
              "heldout": cfg.aft.heldout, "seed": cfg.aft.seed}
    manifest = out / "manifest_aft.json"
    if prepared(manifest, inputs, cfg.overwrite):
        return
    rows = {variant: read_rows(snapshots[variant][0]) for variant in VARIANTS}
    total = len(rows["cot"])
    if len(rows["no-cot"]) != total or any(prompt_of(a["messages"]) != prompt_of(b["messages"])
                                           for a, b in zip(rows["cot"], rows["no-cot"])):
        raise ValueError(f"{sources['cot']} and {sources['no-cot']} do not hold the same prompts in the same order")
    # A chat with an empty reply in either variant would teach the model to answer nothing: it is left out of the draw
    # (4 of the 9,963 released chats, the same in every variant)
    usable = [i for i in range(total) if all(has_replies(rows[v][i]["messages"]) for v in VARIANTS)]
    excluded = sorted(set(range(total)) - set(usable))
    n = len(usable)
    sizes = [n if size == "all" else int(size) for size in cfg.aft.sizes]
    if any(not 0 < size <= n for size in sizes) or not 0 <= cfg.aft.heldout < n:
        raise ValueError(f"aft.sizes {sizes} and aft.heldout {cfg.aft.heldout} must fit in the {n} usable chats")
    order = np.random.default_rng(cfg.aft.seed).permutation(usable).tolist()
    held = order[n - cfg.aft.heldout:]
    overlap = [size for size in sizes if size > n - cfg.aft.heldout]
    if overlap:
        logger.warning(f"The subsets of size {overlap} include held-out chats")
    files, summary = {}, {"chats": total, "usable": n, "excluded_empty_replies": excluded, "sizes": sizes,
                          "heldout": len(held), "trained_tokens": {}}
    for variant in VARIANTS:
        def entries(indices):
            return [{"messages": rows[variant][i]["messages"], "index": int(i)} for i in indices]

        for size in sizes:
            name = f"{variant}_{size}.jsonl"
            files[name] = write_jsonl(out / name, entries(order[:size]))
            if cfg.count_tokens:
                summary["trained_tokens"][name] = trained_tokens(tokenizer, [rows[variant][i] for i in order[:size]])
        if held:
            files[f"{variant}_heldout.jsonl"] = write_jsonl(out / f"{variant}_heldout.jsonl", entries(held))
    save_manifest(manifest, inputs, files, summary)
    logger.info(f"AFT subsets in {out}: sizes {sizes} of {n} usable chats for {', '.join(VARIANTS)}, {len(held)} held "
                f"out; left out {len(excluded)} of {total} with an empty reply (rows {excluded})"
                + "".join(f"\n  {name}: {tokens:,} trained tokens" for name, tokens in summary["trained_tokens"].items()))


def stage_it(cfg: DictConfig, row: dict, tokenizer) -> None:
    directory, revision = snapshot(cfg.it.source, cfg.fetch_manifest)
    split = row["data"]["it_split"]
    out = resolve_path(cfg.out_dir) / "aft" / row["tag"]
    counts_cfg = OmegaConf.to_container(cfg.it.counts) if cfg.it.counts else None
    inputs = {"source": cfg.it.source, "revision": revision, "split": split, "n": cfg.it.n, "counts": counts_cfg,
              "seed": cfg.it.seed}
    manifest = out / f"manifest_it_{cfg.it.n}.json"
    if prepared(manifest, inputs, cfg.overwrite):
        return
    rows = read_rows(directory, split)
    by_source = {}
    for i, r in enumerate(rows):
        by_source.setdefault(r["source"], []).append(i)
    available = {source: len(indices) for source, indices in sorted(by_source.items())}
    counts = counts_cfg or largest_remainder(available, cfg.it.n)
    if sum(counts.values()) != cfg.it.n or set(counts) - set(available) or any(
            counts[s] > available[s] for s in counts):
        raise ValueError(f"it.counts {counts} must sum to it.n={cfg.it.n} over the sources {available}")
    rng = np.random.default_rng(cfg.it.seed)
    chosen = []
    for source in sorted(available):
        chosen += rng.permutation(by_source[source]).tolist()[:counts.get(source, 0)]
    chosen = [chosen[i] for i in rng.permutation(len(chosen))]
    name = f"it_{cfg.it.n}.jsonl"
    files = {name: write_jsonl(out / name, [{"messages": rows[i]["messages"], "source": rows[i]["source"],
                                            "index": int(i)} for i in chosen])}
    summary = {"available": available, "counts": counts}
    if cfg.count_tokens:
        summary["trained_tokens"] = trained_tokens(tokenizer, [rows[i] for i in chosen])
    save_manifest(manifest, inputs, files, summary)
    logger.info(f"Instruction tuning {out / name}: {cfg.it.n} chats from {cfg.it.source}:{split}, per source "
                + ", ".join(f"{s} {c}/{available[s]}" for s, c in counts.items())
                + (f"; {summary['trained_tokens']:,} trained tokens" if cfg.count_tokens else ""))


def stage_smoke(cfg: DictConfig) -> None:
    target = resolve_path(cfg.out_dir) / "distill" / "smoke" / "pairs.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(resolve_path(cfg.smoke_pairs), target)
    logger.info(f"Copied {cfg.smoke_pairs} to {target}")


@hydra.main(version_base=None, config_path="conf", config_name="prepare_data")
def main(cfg: DictConfig) -> None:
    if cfg.stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, not {cfg.stage}")
    if cfg.stage == "smoke":
        stage_smoke(cfg)
        return
    row = model_row(cfg.model)
    from align_distill.eval.adapters import model_template

    tokenizer = load_tokenizer(row["hub_id"], chat_template=model_template(row)[0])
    cfg.model_hub_id = row["hub_id"]
    stages = ("msm", "aft", "it") if cfg.stage == "all" else (cfg.stage,)
    if "msm" in stages:
        stage_msm(cfg, tokenizer)
    if "aft" in stages:
        stage_aft(cfg, row, tokenizer)
    if "it" in stages:
        stage_it(cfg, row, tokenizer)


if __name__ == "__main__":
    main()
