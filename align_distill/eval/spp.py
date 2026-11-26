"""SPP's ConstitutionEval and AIRisk (align_distill/eval/conseval, airisk) on a base model, its adapters, specs in its
system turn and the KD target q, configured by align_distill/conf/eval_spp.yaml.

The models (variants):
    adapters      '<name>=<adapter>' as in eval_am.yaml: paper:<checkpoint>, an adapter directory, a Hugging Face
                  adapter id, or none for the base model
    system_specs  '<name>=<spec file>': the base model with the spec framed as the teacher reads it (tokens.spec_prompt)
                  before the chat template's own system line
    targets       '<name>=<pos spec>,<neg spec>,<alpha>[,<clip>]': the KD target q on the base model, its p+ and p-
                  reading the specs as system_specs do, clipped at the entry's clip (a number, or none for no clip),
                  else at clip (align_distill/eval/logprobs.py)
Every benchmark scores fixed answers by their log probabilities (no sampling, no judge): see each bench.py.

Stages:
    score   score every variant on every benchmark that has no current result (GPU); shard=i/N takes every N-th
            variant, one process each
    report  the tables and the comparisons, from the saved results (no GPU)
    all     both

Results; running again skips every finished result whose settings match, and stops at one whose settings changed
(overwrite=true redoes it):
    <out_dir>/<model>/<label>/<benchmark>/rows.jsonl     one row per item
    <out_dir>/<model>/<label>/<benchmark>/metrics.json
    <out_dir>/<model>/<label>/<benchmark>/meta.json      the settings, written last: the result is finished
    <out_dir>/<model>/summary_<time>.json                 the report's metrics and comparisons

Usage (scripts/align_distill/eval_spp.sh chooses the settings):
    python -m align_distill.eval.spp stage=score shard=0/2
    python -m align_distill.eval.spp stage=report
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from align_distill.eval import stats
from align_distill.eval.adapters import BASE_LABEL, NO_ADAPTER, Adapter, parse_adapters, resolve
from align_distill.eval.airisk import bench as airisk
from align_distill.eval.conseval import bench as conseval
from align_distill.eval.logprobs import Scorer, Target, Variant, default_system
from align_distill.models import model_row
from align_distill.tokens import load_tokenizer, resolve_path, spec_prompt
from align_distill.upstream import ROOT
from msm.datagen.utils import load_json, read_jsonl, save_json, write_jsonl

BENCHMARKS = {conseval.NAME: conseval, airisk.NAME: airisk}
STAGES = ("score", "report", "all")


def sha(value) -> str | None:
    return None if value is None else hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def parse_shard(shard) -> tuple[int, int]:
    """shard=i/N as (i, N); null as (0, 1)."""
    if shard is None:
        return 0, 1
    try:
        i, n = (int(x) for x in str(shard).split("/"))
    except ValueError:
        raise ValueError(f"shard must be i/N, such as 0/2, not {shard!r}") from None
    if not 0 <= i < n:
        raise ValueError(f"shard {shard}: i must be in [0, N)")
    return i, n


# Variants

def split_entry(entry: str, what: str) -> tuple[str, str]:
    name, sep, value = str(entry).partition("=")
    if not sep or not name.strip() or not value.strip():
        raise ValueError(f"{what} entry {entry!r} is not '<name>=<value>'")
    return name.strip(), value.strip()


def parse_target(value: str, clip: float | None) -> tuple[str, str, float, float | None]:
    """A targets entry '<pos spec>,<neg spec>,<alpha>[,<clip>]' as (pos, neg, alpha, clip): its own clip, a positive
    number or none (no clip), else the config's clip."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) not in (3, 4):
        raise ValueError(f"targets entry {value!r} is not '<pos spec>,<neg spec>,<alpha>[,<clip>]'")
    if len(parts) == 4:
        if parts[3].lower() in ("none", "null"):
            clip = None
        else:
            try:
                clip = float(parts[3])
            except ValueError:
                raise ValueError(f"targets entry {value!r}: its clip must be a number or none") from None
    if clip is not None and not clip > 0:
        raise ValueError(f"targets entry {value!r}: its clip must be positive, or none")
    return parts[0], parts[1], float(parts[2]), clip


def target_label(pos: str, neg: str, alpha: float, clip: float | None) -> str:
    return f"q-{Path(pos).stem}-{Path(neg).stem}-a{alpha:g}-clip{'none' if clip is None else f'{clip:g}'}"


def entries(cfg: DictConfig, row: dict) -> list[tuple[str, str, object]]:
    """(name, results directory, what it is) of every variant, in config order, without touching the disk: an Adapter,
    a spec file (system_specs) or (pos spec, neg spec, alpha, clip) (targets)."""
    out = [(a.name, a.label, a) for a in parse_adapters(cfg.adapters, row)] if cfg.adapters else []
    for entry in cfg.system_specs or []:
        name, spec = split_entry(entry, "system_specs")
        out.append((name, f"system-{Path(spec).stem}", spec))
    for entry in cfg.targets or []:
        name, value = split_entry(entry, "targets")
        target = parse_target(value, cfg.clip)
        out.append((name, target_label(*target), target))
    if not out:
        raise ValueError("No variants: set adapters, system_specs or targets")
    for position, kind in ((0, "name"), (1, "results directory")):
        values = [e[position] for e in out]
        repeated = sorted({v for v in values if values.count(v) > 1})
        if repeated:
            raise ValueError(f"Two variants share the {kind} {repeated[0]!r}")
    return out


def variants(cfg: DictConfig, row: dict) -> list[Variant]:
    """Every variant, its adapter resolved (offline) and its system texts built."""
    base = resolve(Adapter("base", NO_ADAPTER, BASE_LABEL), row)
    own = default_system(load_tokenizer(row["hub_id"], base.chat_template), base.chat_template, template_kwargs(cfg))

    def system(spec: str) -> str:
        return spec_prompt(spec, row) + own  # the spec frame ends with a blank line

    out = []
    for name, label, what in entries(cfg, row):
        if isinstance(what, Adapter):
            out.append(Variant(name, label, resolve(what, row)))
        elif isinstance(what, str):
            out.append(Variant(name, label, base, system=system(what)))
        else:
            pos, neg, alpha, clip = what
            out.append(Variant(name, label, base, target=Target(system(pos), system(neg), alpha, clip)))
    return out


def template_kwargs(cfg: DictConfig) -> dict:
    kwargs = cfg.chat_template_kwargs
    return {} if kwargs is None else dict(OmegaConf.to_container(kwargs))


def adapter_id(adapter: Adapter) -> str:
    """The adapter as a result's key records it: a local directory relative to the repository however it was written
    (sweep.sh's records are absolute, eval_spp.sh's lines relative), anything else as written."""
    if adapter.spec == NO_ADAPTER or adapter.repo_id is not None or adapter.path is None:
        return adapter.spec
    path = Path(adapter.path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def variant_key(variant: Variant, row: dict, cfg: DictConfig) -> dict:
    """What a variant's scores depend on, beside the benchmark's own settings."""
    from align_distill.teacher_cache import adapter_digest, revision

    adapter = variant.adapter
    weights = None
    if adapter.spec != NO_ADAPTER:
        weights = adapter.revision or adapter_digest(adapter.path)
    target = variant.target
    return {"model": row["hub_id"], "model_revision": revision(row["hub_id"]), "adapter": adapter_id(adapter),
            "adapter_weights": weights, "chat_template": adapter.template_sha,
            "chat_template_kwargs": template_kwargs(cfg), "system": sha(variant.system),
            "target": None if target is None else {"pos": sha(target.pos), "neg": sha(target.neg),
                                                   "alpha": target.alpha, "clip": target.clip},
            "dtype": cfg.scoring.dtype}


# Results

def result_dir(cfg: DictConfig, row: dict, label: str, benchmark: str) -> Path:
    return resolve_path(cfg.out_dir) / row["tag"] / label / benchmark


def finished(directory: Path, key: dict, overwrite: bool) -> bool:
    """Whether the result exists with these settings; a result made with other settings stops the run, or with
    overwrite is deleted."""
    meta = directory / "meta.json"
    if meta.exists():
        saved = load_json(meta)["key"]
        if saved == key:
            return True
        changed = sorted(k for k in set(saved) | set(key) if saved.get(k) != key.get(k))
        if not overwrite:
            raise RuntimeError(f"{directory} was scored with other settings ({', '.join(changed)}); set "
                               "overwrite=true to redo it, or give the run another out_dir")
        logger.warning(f"{directory}: settings changed ({', '.join(changed)}); scoring it again")
    if directory.exists():
        shutil.rmtree(directory)
    return False


def benchmark_cfg(cfg: DictConfig, name: str) -> DictConfig:
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {name}; choose from {sorted(BENCHMARKS)}")
    return cfg[name]


def score(cfg: DictConfig, row: dict) -> None:
    i, n = parse_shard(cfg.shard)
    chosen = variants(cfg, row)[i::n]
    items = {name: BENCHMARKS[name].load(benchmark_cfg(cfg, name)) for name in cfg.benchmarks}
    bench_keys = {name: {"benchmark": name, "protocol": BENCHMARKS[name].protocol(benchmark_cfg(cfg, name)),
                         "dataset": OmegaConf.to_container(benchmark_cfg(cfg, name)),
                         "items": sha([BENCHMARKS[name].item_id(x) for x in items[name]])}
                  for name in cfg.benchmarks}
    todo = []
    for variant in sorted(chosen, key=lambda v: v.adapter.spec != NO_ADAPTER):  # the base model before any adapter
        key = variant_key(variant, row, cfg)
        names = [name for name in cfg.benchmarks
                 if not finished(result_dir(cfg, row, variant.label, name), {**key, **bench_keys[name]},
                                 cfg.overwrite)]
        if names:
            todo.append((variant, key, names))
    logger.info(f"{sum(len(t[2]) for t in todo)} results to score, {len(chosen)} variants in this shard ({i}/{n})")
    if not todo:
        return
    scorer = Scorer(row["hub_id"], cfg.scoring.dtype, cfg.scoring.device_map,
                    max_batch_tokens=cfg.scoring.max_batch_tokens, max_batch_size=cfg.scoring.max_batch_size,
                    chunk_size=cfg.scoring.chunk_size, min_prefix=cfg.scoring.min_prefix)
    tokenizers, questions = {}, {}
    kwargs = template_kwargs(cfg)
    for variant, key, names in todo:
        template = variant.adapter.chat_template
        if template not in tokenizers:
            tokenizers[template] = load_tokenizer(row["hub_id"], template)
        tokenizer = tokenizers[template]
        scorer.pad_id = tokenizer.pad_token_id
        for name in names:
            bench = BENCHMARKS[name]
            if (name, template) not in questions:
                questions[name, template] = bench.questions(items[name], tokenizer, template, kwargs)
            start = time.time()
            scored = scorer.score(variant, questions[name, template], tokenizer, template, kwargs)
            rows = bench.rows(items[name], scored)
            metrics = bench.metrics(items[name], rows)
            directory = result_dir(cfg, row, variant.label, name)
            write_jsonl(directory / "rows.jsonl", rows)
            save_json(directory / "metrics.json", metrics)
            save_json(directory / "meta.json", {"name": variant.name, "key": {**key, **bench_keys[name]},
                                                "seconds": round(time.time() - start, 1)})
            logger.info(f"{variant.name} · {bench.TITLE}: {bench.PRIMARY} {stats.pct(metrics[bench.PRIMARY])}%, "
                        f"format mass {stats.pct(metrics['format_mass'])}% ({time.time() - start:.0f} s)")


# Report

def load_results(cfg: DictConfig, row: dict) -> dict[tuple[str, str], dict]:
    """(variant name, benchmark) -> its saved metrics and per-item outcomes, for every finished result."""
    results = {}
    for name, label, _ in entries(cfg, row):
        for benchmark in cfg.benchmarks:
            directory = result_dir(cfg, row, label, benchmark)
            if (directory / "meta.json").exists():
                results[name, benchmark] = {"metrics": load_json(directory / "metrics.json"),
                                            "outcomes": BENCHMARKS[benchmark].outcomes(
                                                read_jsonl(directory / "rows.jsonl"))}
    return results


def interval(low: float, high: float) -> str:
    return f"[{stats.pct(low)}, {stats.pct(high)}]"


def compare(results: dict, first: str, second: str, benchmark: str, cfg: DictConfig) -> dict:
    a, b = results[first, benchmark]["outcomes"], results[second, benchmark]["outcomes"]
    if set(a) != set(b):
        raise RuntimeError(f"{first} and {second} were scored on different {benchmark} items")
    ids = sorted(a)
    return stats.paired_binary([a[k] for k in ids], [b[k] for k in ids], cfg.n_bootstrap, cfg.bootstrap_seed)


def report(cfg: DictConfig, row: dict) -> dict:
    results = load_results(cfg, row)
    names = [name for name, _, _ in entries(cfg, row)]
    summary = {"model": row["tag"], "metrics": {}, "comparisons": []}
    for benchmark in cfg.benchmarks:
        bench = BENCHMARKS[benchmark]
        lines = []
        for name in names:
            if (name, benchmark) not in results:
                continue
            result = results[name, benchmark]
            values = list(result["outcomes"].values())
            rate, low, high = stats.bootstrap_rate(values, cfg.n_bootstrap, cfg.bootstrap_seed)
            lines.append([name, f"{stats.pct(rate)} {interval(low, high)}", *bench.table_row(result["metrics"])])
            summary["metrics"].setdefault(benchmark, {})[name] = {**result["metrics"], "interval": [low, high]}
        missing = [name for name in names if (name, benchmark) not in results]
        primary = bench.PRIMARY_TITLE
        print(f"\n{bench.TITLE}: {primary} [95% CI over items]" + (f"; not scored yet: {', '.join(missing)}"
                                                                   if missing else ""))
        if lines:
            print(stats.table(["Model", primary, *bench.TABLE_HEADERS], lines))
        if benchmark == airisk.NAME and lines:
            print("\nAIRisk value Elo (SPP's bootstrap): highest and lowest three value classes")
            elo_lines = []
            for name in names:
                if (name, benchmark) in results:
                    ranking = results[name, benchmark]["metrics"]["value_elo"]
                    elo_lines.append([name, ", ".join(f"{r['value_class']} {r['elo']:.0f}" for r in ranking[:3]),
                                      ", ".join(f"{r['value_class']} {r['elo']:.0f}" for r in ranking[-3:])])
            print(stats.table(["Model", "Highest", "Lowest"], elo_lines))
    comparison_lines, left_out = [], []
    for comparison in cfg.comparisons or []:
        first, sep, second = str(comparison).partition(" vs ")
        if not sep:
            raise ValueError(f"comparison {comparison!r} is not '<first> vs <second>'")
        for benchmark in cfg.benchmarks:
            if (first, benchmark) not in results or (second, benchmark) not in results:
                left_out.append(f"{comparison} ({benchmark})")
                continue
            c = compare(results, first, second, benchmark, cfg)
            summary["comparisons"].append({"first": first, "second": second, "benchmark": benchmark, **c})
            comparison_lines.append([comparison, BENCHMARKS[benchmark].TITLE,
                                     f"{100 * c['difference']:+.1f} [{100 * c['low']:+.1f}, {100 * c['high']:+.1f}]",
                                     c["only_first"], c["only_second"], f"{c['p']:.3g}"])
    if comparison_lines:
        print("\nComparisons, paired by item: first minus second, in points (AIRisk: over the dilemmas that offer a "
              "risky action), with McNemar's exact p")
        print(stats.table(["Comparison", "Benchmark", "Difference [95% CI]", "Only first", "Only second", "p"],
                          comparison_lines))
    if left_out:
        print("Comparisons left out, a model has no result: " + "; ".join(left_out))
    path = resolve_path(cfg.out_dir) / row["tag"] / f"summary_{datetime.now():%Y%m%d_%H%M%S}.json"
    save_json(path, {"config": OmegaConf.to_container(cfg, resolve=True), **summary})
    print(f"\nSaved {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return summary


def check_config(cfg: DictConfig) -> None:
    if cfg.stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, not {cfg.stage}")
    if not cfg.benchmarks:
        raise ValueError("benchmarks is empty")
    for name in cfg.benchmarks:
        benchmark_cfg(cfg, name)
    if cfg.shard is not None and cfg.stage != "score":
        raise ValueError("shard is for stage=score; the report reads every shard's results")
    parse_shard(cfg.shard)


def run(cfg: DictConfig, row: dict) -> dict | None:
    check_config(cfg)
    entries(cfg, row)
    if cfg.stage in ("score", "all"):
        score(cfg, row)
    if cfg.stage in ("report", "all"):
        return report(cfg, row)
    return None


@hydra.main(version_base=None, config_path="../conf", config_name="eval_spp")
def main(cfg: DictConfig) -> None:
    run(cfg, model_row(cfg.model))


if __name__ == "__main__":
    main()
