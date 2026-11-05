"""The teacher caches of spec context distillation read crosswise (docs/spec_context_distill.md): each spec on the
documents written under it and on the documents written under the other.

Each continuation of a minimal pair was written by a writer reading one spec: D+ (pos) under S+, D- (neg) under S-.
On a document, the own spec is the one its writer read and the cross spec the other. The opening was written without
a spec and is the same in both documents of a pair: there neither spec is own, and it serves as a control. The mirror
cache scores every document after S+ and after S-, and the neutral cache scores it with the IT teacher without a spec,
so every position has four log-probabilities of the token the document has there:
    base  p0, the base model (the adapter switched off)
    none  the IT teacher without a spec (the neutral cache's reference)
    S+    the IT teacher after S+
    S-    the IT teacher after S- (the mirror cache's reference)
Each spec's effect is read against the teacher without a spec: e_S = log p_S - log p_none.

1. Four teachers x two branches: the mean log-probability per token, per branch and region (opening, decision,
   after); the spec effects, the part both specs share, (e(S+) + e(S-)) / 2, and own - cross; and own and cross by
   group. own - cross is log p+ - log p- on D+ and its negative on D-: the contrast toward the document's value, which
   q applies to the document's token.
2. Own vs cross, token by token, on the continuation. Positions whose own - cross is beyond threshold are sorted by
   which teacher moved, up or down beyond band, or flat:
       own        the own spec raises the token, the cross spec is flat: copying the writer's spec, or the own value
                  in words the other spec does not object to (the tokens tell which)
       cross      the cross spec lowers it, the own spec is flat: disapproval
       both       the own spec raises it and the cross spec lowers it
       both_up    both raise it, the own spec more
       both_down  both lower it, the cross spec more
       against    own - cross below -threshold: the cross spec predicts the document's token better
   with band at most threshold / 2, every position beyond threshold has exactly one class. Per branch and class: the
   share of the continuation's positions, of its contrast |own - cross| (clipped as q clips it) and of its KL(q || p0)
   (the KD mass at those positions); the same shares by group; and each class's tokens with the most contrast.

CPU only; nothing is recomputed. Outputs, in <out_dir>/<time>/: summary.json (the tables as data) and spec_cross.log.
The tables are printed at the end.

Usage (scripts/msm/distill_spec_cross.sh chooses the settings):
    python -m msm.distill_spec_cross name=main
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from msm.datagen.distill_splits import SPLITS
from msm.datagen.utils import load_json, save_json
from msm.distill_probe import format_rows, pct
from msm.distill_teacher_cache import BRANCHES, GROUPS, OBSERVED, REGIONS, load_cache, mean, row_docs
from msm.train_common import load_tokenizer
from msm.train_data import resolve_path
from msm.train_distill import resolve_cache

TEACHERS = ("base", "none", "S+", "S-")
TEACHER_LABELS = {"base": "base (p0)", "none": "IT, no spec", "S+": "IT + S+", "S-": "IT + S-"}
OWN = {"pos": "S+", "neg": "S-"}  # the spec each branch's writer read
EFFECTS = ("S+", "S-", "shared", "own-cross")
EFFECT_LABELS = {"S+": "e(S+)", "S-": "e(S-)", "shared": "shared (e(S+) + e(S-)) / 2",
                 "own-cross": "own - cross"}
CLASSES = ("own", "cross", "both", "both_up", "both_down", "against", "below")
CLASS_LABELS = {"own": "own up, cross flat", "cross": "cross down, own flat", "both": "own up, cross down",
                "both_up": "both up, own more", "both_down": "both down, cross more",
                "against": "against the document", "below": "within the threshold"}
# The fields of the two caches' keys that must agree for their rows to be the same positions under the same models
SAME_FIELDS = ("model", "model_revision", "tokenizer", "teacher_adapter", "adapter_revision", "layout", "dtype",
               "anchor", "max_seq_len", "documents")


# The caches

def check_caches(mirror: dict, neutral: dict) -> None:
    """The two caches' meta.json: a mirror and a neutral affordability cache of the same documents, models and S+."""
    if mirror.get("reference", "mirror") != "mirror" or neutral.get("reference") != "neutral":
        raise ValueError(f"mirror= must name a mirror cache and neutral= a neutral one, not {mirror['key']} "
                         f"({mirror.get('reference', 'mirror')}) and {neutral['key']} ({neutral.get('reference')})")
    for meta in (mirror, neutral):
        if meta.get("direction", "affordability") != "affordability":
            raise ValueError(f"{meta['key']} is an {meta['direction']} cache, whose S+ reads the other spec: use the "
                             "affordability caches")
    differ = [name for name in SAME_FIELDS if mirror["fields"].get(name) != neutral["fields"].get(name)]
    differ += ["specs.pos"] * (mirror["fields"]["specs"]["pos"] != neutral["fields"]["specs"]["pos"])
    if differ:
        raise ValueError(f"{mirror['key']} and {neutral['key']} differ in {', '.join(differ)}")


def load_rows(mirror_dir: Path, neutral_dir: Path, splits, tolerance: float
              ) -> tuple[list[dict], dict[str, np.ndarray], dict[str, float]]:
    """The documents of the splits and, per row of theirs: the four teachers' log-probabilities of the row's own token,
    KL(q || p0) of the mirror target, the region, branch, document (an index into the documents returned), row within
    the document and token. Also the largest differences in p0 and S+, which both caches hold and must agree on."""
    documents, mirror = load_cache(mirror_dir)
    neutral_documents, neutral = load_cache(neutral_dir)
    if [(d["doc_id"], d["n_rows"]) for d in documents] != [(d["doc_id"], d["n_rows"]) for d in neutral_documents]:
        raise ValueError(f"{mirror_dir.name} and {neutral_dir.name} do not hold the same documents in the same order")
    column = OBSERVED.index
    observed, reference = mirror["observed"], neutral["observed"]
    diffs = {name: float(np.abs(observed[:, column(name)] - reference[:, column(name)]).max())
             for name in ("logp0", "logp_pos")}
    if max(diffs.values()) > tolerance:
        raise ValueError(f"p0 and S+ differ between {mirror_dir.name} and {neutral_dir.name} by up to "
                         f"{max(diffs.values()):.4f} nats, more than tolerance={tolerance:g}")

    kept = np.array([d["split"] in splits for d in documents])
    doc_of = row_docs(documents)
    keep = kept[doc_of]
    rows = {"base": observed[keep, column("logp0")], "none": reference[keep, column("logp_neg")],
            "S+": observed[keep, column("logp_pos")], "S-": observed[keep, column("logp_neg")],
            "kl": observed[keep, column("kl_q_p0")], "region": np.asarray(mirror["region"])[keep],
            "doc": (np.cumsum(kept) - 1)[doc_of][keep],
            "branch": np.array([BRANCHES.index(d["branch"]) for d in documents])[doc_of][keep],
            "row": np.concatenate([np.arange(d["n_rows"]) for d in documents])[keep],
            "token": np.concatenate([np.asarray(d["input_ids"][1:]) for d in documents])[keep]}
    return [d for d, k in zip(documents, kept) if k], rows, diffs


# Part 1: four teachers x two branches

def cluster_mean(values: np.ndarray, doc: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """The mean of values over the masked rows and its standard error, each document one cluster: the positions of a
    document are not independent."""
    if not mask.any():
        return float("nan"), float("nan")
    docs, at = np.unique(doc[mask], return_inverse=True)
    sums, counts = np.bincount(at, weights=values[mask]), np.bincount(at).astype(float)
    m, k = sums.sum() / counts.sum(), len(docs)
    if k < 2:
        return float(m), float("nan")
    return float(m), float(np.sqrt(((sums - m * counts) ** 2).sum() * k / (k - 1)) / counts.sum())


def effects(rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per row, against the teacher without a spec: e(S+), e(S-), the part they share, and the own and cross spec's
    effects and own - cross."""
    e_pos, e_neg = rows["S+"] - rows["none"], rows["S-"] - rows["none"]
    pos = rows["branch"] == BRANCHES.index("pos")
    own, cross = np.where(pos, e_pos, e_neg), np.where(pos, e_neg, e_pos)
    return {"S+": e_pos, "S-": e_neg, "shared": (e_pos + e_neg) / 2, "own": own, "cross": cross,
            "own-cross": own - cross}


def cells(rows: dict[str, np.ndarray]):
    """The columns of part 1: each branch's regions, with the rows of each."""
    for b, branch in enumerate(BRANCHES):
        for r, region in enumerate(REGIONS):
            yield branch, region, (rows["branch"] == b) & (rows["region"] == r)


def teacher_table(rows: dict[str, np.ndarray]) -> list[dict]:
    """The mean log-probability of the document's own token per teacher, branch and region, and the positions."""
    table = [{"teacher": t, **{f"{b}/{r}": mean(rows[t][m]) for b, r, m in cells(rows)}} for t in TEACHERS]
    return table + [{"teacher": "positions", **{f"{b}/{r}": int(m.sum()) for b, r, m in cells(rows)}}]


def effect_table(rows: dict[str, np.ndarray], e: dict[str, np.ndarray]) -> list[dict]:
    """The spec effects per branch and region, with standard errors over documents. own - cross is not defined on the
    opening, which neither spec's writer wrote."""
    table = []
    for name in EFFECTS:
        entry = {"effect": name}
        for b, r, m in cells(rows):
            mu, se = cluster_mean(e[name], rows["doc"], m) if name != "own-cross" or r != "prefix" else (None, None)
            entry[f"{b}/{r}"] = {"mean": mu, "se": se}
        table.append(entry)
    return table


def groups_of(documents: list[dict]) -> list[tuple[str, np.ndarray]]:
    """The cache checks' groups, then one per framing before all: the documents of each."""
    framings = sorted({d["framing"] for d in documents})
    groups = [(label, member) for label, member in GROUPS if label != "all"]
    groups += [(framing, lambda d, f=framing: d["framing"] == f) for framing in framings]
    groups += [(label, member) for label, member in GROUPS if label == "all"]
    return [(label, np.array([member(d) for d in documents])) for label, member in groups]


def group_table(documents: list[dict], rows: dict[str, np.ndarray], e: dict[str, np.ndarray]) -> list[dict]:
    """The own and cross spec effects in the decision and after regions, per group and branch."""
    table = []
    for label, members in groups_of(documents):
        for b, branch in enumerate(BRANCHES):
            in_group = members[rows["doc"]] & (rows["branch"] == b)
            docs = int(sum(bool(m) and d["branch"] == branch for m, d in zip(members, documents)))
            if not docs:
                continue
            entry = {"group": label, "branch": branch, "docs": docs}
            for region in ("decision", "after"):
                m = in_group & (rows["region"] == REGIONS.index(region))
                entry |= {f"own/{region}": mean(e["own"][m]), f"cross/{region}": mean(e["cross"][m])}
            table.append(entry)
    return table


# Part 2: own vs cross, token by token

def classify(own: np.ndarray, cross: np.ndarray, threshold: float, band: float) -> np.ndarray:
    """Each row's index into CLASSES. With band <= threshold / 2, every row beyond threshold has exactly one class."""
    up_own, down_own, up_cross, down_cross = own > band, own < -band, cross > band, cross < -band
    flat_own, flat_cross = ~up_own & ~down_own, ~up_cross & ~down_cross
    toward = own - cross
    strong = toward > threshold
    classes = np.full(len(own), CLASSES.index("below"), dtype=np.int8)
    for name, member in (("own", up_own & flat_cross), ("cross", flat_own & down_cross), ("both", up_own & down_cross),
                         ("both_up", up_own & up_cross), ("both_down", down_own & down_cross)):
        classes[strong & member] = CLASSES.index(name)
    classes[toward < -threshold] = CLASSES.index("against")
    return classes


def class_table(rows: dict[str, np.ndarray], e: dict[str, np.ndarray], classes: np.ndarray,
                contrast: np.ndarray) -> list[dict]:
    """Per branch and class, on the continuation: the share of the positions, of the contrast and of KL(q || p0), and
    the class's mean own and cross effects."""
    table = []
    for b, branch in enumerate(BRANCHES):
        continuation = (rows["branch"] == b) & (rows["region"] > 0)
        n, total, kl = continuation.sum(), contrast[continuation].sum(), rows["kl"][continuation].sum()
        for c, name in enumerate(CLASSES):
            m = continuation & (classes == c)
            table.append({"branch": branch, "class": name, "positions": float(m.sum() / n) if n else float("nan"),
                          "contrast": float(contrast[m].sum() / total) if total else float("nan"),
                          "kl": float(rows["kl"][m].sum() / kl) if kl else float("nan"),
                          "own": mean(e["own"][m]), "cross": mean(e["cross"][m]),
                          "n": int(m.sum()), "of": int(n), "total_contrast": float(total), "total_kl": float(kl)})
    return table


def group_class_table(documents: list[dict], rows: dict[str, np.ndarray], classes: np.ndarray,
                      contrast: np.ndarray) -> list[dict]:
    """Each class's share of the continuation's contrast, per group and branch."""
    table = []
    for label, members in groups_of(documents):
        for b, branch in enumerate(BRANCHES):
            m = members[rows["doc"]] & (rows["branch"] == b) & (rows["region"] > 0)
            total = contrast[m].sum()
            if not total:
                continue
            table.append({"group": label, "branch": branch,
                          **{name: float(contrast[m & (classes == c)].sum() / total) for c, name in enumerate(CLASSES)}})
    return table


def class_tokens(tokenizer, documents: list[dict], rows: dict[str, np.ndarray], e: dict[str, np.ndarray],
                 classes: np.ndarray, contrast: np.ndarray, top: int) -> list[dict]:
    """Per branch and class beyond the threshold, the tokens with the largest share of the branch's continuation
    contrast, their mean own and cross effects, and the context of their position with the most contrast."""
    tokens = []
    for b, branch in enumerate(BRANCHES):
        continuation = (rows["branch"] == b) & (rows["region"] > 0)
        total = contrast[continuation].sum()
        for c, name in enumerate(CLASSES[:-1]):
            at = np.flatnonzero(continuation & (classes == c))
            if not len(at):
                continue
            ids, which = np.unique(rows["token"][at], return_inverse=True)
            mass, n = np.bincount(which, weights=contrast[at]), np.bincount(which)
            own, cross = np.bincount(which, weights=e["own"][at]), np.bincount(which, weights=e["cross"][at])
            for k in np.argsort(-mass)[:top]:
                members = at[which == k]
                strongest = members[np.argmax(contrast[members])]
                doc, j = documents[rows["doc"][strongest]], int(rows["row"][strongest])
                tokens.append({"branch": branch, "class": name, "token": tokenizer.decode([int(ids[k])]),
                               "share": float(mass[k] / total), "n": int(n[k]), "own": float(own[k] / n[k]),
                               "cross": float(cross[k] / n[k]), "doc": doc["doc_id"],
                               "context": tokenizer.decode(doc["input_ids"][max(1, j - 7):j + 1])})
    return tokens


# The report

def f3(x) -> str:
    return "-" if x is None or x != x else f"{x:+.3f}"


def interval(cell: dict) -> str:
    return "-" if cell["mean"] is None else f"{cell['mean']:+.3f} ± {1.96 * cell['se']:.3f}"


def analyse(tokenizer, documents: list[dict], rows: dict[str, np.ndarray], threshold: float, band: float,
            clip: float | None, top_tokens: int) -> tuple[dict, str]:
    """Both parts, as data and as printed text."""
    e = effects(rows)
    contrast = np.abs(e["own-cross"] if clip is None else np.clip(e["own-cross"], -clip, clip))
    classes = classify(e["own"], e["cross"], threshold, band)
    summary = {"documents": len(documents), "rows": int(len(rows["doc"])), "teachers": teacher_table(rows),
               "effects": effect_table(rows, e), "groups": group_table(documents, rows, e),
               "classes": class_table(rows, e, classes, contrast),
               "group_classes": group_class_table(documents, rows, classes, contrast),
               "tokens": class_tokens(tokenizer, documents, rows, e, classes, contrast, top_tokens)}
    columns = [f"{b}/{r}" for b, r, _ in cells(rows)]
    header = [f"{'D+' if b == 'pos' else 'D-'} {'opening' if r == 'prefix' else r}" for b, r, _ in cells(rows)]

    text = ["1. Four teachers x two branches. Own spec: the one the document's writer read (S+ on D+, S- on D-); cross "
            "spec: the other. The opening, written without a spec and the same in both documents, is the control."]
    table = [["", *header]]
    table += [[TEACHER_LABELS[row["teacher"]], *(f"{row[c]:.3f}" for c in columns)] for row in summary["teachers"][:-1]]
    table.append(["positions", *(f"{summary['teachers'][-1][c]:,}" for c in columns)])
    text.append(f"\n1a. Mean log-probability of the document's own token (nats per token)\n{format_rows(table)}")
    table = [["", *header]]
    table += [[EFFECT_LABELS[row["effect"]], *(interval(row[c]) for c in columns)] for row in summary["effects"]]
    text.append("\n1b. Spec effects against the IT teacher without a spec, e_S = log p_S - log p_none (nats per token, "
                "± 95% interval over documents). shared: what having either spec does; own - cross: the contrast "
                f"toward the document's value (log p+ - log p- on D+, its negative on D-)\n{format_rows(table)}")
    table = [["", "branch", "docs", "own decision", "cross decision", "own after", "cross after"]]
    table += [[row["group"], row["branch"], str(row["docs"]), f3(row["own/decision"]), f3(row["cross/decision"]),
               f3(row["own/after"]), f3(row["cross/after"])] for row in summary["groups"]]
    text.append(f"\n1c. Own and cross spec effects by group (nats per token)\n{format_rows(table)}")

    clipped = "" if clip is None else f", clipped at {clip:g} nats as q clips it"
    text.append(f"\n2. Own vs cross at the continuation's own tokens: positions with |own - cross| > {threshold:g} nats, "
                f"by which teacher moved against the one without a spec (up, down: beyond ±{band:g} nats; flat: "
                f"within). contrast: |own - cross|{clipped}; KL: KL(q || p0) at the position, its KD mass. own up, "
                "cross flat: copying the writer's spec, or the own value in words the other spec does not object to; "
                "cross down, own flat: disapproval; against: own - cross < "
                f"-{threshold:g}, the cross spec predicts the token better.")
    for branch in BRANCHES:
        rows_of = [row for row in summary["classes"] if row["branch"] == branch]
        table = [["class", "positions", "contrast", "KL", "mean own", "mean cross"]]
        table += [[f"{row['class']}: {CLASS_LABELS[row['class']]}", pct(row["positions"]), pct(row["contrast"]),
                   pct(row["kl"]), f3(row["own"]), f3(row["cross"])] for row in rows_of]
        own, cross = OWN[branch], OWN["neg" if branch == "pos" else "pos"]
        text.append(f"\n2{'a' if branch == 'pos' else 'b'}. {'D+' if branch == 'pos' else 'D-'} (own = {own}, cross = "
                    f"{cross}): {rows_of[0]['of']:,} continuation positions, summed contrast "
                    f"{rows_of[0]['total_contrast']:,.0f} nats, summed KL {rows_of[0]['total_kl']:,.0f} nats\n"
                    f"{format_rows(table)}")
    table = [["", "branch", *CLASSES]]
    table += [[row["group"], row["branch"], *(pct(row[name]) for name in CLASSES)] for row in summary["group_classes"]]
    text.append(f"\n2c. Each class's share of the continuation's contrast, by group\n{format_rows(table)}")
    table = [["branch", "class", "token", "share", "n", "own", "cross", "after (its position with the most contrast)"]]
    table += [[row["branch"], row["class"], repr(row["token"]), f"{100 * row['share']:.2f}%", str(row["n"]),
               f3(row["own"]), f3(row["cross"]), repr(row["context"][-60:])] for row in summary["tokens"]]
    text.append(f"\n2d. Per branch and class, the tokens with the largest share of the branch's continuation contrast"
                f"\n{format_rows(table)}")
    return summary, "\n".join(text)


def check_config(cfg: DictConfig) -> None:
    if not cfg.splits or set(cfg.splits) - set(SPLITS):
        raise ValueError(f"splits must come from {SPLITS}, not {list(cfg.splits)}")
    if not 0 < cfg.band <= cfg.threshold / 2:
        raise ValueError(f"band must be in (0, threshold / 2], so that every position beyond the threshold has one "
                         f"class, not {cfg.band} with threshold {cfg.threshold}")
    if cfg.top_tokens < 0 or cfg.tolerance < 0:
        raise ValueError("top_tokens and tolerance must not be negative")


@hydra.main(version_base=None, config_path="conf", config_name="distill_spec_cross")
def main(cfg: DictConfig) -> None:
    check_config(cfg)
    mirror_dir, neutral_dir = resolve_cache(cfg.mirror), resolve_cache(cfg.neutral)
    mirror, neutral = load_json(mirror_dir / "meta.json"), load_json(neutral_dir / "meta.json")
    check_caches(mirror, neutral)
    out_dir = resolve_path(cfg.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.add(out_dir / "spec_cross.log")
    logger.info(f"Teacher caches read crosswise: {mirror_dir} and {neutral_dir}\n{OmegaConf.to_yaml(cfg)}")

    documents, rows, diffs = load_rows(mirror_dir, neutral_dir, set(cfg.splits), cfg.tolerance)
    specs = {name: Path(path).stem for name, path in mirror["config"]["specs"].items()}
    logger.info(f"{len(documents)} documents ({', '.join(cfg.splits)}), {len(rows['doc']):,} positions. S+ = "
                f"{specs['pos']}, S- = {specs['neg']}. p0 and S+ agree between the caches to "
                f"{max(diffs.values()):.2g} nats.")
    tokenizer = load_tokenizer(mirror["fields"]["tokenizer"])
    summary, text = analyse(tokenizer, documents, rows, cfg.threshold, cfg.band, mirror["fields"]["clip"],
                            cfg.top_tokens)
    save_json(out_dir / "summary.json", {"config": OmegaConf.to_container(cfg, resolve=True),
                                         "mirror": str(mirror_dir), "neutral": str(neutral_dir), "specs": specs,
                                         "shared_column_diffs": diffs, **summary})
    logger.info(f"Saved the results to {out_dir}")
    print(text + "\n", flush=True)


if __name__ == "__main__":
    main()
