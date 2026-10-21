"""Figures of a msm/score_docs.py run, drawn from its summary_<time>.json alone, so that they can be redrawn without a
GPU. msm/score_docs.py draws them at the end of every run into <out_dir>/figures_<time>/:
    1_differences.png       each run minus the arm's plain MSM run, per group of documents, with 95% intervals
    2_margin.png            the share of (positive, negative) pairs over the SPO margin, per model, with positives
                            from the arm's own corpus or from the paper's corpus
    3_positions.png         each run minus the reference, per token, by position in the document
    4_tokens.png            the tokens that dropped most on held-out negatives, next to the same tokens on positives
    5_value_separation.png  the change from the base model on the paper's two corpora, the arm's value and the other
    6_survival.png          for AFT runs: each run's difference from MSM before AFT, next to the same difference
                            after AFT (e.g. SPO+AFT - MSM+AFT)

Usage:
    python -m msm.plot_scores outputs/msm/score/summary_<time>.json [figure directory]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Colors: the reference palette of the dataviz method, validated (scripts/validate_palette.js): the first three
# categorical slots pass all-pairs in light mode, so runs are colored in this order and keep their color in every
# figure; the two blue steps pass as a two-step ordinal pair (own value / other value, own corpus / paper corpus).
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DARK, LIGHT = "#1c5cab", "#86b6ef"
RING = dict(markeredgecolor=SURFACE, markeredgewidth=1.5)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "font.family": "sans-serif", "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def references(result: dict) -> dict[str, str]:
    """The reference of each compared run of an arm (summaries from before AFT runs name one reference per arm)."""
    if "references" in result:
        return result["references"]
    reference = result.get("reference")
    return {label: reference for label in result["runs"] if reference and label != reference}


def method(label: str) -> str:
    """A run's label without its AFT stage (SPO+AFT · simpo -> SPO · simpo)."""
    head, *tags = label.split(" · ")
    return " · ".join([head.split("+")[0], *tags])


def run_colors(arms: dict) -> dict[str, str]:
    """A color per method (MSM, SPO, SPO · simpo, ...) in the order the methods first appear, the same before and after
    AFT, in every arm and figure; runs from the base model (AFT alone) are gray, as context."""
    labels = list(dict.fromkeys(label for arm in arms.values() for label in arm["runs"]))
    methods = list(dict.fromkeys(method(label) for label in labels if label.startswith(("MSM", "SPO"))))
    if len(methods) > len(SERIES):
        raise ValueError(f"{len(methods)} methods, but only {len(SERIES)} colors: {methods}")
    return {label: SERIES[methods.index(method(label))] if method(label) in methods else MUTED for label in labels}


def comparisons(arms: dict) -> list[str]:
    """The labels of the runs compared with a reference, in order, over all arms."""
    return list(dict.fromkeys(label for arm in arms.values() for label in references(arm)))


def bold_own_value(ax, result: dict, names: list[str]) -> None:
    for tick, name in zip(ax.get_yticklabels(), names):
        if result["groups"][name]["positive"]:
            tick.set_fontweight("bold")
            tick.set_color(INK)


def grid(n_rows: int, n_cols: int, width: float, height: float, sharex: str | bool = False):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width * n_cols, height * n_rows), squeeze=False,
                             sharex=sharex)
    return fig, axes


def blank(ax, text: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", color=MUTED, transform=ax.transAxes)


def zero_line(ax, vertical: bool = True) -> None:
    (ax.axvline if vertical else ax.axhline)(0, color=AXIS, linewidth=1, zorder=1)


def save(fig, path: Path, title: str, subtitle: str, legend_band: float = 0.0) -> Path:
    """Title and subtitle in a band at the top, an optional band at the bottom (inches) for a figure legend."""
    height = fig.get_figheight()
    fig.text(0.01, 1 - 0.22 / height, title, ha="left", va="center", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 1 - 0.46 / height, subtitle, ha="left", va="center", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, legend_band / height, 1, 1 - 0.66 / height))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_differences(arms: dict, colors: dict, path: Path) -> Path:
    labels = comparisons(arms)
    fig, axes = grid(len(arms), len(labels), 6.2 if len(labels) <= 2 else 5.0,
                     0.26 * max(len(a["groups"]) for a in arms.values()) + 1.4, sharex="col")
    for row, (arm, result) in enumerate(arms.items()):
        names = list(result["groups"])
        y = list(range(len(names)))[::-1]
        for col, label in enumerate(labels):
            ax = axes[row][col]
            if label not in references(result):
                blank(ax, f"no {label} run in the {arm} arm")
                continue
            zero_line(ax)
            for yi, name in zip(y, names):
                group = result["groups"][name]
                d = group["differences"][label]
                ax.plot([d["low"], d["high"]], [yi, yi], color=colors[label], linewidth=2, zorder=2)
                ax.plot(d["mean"], yi, "o", markersize=7, zorder=3, color=colors[label],
                        markerfacecolor=colors[label] if group["trained"] else SURFACE,
                        markeredgecolor=colors[label], markeredgewidth=1.5)
            ax.set_yticks(y, names if col == 0 else [""] * len(names))
            if col == 0:
                bold_own_value(ax, result, names)
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{arm} arm: {label} − {references(result)[label]}", loc="left")
            ax.set_xlabel("difference in mean token log-probability (nats/token)")
            ax.tick_params(labelbottom=True)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=INK_2, markersize=7, label="trained on"),
               plt.Line2D([], [], marker="o", linestyle="", color=INK_2, markerfacecolor=SURFACE, markersize=7,
                          label="held out")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    return save(fig, path, "What each run changed, before AFT", legend_band=0.35, subtitle=
                "Mean per-document difference from the plain MSM run of the same stage (MSM, or MSM+AFT), 95% "
                "bootstrap interval; bold rows are the arm's own value. Columns have their own scale.")


def pooled_margin(result: dict, label: str, positives: str) -> float | None:
    """The share of pairs over the margin, pooled over the negative groups of our corpora (the paper's left out)."""
    pairs = [(result["groups"][name]["n"], shares[label].get(positives))
             for name, shares in result["margin"].items() if not result["groups"][name]["paper"]]
    pairs = [(n, s) for n, s in pairs if s is not None]
    return sum(n * s for n, s in pairs) / sum(n for n, _ in pairs) if pairs else None


def plot_margin(arms: dict, margin: dict, path: Path) -> Path:
    n_models = max(len(next(iter(a["margin"].values()))) for a in arms.values())
    fig, axes = grid(1, len(arms), max(5.2, 0.95 * n_models + 1.4), 3.8)
    for ax, (arm, result) in zip(axes[0], arms.items()):
        models = list(next(iter(result["margin"].values())))
        for x, model in enumerate(models):
            points = [(x + dx, value, color, side) for value, color, dx, side in
                      ((pooled_margin(result, model, "trained"), DARK, -0.12, -1),
                       (pooled_margin(result, model, "paper"), LIGHT, 0.12, 1)) if value is not None]
            ax.plot([p[0] for p in points], [100 * p[1] for p in points], color=AXIS, linewidth=2, zorder=1)
            for px, value, color, side in points:
                ax.plot(px, 100 * value, "o", markersize=9, color=color, zorder=3, **RING)
                if model != "Base":
                    ax.annotate(f"{100 * value:.0f}", (px, 100 * value), xytext=(8 * side, 0),
                                textcoords="offset points", va="center", ha="left" if side > 0 else "right",
                                fontsize=8, color=INK_2)
        ax.set_xticks(range(len(models)), [model.replace(" · ", "\n· ") for model in models])
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.set_ylim(-4, 104)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{arm} arm", loc="left")
        ax.set_ylabel("% of pairs over the margin")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=DARK, markersize=8,
                          label="positives from the arm's own corpus (trained on)"),
               plt.Line2D([], [], marker="o", linestyle="", color=LIGHT, markersize=8,
                          label="positives from the paper's corpus (held out)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    beta, gamma = margin["beta"], margin["gamma"]
    return save(fig, path, "Does MSM already separate positives from negatives?", legend_band=0.35, subtitle=
                f"Share of (positive, negative) document pairs with {beta} × (s+ − s−) > {gamma}, negatives "
                f"pooled over our corpora.")


def plot_positions(arms: dict, colors: dict, path: Path) -> Path:
    labels = comparisons(arms)
    fig, axes = grid(len(arms), len(labels), 5.6, 3.4)
    for col, label in enumerate(labels):
        for row, (arm, result) in enumerate(arms.items()):
            ax = axes[row][col]
            if label not in result.get("profile", {}):
                blank(ax, f"no {label} run in the {arm} arm")
                continue
            profile = result["profile"][label]
            x = [start + 1 for start in profile["starts"]]
            zero_line(ax, vertical=False)
            styles = {"negatives · held out": dict(color=colors[label], markerfacecolor=colors[label]),
                      "negatives · trained": dict(color=colors[label], markerfacecolor=SURFACE),
                      "positives · trained": dict(color=MUTED, markerfacecolor=MUTED)}
            for name, style in styles.items():
                values = profile["groups"].get(name)
                if values:
                    points = [(xi, v) for xi, v in zip(x, values) if v is not None]
                    ax.plot(*zip(*points), "-o", linewidth=2, markersize=5, markeredgewidth=1.5,
                            markeredgecolor=style["color"], label=name, **style)
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 4, 16, 64, 256, 1024], ["1", "4", "16", "64", "256", "1024"])
            ax.set_title(f"{arm} arm: {label} − {references(result)[label]}", loc="left")
            ax.set_xlabel("token position in the document (log scale)")
            ax.set_ylabel("mean per-token difference (nats)")
            ax.legend(loc="lower left")
    return save(fig, path, "Where in the document the likelihood drops",
                "Per-token difference from the plain MSM run of the same stage, averaged in doubling position "
                "buckets. Columns differ in scale.")


def plot_tokens(arms: dict, colors: dict, path: Path) -> Path:
    labels = comparisons(arms)
    n = max((len(t) for a in arms.values() for t in a["tokens"].values()), default=1)
    fig, axes = grid(len(arms), len(labels), 5.4, 0.25 * n + 1.3, sharex="col")
    for col, label in enumerate(labels):
        for row, (arm, result) in enumerate(arms.items()):
            ax = axes[row][col]
            entries = result["tokens"].get(label)
            if not entries:
                blank(ax, f"no {label} run in the {arm} arm")
                continue
            zero_line(ax)
            y = list(range(len(entries)))[::-1]
            for yi, e in zip(y, entries):
                ax.plot([0, e["negatives"]], [yi, yi], color=colors[label], linewidth=2, zorder=2)
                ax.plot(e["negatives"], yi, "o", markersize=7, color=colors[label], zorder=3, **RING)
                if e["positives"] is not None:
                    ax.plot(e["positives"], yi, "o", markersize=6, color=MUTED, zorder=4, **RING)
            ax.set_yticks(y, [repr(e["token"]) for e in entries])
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{arm} arm: {label} − {references(result)[label]}", loc="left")
            ax.set_xlabel("mean difference in log-probability of the token (nats)")
            ax.tick_params(labelbottom=True)
    handles = [plt.Line2D([], [], marker="o", color=INK_2, markersize=7,
                          label="held-out negative documents (in the run's color)"),
               plt.Line2D([], [], marker="o", linestyle="", color=MUTED, markersize=6,
                          label="same token in trained positive documents")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    return save(fig, path, "Which tokens are suppressed", legend_band=0.35, subtitle=
                "Tokens with the largest mean drop on held-out negative documents (seen at least the configured number "
                "of times). Columns differ in scale.")


def plot_value_separation(arms: dict, path: Path, clip: float = -1.2) -> Path:
    fig, axes = grid(1, len(arms), 5.6, 0.55 * max(len(a["runs"]) for a in arms.values()) + 2.0)
    for ax, (arm, result) in zip(axes[0], arms.items()):
        groups = result["groups"]
        own = next((g for g in groups.values() if g["positive"] and g["paper"]), None)
        other_name = next((name for name, g in groups.items() if not g["positive"] and g["paper"]), None)
        if own is None or other_name is None or "Base" not in own["scores"]:
            blank(ax, "needs the base model and both paper corpora")
            continue
        other = groups[other_name]
        models = list(result["runs"])
        y = list(range(len(models)))[::-1]
        zero_line(ax)
        for yi, model in zip(y, models):
            a = own["scores"][model] - own["scores"]["Base"]
            b = other["scores"][model] - other["scores"]["Base"]
            shown = [max(v, clip) for v in (a, b)]
            ax.plot(shown, [yi, yi], color=AXIS, linewidth=2, zorder=1)
            for value, show, color in ((a, shown[0], DARK), (b, shown[1], LIGHT)):
                clipped = value < clip
                ax.plot(show, yi, "<" if clipped else "o", markersize=9, color=color, zorder=3, **RING)
                if clipped:
                    ax.annotate(f"{value:+.2f}", (show, yi), xytext=(0, 9), textcoords="offset points", ha="left",
                                fontsize=8, color=INK_2)
            ax.annotate(f"gap {a - b:+.2f}", (1.0, yi), xycoords=("axes fraction", "data"), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK_2)
        ax.set_yticks(y, models)
        ax.set_xlim(clip - 0.1, 0.7)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{arm} arm", loc="left")
        ax.set_xlabel("change from the base model (nats/token)")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=DARK, markersize=8,
                          label="paper corpus of the arm's own value"),
               plt.Line2D([], [], marker="o", linestyle="", color=LIGHT, markersize=8,
                          label="paper corpus of the other value"),
               plt.Line2D([], [], marker="<", linestyle="", color=INK_2, markersize=8,
                          label=f"beyond {clip}, value annotated")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    return save(fig, path, "Does the value distinction carry over to another writer?", legend_band=0.35, subtitle=
                "Mean per-document change from the base model on the paper's corpora, which no run trained on; the gap "
                "is own value minus other value.")


def plot_survival(arms: dict, path: Path) -> Path:
    labels = list(dict.fromkeys(label for arm in arms.values() for label in arm.get("survival", {})))
    fig, axes = grid(len(arms), len(labels), 6.2, 0.26 * max(len(a["groups"]) for a in arms.values()) + 1.6,
                     sharex="col")
    for col, label in enumerate(labels):
        for row, (arm, result) in enumerate(arms.items()):
            ax = axes[row][col]
            entry = result.get("survival", {}).get(label)
            if not entry:
                blank(ax, f"no {label} run in the {arm} arm")
                continue
            names = list(result["groups"])
            y = list(range(len(names)))[::-1]
            zero_line(ax)
            for yi, name in zip(y, names):
                d = entry["groups"][name]
                ax.plot([d["before"], d["after"]], [yi, yi], color=AXIS, linewidth=2, zorder=1)
                ax.plot(d["before"], yi, "o", markersize=8, color=LIGHT, zorder=3, **RING)
                ax.plot(d["after"], yi, "o", markersize=8, color=DARK, zorder=3, **RING)
            ax.set_yticks(y, names if col == 0 else [""] * len(names))
            if col == 0:
                bold_own_value(ax, result, names)
            ax.grid(axis="y", visible=False)
            ax.tick_params(labelbottom=True)
            pooled = entry["groups"].get("all negatives · held out")
            kept = (f"\nheld-out negatives keep {100 * pooled['after'] / pooled['before']:.0f}% of the difference"
                    if pooled and abs(pooled["before"]) >= 0.05 else "")
            ax.set_title(f"{arm} arm: {entry['before']} → {label}{kept}", loc="left")
            ax.set_xlabel("difference from the plain MSM run (nats/token)")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=LIGHT, markersize=8,
                          label="before AFT (run − MSM)"),
               plt.Line2D([], [], marker="o", linestyle="", color=DARK, markersize=8,
                          label="after AFT (run+AFT − MSM+AFT)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    return save(fig, path, "Does the change survive AFT?", legend_band=0.35, subtitle=
                "Mean per-document difference from the plain MSM run, before and after AFT; bold rows are the arm's "
                "own value. Columns have their own scale.")


def plot_summary(summary: dict, fig_dir: Path) -> list[Path]:
    """Draw every figure of a summary into fig_dir and return their paths."""
    arms = summary["arms"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = run_colors(arms)
    paths = [plot_margin(arms, summary["config"]["margin"], fig_dir / "2_margin.png"),
             plot_value_separation(arms, fig_dir / "5_value_separation.png")]
    if comparisons(arms):
        paths += [plot_differences(arms, colors, fig_dir / "1_differences.png"),
                  plot_positions(arms, colors, fig_dir / "3_positions.png"),
                  plot_tokens(arms, colors, fig_dir / "4_tokens.png")]
    if any(arm.get("survival") for arm in arms.values()):
        paths.append(plot_survival(arms, fig_dir / "6_survival.png"))
    return sorted(paths)


def main(argv: list[str]) -> None:
    if not 1 <= len(argv) <= 2:
        sys.exit(__doc__)
    summary_path = Path(argv[0])
    fig_dir = Path(argv[1]) if len(argv) == 2 else \
        summary_path.parent / summary_path.stem.replace("summary", "figures", 1)
    with open(summary_path, encoding="utf-8") as f:
        paths = plot_summary(json.load(f), fig_dir)
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main(sys.argv[1:])
