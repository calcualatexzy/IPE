"""Statistics and console tables shared by the align-distill evals."""

from __future__ import annotations

import math

import numpy as np


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def sem(values) -> float | None:
    """Standard error of the mean (sample standard deviation over sqrt(n))."""
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def bootstrap_mean_rate(counts: list[tuple[int, int]], n_bootstrap: int, seed: int) -> tuple[float, float]:
    """95% interval of the mean rate over settings, resampling the samples within each setting (the settings are the
    eval, not a sample of it). counts: (positives, samples) per setting."""
    rng = np.random.default_rng(seed)
    k, n = np.array(counts, dtype=float).T
    draws = rng.binomial(n.astype(int), k / n, size=(n_bootstrap, len(n))) / n
    low, high = np.percentile(draws.mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def bootstrap_paired_difference(first: list[tuple[int, int]], second: list[tuple[int, int]], n_bootstrap: int,
                                seed: int) -> tuple[float, float, float]:
    """Mean over settings of first's rate minus second's, paired by setting, with a 95% interval that resamples the
    samples within each setting of each model."""
    rng = np.random.default_rng(seed)
    k1, n1 = np.array(first, dtype=float).T
    k2, n2 = np.array(second, dtype=float).T
    d1 = rng.binomial(n1.astype(int), k1 / n1, size=(n_bootstrap, len(n1))) / n1
    d2 = rng.binomial(n2.astype(int), k2 / n2, size=(n_bootstrap, len(n2))) / n2
    low, high = np.percentile((d1 - d2).mean(axis=1), [2.5, 97.5])
    return float((k1 / n1 - k2 / n2).mean()), float(low), float(high)


def cohen_kappa(first: list[bool], second: list[bool]) -> float | None:
    """Agreement of two binary raters beyond chance; None when chance agreement is total."""
    a, b = np.array(first, dtype=bool), np.array(second, dtype=bool)
    if not len(a):
        return None
    observed = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    expected = float(pa * pb + (1 - pa) * (1 - pb))
    return None if expected == 1 else (observed - expected) / (1 - expected)


def pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f}"


def table(headers: list[str], rows: list[list], align: str | None = None) -> str:
    """A plain text table; align holds one character per column, l or r (default: first left, rest right)."""
    cells = [[str(h) for h in headers]] + [["" if c is None else str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    align = align or "l" + "r" * (len(headers) - 1)
    lines = []
    for j, row in enumerate(cells):
        lines.append("  ".join(c.ljust(w) if a == "l" else c.rjust(w) for c, w, a in zip(row, widths, align)))
        if j == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)
