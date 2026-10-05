"""Optional one-off plot helpers.

Figure scripts draw their own panels. These helpers cover the documented API.

Example::

    from datapoints_figures import plot_histogram, generate_all_plots
    plot_histogram(values, ax=ax)
    generate_all_plots(values, outdir="reports/scratch", name="tm1")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .style import DATAPOINTS_COLORS, FILL_ALPHA, REFERENCE_LINE_STYLE


def _as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def plot_histogram(values, ax=None, bins=30, **kwargs):
    """Histogram plus a simple KDE overlay."""
    ax = ax or plt.gca()
    x = _as_array(values)
    x = x[np.isfinite(x)]
    color = kwargs.pop("color", DATAPOINTS_COLORS["blue"])
    ax.hist(x, bins=bins, density=True, color=color, alpha=FILL_ALPHA,
            edgecolor="white", linewidth=0.3, **kwargs)
    try:
        from scipy.stats import gaussian_kde
        if x.size >= 4 and np.unique(x).size > 1:
            grid = np.linspace(x.min(), x.max(), 200)
            ax.plot(grid, gaussian_kde(x)(grid), color=color, lw=1.2)
    except Exception:
        pass
    return ax


def plot_sorted_bars(values, ax=None, labels=None, ref=None, **kwargs):
    """Horizontal bars sorted by value, optional vertical reference line(s)."""
    ax = ax or plt.gca()
    x = _as_array(values)
    order = np.argsort(x)
    y = np.arange(x.size)
    color = kwargs.pop("color", DATAPOINTS_COLORS["blue"])
    ax.barh(y, x[order], color=color, edgecolor="white", linewidth=0.3, **kwargs)
    if labels is not None:
        labs = list(labels)
        ax.set_yticks(y)
        ax.set_yticklabels([labs[i] for i in order])
    if ref is not None:
        refs = ref if np.ndim(ref) else [ref]
        for r in refs:
            ax.axvline(r, ls=REFERENCE_LINE_STYLE, color=DATAPOINTS_COLORS["gray"], lw=0.8)
    return ax


def plot_sorted_area(values, ax=None, **kwargs):
    """Area chart of values sorted ascending."""
    ax = ax or plt.gca()
    x = np.sort(_as_array(values))
    color = kwargs.pop("color", DATAPOINTS_COLORS["blue"])
    ax.fill_between(np.arange(x.size), x, color=color, alpha=FILL_ALPHA)
    ax.plot(np.arange(x.size), x, color=color, lw=1.2, **kwargs)
    return ax


def plot_cdf(values, ax=None, refs=None, **kwargs):
    """Empirical CDF, optional vertical reference lines."""
    ax = ax or plt.gca()
    x = np.sort(_as_array(values))
    x = x[np.isfinite(x)]
    y = np.arange(1, x.size + 1) / max(x.size, 1)
    color = kwargs.pop("color", DATAPOINTS_COLORS["blue"])
    ax.plot(x, y, color=color, lw=1.2, **kwargs)
    if refs is not None:
        for r in np.atleast_1d(refs):
            ax.axvline(r, ls=REFERENCE_LINE_STYLE, color=DATAPOINTS_COLORS["gray"], lw=0.8)
    ax.set_ylim(0, 1)
    return ax


def plot_range_counts(values, ax=None, bins=10, **kwargs):
    """Bar chart of histogram bin counts."""
    ax = ax or plt.gca()
    x = _as_array(values)
    x = x[np.isfinite(x)]
    counts, edges = np.histogram(x, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)
    color = kwargs.pop("color", DATAPOINTS_COLORS["blue"])
    ax.bar(centers, counts, width=width, color=color, edgecolor="white",
           linewidth=0.3, **kwargs)
    return ax


def generate_all_plots(values, outdir, name: str = "value") -> list[Path]:
    """Write histogram, sorted-bar, area, CDF, and range-count PNGs.

    Example::

        generate_all_plots(series, outdir="reports/scratch", name="hic")
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix, fn in (
        ("histogram", plot_histogram),
        ("sorted_bars", plot_sorted_bars),
        ("sorted_area", plot_sorted_area),
        ("cdf", plot_cdf),
        ("range_counts", plot_range_counts),
    ):
        fig, ax = plt.subplots()
        fn(values, ax=ax)
        ax.set_title(name)
        path = out / f"{name}_{suffix}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        written.append(path)
    return written
