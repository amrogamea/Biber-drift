"""Figures. Every function returns a matplotlib ``Figure``; :func:`save_figure` writes it."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from .features.dimensions import DIMENSION_LABELS

REFERENCE_COLOR = "tab:orange"
COHORT_COLOR = "tab:blue"


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 180) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cohort_sizes(counts: pd.Series, title: str = "Documents per year") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(counts)), 5))
    counts.plot(kind="bar", ax=ax, color=COHORT_COLOR)
    ax.set_xlabel("Year")
    ax.set_ylabel("Documents")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_mmd_heatmap(matrix: pd.DataFrame, title: str, annotate_up_to: int = 20) -> plt.Figure:
    """Symmetric cohort x cohort MMD^2 matrix; cells are annotated for up to 20 cohorts."""
    side = max(8, 0.6 * len(matrix))
    fig, ax = plt.subplots(figsize=(side, side))
    image = ax.imshow(matrix.to_numpy())
    labels = [str(c) for c in matrix.columns]
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title(title)
    if len(matrix) <= annotate_up_to:
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                ax.text(j, i, f"{matrix.iloc[i, j]:.4f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="MMD²")
    fig.tight_layout()
    return fig


INCREASE_COLOR = "tab:blue"
DECREASE_COLOR = "tab:orange"


def plot_feature_shifts(shifts: pd.DataFrame, pairs: Sequence[tuple], top: int = 20) -> plt.Figure:
    """Largest marginal Wasserstein distances, one panel per cohort pair (e.g. 2009->2019, 2019->2025).

    The panels share the x axis, so the size of the shifts can be compared across steps; bar colour
    shows whether the feature's mean went up or down.
    """
    fig, axes = plt.subplots(1, len(pairs), figsize=(6.5 * len(pairs), max(4, 0.4 * top)),
                             sharex=True, squeeze=False)
    for ax, (cohort_a, cohort_b) in zip(axes.flat, pairs):
        pair = shifts[(shifts["cohort_a"] == cohort_a) & (shifts["cohort_b"] == cohort_b)]
        pair = pair.nlargest(top, "wasserstein_distance").sort_values("wasserstein_distance")
        colors = np.where(pair["mean_difference_b_minus_a"] >= 0, INCREASE_COLOR, DECREASE_COLOR)
        ax.barh(pair["feature"], pair["wasserstein_distance"], color=colors)
        ax.set_title(f"{cohort_a} → {cohort_b}")
        ax.set_xlabel("Marginal Wasserstein distance (standardised units)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=INCREASE_COLOR), plt.Rectangle((0, 0), 1, 1, color=DECREASE_COLOR)]
    fig.legend(handles, ["mean increased", "mean decreased"], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Largest Biber-feature shifts")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def _its_fit(row: pd.Series, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fitted values and the pre-break trend continued (counterfactual) at times ``t`` since the break."""
    trend = row["intercept"] + row["pre_slope"] * t
    slope = 0.0 if pd.isna(row["slope_change"]) else row["slope_change"]
    post = (t >= -1e-9).astype(float)
    return trend + post * (row["level_change"] + slope * t), trend


def plot_trend_level_changes(its: pd.DataFrame, top: int = 25) -> plt.Figure:
    """Jump at the break beyond the pre-break trend, with its confidence interval, per feature."""
    table = its[its["kind"] == "feature"]
    table = table.reindex(table["level_t"].abs().nlargest(top).index).iloc[::-1]
    significant = table["level_p_adjusted"] < 0.05
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(table))))
    y = np.arange(len(table))
    ax.errorbar(table["level_change"], y, xerr=[table["level_change"] - table["level_ci_lower"],
                                                 table["level_ci_upper"] - table["level_change"]],
                fmt="none", ecolor="0.6", elinewidth=1)
    ax.scatter(table["level_change"][significant], y[significant.to_numpy()], color=COHORT_COLOR, zorder=3,
               label="adjusted p < 0.05")
    ax.scatter(table["level_change"][~significant], y[~significant.to_numpy()], facecolor="white",
               edgecolor=COHORT_COLOR, zorder=3, label="not significant after adjustment")
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_yticks(y, table["series"])
    ax.set_xlabel("Jump at the break beyond the pre-break trend (standard deviations)")
    ax.set_title("Interrupted time series: largest level changes (95% intervals not adjusted)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_trend_series(series: pd.DataFrame, its: pd.DataFrame, break_at: float, names: Sequence[str],
                      title: str = "Interrupted time series", ncols: int = 4) -> plt.Figure:
    """Per-period values with the fitted segmented trend and the pre-break trend continued (dashed)."""
    rows = its.set_index("series")
    names = [n for n in names if n in series.columns and n in rows.index]
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.3 * nrows), squeeze=False)
    time = series["time"].to_numpy(dtype=float)
    t = time - break_at
    dense = len(series) > 40
    for ax, name in zip(axes.flat, names):
        ax.plot(time, series[name], "o", ms=2.5 if dense else 5, color="0.55", alpha=0.6 if dense else 1)
        fitted, counterfactual = _its_fit(rows.loc[name], t)
        before, after = t < -1e-9, t >= -1e-9
        ax.plot(time[before], fitted[before], color=COHORT_COLOR, lw=2)
        ax.plot(time[after], fitted[after], color=COHORT_COLOR, lw=2)
        ax.plot(time[after], counterfactual[after], color=COHORT_COLOR, lw=1.5, ls="--")
        ax.axvline(break_at, color=REFERENCE_COLOR, ls=":", lw=1.2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        row = rows.loc[name]
        detail = f"jump {row['level_change']:+.2f} (adj. p {row['level_p_adjusted']:.3f})"
        if pd.notna(row["slope_change"]):
            detail += f", slope {row['slope_change']:+.2f}/yr"
        ax.set_title(f"{name}\n{detail}", fontsize=9)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.suptitle(f"{title}\n(solid: fitted; dashed: pre-break trend continued; dotted: break)")
    fig.tight_layout()
    return fig


def plot_stability(table: pd.DataFrame, confidence: float = 95.0, ncols: int = 3) -> plt.Figure:
    """One panel per group: CI band of within-group MMD^2 against sample size."""
    groups = list(dict.fromkeys(table["group"]))
    ncols = max(1, min(ncols, len(groups)))
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for ax, group in zip(axes.flat, groups):
        sub = table[table["group"] == group]
        ax.fill_between(sub["n"], sub["ci_lower"], sub["ci_upper"], alpha=0.6, color="tan")
        ax.axhline(0, ls="--", color=COHORT_COLOR, lw=1.2)
        ax.set_title(f"{confidence:g}% Confidence Interval Channel {group} "
                     f"(n_docs={int(sub['documents'].iloc[0])})")
        ax.set_xlabel("Sample Size n")
        ax.set_ylabel("MMD$^2$")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(groups):]:
        fig.delaxes(ax)
    fig.tight_layout()
    return fig


def plot_dimension_violins(scores: pd.DataFrame, x: str, order: Sequence, dimensions: Sequence[str],
                           hue: str | None = None, hue_order: Sequence | None = None,
                           highlight=None, ncols: int = 2, title: str | None = None) -> plt.Figure:
    """Grid of violins, one panel per dimension (Figure 13 of Nieth et al.).

    Without ``hue`` every level is drawn in one colour, except an optional ``highlight`` level
    in a contrasting colour; with ``hue`` the levels of a second factor are drawn side by side.
    """
    order = [str(o) for o in order]
    data = scores.assign(**{x: scores[x].astype(str)})
    if hue is not None:
        data[hue] = data[hue].astype(str)
        hue_order = [str(h) for h in hue_order] if hue_order is not None else None

    nrows = int(np.ceil(len(dimensions) / ncols))
    width = max(9.0, 0.55 * len(order) + 1) if hue is None else max(9.0, 1.6 * len(order))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, 4.5 * nrows), squeeze=False)
    for i, (ax, dimension) in enumerate(zip(axes.flat, dimensions)):
        if hue is None:
            palette = {level: (REFERENCE_COLOR if level == str(highlight) else COHORT_COLOR)
                       for level in order}
            sns.violinplot(data=data, x=x, y=dimension, order=order, hue=x, hue_order=order,
                           palette=palette, legend=False, ax=ax, cut=0, linewidth=0.8)
        else:
            sns.violinplot(data=data, x=x, y=dimension, order=order, hue=hue, hue_order=hue_order,
                           ax=ax, cut=0, linewidth=0.8)
            if i and ax.get_legend() is not None:
                ax.get_legend().remove()
        positive, negative = DIMENSION_LABELS.get(dimension, (None, None))
        ax.set_title(dimension.replace("_", " "))
        ax.set_xlabel(x)
        ax.set_ylabel(f"Dimension {dimension[3]}")
        if positive:
            ax.text(0.01, 0.98, positive, transform=ax.transAxes, ha="left", va="top",
                    style="italic", fontsize=8)
        if negative:
            ax.text(0.01, 0.02, negative, transform=ax.transAxes, ha="left", va="bottom",
                    style="italic", fontsize=8)
        ax.tick_params(axis="x", rotation=45 if len(order) > 6 else 0)
        ax.grid(axis="y", alpha=0.3)
    for ax in list(axes.flat)[len(dimensions):]:
        fig.delaxes(ax)
    if title:
        fig.suptitle(title, y=1.01, fontsize=15)
    fig.tight_layout()
    return fig


def plot_quality_metrics(metrics: pd.DataFrame, thresholds: dict[str, float]) -> plt.Figure:
    """Histogram of every malformed-text metric (log scale) with its threshold."""
    names = list(thresholds)
    ncols = 4
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for ax, metric in zip(axes.flat, names):
        values = metrics[metric]
        threshold = thresholds[metric]
        upper = max(values.quantile(0.999), threshold * 1.5, 1e-9)
        ax.hist(values.clip(upper=upper), bins=80, color=COHORT_COLOR, alpha=0.8)
        ax.axvline(threshold, color="tab:red", ls="--", lw=1.5, label=f"threshold = {threshold:g}")
        ax.set_yscale("log")
        ax.set_title(f"{metric}  ({int((values > threshold).sum())} docs above)")
        ax.set_ylabel("documents (log)")
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(names):]:
        fig.delaxes(ax)
    fig.suptitle("Malformed-text metrics (values above the 99.9th percentile are clipped)", fontsize=14)
    fig.tight_layout()
    return fig


def plot_variance(table: pd.DataFrame, confidence: float = 95.0, title: str | None = None) -> plt.Figure:
    """Total variance of standardised features per cohort, resampled at one common size."""
    pooled = table[table["cohort"] == "all documents"]
    rows = table[(table["cohort"] != "all documents") & table["mean_total_variance"].notna()]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(rows) + 2), 4.5))
    x = np.arange(len(rows))
    ax.errorbar(x, rows["mean_total_variance"],
                yerr=[rows["mean_total_variance"] - rows["ci_lower"], rows["ci_upper"] - rows["mean_total_variance"]],
                fmt="o", color=COHORT_COLOR, ecolor=COHORT_COLOR, capsize=3)
    if not pooled.empty and pd.notna(pooled["mean_total_variance"].iloc[0]):
        ax.axhline(pooled["mean_total_variance"].iloc[0], ls="--", color=REFERENCE_COLOR, lw=1.2,
                   label="all documents")
        ax.legend(fontsize=8)
    ax.set_xticks(x, rows["cohort"].astype(str), rotation=45 if len(rows) > 6 else 0)
    n = int(rows["sample_size"].iloc[0]) if len(rows) else 0
    ax.set_ylabel(f"Sum of feature variances (n = {n}, {confidence:g}% CI)")
    ax.set_title(title or "Total variance of standardised features")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_flag_rates(rates: pd.DataFrame, columns: Sequence[str] | None = None) -> plt.Figure:
    columns = list(columns or [c for c in rates.columns if c.startswith("flag_")])
    fig, ax = plt.subplots(figsize=(14, 5))
    for column in columns:
        ax.plot(rates.index, rates[column], marker="o", label=column)
    ax.set_xlabel("Year")
    ax.set_ylabel("% of documents flagged")
    ax.set_title("Malformed-text flag rate by year")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig
