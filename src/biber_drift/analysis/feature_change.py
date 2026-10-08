"""Which Biber features changed most between key years?

For each comparison (consecutive key years, plus first -> last) every feature gets an absolute
change (difference of yearly means) and a relative change (percent of the earlier mean). The
two magnitudes are min-max scaled to [0, 1] within the comparison and combined into one score,
so a feature ranks high only if it moved a lot both in raw units and relative to its level.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def minmax_scale(series: pd.Series) -> pd.Series:
    """Scale to [0, 1]; a constant series maps to 0, an all-NaN series stays NaN."""
    low, high = series.min(), series.max()
    if pd.isna(low) or pd.isna(high):
        return pd.Series(np.nan, index=series.index)
    if high == low:
        return pd.Series(0.0, index=series.index)
    return (series - low) / (high - low)


def percent_change(start: pd.Series, change: pd.Series) -> pd.Series:
    """``change`` as a percentage of ``|start|``; NaN where the start mean is ~0."""
    return pd.Series(
        np.where(start.abs() > 1e-10, change / start.abs() * 100, np.nan), index=start.index
    )


@dataclass
class FeatureChangeResult:
    """Result for one corpus (``group=None``) or one level of the group column."""

    group: str | None
    means_by_year: pd.DataFrame
    """Mean of every feature (rows) in every year present (columns)."""
    changes: pd.DataFrame
    """Means of the key years plus ``change_<a>_to_<b>`` and ``pct_<a>_to_<b>`` per comparison."""
    scaled: dict[str, pd.DataFrame]
    """Per comparison: magnitudes, scaled magnitudes and ``combined_score``."""

    def combined_scores(self) -> pd.DataFrame:
        return pd.DataFrame({name: table["combined_score"] for name, table in self.scaled.items()})

    def top(self, comparison: str, by: str = "combined_score", k: int = 15) -> pd.DataFrame:
        """Top ``k`` features of a comparison by ``combined_score``, ``absolute`` or ``relative``."""
        table = self.scaled[comparison]
        column = {"absolute": "absolute_magnitude", "relative": "relative_magnitude"}.get(by, by)
        return table.sort_values(column, ascending=False).head(k)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory) / (f"group={self.group}" if self.group is not None else "all")
        directory.mkdir(parents=True, exist_ok=True)
        self.means_by_year.to_csv(directory / "means_by_year.csv")
        self.changes.to_csv(directory / "changes.csv")
        self.combined_scores().to_csv(directory / "combined_scores.csv")
        for name, table in self.scaled.items():
            table.to_csv(directory / f"scaled_{name}.csv")


class FeatureChangeAnalysis:
    """Absolute, relative and combined feature change between key years."""

    def __init__(self, years: list[int], absolute_weight: float = 0.5,
                 relative_weight: float = 0.5) -> None:
        if len(years) < 2:
            raise ValueError("Feature change needs at least two years")
        self.years = sorted(years)
        self.absolute_weight = absolute_weight
        self.relative_weight = relative_weight

    @property
    def comparisons(self) -> dict[str, tuple[int, int]]:
        """Consecutive key years, plus the full span when there are more than two."""
        pairs = list(zip(self.years[:-1], self.years[1:]))
        if len(self.years) > 2:
            pairs.append((self.years[0], self.years[-1]))
        return {f"{a}_to_{b}": (a, b) for a, b in pairs}

    def run(self, features: pd.DataFrame, years: pd.Series, group: str | None = None
            ) -> FeatureChangeResult:
        """``features`` are raw (unstandardised) values; ``years`` aligns with its rows."""
        years = years.loc[features.index]
        means = features.groupby(years).mean().T
        missing = [y for y in self.years if y not in means.columns]
        if missing:
            raise ValueError(f"No documents for key year(s) {missing}" + (f" in group {group}" if group else ""))

        changes = pd.DataFrame(index=pd.Index(features.columns, name="feature"))
        for year in self.years:
            changes[f"{year}_mean"] = means[year]
        for name, (a, b) in self.comparisons.items():
            change = means[b] - means[a]
            changes[f"change_{name}"] = change
            changes[f"pct_{name}"] = percent_change(means[a], change)

        mean_columns = [f"{y}_mean" for y in self.years]
        scaled = {}
        for name in self.comparisons:
            table = changes[mean_columns].copy()
            table["absolute_magnitude"] = changes[f"change_{name}"].abs()
            table["relative_magnitude"] = changes[f"pct_{name}"].abs()
            table["absolute_scaled"] = minmax_scale(table["absolute_magnitude"])
            table["relative_scaled"] = minmax_scale(table["relative_magnitude"])
            table["combined_score"] = (self.absolute_weight * table["absolute_scaled"]
                                       + self.relative_weight * table["relative_scaled"])
            scaled[name] = table
        return FeatureChangeResult(group=group, means_by_year=means, changes=changes, scaled=scaled)
