"""Interrupted time series on per-period means.

Every series is one number per period (a year, or a month for dated documents): the mean of a
standardised feature, of a Biber dimension score, or the total variance of the standardised
features. Around a fixed break (e.g. ChatGPT, Nov 2022), with ``t`` = time since the break and
``post`` = 1 from the break on, ``y = b0 + b1 t + b2 post (+ b3 post t)``. ``b2`` is the jump at the
break beyond the pre-break trend, ``b3`` the change of slope (fitted only with at least three
post-break periods). ``post_departure`` is the mean gap between the fitted post-break line and the
pre-break trend continued, over the post-break periods (``b2 + b3 * mean post-break t``): one number
for "how much did it change", counting jumps and bends alike. Placebo breaks inside the pre-break
period, fitted on pre-break data only, show how often a jump as large appears where nothing happened.

Periods are equally weighted (one observation each); with monthly data the standard errors are
Newey-West (HAC) to allow for autocorrelation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .multiple_testing import adjust_pvalues

_EPS = 1e-9


def break_time(break_at, time_unit: str) -> float:
    """The first post-break period in (fractional) years: 2023 -> 2023.0, '2022-12' -> 2022.917."""
    if time_unit == "year":
        return float(int(break_at))
    year, _, month = str(break_at).partition("-")
    return int(year) + (int(month or 1) - 1) / 12


def document_times(documents: pd.DataFrame, time_unit: str, date_column: str | None = None,
                   year_column: str = "year") -> pd.DataFrame:
    """Per document: ``period`` label and ``time`` (start of its period, in years). Undated rows are dropped."""
    if time_unit == "year":
        years = documents[year_column].astype(int)
        return pd.DataFrame({"period": years.astype(str), "time": years.astype(float)}, index=documents.index)
    dates = pd.to_datetime(documents[date_column], errors="coerce").dropna()
    return pd.DataFrame({"period": dates.dt.strftime("%Y-%m"),
                         "time": dates.dt.year + (dates.dt.month - 1) / 12}, index=dates.index)


def period_series(values: pd.DataFrame, times: pd.DataFrame, min_documents: int,
                  total_variance_of: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per period with at least ``min_documents``: ``time``, ``documents`` and each column's mean.

    ``total_variance_of`` (standardised features) adds a ``total_variance`` column: the sum of the
    per-feature variances inside each period.
    """
    periods = times["period"]
    grouped = values.loc[times.index].groupby(periods)
    table = pd.concat([times.groupby("period")["time"].first(), grouped.size().rename("documents"),
                       grouped.mean()], axis=1)
    if total_variance_of is not None:
        variance = total_variance_of.loc[times.index].groupby(periods).var(ddof=1).sum(axis=1)
        table["total_variance"] = variance
    table = table.sort_values("time")
    table.index.name = "period"
    return table[table["documents"] >= min_documents]


@dataclass
class InterruptedTimeSeries:
    """Segmented regression of one series around a fixed break."""

    break_time: float
    hac_lags: int = 0
    min_pre: int = 3
    min_segment: int = 2
    slope_min_post: int = 3
    confidence: float = 95.0

    def _fit(self, time: np.ndarray, y: np.ndarray, at: float, with_slope: bool):
        t = time - at
        post = (time >= at - _EPS).astype(float)
        columns = [np.ones_like(t), t, post] + ([post * t] if with_slope else [])
        names = ["const", "t", "post"] + (["post_t"] if with_slope else [])
        model = sm.OLS(y, pd.DataFrame(np.column_stack(columns), columns=names))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.hac_lags:
                return model.fit(cov_type="HAC", cov_kwds={"maxlags": self.hac_lags})
            return model.fit()

    def fit(self, time: np.ndarray, y: np.ndarray) -> tuple[dict | None, pd.DataFrame]:
        """The break's coefficients and its placebo breaks; ``None`` when a side has too few periods."""
        pre = time < self.break_time - _EPS
        n_pre, n_post = int(pre.sum()), int((~pre).sum())
        if n_pre < self.min_pre or n_post < 1:
            return None, pd.DataFrame()
        with_slope = n_post >= self.slope_min_post
        fit = self._fit(time, y, self.break_time, with_slope)
        alpha = 1 - self.confidence / 100
        ci = fit.conf_int(alpha)
        row = {
            "periods": len(y), "n_pre": n_pre, "n_post": n_post,
            "intercept": fit.params["const"], "pre_slope": fit.params["t"],
            "level_change": fit.params["post"], "level_se": fit.bse["post"],
            "level_ci_lower": ci.loc["post", 0], "level_ci_upper": ci.loc["post", 1],
            "level_t": fit.tvalues["post"], "level_p": fit.pvalues["post"],
            "slope_change": fit.params["post_t"] if with_slope else np.nan,
            "slope_p": fit.pvalues["post_t"] if with_slope else np.nan,
            "post_departure": fit.params["post"] + (
                fit.params["post_t"] * float(np.mean(time[~pre] - self.break_time)) if with_slope else 0.0),
            "r2": fit.rsquared,
        }
        placebos = self._placebos(time[pre], y[pre], with_slope)
        if len(placebos):
            beyond = np.count_nonzero(np.abs(placebos["level_t"]) >= abs(row["level_t"]))
            row["placebos"] = len(placebos)
            row["placebo_p"] = (1 + beyond) / (1 + len(placebos))
        else:
            row["placebos"], row["placebo_p"] = 0, np.nan
        return row, placebos

    def _placebos(self, time: np.ndarray, y: np.ndarray, with_slope: bool) -> pd.DataFrame:
        """Fake breaks inside the pre-break period, fitted on pre-break data only (same model form)."""
        rows = []
        need_after = max(self.min_segment, self.slope_min_post if with_slope else 1)
        for at in time:
            before = np.count_nonzero(time < at - _EPS)
            if before < max(self.min_pre, self.min_segment) or len(time) - before < need_after:
                continue
            fit = self._fit(time, y, at, with_slope)
            rows.append({"placebo_break": at, "level_change": fit.params["post"], "level_t": fit.tvalues["post"]})
        return pd.DataFrame(rows)


@dataclass
class TrendResult:
    series: pd.DataFrame
    its: pd.DataFrame
    placebos: pd.DataFrame

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.series.to_csv(directory / "series.csv")
        self.its.to_csv(directory / "its.csv", index=False)
        self.placebos.to_csv(directory / "its_placebos.csv", index=False)


@dataclass
class TrendAnalysis:
    """Interrupted time series for every series of a period table."""

    break_time: float
    hac_lags: int = 0
    min_segment: int = 2
    confidence: float = 95.0
    p_adjust: str = "fdr_bh"

    def run(self, table: pd.DataFrame, kinds: dict[str, str]) -> TrendResult:
        its = InterruptedTimeSeries(self.break_time, self.hac_lags, min_segment=self.min_segment,
                                    confidence=self.confidence)
        time = table["time"].to_numpy(dtype=float)
        its_rows, placebo_tables = [], []
        for name, kind in kinds.items():
            if name not in table:
                continue
            y = table[name].to_numpy(dtype=float)
            row, placebos = its.fit(time, y)
            if row is not None:
                its_rows.append({"series": name, "kind": kind, **row})
                placebo_tables.append(placebos.assign(series=name))
        its_table = pd.DataFrame(its_rows)
        if len(its_table):
            its_table["level_p_adjusted"] = adjust_pvalues(its_table["level_p"], self.p_adjust)
        return TrendResult(
            series=table, its=its_table,
            placebos=pd.concat(placebo_tables, ignore_index=True) if placebo_tables else pd.DataFrame(),
        )
