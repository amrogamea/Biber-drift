import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from biber_drift.analysis import (
    InterruptedTimeSeries, adjust_pvalues, break_time, document_times,
    period_series,
)
from biber_drift.config import ConfigError, TrendConfig
from biber_drift.pipeline import trend_panels


@pytest.mark.parametrize("method", ["fdr_bh", "holm"])
def test_adjusted_pvalues_match_statsmodels(rng, method):
    p = rng.uniform(0, 0.2, size=30)
    np.testing.assert_allclose(adjust_pvalues(p, method), multipletests(p, method=method)[1])


def test_adjustment_skips_missing_values():
    adjusted = adjust_pvalues([0.01, np.nan, 0.04], "fdr_bh")
    assert np.isnan(adjusted[1])
    np.testing.assert_allclose(adjusted[[0, 2]], [0.02, 0.04])


def test_break_time_and_document_times():
    assert break_time(2023, "year") == 2023.0
    assert break_time("2022-12", "month") == pytest.approx(2022 + 11 / 12)
    docs = pd.DataFrame({"year": [2022, 2023], "date": ["2022-12-30", "not a date"]})
    months = document_times(docs, "month", "date")
    assert list(months["period"]) == ["2022-12"]          # undated rows dropped
    assert list(document_times(docs, "year")["time"]) == [2022.0, 2023.0]


def test_period_series_means_counts_and_total_variance():
    times = pd.DataFrame({"period": ["a", "a", "b", "b", "b"], "time": [1.0, 1.0, 2.0, 2.0, 2.0]})
    values = pd.DataFrame({"x": [1.0, 3.0, 0.0, 0.0, 3.0]})
    table = period_series(values, times, min_documents=3, total_variance_of=values)
    assert list(table.index) == ["b"] and table.loc["b", "x"] == 1.0
    assert table.loc["b", "total_variance"] == pytest.approx(3.0)


def test_its_recovers_a_jump_and_its_placebos_stay_small(rng):
    time = np.arange(2010, 2025, dtype=float)
    y = 0.05 * (time - 2010) + 1.0 * (time >= 2021) + rng.normal(0, 0.05, len(time))
    row, placebos = InterruptedTimeSeries(2021.0).fit(time, y)
    assert row["level_change"] == pytest.approx(1.0, abs=0.2)
    assert row["level_p"] < 0.001 and not np.isnan(row["slope_change"])
    assert len(placebos) == row["placebos"] > 0
    assert row["placebo_p"] == pytest.approx(1 / (1 + len(placebos)))   # beats every placebo
    post_t = time[time >= 2021] - 2021
    assert row["post_departure"] == pytest.approx(row["level_change"] + row["slope_change"] * post_t.mean())


def test_its_fits_a_level_change_only_with_few_post_periods():
    time = np.arange(2013, 2025, dtype=float)
    row, _ = InterruptedTimeSeries(2023.0).fit(time, time * 0.1 + (time >= 2023))
    assert row["n_post"] == 2 and np.isnan(row["slope_change"])
    assert row["post_departure"] == row["level_change"]                    # no slope term: the jump
    assert InterruptedTimeSeries(2023.0).fit(time[:4], time[:4])[0] is None   # no post periods


def test_trend_config_defaults_and_validation():
    monthly = TrendConfig(time_unit="month", date_column="chapter_date", break_at="2022-12")
    assert (monthly.hac_lags, monthly.min_segment) == (12, 12)
    assert TrendConfig().min_segment == 2
    with pytest.raises(ConfigError):
        TrendConfig(time_unit="month")


def test_trend_panels_split_dimensions_from_the_most_changed_features():
    its = pd.DataFrame({
        "series": ["f_a", "f_b", "f_c", "Dim1", "Dim2", "total_variance"],
        "kind": ["feature", "feature", "feature", "dimension", "dimension", "diversity"],
        "level_change": [0.9, 0.1, -0.2, 1.0, 1.0, 5.0],
        "post_departure": [0.1, 0.5, -0.8, 1.0, 1.0, 5.0],     # f_b bends, f_c jumps and bends
    })
    diversity, features = trend_panels(its, n_features=2)
    assert diversity == ["total_variance", "Dim1", "Dim2"]
    assert features == ["f_c", "f_b"]                         # by |departure|, not by the jump
