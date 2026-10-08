import numpy as np
import pandas as pd
import pytest
from scipy.stats import wasserstein_distance

from biber_drift.analysis import (
    CohortMMDAnalysis, RBFKernelMMD, VarianceAnalysis, confidence_interval, median_heuristic,
    permutation_p_value, rbf_mmd2, standardize, total_variance, within_group_baseline,
)
from biber_drift.config import MMDConfig


def brute_force_mmd2(x, y, sigma):
    k = lambda a, b: np.exp(-np.sum((a - b) ** 2) / (2 * sigma ** 2))
    xx = np.mean([k(a, b) for a in x for b in x])
    yy = np.mean([k(a, b) for a in y for b in y])
    xy = np.mean([k(a, b) for a in x for b in y])
    return xx + yy - 2 * xy


def test_mmd2_matches_brute_force(rng):
    x, y = rng.normal(size=(12, 3)), rng.normal(0.5, 1.0, size=(9, 3))
    assert rbf_mmd2(x, y, 1.3) == pytest.approx(brute_force_mmd2(x, y, 1.3))


def test_mmd2_is_zero_for_identical_samples_and_symmetric(rng):
    x, y = rng.normal(size=(20, 4)), rng.normal(size=(15, 4))
    assert rbf_mmd2(x, x, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert rbf_mmd2(x, y, 1.0) == pytest.approx(rbf_mmd2(y, x, 1.0))


def test_median_heuristic_matches_definition(rng):
    pooled = rng.normal(size=(50, 3))
    d2 = [np.sum((a - b) ** 2) for i, a in enumerate(pooled) for b in pooled[i + 1:]]
    expected = np.sqrt(0.5 * np.median(d2))
    assert median_heuristic(pooled, np.random.default_rng(0)) == pytest.approx(expected)


def test_baseline_draws_disjoint_halves_without_replacement(rng):
    x = rng.normal(size=(40, 2))
    values = within_group_baseline(x, 10, 1.0, 3, np.random.default_rng(7))
    check = np.random.default_rng(7)
    for value in values:
        selected = check.permutation(len(x))[:20]
        assert len(set(selected)) == 20
        assert value == pytest.approx(rbf_mmd2(x[selected[:10]], x[selected[10:]], 1.0))


def test_baseline_rejects_too_large_samples(rng):
    with pytest.raises(ValueError):
        within_group_baseline(rng.normal(size=(10, 2)), 6, 1.0, 1, rng)


def test_confidence_is_a_percentage():
    values = np.arange(101)
    assert confidence_interval(values, 95) == (2.5, 97.5)
    with pytest.raises(ValueError):
        confidence_interval(values, 0.95)


def test_permutation_test_detects_a_shift(rng):
    x, y = rng.normal(size=(40, 3)), rng.normal(2.0, 1.0, size=(40, 3))
    mmd = RBFKernelMMD.from_median_heuristic(np.vstack([x, y]), rng)
    p = permutation_p_value(x, y, mmd.mmd2(x, y), mmd.sigma, 99, rng)
    assert p == pytest.approx(1 / 100)


def _cohorts(rng, sizes, shifts):
    frames, labels = [], []
    for year, (n, shift) in enumerate(zip(sizes, shifts), start=2000):
        frames.append(rng.normal(shift, 1.0, size=(n, 4)))
        labels += [year] * n
    features = pd.DataFrame(np.vstack(frames), columns=[f"f_{i}" for i in range(4)])
    features.index = features.index.astype(str)
    return features, pd.Series(labels, index=features.index)


def _config(**kwargs):
    defaults = dict(sample_size=20, bootstrap_resamples=20, permutations=20, subsample="random", seed=1)
    return MMDConfig(**{**defaults, **kwargs})


def test_cohort_analysis_ranks_the_shifted_cohort_highest(rng):
    features, labels = _cohorts(rng, [60, 60, 60], [0.0, 0.0, 1.5])
    result = CohortMMDAnalysis(_config()).run(features, labels)

    assert result.sample_size == 20
    assert list(result.matrix.index) == [2000, 2001, 2002]
    pair = result.pairwise.set_index(["cohort_a", "cohort_b"])["observed_mmd2"]
    assert pair[(2000, 2002)] > pair[(2000, 2001)]
    assert pair[(2001, 2002)] > pair[(2000, 2001)]
    verdict = result.interpretation().set_index("comparison")
    assert verdict.loc["2000 -> 2002", "above_pooled_baseline"]
    assert verdict.loc["2000 -> 2002", "effect_vs_pooled_baseline"] > verdict.loc["2000 -> 2001", "effect_vs_pooled_baseline"]
    assert set(result.feature_shifts["feature"]) == set(features.columns)


def test_observed_value_is_the_statistic_on_the_full_fixed_samples(rng):
    features, labels = _cohorts(rng, [50, 70], [0.0, 0.5])
    result = CohortMMDAnalysis(_config()).run(features, labels)

    samples = result.samples.groupby("cohort")["doc_id"].apply(list)
    assert samples.map(len).tolist() == [20, 20]
    assert all(len(set(ids)) == 20 for ids in samples)
    z = pd.DataFrame(standardize(features.to_numpy()), index=features.index)
    expected = rbf_mmd2(z.loc[samples[2000]].to_numpy(), z.loc[samples[2001]].to_numpy(), result.sigma)
    assert result.pairwise["observed_mmd2"].iloc[0] == pytest.approx(expected)


def test_small_cohorts_are_dropped_not_shrinking_n_and_runs_are_reproducible(rng):
    features, labels = _cohorts(rng, [60, 30, 10], [0.0, 0.5, 0.0])
    first = CohortMMDAnalysis(_config()).run(features, labels)
    second = CohortMMDAnalysis(_config()).run(features, labels)
    assert list(first.cohort_counts) == [2000, 2001]          # 10 < sample_size: dropped
    assert first.sample_size == 20                             # n stays fixed
    baselines = first.baselines.set_index("cohort")
    assert pd.notna(baselines.loc[2000, "ci_upper"])           # 60 >= 2n
    assert pd.isna(baselines.loc[2001, "ci_upper"])            # 30 < 2n: pooled baseline only
    pd.testing.assert_frame_equal(first.pairwise, second.pairwise)


def test_representative_sample_matches_the_cohort_better_than_random(rng):
    features, labels = _cohorts(rng, [400, 400], [0.0, 0.0])
    dims = pd.DataFrame(rng.normal(size=(800, 3)), index=features.index)
    analysis = CohortMMDAnalysis(_config(subsample="representative", subsample_draws=200))
    ids = labels.index[labels == 2000]

    def distance(sample):
        return np.mean([wasserstein_distance(dims.loc[ids, d], dims.loc[sample, d]) for d in dims])

    chosen = analysis._fixed_sample(ids, 20, np.random.default_rng(0), dims)
    random_draws = [ids[np.random.default_rng(s).choice(len(ids), 20, replace=False)] for s in range(20)]
    assert distance(chosen) < np.median([distance(r) for r in random_draws])


def test_reference_is_the_pooled_scaler_and_median_heuristic(rng):
    from biber_drift.analysis import FeatureReference

    features, _ = _cohorts(rng, [80, 80], [0.0, 1.0])
    reference = FeatureReference.fit(features, np.random.default_rng(3))
    np.testing.assert_allclose(reference.transform(features).to_numpy(), standardize(features.to_numpy()))
    expected_sigma = median_heuristic(standardize(features.to_numpy()), np.random.default_rng(3))
    assert reference.sigma == pytest.approx(expected_sigma)


def test_cohort_analysis_measures_with_a_given_reference(rng):
    from biber_drift.analysis import FeatureReference

    features, labels = _cohorts(rng, [60, 60, 60], [0.0, 0.5, 1.0])
    reference = FeatureReference.fit(features, np.random.default_rng(0))     # all three cohorts
    two = labels[labels < 2002]
    result = CohortMMDAnalysis(_config()).run(features.loc[two.index], two, reference=reference,
                                             pool=features)
    assert result.sigma == reference.sigma                 # not refitted on the two cohorts
    assert result.pooled_baseline["documents"] == 180      # baseline drawn from the whole pool
    samples = result.samples.groupby("cohort")["doc_id"].apply(list)
    z = reference.transform(features)
    expected = rbf_mmd2(z.loc[samples[2000]].to_numpy(), z.loc[samples[2001]].to_numpy(), reference.sigma)
    assert result.pairwise["observed_mmd2"].iloc[0] == pytest.approx(expected)


def test_stability_uses_the_given_sigma(rng):
    from biber_drift.analysis import SampleSizeStability
    from biber_drift.config import StabilityConfig

    stability = SampleSizeStability(StabilityConfig(sample_sizes=[5, 10], resamples=3))
    table = stability.run({"all": standardize(rng.normal(size=(40, 3)))}, "all", sigma=1.234)
    assert (table["sigma"] == 1.234).all()


def test_config_rejects_cohort_minimum_below_sample_size():
    from biber_drift.config import ConfigError
    with pytest.raises(ConfigError):
        MMDConfig(sample_size=600, min_cohort_documents=100)


def test_total_variance_of_standardised_features(rng):
    values = standardize(rng.normal(size=(5000, 6)) * [1, 10, 100, 1, 1, 1])
    assert total_variance(values) == pytest.approx(6.0, rel=0.01)  # each feature counts once


def test_variance_analysis_resamples_at_one_size(rng):
    frame = pd.DataFrame(standardize(rng.normal(size=(300, 4))), index=[str(i) for i in range(300)])
    groups = {"a": frame.index[:200], "b": frame.index[200:230]}
    table = VarianceAnalysis(sample_size=50, resamples=30, seed=0).run(frame, groups, "year").set_index("cohort")
    assert list(table.index) == ["all documents", "a", "b"]
    assert pd.notna(table.loc["a", "ci_upper"]) and pd.isna(table.loc["b", "mean_total_variance"])
    assert table.loc["all documents", "total_variance_all_documents"] == pytest.approx(4.0, rel=0.01)
