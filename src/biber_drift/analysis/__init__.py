"""Statistical analyses: cohort MMD, sample-size stability, feature change, variance, multiple-testing
correction and interrupted time series."""

from .cohort_mmd import CohortMMDAnalysis, CohortMMDResult
from .feature_change import FeatureChangeAnalysis, FeatureChangeResult, minmax_scale, percent_change
from .mmd import (
    RBFKernelMMD, bootstrap_mmd2, confidence_interval, median_heuristic, permutation_p_value,
    rbf_mmd2, standardize, within_group_baseline,
)
from .multiple_testing import P_ADJUST_METHODS, adjust_pvalues
from .reference import FeatureReference
from .stability import SampleSizeStability
from .trend import (
    InterruptedTimeSeries, TrendAnalysis, TrendResult, break_time, document_times,
    period_series,
)
from .variance import VarianceAnalysis, total_variance

__all__ = [
    "P_ADJUST_METHODS", "CohortMMDAnalysis", "CohortMMDResult",
    "FeatureChangeAnalysis", "FeatureChangeResult", "FeatureReference",
    "InterruptedTimeSeries", "RBFKernelMMD", "SampleSizeStability", "TrendAnalysis", "TrendResult",
    "VarianceAnalysis", "adjust_pvalues", "bootstrap_mmd2",
    "break_time", "confidence_interval", "document_times", "median_heuristic",
    "minmax_scale", "percent_change", "period_series", "permutation_p_value", "rbf_mmd2", "standardize",
    "total_variance", "within_group_baseline",
]
