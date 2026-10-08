"""Statistical analyses: cohort MMD, sample-size stability, feature change and variance."""

from .cohort_mmd import CohortMMDAnalysis, CohortMMDResult
from .feature_change import FeatureChangeAnalysis, FeatureChangeResult, minmax_scale, percent_change
from .mmd import (
    RBFKernelMMD, bootstrap_mmd2, confidence_interval, median_heuristic, permutation_p_value,
    rbf_mmd2, standardize, within_group_baseline,
)
from .reference import FeatureReference
from .stability import SampleSizeStability
from .variance import VarianceAnalysis, total_variance

__all__ = [
    "CohortMMDAnalysis", "CohortMMDResult", "FeatureChangeAnalysis", "FeatureChangeResult",
    "FeatureReference", "RBFKernelMMD", "SampleSizeStability", "VarianceAnalysis", "bootstrap_mmd2",
    "confidence_interval", "median_heuristic", "minmax_scale", "percent_change",
    "permutation_p_value", "rbf_mmd2", "standardize", "total_variance", "within_group_baseline",
]
