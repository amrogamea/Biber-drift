"""Total linguistic variability per cohort, as in Nieth et al.'s variance analysis.

Total variance = sum over features of each feature's variance (ddof 1) AFTER z-scoring against the
pooled corpus. Standardising first makes every feature count equally; on raw rates the sum is
dominated by the few high-frequency features (nouns, prepositions, past tense).

Because a sample variance is noisier for small samples, cohorts are also resampled at one common
size (``sample_size`` documents, without replacement) -- the paper's human-human variance baseline --
so cohorts of different sizes are compared on equal terms.
"""

from __future__ import annotations

from typing import Hashable, Mapping

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .mmd import confidence_interval

logger = get_logger(__name__)


def total_variance(values: np.ndarray) -> float:
    """Sum of the per-feature variances (ddof 1) of a documents x features matrix."""
    return float(np.var(values, axis=0, ddof=1).sum())


class VarianceAnalysis:
    """Total variance of standardised features, per cohort and for the pooled corpus."""

    def __init__(self, sample_size: int, resamples: int = 500, confidence: float = 95.0,
                 seed: int = 42) -> None:
        self.sample_size = sample_size
        self.resamples = resamples
        self.confidence = confidence
        self.seed = seed

    def resample(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = self.sample_size
        return np.array([total_variance(values[rng.permutation(len(values))[:n]])
                         for _ in range(self.resamples)])

    def run(self, standardized: pd.DataFrame, groups: Mapping[str, pd.Index], grouping: str
            ) -> pd.DataFrame:
        """``standardized``: features z-scored against the pooled corpus (rows = documents).
        ``groups``: cohort label -> its doc_ids. Returns one row per cohort plus the pooled corpus."""
        rng = np.random.default_rng(self.seed)
        n = self.sample_size
        rows = []
        cohorts: dict[Hashable, np.ndarray] = {"all documents": standardized.to_numpy()}
        cohorts.update({label: standardized.loc[ids].to_numpy() for label, ids in groups.items()})
        for label, values in cohorts.items():
            row = {"grouping": grouping, "cohort": label, "documents": len(values),
                   "features": values.shape[1], "sample_size": n,
                   "total_variance_all_documents": total_variance(values) if len(values) > 1 else np.nan}
            if len(values) >= n:
                draws = self.resample(values, rng)
                lower, upper = confidence_interval(draws, self.confidence)
                row.update({"mean_total_variance": draws.mean(), "std_total_variance": draws.std(ddof=1),
                            "ci_lower": lower, "ci_upper": upper})
            else:
                logger.info("[variance] %s has %d documents (< %d): no resampled estimate", label, len(values), n)
            rows.append(row)
        return pd.DataFrame(rows)
