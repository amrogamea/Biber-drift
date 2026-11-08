"""One scaler and one kernel bandwidth for a whole run -- the "ruler" every analysis measures with.

As in Nieth et al., the z-score mean / sd and the RBF bandwidth sigma are fitted ONCE on the full
corpus and reused everywhere (pairwise comparisons, their baselines, the sample-size curve and the
variance analysis). MMD^2 values depend on both, so only numbers measured with the same reference
(and the same sample size) can be compared.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .mmd import median_heuristic


@dataclass(frozen=True)
class FeatureReference:
    """Feature means / sds (population sd, as sklearn's StandardScaler) and the RBF bandwidth."""

    mean: pd.Series
    std: pd.Series
    sigma: float
    documents: int

    @classmethod
    def fit(cls, features: pd.DataFrame, rng: np.random.Generator, max_points: int = 5000) -> "FeatureReference":
        """Fit on ``features`` (rows = documents): z-score them, then the median heuristic."""
        values = features.to_numpy(dtype=float)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std == 0] = 1.0
        sigma = median_heuristic((values - mean) / std, rng, max_points)
        return cls(pd.Series(mean, index=features.columns), pd.Series(std, index=features.columns),
                   sigma, len(features))

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """z-scores of ``features`` on this reference (columns aligned by name)."""
        mean, std = self.mean.loc[features.columns], self.std.loc[features.columns]
        return (features.astype(float) - mean) / std

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"mean": self.mean, "std": self.std}).rename_axis("feature").to_csv(
            directory / "reference_scaler.csv")
        (directory / "reference.json").write_text(json.dumps(
            {"sigma": self.sigma, "documents": self.documents, "features": len(self.mean)}, indent=2))
