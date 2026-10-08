"""How within-cohort MMD^2 shrinks with sample size -- used to pick ``mmd.sample_size``."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from ..config import StabilityConfig
from ..logging_utils import get_logger
from .mmd import RBFKernelMMD, confidence_interval, standardize

logger = get_logger(__name__)


class SampleSizeStability:
    """Within-group MMD^2 (disjoint random halves) across a range of sample sizes.

    Two samples from the same distribution should have MMD^2 near zero; the band shows how much
    MMD^2 a given sample size produces by chance. Pick a sample size past the "elbow", where
    the band has flattened.
    """

    def __init__(self, config: StabilityConfig, confidence: float = 95.0) -> None:
        self.config = config
        self.confidence = confidence

    def run(self, groups: Mapping[str, np.ndarray], grouping: str, sigma: float | None = None
            ) -> pd.DataFrame:
        """``groups`` maps a group label to its feature matrix. With ``scaling == 'pooled'`` the
        matrices are already z-scored on the run's reference and ``sigma`` is the run's bandwidth,
        so the curves sit on the same ruler as the cohort comparisons. With ``per_group`` each
        group is re-standardised and gets its own sigma (its own ruler). Returns one row per
        (group, sample size)."""
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        rows = []
        for label, values in groups.items():
            values = np.asarray(values, dtype=float)
            if len(values) < 4:
                logger.warning("[%s=%s] only %d documents -- skipped", grouping, label, len(values))
                continue
            if cfg.scaling == "per_group":
                values = standardize(values)
                mmd = RBFKernelMMD.from_median_heuristic(values, rng)
            elif sigma is not None:
                mmd = RBFKernelMMD(sigma)
            else:
                mmd = RBFKernelMMD.from_median_heuristic(values, rng)

            usable = [n for n in cfg.sample_sizes if n <= len(values) // 2]
            skipped = [n for n in cfg.sample_sizes if n > len(values) // 2]
            if skipped:
                logger.info("[%s=%s] %d documents -> skipping sample sizes %s",
                            grouping, label, len(values), skipped)
            for n in usable:
                baseline = mmd.baseline(values, n, cfg.resamples, rng)
                lower, upper = confidence_interval(baseline, self.confidence)
                rows.append({
                    "grouping": grouping, "group": label, "documents": len(values), "n": n,
                    "mean_mmd2": baseline.mean(), "ci_lower": lower, "ci_upper": upper,
                    "sigma": mmd.sigma,
                })
        return pd.DataFrame(rows)
