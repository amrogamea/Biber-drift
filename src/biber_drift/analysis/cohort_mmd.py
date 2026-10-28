"""Pairwise MMD^2 between cohorts of documents (years, or era x group combinations).

The procedure follows Nieth et al. (2026): every cohort is represented by one fixed sample of
``sample_size`` documents (their 600-document human subsample), and a pair's observed MMD^2 is
the statistic on the two fixed samples in full. Uncertainty comes from resampling those samples,
and chance level from baselines of two disjoint random halves of the same size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Hashable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from ..config import MMDConfig
from ..logging_utils import get_logger
from .mmd import RBFKernelMMD, confidence_interval, standardize
from .multiple_testing import adjust_pvalues
from .reference import FeatureReference

logger = get_logger(__name__)


def _summary(values: np.ndarray, confidence: float) -> dict[str, float]:
    lower, upper = confidence_interval(values, confidence)
    return {"mean_mmd2": float(values.mean()), "std_mmd2": float(values.std(ddof=1)),
            "ci_lower": lower, "ci_upper": upper}


@dataclass
class CohortMMDResult:
    """Everything a cohort comparison produces. Each table is saved as ``<prefix>_<table>.csv``."""

    pairwise: pd.DataFrame
    """One row per cohort pair: observed MMD^2 on the fixed samples, bootstrap CI, p-value (raw and
    adjusted over all pairs), effect."""
    baselines: pd.DataFrame
    """One row per cohort: within-cohort MMD^2 at ``sample_size`` (needs 2 x sample_size docs)."""
    pooled_baseline: dict[str, float]
    """MMD^2 between disjoint halves of the pooled corpus at ``sample_size`` (the paper's
    Human-Human baseline): the chance level every pair is compared against."""
    samples: pd.DataFrame
    """The fixed sample of every cohort: ``cohort``, ``doc_id``."""
    feature_shifts: pd.DataFrame
    """Per pair and feature: means (standardised units) and the marginal Wasserstein distance,
    computed on the full cohorts."""
    matrix: pd.DataFrame
    """Symmetric cohort x cohort matrix of observed MMD^2."""
    sigma: float
    sample_size: int
    cohort_counts: dict[Hashable, int] = field(default_factory=dict)

    def interpretation(self) -> pd.DataFrame:
        """Each pair against the pooled baseline and, where available, both cohorts' baselines."""
        has_upper = not self.baselines.empty and "ci_upper" in self.baselines.columns
        upper = self.baselines.set_index("cohort")["ci_upper"] if has_upper else pd.Series(dtype=float)
        rows = []
        for row in self.pairwise.itertuples(index=False):
            within = [upper.get(row.cohort_a, np.nan), upper.get(row.cohort_b, np.nan)]
            within_max = np.nan if np.isnan(within).any() else max(within)
            rows.append({
                "comparison": f"{row.cohort_a} -> {row.cohort_b}",
                "mmd2": row.observed_mmd2,
                "ci": f"[{row.ci_lower:.4f}, {row.ci_upper:.4f}]",
                "permutation_p_value": row.permutation_p_value,
                "p_adjusted": row.p_adjusted,
                "effect_vs_pooled_baseline": row.effect_vs_pooled_baseline,
                "pooled_baseline_ci_upper": self.pooled_baseline.get("ci_upper", np.nan),
                "above_pooled_baseline": row.observed_mmd2 > self.pooled_baseline.get("ci_upper", np.inf),
                "max_within_cohort_ci_upper": within_max,
                "above_within_cohort_baselines": (np.nan if np.isnan(within_max)
                                                  else bool(row.observed_mmd2 > within_max)),
            })
        return pd.DataFrame(rows)

    def save(self, directory: str | Path, prefix: str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.pairwise.to_csv(directory / f"{prefix}_pairwise.csv", index=False)
        self.baselines.to_csv(directory / f"{prefix}_baselines.csv", index=False)
        pd.DataFrame([self.pooled_baseline]).to_csv(directory / f"{prefix}_pooled_baseline.csv", index=False)
        self.samples.to_csv(directory / f"{prefix}_samples.csv", index=False)
        self.feature_shifts.to_csv(directory / f"{prefix}_feature_shifts.csv", index=False)
        self.matrix.to_csv(directory / f"{prefix}_matrix.csv")
        self.interpretation().to_csv(directory / f"{prefix}_interpretation.csv", index=False)


class CohortMMDAnalysis:
    """Compares every pair of cohorts with an RBF-kernel MMD^2 on fixed samples.

    1. Cohorts with fewer than ``min_cohort_documents`` (default ``sample_size``) are dropped, so
       every pair is compared at the same n -- biased MMD^2 shrinks with n.
    2. Features are z-scored and sigma set by the run's :class:`FeatureReference` (one scaler and
       one bandwidth fitted on the full corpus, as in the paper); without one, both are fitted on
       the cohorts compared.
    3. Each cohort gets ONE fixed sample of ``sample_size`` documents: the most representative of
       ``subsample_draws`` random draws (closest Biber-dimension distributions to the full
       cohort), or a single random draw.
    4. Per pair: observed MMD^2 on the two fixed samples in full; bootstrap CI from with-replacement
       resamples of the fixed samples; permutation p-value on the fixed samples; effect size
       ``(observed - pooled baseline mean) / pooled baseline sd``.
    5. Baselines: disjoint halves of the pooled corpus (always) and of each cohort (when it has
       ``2 x sample_size`` documents).
    """

    def __init__(self, config: MMDConfig, p_adjust: str = "fdr_bh") -> None:
        self.config = config
        self.p_adjust = p_adjust

    def run(self, features: pd.DataFrame, labels: pd.Series, order: Sequence[Hashable] | None = None,
            dimension_scores: pd.DataFrame | None = None, with_feature_shifts: bool = True,
            reference: FeatureReference | None = None, pool: pd.DataFrame | None = None
            ) -> CohortMMDResult:
        """Compare every pair of cohorts.

        Args:
            reference: the run's scaler and sigma. Without one, both are fitted on the cohorts
                compared here (numbers are then only comparable within this call).
            pool: documents for the pooled baseline (default: the cohorts compared). The pipeline
                passes the whole corpus, as the paper's human-human baseline uses the full corpus.
        """
        cfg = self.config
        n = cfg.sample_size
        rng = np.random.default_rng(cfg.seed)
        labels = labels.loc[features.index]
        order = list(order) if order is not None else sorted(labels.dropna().unique())

        members: dict[Hashable, pd.Index] = {}
        dropped = []
        for cohort in order:
            ids = labels.index[labels == cohort]
            if len(ids) < cfg.min_cohort_documents:
                dropped.append((cohort, len(ids)))
                continue
            members[cohort] = ids
        if dropped:
            logger.warning("Dropping cohorts with fewer than %d documents (sample_size %d): %s",
                           cfg.min_cohort_documents, n, dropped)
        names = [c for c in order if c in members]
        if len(names) < 2:
            raise ValueError(f"Need at least two cohorts with >= {cfg.min_cohort_documents} documents "
                             "for pairwise MMD; lower mmd.sample_size or mmd.min_cohort_documents.")
        counts = {name: len(members[name]) for name in names}

        used = pd.Index(np.concatenate([members[name] for name in names]))
        if reference is not None:
            z = reference.transform(features.loc[used])
            mmd = RBFKernelMMD(reference.sigma)
            pool_z = reference.transform(pool if pool is not None else features.loc[used]).to_numpy()
        else:
            z = pd.DataFrame(standardize(features.loc[used].to_numpy(dtype=float)),
                             index=used, columns=features.columns)
            mmd = RBFKernelMMD.from_median_heuristic(z.to_numpy(), rng, cfg.bandwidth_max_points)
            pool_z = z.to_numpy()
        logger.info("Cohorts: %s | fixed samples of %d | sigma %.4f", counts, n, mmd.sigma)

        samples = {name: self._fixed_sample(members[name], n, rng, dimension_scores) for name in names}
        sample_values = {name: z.loc[samples[name]].to_numpy() for name in names}

        if len(pool_z) >= 2 * n:
            pooled_baseline = {"documents": len(pool_z), "sample_size": n,
                               **_summary(mmd.baseline(pool_z, n, cfg.bootstrap_resamples, rng),
                                          cfg.confidence)}
        else:
            pooled_baseline = {"documents": len(pool_z), "sample_size": n}
            logger.warning("Pooled corpus has fewer than %d documents: no pooled baseline", 2 * n)

        baseline_rows = []
        for name in tqdm(names, desc="Within-cohort baselines"):
            row = {"cohort": name, "documents": counts[name], "sample_size": n, "sigma": mmd.sigma}
            if counts[name] >= 2 * n:
                values = mmd.baseline(z.loc[members[name]].to_numpy(), n, cfg.bootstrap_resamples, rng)
                row.update(_summary(values, cfg.confidence))
            baseline_rows.append(row)
        missing = [name for name in names if counts[name] < 2 * n]
        if missing:
            logger.info("No within-cohort baseline (needs %d documents) for %s; they are judged "
                        "against the pooled baseline only", 2 * n, missing)

        base_mean = pooled_baseline.get("mean_mmd2", np.nan)
        base_std = pooled_baseline.get("std_mmd2", np.nan)
        pair_rows, shift_rows = [], []
        feature_names = list(features.columns)
        for a, b in tqdm(list(combinations(names, 2)), desc="Pairwise MMD"):
            xa, xb = sample_values[a], sample_values[b]
            observed = mmd.mmd2(xa, xb)
            boot = mmd.bootstrap(xa, xb, n, cfg.bootstrap_resamples, rng)
            lower, upper = confidence_interval(boot, cfg.confidence)
            p_value = mmd.permutation_test(xa, xb, observed, cfg.permutations, rng)
            pair_rows.append({
                "cohort_a": a, "cohort_b": b, "documents_a": counts[a], "documents_b": counts[b],
                "sample_size": n, "observed_mmd2": observed,
                "bootstrap_mean_mmd2": boot.mean(), "bootstrap_std_mmd2": boot.std(ddof=1),
                "ci_lower": lower, "ci_upper": upper, "permutation_p_value": p_value,
                "effect_vs_pooled_baseline": (observed - base_mean) / base_std if base_std else np.nan,
                "sigma": mmd.sigma, "features": len(feature_names),
            })
            if with_feature_shifts:
                full_a, full_b = z.loc[members[a]].to_numpy(), z.loc[members[b]].to_numpy()
                for j, feature in enumerate(feature_names):
                    shift_rows.append({
                        "cohort_a": a, "cohort_b": b, "feature": feature,
                        "mean_a": full_a[:, j].mean(), "mean_b": full_b[:, j].mean(),
                        "mean_difference_b_minus_a": full_b[:, j].mean() - full_a[:, j].mean(),
                        "wasserstein_distance": wasserstein_distance(full_a[:, j], full_b[:, j]),
                    })

        pairwise = pd.DataFrame(pair_rows)
        pairwise.insert(pairwise.columns.get_loc("permutation_p_value") + 1, "p_adjusted",
                        adjust_pvalues(pairwise["permutation_p_value"], self.p_adjust))
        matrix = pd.DataFrame(0.0, index=names, columns=names)
        for row in pairwise.itertuples(index=False):
            matrix.loc[row.cohort_a, row.cohort_b] = row.observed_mmd2
            matrix.loc[row.cohort_b, row.cohort_a] = row.observed_mmd2
        sample_table = pd.DataFrame(
            [(name, doc_id) for name in names for doc_id in samples[name]], columns=["cohort", "doc_id"])

        return CohortMMDResult(
            pairwise=pairwise, baselines=pd.DataFrame(baseline_rows), pooled_baseline=pooled_baseline,
            samples=sample_table, feature_shifts=pd.DataFrame(shift_rows), matrix=matrix,
            sigma=mmd.sigma, sample_size=n, cohort_counts=counts,
        )

    def _fixed_sample(self, ids: pd.Index, n: int, rng: np.random.Generator,
                      dimension_scores: pd.DataFrame | None) -> pd.Index:
        """One fixed sample of ``n`` documents: representative (paper) or random."""
        if len(ids) == n:
            return ids
        if self.config.subsample == "random" or dimension_scores is None:
            return ids[rng.choice(len(ids), size=n, replace=False)]

        full = dimension_scores.loc[ids].to_numpy(dtype=float)
        best, best_score = None, np.inf
        for _ in range(self.config.subsample_draws):
            pick = rng.choice(len(ids), size=n, replace=False)
            score = np.mean([wasserstein_distance(full[:, d], full[pick, d]) for d in range(full.shape[1])])
            if score < best_score:
                best, best_score = pick, score
        return ids[best]
