"""RBF-kernel maximum mean discrepancy (MMD) and its resampling procedures.

Conventions (identical to the reference implementation used in Nieth et al.):

* kernel ``k(x, y) = exp(-||x - y||^2 / (2 sigma^2))``;
* bandwidth by the median heuristic, ``sigma = sqrt(median(||x_i - x_j||^2) / 2)`` over the
  positive pairwise distances of the (sub-sampled) pooled data;
* the biased (V-statistic) estimator of MMD^2, which includes the kernel diagonal;
* the within-group baseline draws two *disjoint* samples without replacement;
* the bootstrap draws each sample *with* replacement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(
        (a * a).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None, :] - 2.0 * a @ b.T,
        0.0,
    )


def rbf_mmd2(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """Biased squared MMD between samples ``x`` and ``y`` with an RBF kernel."""

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.exp(-_squared_distances(a, b) / (2.0 * sigma * sigma))

    return float(kernel(x, x).mean() - 2.0 * kernel(x, y).mean() + kernel(y, y).mean())


def median_heuristic(pooled: np.ndarray, rng: np.random.Generator, max_points: int = 5000) -> float:
    """Kernel bandwidth from the median pairwise squared distance (sub-sampled for speed)."""
    if len(pooled) > max_points:
        pooled = pooled[rng.choice(len(pooled), size=max_points, replace=False)]
    distances = _squared_distances(pooled, pooled)
    positive = distances[np.triu_indices_from(distances, k=1)]
    positive = positive[positive > 0]
    if not len(positive):
        raise ValueError("Cannot estimate an RBF bandwidth from identical feature vectors.")
    return float(np.sqrt(0.5 * np.median(positive)))


def confidence_interval(values: np.ndarray, confidence: float = 95.0) -> tuple[float, float]:
    """Percentile interval; ``confidence`` is in percent (95, not 0.95)."""
    if not 1 < confidence < 100:
        raise ValueError(f"confidence is a percentage such as 95, got {confidence}")
    alpha = (100 - confidence) / 2
    return float(np.percentile(values, alpha)), float(np.percentile(values, 100 - alpha))


def within_group_baseline(x: np.ndarray, sample_size: int, sigma: float, resamples: int,
                          rng: np.random.Generator, centering: bool = False) -> np.ndarray:
    """MMD^2 between two disjoint random halves of ``x`` (no replacement), ``resamples`` times.

    Both halves come from the same distribution, so these values show how much MMD^2 a sample
    of this size produces by chance. ``centering`` removes each half's mean first, isolating
    differences in spread from differences in location.
    """
    if 2 * sample_size > len(x):
        raise ValueError(f"Need at least {2 * sample_size} rows for disjoint samples, got {len(x)}")
    values = np.empty(resamples, dtype=float)
    for i in range(resamples):
        selected = rng.permutation(len(x))[: 2 * sample_size]
        a, b = x[selected[:sample_size]], x[selected[sample_size:]]
        if centering:
            a, b = a - a.mean(axis=0), b - b.mean(axis=0)
        values[i] = rbf_mmd2(a, b, sigma)
    return values


def bootstrap_mmd2(x: np.ndarray, y: np.ndarray, sample_size: int, sigma: float,
                   resamples: int, rng: np.random.Generator) -> np.ndarray:
    """MMD^2 between independent with-replacement samples of ``x`` and ``y``."""
    values = np.empty(resamples, dtype=float)
    for i in range(resamples):
        xs = x[rng.choice(len(x), size=sample_size, replace=True)]
        ys = y[rng.choice(len(y), size=sample_size, replace=True)]
        values[i] = rbf_mmd2(xs, ys, sigma)
    return values


def permutation_p_value(x: np.ndarray, y: np.ndarray, observed: float, sigma: float,
                        permutations: int, rng: np.random.Generator) -> float:
    """One-sided permutation p-value for ``observed`` MMD^2 (with the +1 correction)."""
    combined = np.vstack([x, y])
    n_x = len(x)
    null = np.empty(permutations, dtype=float)
    for i in range(permutations):
        shuffled = combined[rng.permutation(len(combined))]
        null[i] = rbf_mmd2(shuffled[:n_x], shuffled[n_x:], sigma)
    return float((1 + np.count_nonzero(null >= observed)) / (permutations + 1))


def standardize(values: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """z-scores against ``reference`` (default: ``values``); population sd, zero sd -> 1."""
    reference = values if reference is None else reference
    mean = reference.mean(axis=0)
    std = reference.std(axis=0)
    std[std == 0] = 1.0
    return (values - mean) / std


@dataclass
class RBFKernelMMD:
    """An RBF-kernel MMD with a fixed bandwidth."""

    sigma: float

    @classmethod
    def from_median_heuristic(cls, pooled: np.ndarray, rng: np.random.Generator,
                              max_points: int = 5000) -> "RBFKernelMMD":
        return cls(median_heuristic(pooled, rng, max_points))

    def mmd2(self, x: np.ndarray, y: np.ndarray) -> float:
        return rbf_mmd2(x, y, self.sigma)

    def baseline(self, x: np.ndarray, sample_size: int, resamples: int,
                 rng: np.random.Generator, centering: bool = False) -> np.ndarray:
        return within_group_baseline(x, sample_size, self.sigma, resamples, rng, centering)

    def bootstrap(self, x: np.ndarray, y: np.ndarray, sample_size: int, resamples: int,
                  rng: np.random.Generator) -> np.ndarray:
        return bootstrap_mmd2(x, y, sample_size, self.sigma, resamples, rng)

    def permutation_test(self, x: np.ndarray, y: np.ndarray, observed: float, permutations: int,
                         rng: np.random.Generator) -> float:
        return permutation_p_value(x, y, observed, self.sigma, permutations, rng)
