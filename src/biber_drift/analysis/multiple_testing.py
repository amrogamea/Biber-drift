"""Multiple-testing correction for families of p-values (e.g. every year pair, every feature)."""

from __future__ import annotations

import numpy as np

P_ADJUST_METHODS = ("fdr_bh", "holm", "none")


def adjust_pvalues(pvalues, method: str = "fdr_bh") -> np.ndarray:
    """Adjusted p-values for one family of tests; NaNs are left out of the family and kept as NaN.

    ``fdr_bh``: Benjamini-Hochberg false discovery rate (valid under positive dependence, as between
    correlated features); ``holm``: Holm's step-down family-wise error rate; ``none``: unchanged.
    """
    if method not in P_ADJUST_METHODS:
        raise ValueError(f"p-value adjustment must be one of {P_ADJUST_METHODS}, got {method!r}")
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    tested = ~np.isnan(p)
    values = p[tested]
    m = len(values)
    if method == "none" or m == 0:
        adjusted[tested] = values
        return adjusted
    order = np.argsort(values)
    ranked = values[order]
    if method == "fdr_bh":
        ranked = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    else:
        ranked = np.maximum.accumulate(ranked * (m - np.arange(m)))
    result = np.empty(m)
    result[order] = np.minimum(ranked, 1.0)
    adjusted[tested] = result
    return adjusted
