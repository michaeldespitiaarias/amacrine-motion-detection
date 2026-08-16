"""Multiple-comparison corrections shared across the inference modules.

Two corrections live here:

- ``bh_fdr`` — Benjamini-Hochberg FDR across every variable tested in one
  dataset's table.
- ``holm_adjust`` — Holm-Bonferroni step-down across the pairwise
  post-hoc comparisons of one repeated-measures family (see
  ``rm_anova.py``), used only when that family's Condition x Level
  interaction (or main effect) clears its own omnibus gate first —
  simple effects are never tested unguarded across every level.
"""

from __future__ import annotations

import numpy as np


def bh_fdr(p_values) -> list[float]:
    """Benjamini-Hochberg FDR-adjusted q-values.

    Corrects across the p-values passed in as one "family" — the caller
    decides what that family is (here: every variable tested in one
    dataset's contrast table). NaN entries pass through as NaN and do
    not participate in the correction (neither counted toward *m* nor
    assigned a rank). Monotonicity is enforced (q-values are
    non-decreasing when read in ascending p-value order), the standard
    BH step-up guarantee.

    Parameters
    ----------
    p_values : array-like
        Raw p-values, one per test in the family.

    Returns
    -------
    list[float]
        q-values in the same order and length as *p_values*; NaN where
        the input was NaN.
    """
    p_arr = np.array([
        float(p) if p is not None and not np.isnan(float(p)) else np.nan
        for p in p_values
    ])
    valid_mask = ~np.isnan(p_arr)
    valid_p = p_arr[valid_mask]
    m = len(valid_p)
    if m == 0:
        return list(p_arr)

    order = np.argsort(valid_p)
    ranked = valid_p[order]
    q_sorted = ranked * m / (np.arange(m) + 1)
    # Enforce monotonicity: right-to-left cumulative minimum.
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q_valid = np.empty(m)
    q_valid[order] = q_sorted
    q_full = np.full_like(p_arr, np.nan)
    q_full[valid_mask] = q_valid
    return list(q_full)


def holm_adjust(p_values) -> list[float]:
    """Holm-Bonferroni step-down correction.

    Less conservative than plain Bonferroni while still controlling the
    family-wise error rate exactly, no independence assumption needed.

    Parameters
    ----------
    p_values : array-like
        Raw p-values from one family of pairwise comparisons.

    Returns
    -------
    list[float]
        Adjusted p-values, same order and length as *p_values*, each
        capped at 1.0 and monotonically non-decreasing in rank order.
    """
    p_values = list(p_values)
    n = len(p_values)
    order = np.argsort(p_values)
    adj = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        p_adj = (n - rank) * p_values[idx]
        running_max = max(running_max, p_adj)
        adj[idx] = min(running_max, 1.0)
    return adj
